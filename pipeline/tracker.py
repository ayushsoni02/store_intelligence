"""
ByteTrack-based multi-object tracker with visitor_id assignment.

Design:
  - Uses ultralytics built-in ByteTrack (no separate install needed)
  - One PersonTracker instance per camera per video processing run
  - Assigns stable visitor_id tokens (VIS_xxxxxx) to track IDs
  - Tracks centroid history per visitor for zone assignment
  - Detects ENTRY/EXIT direction at entry cameras via
    vertical centroid movement (top→bottom = entry, bottom→top = exit)
  - Detects re-entry: same physical appearance after a prior EXIT
    (approximated by track_id reappearance within session window)
"""

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
from ultralytics import YOLO

from pipeline.detect import (
    DetectedPerson, BoundingBox, detect_persons,
    ENTRY_CAMERAS, BILLING_CAMERAS, ZONE_CAMERAS,
    YOLO_MODEL, DEFAULT_CONF,
)

# ── Configuration ──────────────────────────────────────────────────────────────

ENTRY_LINE_Y_NORM   = 0.5    # normalised Y threshold for entry/exit classification
                              # persons crossing from cy < 0.5 → cy > 0.5 = ENTRY
                              # persons crossing from cy > 0.5 → cy < 0.5 = EXIT
                              # Adjust if your entry camera is oriented differently

REENTRY_WINDOW_SECS = 120    # seconds: if same track_id reappears within this
                              # window after an EXIT, it is a REENTRY not new ENTRY

TRAJECTORY_BUFFER   = 30     # number of recent centroids to keep per track

MIN_ASPECT_RATIO    = 1.1    # filter out detections with aspect < this
                              # removes boxes that are wider than tall (carts, shelves)

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Live state for one tracked person."""
    track_id: int
    visitor_id: str
    camera_id: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float           # wall-clock time (or frame-derived)
    centroid_history: deque          # deque of (cx_norm, cy_norm, frame_idx)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=200))
    current_zone: Optional[str] = None
    zone_entry_frame: Optional[int] = None
    is_staff: Optional[bool] = None
    exited: bool = False             # True once EXIT event has been emitted
    exit_frame: Optional[int] = None

    @property
    def last_centroid(self) -> Optional[tuple[float, float]]:
        if self.centroid_history:
            cx, cy, _ = self.centroid_history[-1]
            return cx, cy
        return None

    @property
    def first_centroid(self) -> Optional[tuple[float, float]]:
        if self.centroid_history:
            cx, cy, _ = self.centroid_history[0]
            return cx, cy
        return None

    @property
    def aspect_ratio_history(self) -> list[float]:
        """Compute aspect ratios from bbox_history. h/w for each box."""
        ratios = []
        for x1, y1, x2, y2, _ in self.bbox_history:
            w = x2 - x1
            h = y2 - y1
            if w > 0:
                ratios.append(h / w)
        return ratios


@dataclass
class DirectionResult:
    """Result of entry/exit direction classification."""
    direction: str          # "ENTRY", "EXIT", or "UNKNOWN"
    confidence: float       # 0.0–1.0 based on trajectory clarity
    cy_start: float
    cy_end: float


# ── visitor_id generator ───────────────────────────────────────────────────────

def make_visitor_id(track_id: int, camera_id: str, session_salt: str) -> str:
    """
    Generate a stable VIS_xxxxxx token from track_id + camera + salt.
    Same track_id + camera in the same session always maps to the same visitor_id.
    Different sessions (different video runs) get different visitor_ids
    because session_salt changes.
    """
    raw = f"{track_id}:{camera_id}:{session_salt}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"VIS_{digest}"


# ── Direction classifier ───────────────────────────────────────────────────────

def classify_direction(
    track: TrackState,
    entry_line_y: float = ENTRY_LINE_Y_NORM,
) -> DirectionResult:
    """
    Classify movement direction using centroid trajectory.

    Logic:
      - Take first and last centroid in history
      - If cy moved from < entry_line_y to > entry_line_y → ENTRY (moving inward)
      - If cy moved from > entry_line_y to < entry_line_y → EXIT (moving outward)
      - If trajectory is too short or ambiguous → UNKNOWN

    Only meaningful for ENTRY_CAMERAS. For other cameras always returns UNKNOWN.
    """
    if len(track.centroid_history) < 3:
        return DirectionResult("UNKNOWN", 0.0, 0.0, 0.0)

    centroids = list(track.centroid_history)
    cy_values = [c[1] for c in centroids]
    cy_start  = np.mean(cy_values[:3])    # average of first 3 frames
    cy_end    = np.mean(cy_values[-3:])   # average of last 3 frames
    delta     = cy_end - cy_start

    # Require minimum movement to avoid noise
    if abs(delta) < 0.08:
        return DirectionResult("UNKNOWN", abs(delta) / 0.08, cy_start, cy_end)

    direction  = "ENTRY" if delta > 0 else "EXIT"
    confidence = min(1.0, abs(delta) / 0.3)
    return DirectionResult(direction, confidence, cy_start, cy_end)


# ── Re-entry detector ──────────────────────────────────────────────────────────

def is_reentry(
    track_id: int,
    exited_tracks: dict[int, TrackState],
    current_frame: int,
    fps: float,
    window_secs: float = REENTRY_WINDOW_SECS,
) -> Optional[TrackState]:
    """
    Check if a newly appeared track_id matches a recently exited track.
    Returns the prior TrackState if re-entry detected, else None.

    Matching logic: same track_id reappears within window_secs worth of frames.
    Note: ByteTrack may reassign track_ids after long gaps — this is a known
    limitation documented in .ai_history.log.
    """
    if track_id not in exited_tracks:
        return None
    prior = exited_tracks[track_id]
    if prior.exit_frame is None:
        return None
    frames_since_exit = current_frame - prior.exit_frame
    secs_since_exit   = frames_since_exit / fps if fps > 0 else 999
    if secs_since_exit <= window_secs:
        return prior
    return None


# ── Main tracker class ─────────────────────────────────────────────────────────

class PersonTracker:
    """
    Stateful multi-person tracker for one camera's video stream.

    Usage:
        tracker = PersonTracker(camera_id="CAM_3_ENTRY", fps=15.0)
        for frame_idx, frame in enumerate(frames):
            results = tracker.update(frame, frame_idx)
            # results: list of TrackState with updated centroid history
    """

    def __init__(
        self,
        camera_id: str,
        fps: float = 15.0,
        model_path: str = YOLO_MODEL,
        conf_threshold: float = DEFAULT_CONF,
        session_salt: Optional[str] = None,
    ):
        self.camera_id      = camera_id
        self.fps            = fps
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.session_salt   = session_salt or str(time.time())

        # Load model with tracking enabled
        self._model = YOLO(model_path)

        # Active tracks: track_id → TrackState
        self.active_tracks:  dict[int, TrackState] = {}
        # Exited tracks: track_id → TrackState (for re-entry detection)
        self.exited_tracks:  dict[int, TrackState] = {}
        # All visitor_ids ever assigned this session
        self.visitor_registry: dict[str, str] = {}  # visitor_id → track_id str

        self._frame_count = 0

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> list[TrackState]:
        """
        Process one frame. Returns list of currently active TrackStates.

        Steps:
          1. Run YOLOv8 + ByteTrack on frame
          2. Filter to person class, apply aspect ratio filter
          3. For each tracked box:
             a. If track_id is new → check re-entry → assign visitor_id
             b. Update centroid history
             c. Update last_seen_frame
          4. Mark tracks not seen this frame as potentially exited
             (caller / emit.py decides when to emit EXIT event)
        """
        if frame is None or frame.size == 0:
            return list(self.active_tracks.values())

        self._frame_count += 1
        h, w = frame.shape[:2]

        # Run ByteTrack via ultralytics persist=True
        results = self._model.track(
            frame,
            classes=[0],          # person only
            conf=0.1,
            iou=0.45,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        seen_track_ids: set[int] = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Skip if no track_id assigned yet
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Clamp to frame
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                # Aspect ratio filter — skip implausible boxes
                aspect = bh / bw
                if aspect < MIN_ASPECT_RATIO:
                    continue

                cx_norm = ((x1 + x2) / 2) / w
                cy_norm = ((y1 + y2) / 2) / h

                seen_track_ids.add(track_id)

                if track_id not in self.active_tracks:
                    # Check re-entry before assigning new visitor_id
                    prior = is_reentry(
                        track_id, self.exited_tracks,
                        frame_idx, self.fps
                    )
                    if prior:
                        # Re-entry: reuse same visitor_id
                        visitor_id = prior.visitor_id
                        prior.exited = False
                        prior.exit_frame = None
                    else:
                        visitor_id = make_visitor_id(
                            track_id, self.camera_id, self.session_salt
                        )

                    self.active_tracks[track_id] = TrackState(
                        track_id=track_id,
                        visitor_id=visitor_id,
                        camera_id=self.camera_id,
                        first_seen_frame=frame_idx,
                        last_seen_frame=frame_idx,
                        first_seen_time=frame_idx / self.fps,
                        centroid_history=deque(maxlen=TRAJECTORY_BUFFER),
                    )

                track = self.active_tracks[track_id]
                track.last_seen_frame = frame_idx
                track.centroid_history.append((cx_norm, cy_norm, frame_idx))
                track.bbox_history.append((x1, y1, x2, y2, frame_idx))

        # Mark tracks not seen this frame
        lost_ids = set(self.active_tracks.keys()) - seen_track_ids
        for lost_id in lost_ids:
            lost_track = self.active_tracks[lost_id]
            # Only move to exited if not seen for >1 second (fps frames)
            frames_missing = frame_idx - lost_track.last_seen_frame
            if frames_missing > int(self.fps):
                lost_track.exited = True
                lost_track.exit_frame = lost_track.last_seen_frame
                self.exited_tracks[lost_id] = lost_track
                del self.active_tracks[lost_id]

        return list(self.active_tracks.values())

    def get_all_tracks(self) -> list[TrackState]:
        """Return active + exited tracks for end-of-video processing."""
        return list(self.active_tracks.values()) + list(self.exited_tracks.values())
