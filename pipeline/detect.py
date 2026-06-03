"""
YOLOv8 detection layer.
Responsibilities:
  - Load YOLOv8n (nano) model — smallest variant, fast on CPU
  - Run inference on a single BGR frame (numpy array)
  - Return only person detections (COCO class 0)
  - Normalise bounding box to frame dimensions
  - Assign a per-detection confidence score
  - Flag detections below confidence threshold rather than dropping them
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO
import cv2

# ── Configuration ─────────────────────────────────────────────────────────────

YOLO_MODEL      = "yolov8n.pt"   # auto-downloads on first run
PERSON_CLASS_ID = 0              # COCO class 0 = person
DEFAULT_CONF    = 0.25           # detections below this are flagged, NOT dropped
IOU_THRESHOLD   = 0.45           # NMS IoU threshold

# ── Output schema ──────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Pixel-space bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    frame_w: int
    frame_h: int

    @property
    def cx_norm(self) -> float:
        """Normalised centroid x in [0, 1]."""
        return ((self.x1 + self.x2) / 2) / self.frame_w

    @property
    def cy_norm(self) -> float:
        """Normalised centroid y in [0, 1]."""
        return ((self.y1 + self.y2) / 2) / self.frame_h

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def aspect_ratio(self) -> float:
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return h / w if w > 0 else 0.0


@dataclass
class DetectedPerson:
    """Single person detection from one frame."""
    bbox: BoundingBox
    confidence: float
    low_confidence: bool        # True if below DEFAULT_CONF threshold
    frame_idx: int              # which frame this came from
    camera_id: str              # e.g. "CAM_3_ENTRY"
    track_id: Optional[int] = None   # populated by tracker in Phase 5
    is_staff: Optional[bool] = None  # populated by staff classifier in Phase 7


# ── Model loader (singleton) ───────────────────────────────────────────────────

_model_cache: dict[str, YOLO] = {}

def load_model(model_path: str = YOLO_MODEL) -> YOLO:
    """
    Load YOLOv8 model. Cache it in module-level dict to avoid
    reloading on every call. Thread-safe for single-process use.
    """
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


# ── Core detection function ────────────────────────────────────────────────────

def detect_persons(
    frame: np.ndarray,
    frame_idx: int,
    camera_id: str,
    conf_threshold: float = DEFAULT_CONF,
    model_path: str = YOLO_MODEL,
) -> list[DetectedPerson]:
    """
    Run YOLOv8 inference on a single BGR frame.
    Returns ALL person detections — low-confidence ones are flagged,
    not filtered out. Caller decides what to do with low-conf detections.

    Args:
        frame:          BGR numpy array (H, W, 3)
        frame_idx:      frame number in the source video
        camera_id:      camera identifier string
        conf_threshold: detections below this set low_confidence=True
        model_path:     YOLO weights file

    Returns:
        List of DetectedPerson, sorted by confidence descending.
        Empty list if frame is None or contains no persons.
    """
    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    model = load_model(model_path)

    results = model(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=0.1,           # intentionally lower than threshold — we flag, not drop
        iou=IOU_THRESHOLD,
        verbose=False,
    )

    detections: list[DetectedPerson] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp to frame bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                frame_w=w, frame_h=h
            )

            detections.append(DetectedPerson(
                bbox=bbox,
                confidence=round(conf, 4),
                low_confidence=(conf < conf_threshold),
                frame_idx=frame_idx,
                camera_id=camera_id,
            ))

    return sorted(detections, key=lambda d: d.confidence, reverse=True)


# ── Frame extractor ────────────────────────────────────────────────────────────

CAMERA_ID_MAP = {
    "CAM 1 - zone.mp4":    "CAM_1_ZONE",
    "CAM 2 - zone.mp4":    "CAM_2_ZONE",
    "CAM 3 - entry.mp4":   "CAM_3_ENTRY",
    "CAM 5 - billing.mp4": "CAM_5_BILLING",
    "billing_area.mp4":    "CAM_4_BILLING_AREA",
    "entry 2.mp4":         "CAM_3B_ENTRY",
    "zone.mp4":            "CAM_0_ZONE",
    "entry 1.mp4":         "CAM_3C_ENTRY",
}

ENTRY_CAMERAS   = {"CAM_3_ENTRY", "CAM_3B_ENTRY", "CAM_3C_ENTRY"}
BILLING_CAMERAS = {"CAM_5_BILLING", "CAM_4_BILLING_AREA"}
ZONE_CAMERAS    = {"CAM_1_ZONE", "CAM_2_ZONE", "CAM_0_ZONE"}

def extract_frame(video_path: str, frame_idx: int = 0) -> Optional[np.ndarray]:
    """
    Extract a single frame from a video file by frame index.
    Returns BGR numpy array or None if extraction fails.
    Does NOT hold the video open — opens, seeks, reads, closes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def get_camera_id(video_filename: str) -> str:
    """
    Map a video filename to its camera_id string.
    Falls back to uppercased stem if not in CAMERA_ID_MAP.
    """
    from pathlib import Path
    name = Path(video_filename).name
    return CAMERA_ID_MAP.get(name, Path(video_filename).stem.upper().replace(" ", "_"))
