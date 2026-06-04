"""
Event emitter: translates TrackState + tracker context → StoreEvent stream.

Emission rules per event type:
  ENTRY            → emitted once when a new track is confirmed on entry camera
                     OR when a new track appears on any camera (non-entry default)
  EXIT             → emitted once when a track moves to exited_tracks
  ZONE_ENTER       → emitted when current_zone changes to a new zone
  ZONE_EXIT        → emitted when current_zone changes away from prior zone
  ZONE_DWELL       → emitted every 30s of continuous zone occupancy
  BILLING_QUEUE_JOIN → emitted when track enters billing zone and queue_depth > 0
  BILLING_QUEUE_ABANDON → emitted when track exits billing zone with no
                          subsequent POS transaction in 5-minute window
  REENTRY          → emitted instead of ENTRY when is_reentry=True
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict

from app.models import StoreEvent, EventType, EventMetadata
from pipeline.tracker import TrackState, DirectionResult, classify_direction
from pipeline.detect import ENTRY_CAMERAS, BILLING_CAMERAS, ZONE_CAMERAS

# ── Constants ──────────────────────────────────────────────────────────────────

CANONICAL_STORE_ID = "PURPLLE_MUM_1076"

ZONE_PREFIX = "PURPLLE_MUM_1076"

# Zone IDs from canonical_zones.json
ZONE_IDS = {
    "Z01":      f"{ZONE_PREFIX}_Z01",
    "Z02":      f"{ZONE_PREFIX}_Z02",
    "Z03":      f"{ZONE_PREFIX}_Z03",
    "BILLING":  f"{ZONE_PREFIX}_Z_BILLING_01",
    "ENTRY":    "ENTRY_THRESHOLD",
}

# Cameras where ENTRY/EXIT direction from classify_direction() is inverted
INVERTED_ENTRY_CAMERAS = {"CAM_3B_ENTRY"}

# Dwell emission interval in seconds
DWELL_INTERVAL_SECS = 30.0

# Billing abandon window: seconds after billing zone exit to wait for POS match
BILLING_ABANDON_WINDOW_SECS = 300.0


# ── Zone assignment heuristic ──────────────────────────────────────────────────

def assign_zone(camera_id: str, cx_norm: float) -> str:
    """
    Assign a zone_id based on camera type and normalised centroid x.

    For zone cameras: divide frame into thirds horizontally.
    For billing cameras: always billing zone.
    For entry cameras: always entry threshold.
    """
    if camera_id in BILLING_CAMERAS:
        return ZONE_IDS["BILLING"]
    if camera_id in ENTRY_CAMERAS:
        return ZONE_IDS["ENTRY"]
    # Zone cameras — thirds heuristic
    if cx_norm < 0.33:
        return ZONE_IDS["Z01"]
    elif cx_norm < 0.66:
        return ZONE_IDS["Z02"]
    else:
        return ZONE_IDS["Z03"]


# ── Timestamp from frame ───────────────────────────────────────────────────────

def frame_to_timestamp(
    frame_idx: int,
    fps: float,
    clip_start_time: datetime,
) -> datetime:
    """Convert a frame index to a UTC datetime using clip start time + offset."""
    offset_secs = frame_idx / fps if fps > 0 else 0
    return clip_start_time + timedelta(seconds=offset_secs)


# ── Core event factories ───────────────────────────────────────────────────────

def _base_event(
    track: TrackState,
    event_type: EventType,
    timestamp: datetime,
    zone_id: Optional[str],
    dwell_ms: int,
    confidence: float,
    session_seq: int,
    metadata_extras: Optional[dict] = None,
) -> StoreEvent:
    """Build a StoreEvent from a TrackState. Internal helper."""
    allowed = {"queue_depth", "sku_zone", "session_seq", "direction", "reentry_count"}
    safe_extras = {k: v for k, v in (metadata_extras or {}).items() if k in allowed}
    meta = EventMetadata(
        sku_zone=zone_id,
        session_seq=session_seq,
        **safe_extras,
    )
    return StoreEvent(
        event_id=str(uuid.uuid4()),
        store_id=CANONICAL_STORE_ID,
        camera_id=track.camera_id,
        visitor_id=track.visitor_id,
        event_type=event_type,
        timestamp=timestamp,
        zone_id=zone_id,
        dwell_ms=dwell_ms,
        is_staff=bool(track.is_staff) if track.is_staff is not None else False,
        confidence=round(min(1.0, max(0.0, confidence)), 4),
        metadata=meta,
    )


# ── Session-level event emitter ────────────────────────────────────────────────

class SessionEventEmitter:
    """
    Stateful emitter for one camera session.
    Maintains per-visitor state needed for dwell timing,
    zone transitions, billing queue logic, and session_seq counters.

    One instance per camera per video run.
    """

    def __init__(
        self,
        camera_id: str,
        fps: float,
        clip_start_time: datetime,
        queue_depth_provider: Optional[callable] = None,
    ):
        self.camera_id         = camera_id
        self.fps               = fps
        self.clip_start_time   = clip_start_time
        # Optional callable: (frame_idx) → int, returns current billing queue depth
        self.queue_depth_provider = queue_depth_provider or (lambda _: 0)

        # Per-visitor tracking state
        self._session_seq:     dict[str, int]            = defaultdict(int)
        self._emitted_entry:   set[str]                  = set()
        self._emitted_exit:    set[str]                  = set()
        self._last_zone:       dict[str, Optional[str]]  = {}
        self._zone_entry_frame: dict[str, int]           = {}
        self._last_dwell_emit:  dict[str, float]         = {}
        self._billing_entry_frame: dict[str, int]        = {}
        self._pending_abandon:  dict[str, datetime]      = {}

        self.emitted_events: list[StoreEvent] = []

    def _next_seq(self, visitor_id: str) -> int:
        self._session_seq[visitor_id] += 1
        return self._session_seq[visitor_id]

    def _ts(self, frame_idx: int) -> datetime:
        return frame_to_timestamp(frame_idx, self.fps, self.clip_start_time)

    def _emit(self, event: StoreEvent) -> None:
        self.emitted_events.append(event)

    # ── Entry / Exit ───────────────────────────────────────────────────────────

    def maybe_emit_entry(
        self,
        track: TrackState,
        frame_idx: int,
        is_reentry: bool = False,
    ) -> None:
        """
        Emit ENTRY or REENTRY event for a track.
        Only emits once per visitor_id per session.
        For entry cameras: uses direction classification.
        For non-entry cameras: always emits ENTRY (visitor already inside).
        """
        vid = track.visitor_id
        if vid in self._emitted_entry:
            return

        confidence = 0.85  # default for non-entry camera appearances

        if self.camera_id in ENTRY_CAMERAS:
            direction_result = classify_direction(track)
            direction = direction_result.direction

            # Handle inversion
            if self.camera_id in INVERTED_ENTRY_CAMERAS:
                if direction == "ENTRY":
                    direction = "EXIT"
                elif direction == "EXIT":
                    direction = "ENTRY"

            if direction == "EXIT":
                # This is an exit event, not entry — handle in maybe_emit_exit
                return

            if direction == "UNKNOWN":
                confidence = direction_result.confidence * 0.5
                confidence = max(0.15, confidence)  # floor at 0.15, never zero

        event_type = EventType.REENTRY if is_reentry else EventType.ENTRY
        zone_id    = assign_zone(self.camera_id, 0.5)  # entry centroid default

        event = _base_event(
            track=track,
            event_type=event_type,
            timestamp=self._ts(track.first_seen_frame),
            zone_id=None,          # ENTRY/EXIT have null zone_id per spec
            dwell_ms=0,
            confidence=confidence,
            session_seq=self._next_seq(vid),
        )
        self._emit(event)
        self._emitted_entry.add(vid)

    def maybe_emit_exit(
        self,
        track: TrackState,
        frame_idx: int,
    ) -> None:
        """Emit EXIT event when track moves to exited state."""
        vid = track.visitor_id
        if vid in self._emitted_exit:
            return
        if not track.exited:
            return

        event = _base_event(
            track=track,
            event_type=EventType.EXIT,
            timestamp=self._ts(track.exit_frame or frame_idx),
            zone_id=None,
            dwell_ms=0,
            confidence=0.85,
            session_seq=self._next_seq(vid),
        )
        self._emit(event)
        self._emitted_exit.add(vid)

    # ── Zone transitions ───────────────────────────────────────────────────────

    def maybe_emit_zone_events(
        self,
        track: TrackState,
        frame_idx: int,
    ) -> None:
        """
        Check if visitor has changed zones or dwelled 30+ seconds.
        Emits ZONE_ENTER, ZONE_EXIT, ZONE_DWELL as appropriate.
        Skips for entry cameras (zone logic only on floor/billing).
        """
        vid = track.visitor_id
        if self.camera_id in ENTRY_CAMERAS:
            return
        if not track.last_centroid:
            return

        cx, cy = track.last_centroid
        new_zone = assign_zone(self.camera_id, cx)
        prev_zone = self._last_zone.get(vid)

        current_time_secs = frame_idx / self.fps

        # Zone transition
        if new_zone != prev_zone:
            # Emit ZONE_EXIT for previous zone
            if prev_zone is not None:
                entry_frame = self._zone_entry_frame.get(vid, frame_idx)
                dwell_ms = int(((frame_idx - entry_frame) / self.fps) * 1000)
                self._emit(_base_event(
                    track=track,
                    event_type=EventType.ZONE_EXIT,
                    timestamp=self._ts(frame_idx),
                    zone_id=prev_zone,
                    dwell_ms=dwell_ms,
                    confidence=0.80,
                    session_seq=self._next_seq(vid),
                ))

                # Check billing abandon
                if prev_zone == ZONE_IDS["BILLING"]:
                    self._pending_abandon[vid] = self._ts(frame_idx)

            # Emit ZONE_ENTER for new zone
            billing_queue_depth = 0
            extra = {}
            if new_zone == ZONE_IDS["BILLING"]:
                billing_queue_depth = self.queue_depth_provider(frame_idx)
                if billing_queue_depth > 0:
                    # Emit BILLING_QUEUE_JOIN instead of regular ZONE_ENTER
                    extra = {"queue_depth": billing_queue_depth}
                    self._emit(_base_event(
                        track=track,
                        event_type=EventType.BILLING_QUEUE_JOIN,
                        timestamp=self._ts(frame_idx),
                        zone_id=new_zone,
                        dwell_ms=0,
                        confidence=0.80,
                        session_seq=self._next_seq(vid),
                        metadata_extras=extra,
                    ))
                self._billing_entry_frame[vid] = frame_idx
                # Cancel any pending abandon for this visitor
                self._pending_abandon.pop(vid, None)

            self._emit(_base_event(
                track=track,
                event_type=EventType.ZONE_ENTER,
                timestamp=self._ts(frame_idx),
                zone_id=new_zone,
                dwell_ms=0,
                confidence=0.80,
                session_seq=self._next_seq(vid),
            ))

            self._last_zone[vid] = new_zone
            self._zone_entry_frame[vid] = frame_idx
            self._last_dwell_emit[vid] = current_time_secs

        # Dwell check — emit every 30 seconds of continuous occupancy
        else:
            last_dwell = self._last_dwell_emit.get(vid, current_time_secs)
            if current_time_secs - last_dwell >= DWELL_INTERVAL_SECS:
                entry_frame = self._zone_entry_frame.get(vid, frame_idx)
                dwell_ms = int(((frame_idx - entry_frame) / self.fps) * 1000)
                self._emit(_base_event(
                    track=track,
                    event_type=EventType.ZONE_DWELL,
                    timestamp=self._ts(frame_idx),
                    zone_id=new_zone,
                    dwell_ms=max(1, dwell_ms),  # must be > 0 per schema validator
                    confidence=0.75,
                    session_seq=self._next_seq(vid),
                ))
                self._last_dwell_emit[vid] = current_time_secs

    # ── Billing abandon ────────────────────────────────────────────────────────

    def flush_billing_abandons(
        self,
        pos_timestamps: list[datetime],
        flush_at_frame: int,
    ) -> None:
        """
        Called at end of video or periodically.
        For any visitor with a pending_abandon timestamp,
        check if a POS transaction occurred within BILLING_ABANDON_WINDOW_SECS.
        If not → emit BILLING_QUEUE_ABANDON.
        pos_timestamps: list of UTC datetimes from pos_transactions.csv
        """
        flush_time = self._ts(flush_at_frame)
        for vid, abandon_time in list(self._pending_abandon.items()):
            # Check if enough time has passed to make a verdict
            if (flush_time - abandon_time).total_seconds() < BILLING_ABANDON_WINDOW_SECS:
                continue  # too early to decide
            # Look for any POS transaction within window after billing exit
            window_end = abandon_time + timedelta(seconds=BILLING_ABANDON_WINDOW_SECS)
            matched = any(
                abandon_time <= ts <= window_end
                for ts in pos_timestamps
            )
            if not matched:
                # Find the track (check active and any available)
                # Emit with a synthetic TrackState stub if track is gone
                track_stub = TrackState(
                    track_id=-1,
                    visitor_id=vid,
                    camera_id=self.camera_id,
                    first_seen_frame=flush_at_frame,
                    last_seen_frame=flush_at_frame,
                    first_seen_time=flush_at_frame / self.fps,
                    centroid_history=__import__('collections').deque(),
                )
                self._emit(_base_event(
                    track=track_stub,
                    event_type=EventType.BILLING_QUEUE_ABANDON,
                    timestamp=abandon_time,
                    zone_id=ZONE_IDS["BILLING"],
                    dwell_ms=0,
                    confidence=0.70,
                    session_seq=self._next_seq(vid),
                ))
            del self._pending_abandon[vid]

    # ── Main update — called each frame ───────────────────────────────────────

    def process_frame(
        self,
        active_tracks: list[TrackState],
        exited_tracks: list[TrackState],
        frame_idx: int,
    ) -> list[StoreEvent]:
        """
        Process one frame's worth of tracker output.
        Returns list of newly emitted events this frame.

        Call this every frame inside your video processing loop.
        """
        before = len(self.emitted_events)

        # Process active tracks
        for track in active_tracks:
            self.maybe_emit_entry(track, frame_idx)
            self.maybe_emit_zone_events(track, frame_idx)

        # Process newly exited tracks
        for track in exited_tracks:
            self.maybe_emit_zone_events(track, frame_idx)
            self.maybe_emit_exit(track, frame_idx)

        return self.emitted_events[before:]
