# PROMPT: Generate tests for pipeline components: event schema
# validation, staff classifier, direction classification,
# visitor_id determinism, POS correlation logic.
# CHANGES MADE: Removed video I/O tests (too slow for CI),
# added BILLING_QUEUE_JOIN validator test missing from first draft,
# fixed TrackState deque import. Fixed pd.Timestamp tz parameter.

"""Tests for pipeline components (no video I/O)."""

import pytest
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from app.models import StoreEvent, EventType, IngestRequest
from pipeline.tracker import (
    TrackState, make_visitor_id, classify_direction, is_reentry,
    DirectionResult,
)
from pipeline.staff import (
    classify_staff, classify_staff_batch,
    _signal_dwell_duration, _signal_zone_breadth,
    _signal_movement_frequency, _signal_early_presence,
    _signal_aspect_consistency, _combine_signals,
    StaffSignal,
)
from pipeline.pos_correlator import (
    build_visitor_sessions, correlate_conversions,
    VisitorSession, compute_conversion_report,
)
import pandas as pd

BASE_TIME  = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
STORE      = "PURPLLE_MUM_1076"


# ── Schema validation ──────────────────────────────────────────────────────────

class TestEventSchema:
    def test_valid_entry_event(self):
        ev = StoreEvent(
            event_id   = str(uuid.uuid4()),
            store_id   = STORE,
            camera_id  = "CAM_3_ENTRY",
            visitor_id = "VIS_abc123",
            event_type = EventType.ENTRY,
            timestamp  = BASE_TIME,
            zone_id    = None,
            dwell_ms   = 0,
            is_staff   = False,
            confidence = 0.85,
            metadata   = {"session_seq": 1},
        )
        assert ev.event_type == EventType.ENTRY
        assert ev.zone_id is None

    def test_zone_dwell_requires_positive_dwell_ms(self):
        with pytest.raises(Exception):
            StoreEvent(
                event_id   = str(uuid.uuid4()),
                store_id   = STORE,
                camera_id  = "CAM_1_ZONE",
                visitor_id = "VIS_test",
                event_type = EventType.ZONE_DWELL,
                timestamp  = BASE_TIME,
                dwell_ms   = 0,
                is_staff   = False,
                confidence = 0.9,
                metadata   = {"session_seq": 1},
            )

    def test_billing_queue_join_requires_queue_depth(self):
        with pytest.raises(Exception):
            StoreEvent(
                event_id   = str(uuid.uuid4()),
                store_id   = STORE,
                camera_id  = "CAM_5_BILLING",
                visitor_id = "VIS_test",
                event_type = EventType.BILLING_QUEUE_JOIN,
                timestamp  = BASE_TIME,
                dwell_ms   = 0,
                is_staff   = False,
                confidence = 0.8,
                metadata   = {
                    "session_seq": 1,
                    "queue_depth": None,
                },
            )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(Exception):
            StoreEvent(
                event_id   = str(uuid.uuid4()),
                store_id   = STORE,
                camera_id  = "CAM_1_ZONE",
                visitor_id = "VIS_test",
                event_type = EventType.ENTRY,
                timestamp  = BASE_TIME,
                dwell_ms   = 0,
                is_staff   = False,
                confidence = 1.5,
                metadata   = {"session_seq": 1},
            )

    def test_invalid_uuid_event_id_rejected(self):
        with pytest.raises(Exception):
            StoreEvent(
                event_id   = "not-a-uuid",
                store_id   = STORE,
                camera_id  = "CAM_1_ZONE",
                visitor_id = "VIS_test",
                event_type = EventType.ENTRY,
                timestamp  = BASE_TIME,
                dwell_ms   = 0,
                is_staff   = False,
                confidence = 0.8,
                metadata   = {"session_seq": 1},
            )

    def test_ingest_request_empty_list_rejected(self):
        with pytest.raises(Exception):
            IngestRequest(events=[])

    def test_ingest_request_over_500_rejected(self):
        events = []
        for _ in range(501):
            events.append(StoreEvent(
                event_id   = str(uuid.uuid4()),
                store_id   = STORE,
                camera_id  = "CAM_1_ZONE",
                visitor_id = "VIS_x",
                event_type = EventType.ENTRY,
                timestamp  = BASE_TIME,
                dwell_ms   = 0,
                is_staff   = False,
                confidence = 0.8,
                metadata   = {"session_seq": 1},
            ))
        with pytest.raises(Exception):
            IngestRequest(events=events)

    def test_valid_zone_dwell_positive(self):
        """ZONE_DWELL with dwell_ms > 0 should pass."""
        ev = StoreEvent(
            event_id   = str(uuid.uuid4()),
            store_id   = STORE,
            camera_id  = "CAM_1_ZONE",
            visitor_id = "VIS_dwell",
            event_type = EventType.ZONE_DWELL,
            timestamp  = BASE_TIME,
            zone_id    = "Z01",
            dwell_ms   = 5000,
            is_staff   = False,
            confidence = 0.9,
            metadata   = {"session_seq": 1},
        )
        assert ev.dwell_ms == 5000

    def test_valid_billing_queue_join(self):
        """BILLING_QUEUE_JOIN with valid queue_depth should pass."""
        ev = StoreEvent(
            event_id   = str(uuid.uuid4()),
            store_id   = STORE,
            camera_id  = "CAM_5_BILLING",
            visitor_id = "VIS_bill",
            event_type = EventType.BILLING_QUEUE_JOIN,
            timestamp  = BASE_TIME,
            dwell_ms   = 0,
            is_staff   = False,
            confidence = 0.8,
            metadata   = {"session_seq": 1, "queue_depth": 3},
        )
        assert ev.metadata.queue_depth == 3

    def test_reentry_event_type(self):
        ev = StoreEvent(
            event_id   = str(uuid.uuid4()),
            store_id   = STORE,
            camera_id  = "CAM_3_ENTRY",
            visitor_id = "VIS_re",
            event_type = EventType.REENTRY,
            timestamp  = BASE_TIME,
            dwell_ms   = 0,
            is_staff   = False,
            confidence = 0.7,
            metadata   = {"session_seq": 2, "reentry_count": 1},
        )
        assert ev.event_type == EventType.REENTRY


# ── Tracker ────────────────────────────────────────────────────────────────────

class TestTracker:
    def test_visitor_id_deterministic(self):
        id1 = make_visitor_id(42, "CAM_3_ENTRY", "salt_abc")
        id2 = make_visitor_id(42, "CAM_3_ENTRY", "salt_abc")
        assert id1 == id2
        assert id1.startswith("VIS_")

    def test_visitor_id_unique_per_track(self):
        id1 = make_visitor_id(1, "CAM_3_ENTRY", "salt")
        id2 = make_visitor_id(2, "CAM_3_ENTRY", "salt")
        assert id1 != id2

    def test_visitor_id_unique_per_camera(self):
        id1 = make_visitor_id(1, "CAM_3_ENTRY", "salt")
        id2 = make_visitor_id(1, "CAM_5_BILLING", "salt")
        assert id1 != id2

    def test_visitor_id_unique_per_salt(self):
        id1 = make_visitor_id(1, "CAM_3_ENTRY", "salt_a")
        id2 = make_visitor_id(1, "CAM_3_ENTRY", "salt_b")
        assert id1 != id2

    def test_classify_direction_entry(self):
        track = TrackState(
            track_id=1, visitor_id="VIS_t",
            camera_id="CAM_3_ENTRY",
            first_seen_frame=0, last_seen_frame=30,
            first_seen_time=0.0,
            centroid_history=deque(maxlen=30),
        )
        for i in range(20):
            track.centroid_history.append((0.5, 0.2 + i * 0.03, i))
        result = classify_direction(track)
        assert result.direction == "ENTRY"
        assert result.confidence > 0

    def test_classify_direction_exit(self):
        track = TrackState(
            track_id=2, visitor_id="VIS_t2",
            camera_id="CAM_3_ENTRY",
            first_seen_frame=0, last_seen_frame=30,
            first_seen_time=0.0,
            centroid_history=deque(maxlen=30),
        )
        for i in range(20):
            track.centroid_history.append((0.5, 0.8 - i * 0.03, i))
        result = classify_direction(track)
        assert result.direction == "EXIT"

    def test_classify_direction_unknown_short_trajectory(self):
        track = TrackState(
            track_id=3, visitor_id="VIS_t3",
            camera_id="CAM_3_ENTRY",
            first_seen_frame=0, last_seen_frame=2,
            first_seen_time=0.0,
            centroid_history=deque([(0.5, 0.5, 0), (0.5, 0.51, 1)]),
        )
        result = classify_direction(track)
        assert result.direction == "UNKNOWN"

    def test_classify_direction_ambiguous(self):
        """Small delta → UNKNOWN."""
        track = TrackState(
            track_id=4, visitor_id="VIS_t4",
            camera_id="CAM_3_ENTRY",
            first_seen_frame=0, last_seen_frame=10,
            first_seen_time=0.0,
            centroid_history=deque(
                [(0.5, 0.5 + (i % 2) * 0.01, i) for i in range(10)]
            ),
        )
        result = classify_direction(track)
        assert result.direction == "UNKNOWN"

    def test_track_state_properties(self):
        track = TrackState(
            track_id=5, visitor_id="VIS_props",
            camera_id="CAM_1_ZONE",
            first_seen_frame=0, last_seen_frame=10,
            first_seen_time=0.0,
            centroid_history=deque([(0.3, 0.4, 0), (0.6, 0.7, 10)]),
        )
        assert track.first_centroid == (0.3, 0.4)
        assert track.last_centroid == (0.6, 0.7)

    def test_track_state_empty_centroid(self):
        track = TrackState(
            track_id=6, visitor_id="VIS_empty",
            camera_id="CAM_1_ZONE",
            first_seen_frame=0, last_seen_frame=0,
            first_seen_time=0.0,
            centroid_history=deque(),
        )
        assert track.first_centroid is None
        assert track.last_centroid is None

    def test_track_state_aspect_ratio(self):
        track = TrackState(
            track_id=7, visitor_id="VIS_ar",
            camera_id="CAM_1_ZONE",
            first_seen_frame=0, last_seen_frame=2,
            first_seen_time=0.0,
            centroid_history=deque([(0.5, 0.5, 0)]),
        )
        # Add bbox: x1, y1, x2, y2, frame
        track.bbox_history.append((100, 100, 150, 250, 0))  # w=50, h=150 → ratio=3.0
        ratios = track.aspect_ratio_history
        assert len(ratios) == 1
        assert abs(ratios[0] - 3.0) < 0.1

    def test_is_reentry_no_prior(self):
        result = is_reentry(99, {}, 100, 15.0)
        assert result is None

    def test_is_reentry_within_window(self):
        prior = TrackState(
            track_id=1, visitor_id="VIS_re",
            camera_id="CAM_3_ENTRY",
            first_seen_frame=0, last_seen_frame=50,
            first_seen_time=0.0,
            centroid_history=deque(),
        )
        prior.exited = True
        prior.exit_frame = 50
        result = is_reentry(1, {1: prior}, 100, 15.0)
        assert result is not None
        assert result.visitor_id == "VIS_re"

    def test_is_reentry_outside_window(self):
        prior = TrackState(
            track_id=1, visitor_id="VIS_old",
            camera_id="CAM_3_ENTRY",
            first_seen_frame=0, last_seen_frame=50,
            first_seen_time=0.0,
            centroid_history=deque(),
        )
        prior.exited = True
        prior.exit_frame = 50
        # 100000 frames later at 15fps = way past 120s window
        result = is_reentry(1, {1: prior}, 100000, 15.0)
        assert result is None


# ── Staff classifier ───────────────────────────────────────────────────────────

class TestStaffClassifier:
    def _make_track(self, first_frame, last_frame, vid="VIS_test"):
        return TrackState(
            track_id=1, visitor_id=vid,
            camera_id="CAM_1_ZONE",
            first_seen_frame=first_frame,
            last_seen_frame=last_frame,
            first_seen_time=first_frame / 15.0,
            centroid_history=deque(
                [(0.5, 0.5, i) for i in range(
                    min(30, max(1, last_frame - first_frame))
                )],
                maxlen=30
            ),
        )

    def test_long_duration_is_staff(self):
        track  = self._make_track(0, 9001)
        result = classify_staff(
            track, {"Z01", "Z02", "Z03"}, [1.7] * 50, fps=15.0
        )
        assert result.is_staff is True

    def test_short_duration_is_customer(self):
        track  = self._make_track(100, 400)
        result = classify_staff(
            track, {"Z01"}, [1.8] * 10, fps=15.0
        )
        assert result.is_staff is False

    def test_early_presence_alone_not_staff(self):
        track  = self._make_track(0, 30)
        result = classify_staff(
            track, {"Z01"}, [1.7] * 10, fps=15.0
        )
        assert result.is_staff is False

    def test_staff_classification_has_reason(self):
        track  = self._make_track(0, 9001)
        result = classify_staff(
            track, {"Z01", "Z02", "Z03"}, [1.7] * 50, fps=15.0
        )
        assert result.reason != ""
        assert result.combined_confidence > 0

    def test_signal_dwell_duration_fired(self):
        track = self._make_track(0, 9001)
        sig = _signal_dwell_duration(track, 15.0)
        assert sig.fired is True
        assert sig.confidence > 0

    def test_signal_dwell_duration_not_fired(self):
        track = self._make_track(0, 100)
        sig = _signal_dwell_duration(track, 15.0)
        assert sig.fired is False

    def test_signal_zone_breadth_fired(self):
        track = self._make_track(0, 9001)
        sig = _signal_zone_breadth(track, {"Z01", "Z02", "Z03"}, 15.0)
        assert sig.fired is True

    def test_signal_zone_breadth_not_fired_few_zones(self):
        track = self._make_track(0, 9001)
        sig = _signal_zone_breadth(track, {"Z01"}, 15.0)
        assert sig.fired is False

    def test_signal_early_presence(self):
        track = self._make_track(0, 100)
        sig = _signal_early_presence(track)
        assert sig.fired is True
        assert sig.confidence == 0.40

    def test_signal_early_presence_late(self):
        track = self._make_track(100, 200)
        sig = _signal_early_presence(track)
        assert sig.fired is False

    def test_signal_aspect_consistency_fired(self):
        sig = _signal_aspect_consistency([1.7] * 20)
        assert sig.fired is True

    def test_signal_aspect_consistency_too_few(self):
        sig = _signal_aspect_consistency([1.7, 1.8])
        assert sig.fired is False

    def test_signal_movement_frequency_short(self):
        track = self._make_track(0, 5)
        sig = _signal_movement_frequency(track, 15.0)
        assert sig.fired is False

    def test_combine_signals_no_fired(self):
        signals = [
            StaffSignal("A", False, 0.0, ""),
            StaffSignal("B", False, 0.0, ""),
        ]
        conf, reason = _combine_signals(signals)
        assert conf == 0.0

    def test_combine_signals_one_strong(self):
        signals = [
            StaffSignal("DWELL_DURATION", True, 0.85, "long"),
            StaffSignal("ASPECT_CONSISTENCY", True, 0.10, "ok"),
        ]
        conf, reason = _combine_signals(signals)
        assert conf >= 0.85

    def test_classify_staff_batch(self):
        track1 = self._make_track(0, 9001, "VIS_s1")
        track2 = self._make_track(100, 200, "VIS_c1")
        results = classify_staff_batch(
            [track1, track2],
            {"VIS_s1": {"Z01", "Z02", "Z03"}, "VIS_c1": {"Z01"}},
            {"VIS_s1": [1.7] * 50, "VIS_c1": [1.8] * 5},
            fps=15.0,
        )
        assert results["VIS_s1"].is_staff is True
        assert results["VIS_c1"].is_staff is False

    def test_signal_dwell_empty_centroid(self):
        track = TrackState(
            track_id=1, visitor_id="VIS_empty",
            camera_id="CAM_1_ZONE",
            first_seen_frame=0, last_seen_frame=0,
            first_seen_time=0.0,
            centroid_history=deque(),
        )
        sig = _signal_dwell_duration(track, 15.0)
        assert sig.fired is False


# ── POS Correlator ─────────────────────────────────────────────────────────────

class TestPOSCorrelator:
    def _make_events(self) -> list[StoreEvent]:
        events = []
        vid = "VIS_buyer_01"
        events.append(StoreEvent(
            event_id=str(uuid.uuid4()), store_id=STORE,
            camera_id="CAM_5_BILLING", visitor_id=vid,
            event_type=EventType.ENTRY, timestamp=BASE_TIME,
            dwell_ms=0, is_staff=False, confidence=0.9,
            metadata={"session_seq": 1},
        ))
        events.append(StoreEvent(
            event_id=str(uuid.uuid4()), store_id=STORE,
            camera_id="CAM_5_BILLING", visitor_id=vid,
            event_type=EventType.ZONE_ENTER,
            timestamp=BASE_TIME + timedelta(minutes=3),
            zone_id="PURPLLE_MUM_1076_Z_BILLING_01",
            dwell_ms=0, is_staff=False, confidence=0.9,
            metadata={"session_seq": 2},
        ))
        return events

    def test_build_visitor_sessions(self):
        events = self._make_events()
        sessions = build_visitor_sessions(events)
        assert "VIS_buyer_01" in sessions
        s = sessions["VIS_buyer_01"]
        assert s.entry_time == BASE_TIME
        assert s.billing_entry_time is not None

    def test_conversion_within_window(self):
        events   = self._make_events()
        sessions = build_visitor_sessions(events)
        pos_df   = pd.DataFrame([{
            "store_id":         STORE,
            "transaction_id":   "TXN_001",
            "timestamp":        pd.Timestamp(
                BASE_TIME + timedelta(minutes=5)
            ),
            "basket_value_inr": 1000.0,
        }])
        sessions = correlate_conversions(sessions, pos_df)
        assert sessions["VIS_buyer_01"].converted is True

    def test_no_conversion_outside_window(self):
        events   = self._make_events()
        sessions = build_visitor_sessions(events)
        pos_df   = pd.DataFrame([{
            "store_id":         STORE,
            "transaction_id":   "TXN_002",
            "timestamp":        pd.Timestamp(
                BASE_TIME + timedelta(hours=2)
            ),
            "basket_value_inr": 500.0,
        }])
        sessions = correlate_conversions(sessions, pos_df)
        assert sessions["VIS_buyer_01"].converted is False

    def test_staff_not_counted_as_conversion(self):
        events = self._make_events()
        staff_events = [
            e.model_copy(update={"is_staff": True}) for e in events
        ]
        sessions = build_visitor_sessions(staff_events)
        pos_df   = pd.DataFrame([{
            "store_id":         STORE,
            "transaction_id":   "TXN_003",
            "timestamp":        pd.Timestamp(
                BASE_TIME + timedelta(minutes=4)
            ),
            "basket_value_inr": 800.0,
        }])
        sessions = correlate_conversions(sessions, pos_df)
        assert sessions["VIS_buyer_01"].converted is False

    def test_empty_pos_no_conversions(self):
        events   = self._make_events()
        sessions = build_visitor_sessions(events)
        pos_df   = pd.DataFrame(columns=[
            "store_id", "transaction_id", "timestamp", "basket_value_inr"
        ])
        sessions = correlate_conversions(sessions, pos_df)
        assert sessions["VIS_buyer_01"].converted is False

    def test_abandonment_marked(self):
        events   = self._make_events()
        sessions = build_visitor_sessions(events)
        # Use a non-empty pos_df with timestamp far outside window
        # (empty DF triggers early return before abandonment marking)
        pos_df   = pd.DataFrame([{
            "store_id":         STORE,
            "transaction_id":   "TXN_FAR",
            "timestamp":        pd.Timestamp(
                BASE_TIME + timedelta(hours=24)
            ),
            "basket_value_inr": 100.0,
        }])
        sessions = correlate_conversions(sessions, pos_df)
        assert sessions["VIS_buyer_01"].abandoned_billing is True

    def test_compute_conversion_report(self):
        events   = self._make_events()
        sessions = build_visitor_sessions(events)
        pos_df   = pd.DataFrame([{
            "store_id":         STORE,
            "transaction_id":   "TXN_R01",
            "timestamp":        pd.Timestamp(
                BASE_TIME + timedelta(minutes=5)
            ),
            "basket_value_inr": 1200.0,
        }])
        sessions = correlate_conversions(sessions, pos_df)
        report   = compute_conversion_report(sessions, pos_df, STORE, events)
        assert report.total_visitors == 1
        assert report.converted_visitors == 1
        assert report.conversion_rate > 0
        assert report.total_basket_value == 1200.0

    def test_session_exit_time(self):
        events = self._make_events()
        events.append(StoreEvent(
            event_id=str(uuid.uuid4()), store_id=STORE,
            camera_id="CAM_3_ENTRY", visitor_id="VIS_buyer_01",
            event_type=EventType.EXIT,
            timestamp=BASE_TIME + timedelta(minutes=10),
            dwell_ms=0, is_staff=False, confidence=0.9,
            metadata={"session_seq": 3},
        ))
        sessions = build_visitor_sessions(events)
        assert sessions["VIS_buyer_01"].exit_time is not None

    def test_session_billing_exit(self):
        events = self._make_events()
        events.append(StoreEvent(
            event_id=str(uuid.uuid4()), store_id=STORE,
            camera_id="CAM_5_BILLING", visitor_id="VIS_buyer_01",
            event_type=EventType.ZONE_EXIT,
            timestamp=BASE_TIME + timedelta(minutes=8),
            zone_id="PURPLLE_MUM_1076_Z_BILLING_01",
            dwell_ms=300000, is_staff=False, confidence=0.9,
            metadata={"session_seq": 4},
        ))
        sessions = build_visitor_sessions(events)
        assert sessions["VIS_buyer_01"].billing_exit_time is not None
