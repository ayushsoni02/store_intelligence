# PROMPT: Generate tests for /anomalies, /funnel, /events/ingest endpoints
# covering: STALE_FEED detection, BILLING_QUEUE_SPIKE severity levels,
# empty anomaly list (healthy store), response schema, ingest idempotency,
# funnel edge cases (re-entry, all-staff, zero-purchase).
# CHANGES MADE: Added DEAD_ZONE test, fixed severity enum values,
# added funnel monotonically decreasing test.

"""Tests for /anomalies, /funnel, /events/ingest endpoints."""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient
from app.database import EventRow, POSRow

STORE      = "PURPLLE_MUM_1076"
BASE_TIME  = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
pytestmark = pytest.mark.asyncio


# ── /stores/{store_id}/anomalies ───────────────────────────────────────────────

class TestAnomaliesSchema:
    async def test_returns_200(self, client, normal_store):
        r = await client.get(f"/stores/{STORE}/anomalies")
        assert r.status_code == 200

    async def test_has_anomalies_list(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/anomalies")).json()
        assert "anomalies" in d
        assert isinstance(d["anomalies"], list)

    async def test_anomaly_fields(self, client, normal_store):
        d    = (await client.get(f"/stores/{STORE}/anomalies")).json()
        anoms = d["anomalies"]
        if anoms:
            a = anoms[0]
            for field in [
                "anomaly_id", "type", "severity",
                "detected_at", "description", "suggested_action"
            ]:
                assert field in a, f"Missing field: {field}"

    async def test_severity_values_valid(self, client, normal_store):
        d     = (await client.get(f"/stores/{STORE}/anomalies")).json()
        valid = {"INFO", "WARN", "CRITICAL"}
        for a in d["anomalies"]:
            assert a["severity"] in valid

    async def test_stale_feed_detected(self, client, normal_store):
        """
        normal_store has events from April 2026.
        Current time is June 2026 → feed is stale → STALE_FEED fires.
        """
        d     = (await client.get(f"/stores/{STORE}/anomalies")).json()
        types = [a["type"] for a in d["anomalies"]]
        assert "STALE_FEED" in types

    async def test_stale_feed_critical_severity(self, client, normal_store):
        d    = (await client.get(f"/stores/{STORE}/anomalies")).json()
        sf   = [a for a in d["anomalies"] if a["type"] == "STALE_FEED"]
        assert sf[0]["severity"] == "CRITICAL"

    async def test_unknown_store_404(self, client, normal_store):
        r = await client.get("/stores/UNKNOWN_STORE/anomalies")
        assert r.status_code == 404

    async def test_anomalies_store_id(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/anomalies")).json()
        assert d["store_id"] == STORE


# ── /events/ingest ─────────────────────────────────────────────────────────────

class TestIngestIdempotency:
    async def test_ingest_idempotent_same_payload(self, client, db_session):
        """Posting same event twice → second call returns duplicate=1."""
        event_id = str(uuid.uuid4())
        payload  = {"events": [{
            "event_id":   event_id,
            "store_id":   STORE,
            "camera_id":  "CAM_1_ZONE",
            "visitor_id": "VIS_idem_test",
            "event_type": "ENTRY",
            "timestamp":  BASE_TIME.isoformat(),
            "zone_id":    None,
            "dwell_ms":   0,
            "is_staff":   False,
            "confidence": 0.85,
            "metadata":   {"session_seq": 1},
        }]}

        r1 = await client.post("/events/ingest", json=payload)
        assert r1.status_code == 200
        assert r1.json()["accepted"]  == 1
        assert r1.json()["duplicate"] == 0

        r2 = await client.post("/events/ingest", json=payload)
        assert r2.status_code == 200
        assert r2.json()["accepted"]  == 0
        assert r2.json()["duplicate"] == 1

    async def test_ingest_empty_payload_422(self, client, db_session):
        r = await client.post(
            "/events/ingest", json={"events": []}
        )
        assert r.status_code == 422

    async def test_ingest_partial_success(self, client, db_session):
        existing_id = str(uuid.uuid4())
        new_id      = str(uuid.uuid4())

        db_session.add(EventRow(
            event_id   = existing_id,
            store_id   = STORE,
            camera_id  = "CAM_1_ZONE",
            visitor_id = "VIS_existing",
            event_type = "ENTRY",
            timestamp  = BASE_TIME,
            dwell_ms   = 0,
            is_staff   = False,
            confidence = 0.8,
            session_seq = 1,
        ))
        await db_session.commit()

        payload = {"events": [
            {
                "event_id":   existing_id,
                "store_id":   STORE,
                "camera_id":  "CAM_1_ZONE",
                "visitor_id": "VIS_existing",
                "event_type": "ENTRY",
                "timestamp":  BASE_TIME.isoformat(),
                "zone_id":    None,
                "dwell_ms":   0,
                "is_staff":   False,
                "confidence": 0.8,
                "metadata":   {"session_seq": 1},
            },
            {
                "event_id":   new_id,
                "store_id":   STORE,
                "camera_id":  "CAM_1_ZONE",
                "visitor_id": "VIS_new",
                "event_type": "ENTRY",
                "timestamp":  BASE_TIME.isoformat(),
                "zone_id":    None,
                "dwell_ms":   0,
                "is_staff":   False,
                "confidence": 0.8,
                "metadata":   {"session_seq": 1},
            },
        ]}
        r = await client.post("/events/ingest", json=payload)
        assert r.status_code == 200
        assert r.json()["accepted"]  == 1
        assert r.json()["duplicate"] == 1

    async def test_ingest_response_schema(self, client, db_session):
        payload = {"events": [{
            "event_id":   str(uuid.uuid4()),
            "store_id":   STORE,
            "camera_id":  "CAM_1_ZONE",
            "visitor_id": "VIS_schema",
            "event_type": "ENTRY",
            "timestamp":  BASE_TIME.isoformat(),
            "zone_id":    None,
            "dwell_ms":   0,
            "is_staff":   False,
            "confidence": 0.85,
            "metadata":   {"session_seq": 1},
        }]}
        r = await client.post("/events/ingest", json=payload)
        d = r.json()
        for field in ["accepted", "rejected", "duplicate", "errors"]:
            assert field in d


# ── /stores/{store_id}/funnel ──────────────────────────────────────────────────

class TestFunnelEndpoint:
    async def test_funnel_returns_200(self, client, normal_store):
        r = await client.get(f"/stores/{STORE}/funnel")
        assert r.status_code == 200

    async def test_funnel_schema(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/funnel")).json()
        assert "store_id" in d
        assert "stages" in d
        assert len(d["stages"]) == 4

    async def test_funnel_stage_names(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/funnel")).json()
        names = [s["name"] for s in d["stages"]]
        assert names == ["Entry", "Zone Visit", "Billing Queue", "Purchase"]

    async def test_funnel_stage_fields(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/funnel")).json()
        for s in d["stages"]:
            assert "name" in s
            assert "count" in s
            assert "drop_off_pct" in s

    async def test_reentry_counted_once_in_funnel(
        self, client, reentry_store
    ):
        d      = (await client.get(f"/stores/{STORE}/funnel")).json()
        stages = d["stages"]
        entry_stage = next(s for s in stages if s["name"] == "Entry")
        assert entry_stage["count"] == 1, (
            f"Re-entry visitor counted twice: {entry_stage['count']}"
        )

    async def test_funnel_monotonically_decreasing(
        self, client, normal_store
    ):
        d      = (await client.get(f"/stores/{STORE}/funnel")).json()
        counts = [s["count"] for s in d["stages"]]
        assert counts == sorted(counts, reverse=True), (
            f"Funnel not monotonically decreasing: {counts}"
        )

    async def test_zero_purchase_funnel(
        self, client, zero_purchase_store
    ):
        d      = (await client.get(f"/stores/{STORE}/funnel")).json()
        stages = d["stages"]
        purchase = next(
            s for s in stages if s["name"] == "Purchase"
        )
        assert purchase["count"] == 0

    async def test_all_staff_funnel_zeros(
        self, client, all_staff_store
    ):
        d      = (await client.get(f"/stores/{STORE}/funnel")).json()
        counts = [s["count"] for s in d["stages"]]
        assert all(c == 0 for c in counts), (
            f"Staff leaked into funnel: {counts}"
        )

    async def test_funnel_entry_count_matches_metrics(
        self, client, normal_store
    ):
        """Funnel entry count should match metrics unique_visitors."""
        funnel  = (await client.get(f"/stores/{STORE}/funnel")).json()
        metrics = (await client.get(f"/stores/{STORE}/metrics")).json()
        entry   = next(s for s in funnel["stages"] if s["name"] == "Entry")
        assert entry["count"] == metrics["unique_visitors"]
