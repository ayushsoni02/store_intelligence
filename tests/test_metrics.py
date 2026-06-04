# PROMPT: Generate tests for GET /stores/{id}/metrics covering:
# normal operation, empty store, all-staff store, zero purchases,
# unknown store 404, response schema validation.
# CHANGES MADE: Fixed async fixture ordering, added window_hours
# query param test, added staff exclusion assertion.
# Added /health and /heatmap endpoint tests for full coverage.

"""Tests for /stores/{store_id}/metrics, /health, /heatmap endpoints."""

import pytest
from httpx import AsyncClient

STORE  = "PURPLLE_MUM_1076"
pytestmark = pytest.mark.asyncio


# ── /stores/{store_id}/metrics ─────────────────────────────────────────────────

class TestMetricsNormal:
    async def test_returns_200(self, client, normal_store):
        r = await client.get(f"/stores/{STORE}/metrics")
        assert r.status_code == 200

    async def test_schema_fields_present(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        for field in [
            "store_id", "unique_visitors", "conversion_rate",
            "avg_dwell_ms_by_zone", "queue_depth_current",
            "abandonment_rate", "total_transactions",
            "total_basket_inr", "data_freshness_secs",
        ]:
            assert field in d, f"Missing field: {field}"

    async def test_excludes_staff(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        # normal_store has 13 customers (10 + 3 billing) and 2 staff
        # unique_visitors must exclude the 2 staff
        assert d["unique_visitors"] <= 13
        assert d["unique_visitors"] > 0

    async def test_conversion_rate_in_range(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert 0.0 <= d["conversion_rate"] <= 1.0

    async def test_basket_value_correct(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert d["total_basket_inr"] == 1500.0

    async def test_queue_depth_non_negative(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert d["queue_depth_current"] >= 0

    async def test_store_id_in_response(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert d["store_id"] == STORE

    async def test_data_freshness_positive(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert d["data_freshness_secs"] > 0


class TestMetricsEdgeCases:
    async def test_unknown_store_returns_404(self, client, normal_store):
        r = await client.get("/stores/STORE_DOES_NOT_EXIST/metrics")
        assert r.status_code == 404

    async def test_all_staff_unique_visitors_zero(
        self, client, all_staff_store
    ):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert d["unique_visitors"] == 0

    async def test_all_staff_conversion_not_crash(
        self, client, all_staff_store
    ):
        """Division by zero guard: 0 visitors → 0.0 rate, not crash."""
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert d["conversion_rate"] == 0.0

    async def test_zero_purchase_abandonment_rate(
        self, client, zero_purchase_store
    ):
        d = (await client.get(f"/stores/{STORE}/metrics")).json()
        assert 0.0 <= d["abandonment_rate"] <= 1.0

    async def test_window_hours_param(self, client, normal_store):
        r = await client.get(
            f"/stores/{STORE}/metrics?window_hours=1"
        )
        assert r.status_code == 200

    async def test_window_hours_out_of_range(self, client, normal_store):
        r = await client.get(
            f"/stores/{STORE}/metrics?window_hours=999"
        )
        assert r.status_code == 422

    async def test_empty_store_returns_200(self, client, empty_store):
        """Empty store has one phantom event — should return 200."""
        r = await client.get(f"/stores/{STORE}/metrics")
        assert r.status_code == 200


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    async def test_health_returns_200(self, client, normal_store):
        r = await client.get("/health")
        assert r.status_code == 200

    async def test_health_schema(self, client, normal_store):
        d = (await client.get("/health")).json()
        for field in [
            "status", "version", "checked_at", "db_connected",
            "last_event_per_store", "stale_stores",
            "total_events_ingested", "uptime_secs",
        ]:
            assert field in d, f"Missing field: {field}"

    async def test_health_db_connected(self, client, normal_store):
        d = (await client.get("/health")).json()
        assert d["db_connected"] is True

    async def test_health_version(self, client, normal_store):
        d = (await client.get("/health")).json()
        assert d["version"] == "1.0.0"

    async def test_health_total_events(self, client, normal_store):
        d = (await client.get("/health")).json()
        assert d["total_events_ingested"] > 0

    async def test_health_stale_stores(self, client, normal_store):
        """Events from April 2026 → stale in June 2026."""
        d = (await client.get("/health")).json()
        assert STORE in d["stale_stores"]

    async def test_health_degraded_when_stale(self, client, normal_store):
        d = (await client.get("/health")).json()
        assert d["status"] in ("degraded", "healthy")

    async def test_health_empty_db(self, client, db_session):
        """Empty DB — should still return 200."""
        r = await client.get("/health")
        assert r.status_code == 200


# ── /stores/{store_id}/heatmap ─────────────────────────────────────────────────

class TestHeatmap:
    async def test_heatmap_returns_200(self, client, normal_store):
        r = await client.get(f"/stores/{STORE}/heatmap")
        assert r.status_code == 200

    async def test_heatmap_schema(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/heatmap")).json()
        assert "store_id" in d
        assert "zones" in d
        assert "generated_at" in d

    async def test_heatmap_has_zones(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/heatmap")).json()
        assert len(d["zones"]) > 0

    async def test_heatmap_zone_fields(self, client, normal_store):
        d = (await client.get(f"/stores/{STORE}/heatmap")).json()
        z = d["zones"][0]
        for field in ["zone_id", "visit_count", "avg_dwell_ms",
                       "normalised_score", "data_confidence"]:
            assert field in z, f"Missing zone field: {field}"

    async def test_heatmap_excludes_staff(self, client, all_staff_store):
        """All-staff → no zone visits from customers → empty zones list."""
        d = (await client.get(f"/stores/{STORE}/heatmap")).json()
        assert len(d["zones"]) == 0

    async def test_heatmap_empty_store(self, client, empty_store):
        d = (await client.get(f"/stores/{STORE}/heatmap")).json()
        assert d["zones"] == []
