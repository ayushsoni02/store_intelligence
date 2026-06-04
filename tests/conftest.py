# PROMPT: Generate pytest fixtures for store intelligence API tests.
# Cover: empty store, all-staff scenario, normal scenario,
# re-entry scenario, zero-purchase scenario.
# CHANGES MADE: Added async DB fixtures, fixed timezone handling,
# added re-entry fixture missing from first draft.
# Switched to StaticPool for in-memory SQLite sharing.
# Added tests for /health, /heatmap, /funnel endpoints.

"""
Shared pytest fixtures.
All fixtures use an in-memory SQLite DB — no file I/O.
All timestamps use real UTC datetimes to avoid timezone issues.
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, EventRow, POSRow, get_db
from app.main import app
from app.models import EventType

# ── In-memory DB engine ────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_engine():
    """Create a fresh in-memory DB engine with StaticPool for connection sharing."""
    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    # Drop all tables before disposing to prevent leaks between tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    """Direct DB session for inserting test data."""
    session_factory = async_sessionmaker(
        test_engine, expire_on_commit=False, class_=AsyncSession
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(test_engine):
    """
    AsyncClient with app's get_db overridden to use the SAME in-memory DB.
    Lifespan is skipped — no JSONL seeding in tests.
    StaticPool ensures all sessions share the same underlying connection.
    """
    session_factory = async_sessionmaker(
        test_engine, expire_on_commit=False, class_=AsyncSession
    )

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Event factory ──────────────────────────────────────────────────────────────

BASE_TIME = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
STORE_ID  = "PURPLLE_MUM_1076"


def make_event_row(
    visitor_id:  str,
    event_type:  str,
    timestamp:   datetime,
    zone_id:     str  = None,
    is_staff:    bool = False,
    dwell_ms:    int  = 0,
    queue_depth: int  = None,
    camera_id:   str  = "CAM_1_ZONE",
    confidence:  float = 0.85,
) -> EventRow:
    return EventRow(
        event_id    = str(uuid.uuid4()),
        store_id    = STORE_ID,
        camera_id   = camera_id,
        visitor_id  = visitor_id,
        event_type  = event_type,
        timestamp   = timestamp,
        zone_id     = zone_id,
        dwell_ms    = dwell_ms,
        is_staff    = is_staff,
        confidence  = confidence,
        queue_depth = queue_depth,
        sku_zone    = zone_id,
        session_seq = 1,
    )


# ── Scenario fixtures ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def normal_store(db_session):
    """
    10 customers, 2 staff, 3 reach billing, 1 POS transaction.
    Also includes ZONE_DWELL events for heatmap coverage.
    Covers: metrics, funnel, heatmap, basic anomaly checks.
    """
    rows = []

    # 10 customers with ENTRY → ZONE_ENTER → ZONE_DWELL → ZONE_EXIT → EXIT
    for i in range(10):
        vid = f"VIS_cust_{i:02d}"
        t   = BASE_TIME + timedelta(minutes=i)
        rows.append(make_event_row(vid, "ENTRY", t))
        rows.append(make_event_row(
            vid, "ZONE_ENTER", t + timedelta(seconds=30),
            zone_id="PURPLLE_MUM_1076_Z01"
        ))
        rows.append(make_event_row(
            vid, "ZONE_DWELL", t + timedelta(minutes=3),
            zone_id="PURPLLE_MUM_1076_Z01",
            dwell_ms=270000
        ))
        rows.append(make_event_row(
            vid, "ZONE_EXIT", t + timedelta(minutes=5),
            zone_id="PURPLLE_MUM_1076_Z01",
            dwell_ms=270000
        ))
        rows.append(make_event_row(vid, "EXIT", t + timedelta(minutes=6)))

    # 3 customers reach billing
    for i in range(3):
        vid = f"VIS_bill_{i:02d}"
        t   = BASE_TIME + timedelta(minutes=i, seconds=15)
        rows.append(make_event_row(vid, "ENTRY", t))
        rows.append(make_event_row(
            vid, "BILLING_QUEUE_JOIN",
            t + timedelta(minutes=3),
            zone_id="PURPLLE_MUM_1076_Z_BILLING_01",
            queue_depth=i + 1,
        ))
        rows.append(make_event_row(
            vid, "ZONE_EXIT",
            t + timedelta(minutes=8),
            zone_id="PURPLLE_MUM_1076_Z_BILLING_01",
            dwell_ms=300000,
        ))
        rows.append(make_event_row(vid, "EXIT", t + timedelta(minutes=9)))

    # 2 staff
    for i in range(2):
        vid = f"VIS_staff_{i:02d}"
        rows.append(make_event_row(
            vid, "ENTRY", BASE_TIME, is_staff=True
        ))
        rows.append(make_event_row(
            vid, "ZONE_ENTER", BASE_TIME + timedelta(minutes=1),
            zone_id="PURPLLE_MUM_1076_Z01", is_staff=True
        ))

    # 1 POS transaction
    pos = POSRow(
        transaction_id    = "TXN_TEST_001",
        store_id          = STORE_ID,
        timestamp         = BASE_TIME + timedelta(minutes=4),
        basket_value_inr  = 1500.0,
    )

    for row in rows:
        db_session.add(row)
    db_session.add(pos)
    await db_session.commit()
    return {"customers": 10, "billing": 3, "staff": 2}


@pytest_asyncio.fixture
async def empty_store(db_session):
    """Single phantom event — store exists but essentially empty."""
    db_session.add(make_event_row(
        "VIS_phantom", "ENTRY",
        BASE_TIME, is_staff=False
    ))
    await db_session.commit()


@pytest_asyncio.fixture
async def all_staff_store(db_session):
    """
    All visitors are staff — unique_visitors must be 0.
    Conversion rate must be 0.0, not division-by-zero crash.
    """
    for i in range(5):
        vid = f"VIS_staff_only_{i:02d}"
        db_session.add(make_event_row(
            vid, "ENTRY", BASE_TIME + timedelta(minutes=i),
            is_staff=True
        ))
        db_session.add(make_event_row(
            vid, "ZONE_ENTER",
            BASE_TIME + timedelta(minutes=i, seconds=30),
            zone_id="PURPLLE_MUM_1076_Z01", is_staff=True
        ))
    await db_session.commit()


@pytest_asyncio.fixture
async def zero_purchase_store(db_session):
    """
    Visitors reach billing but zero POS transactions.
    Abandonment rate must be computable without crash.
    """
    for i in range(5):
        vid = f"VIS_nobuy_{i:02d}"
        t   = BASE_TIME + timedelta(minutes=i)
        db_session.add(make_event_row(vid, "ENTRY", t))
        db_session.add(make_event_row(
            vid, "BILLING_QUEUE_JOIN",
            t + timedelta(minutes=2),
            zone_id="PURPLLE_MUM_1076_Z_BILLING_01",
            queue_depth=2,
        ))
        db_session.add(make_event_row(
            vid, "BILLING_QUEUE_ABANDON",
            t + timedelta(minutes=5),
            zone_id="PURPLLE_MUM_1076_Z_BILLING_01",
        ))
        db_session.add(make_event_row(
            vid, "EXIT", t + timedelta(minutes=6)
        ))
    await db_session.commit()


@pytest_asyncio.fixture
async def reentry_store(db_session):
    """
    One visitor enters, exits, re-enters (REENTRY event).
    Funnel must count them as ONE unique visitor, not two.
    """
    vid = "VIS_reentry_001"
    # First visit
    db_session.add(make_event_row(vid, "ENTRY", BASE_TIME))
    db_session.add(make_event_row(
        vid, "ZONE_ENTER", BASE_TIME + timedelta(minutes=1),
        zone_id="PURPLLE_MUM_1076_Z01"
    ))
    db_session.add(make_event_row(
        vid, "EXIT", BASE_TIME + timedelta(minutes=5)
    ))
    # Re-entry
    db_session.add(make_event_row(
        vid, "REENTRY", BASE_TIME + timedelta(minutes=10)
    ))
    db_session.add(make_event_row(
        vid, "ZONE_ENTER", BASE_TIME + timedelta(minutes=11),
        zone_id="PURPLLE_MUM_1076_Z02"
    ))
    db_session.add(make_event_row(
        vid, "EXIT", BASE_TIME + timedelta(minutes=15)
    ))
    await db_session.commit()
