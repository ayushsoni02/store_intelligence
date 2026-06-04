"""
Async SQLite database layer via SQLAlchemy + aiosqlite.
Single file DB at data/store_intelligence.db.
Auto-seeds from all_events.jsonl on first startup.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import String, Float, Boolean, Integer, DateTime, Text
from sqlalchemy import text
from datetime import datetime
from typing import Optional, AsyncGenerator
from pathlib import Path
import json

DB_PATH = "data/store_intelligence.db"
DB_URL  = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(
    DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────────

class EventRow(Base):
    __tablename__ = "events"

    event_id:    Mapped[str]            = mapped_column(String, primary_key=True)
    store_id:    Mapped[str]            = mapped_column(String, index=True)
    camera_id:   Mapped[str]            = mapped_column(String)
    visitor_id:  Mapped[str]            = mapped_column(String, index=True)
    event_type:  Mapped[str]            = mapped_column(String, index=True)
    timestamp:   Mapped[datetime]       = mapped_column(DateTime(timezone=True), index=True)
    zone_id:     Mapped[Optional[str]]  = mapped_column(String, nullable=True)
    dwell_ms:    Mapped[int]            = mapped_column(Integer, default=0)
    is_staff:    Mapped[bool]           = mapped_column(Boolean, default=False)
    confidence:  Mapped[float]          = mapped_column(Float)
    queue_depth: Mapped[Optional[int]]  = mapped_column(Integer, nullable=True)
    sku_zone:    Mapped[Optional[str]]  = mapped_column(String, nullable=True)
    session_seq: Mapped[int]            = mapped_column(Integer, default=1)
    direction:   Mapped[Optional[str]]  = mapped_column(String, nullable=True)


class POSRow(Base):
    __tablename__ = "pos_transactions"

    transaction_id:  Mapped[str]      = mapped_column(String, primary_key=True)
    store_id:        Mapped[str]      = mapped_column(String, index=True)
    timestamp:       Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    basket_value_inr: Mapped[float]   = mapped_column(Float)


# ── DB lifecycle ───────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables. Called on app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Seed from JSONL ────────────────────────────────────────────────────────────

async def seed_from_jsonl(
    events_path: str = "data/processed/all_events.jsonl",
    pos_path:    str = "data/raw/pos_transactions.csv",
) -> dict:
    """
    Seed the database from all_events.jsonl and pos_transactions.csv.
    Idempotent: skips if events table already has rows.
    Returns {"events_seeded": N, "pos_seeded": N, "skipped": bool}
    """
    import pandas as pd
    from app.models import StoreEvent

    async with AsyncSessionLocal() as session:
        # Check if already seeded
        result = await session.execute(text("SELECT COUNT(*) FROM events"))
        count  = result.scalar()
        if count and count > 0:
            return {"events_seeded": 0, "pos_seeded": 0, "skipped": True}

        # Seed events
        events_seeded = 0
        events_file   = Path(events_path)
        if events_file.exists():
            for line in events_file.read_text().splitlines():
                if not line.strip():
                    continue
                ev = StoreEvent.model_validate_json(line)
                row = EventRow(
                    event_id   = ev.event_id,
                    store_id   = ev.store_id,
                    camera_id  = ev.camera_id,
                    visitor_id = ev.visitor_id,
                    event_type = ev.event_type.value,
                    timestamp  = ev.timestamp,
                    zone_id    = ev.zone_id,
                    dwell_ms   = ev.dwell_ms,
                    is_staff   = ev.is_staff,
                    confidence = ev.confidence,
                    queue_depth = ev.metadata.queue_depth,
                    sku_zone    = ev.metadata.sku_zone,
                    session_seq = ev.metadata.session_seq,
                )
                session.add(row)
                events_seeded += 1

        # Seed POS
        pos_seeded = 0
        pos_file   = Path(pos_path)
        if pos_file.exists():
            df = pd.read_csv(pos_file)
            df.columns = df.columns.str.strip()
            
            # Handle possible alternate POS schemas
            if "order_date" in df.columns and "order_time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["order_date"] + " " + df["order_time"], format="%d-%m-%Y %H:%M:%S", utc=True)
                df["basket_value_inr"] = df["total_amount"]
                df["transaction_id"] = df["order_id"]
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                
            for _, row in df.iterrows():
                pos_row = POSRow(
                    transaction_id   = str(row["transaction_id"]).strip(),
                    store_id         = str(row["store_id"]).strip(),
                    timestamp        = row["timestamp"].to_pydatetime(),
                    basket_value_inr = float(row["basket_value_inr"]),
                )
                session.add(pos_row)
                pos_seeded += 1

        await session.commit()
        return {
            "events_seeded": events_seeded,
            "pos_seeded":    pos_seeded,
            "skipped":       False,
        }
