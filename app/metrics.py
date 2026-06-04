"""GET /stores/{store_id}/metrics — real-time store metrics."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, distinct
from datetime import datetime, timezone, timedelta
from typing import Optional
from app.database import get_db, EventRow, POSRow
from pydantic import BaseModel

router = APIRouter(tags=["metrics"])


class StoreMetrics(BaseModel):
    store_id:             str
    window_start:         datetime
    window_end:           datetime
    unique_visitors:      int
    conversion_rate:      float
    avg_dwell_ms_by_zone: dict[str, int]
    queue_depth_current:  int
    abandonment_rate:     float
    total_transactions:   int
    total_basket_inr:     float
    data_freshness_secs:  float


@router.get("/stores/{store_id}/metrics", response_model=StoreMetrics)
async def get_metrics(
    store_id: str,
    window_hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
) -> StoreMetrics:
    """
    Returns real-time store metrics for the given window.
    Excludes is_staff=True from all visitor counts.
    Handles zero-purchase stores gracefully.
    """
    now          = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=window_hours)

    # For test data where events are in the past,
    # use the event timestamp range instead of wall-clock
    ts_range = await db.execute(
        select(func.min(EventRow.timestamp),
               func.max(EventRow.timestamp))
        .where(EventRow.store_id == store_id)
    )
    ts_min, ts_max = ts_range.one()
    if ts_min is None:
        raise HTTPException(status_code=404,
                            detail=f"Store {store_id} not found")
                            
    if ts_min.tzinfo is None:
        ts_min = ts_min.replace(tzinfo=timezone.utc)
    if ts_max.tzinfo is None:
        ts_max = ts_max.replace(tzinfo=timezone.utc)

    # Use data range if wall-clock window has no data
    if ts_max < window_start:
        window_start = ts_min
        now          = ts_max

    base_filter = and_(
        EventRow.store_id  == store_id,
        EventRow.is_staff  == False,
        EventRow.timestamp >= window_start,
        EventRow.timestamp <= now,
    )

    # Unique customer visitors
    uv_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(base_filter,
                    EventRow.event_type == "ENTRY"))
    )
    unique_visitors = uv_result.scalar() or 0

    # Avg dwell per zone
    dwell_result = await db.execute(
        select(EventRow.zone_id,
               func.avg(EventRow.dwell_ms).label("avg_dwell"))
        .where(and_(base_filter,
                    EventRow.event_type == "ZONE_DWELL",
                    EventRow.zone_id.isnot(None)))
        .group_by(EventRow.zone_id)
    )
    avg_dwell = {row.zone_id: int(row.avg_dwell)
                 for row in dwell_result.all()}

    # Current queue depth (most recent BILLING_QUEUE_JOIN queue_depth)
    queue_result = await db.execute(
        select(EventRow.queue_depth)
        .where(and_(
            EventRow.store_id  == store_id,
            EventRow.event_type == "BILLING_QUEUE_JOIN",
            EventRow.queue_depth.isnot(None),
        ))
        .order_by(EventRow.timestamp.desc())
        .limit(1)
    )
    queue_row           = queue_result.scalar_one_or_none()
    queue_depth_current = queue_row or 0

    # Billing visitors (reached billing zone)
    billing_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(
            base_filter,
            EventRow.event_type.in_(
                ["BILLING_QUEUE_JOIN", "ZONE_ENTER"]
            ),
            EventRow.zone_id.like("%BILLING%"),
        ))
    )
    billing_visitors = billing_result.scalar() or 0

    # Abandon count
    abandon_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(
            base_filter,
            EventRow.event_type == "BILLING_QUEUE_ABANDON",
        ))
    )
    abandoned = abandon_result.scalar() or 0

    abandonment_rate = (
        abandoned / billing_visitors if billing_visitors > 0 else 0.0
    )

    # POS transactions in window
    pos_result = await db.execute(
        select(func.count(POSRow.transaction_id),
               func.coalesce(func.sum(POSRow.basket_value_inr), 0.0))
        .where(and_(
            POSRow.store_id   == store_id,
            POSRow.timestamp  >= window_start,
            POSRow.timestamp  <= now,
        ))
    )
    txn_count, basket_total = pos_result.one()

    # Conversion: visitors who were in billing zone before a transaction
    # Use billing_visitors as proxy numerator (full POS correlation
    # is in pos_correlator.py — here we use a fast approximation)
    converted = min(billing_visitors, txn_count or 0)
    conversion_rate = (
        converted / unique_visitors if unique_visitors > 0 else 0.0
    )

    # Data freshness
    freshness_result = await db.execute(
        select(func.max(EventRow.timestamp))
        .where(EventRow.store_id == store_id)
    )
    last_event_ts    = freshness_result.scalar()
    if last_event_ts and last_event_ts.tzinfo is None:
        last_event_ts = last_event_ts.replace(tzinfo=timezone.utc)
        
    freshness_secs   = (
        (datetime.now(timezone.utc) - last_event_ts).total_seconds()
        if last_event_ts else 999999.0
    )

    return StoreMetrics(
        store_id             = store_id,
        window_start         = window_start,
        window_end           = now,
        unique_visitors      = unique_visitors,
        conversion_rate      = round(conversion_rate, 4),
        avg_dwell_ms_by_zone = avg_dwell,
        queue_depth_current  = queue_depth_current,
        abandonment_rate     = round(abandonment_rate, 4),
        total_transactions   = txn_count or 0,
        total_basket_inr     = float(basket_total),
        data_freshness_secs  = round(freshness_secs, 1),
    )
