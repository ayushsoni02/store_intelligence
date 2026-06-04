"""GET /stores/{store_id}/funnel — customer conversion funnel."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, distinct, func
from datetime import timedelta
from typing import List
from app.database import get_db, EventRow, POSRow
from pydantic import BaseModel

router = APIRouter(tags=["funnel"])

class FunnelStage(BaseModel):
    name: str
    count: int
    drop_off_pct: float

class FunnelResponse(BaseModel):
    store_id: str
    stages: List[FunnelStage]

@router.get("/stores/{store_id}/funnel", response_model=FunnelResponse)
async def get_funnel(
    store_id: str,
    db: AsyncSession = Depends(get_db),
) -> FunnelResponse:
    base_filter = and_(
        EventRow.store_id == store_id,
        EventRow.is_staff == False,
    )

    # Stage 1: Entry
    entry_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(base_filter, EventRow.event_type == "ENTRY"))
    )
    entry_count = entry_result.scalar() or 0

    # Stage 2: Zone Visit
    # Exclude ENTRY_THRESHOLD zones, usually named like %ENTRY% but we just need ZONE_ENTER
    zone_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(
            base_filter,
            EventRow.event_type == "ZONE_ENTER",
            ~EventRow.zone_id.like("%ENTRY%")
        ))
    )
    zone_count = zone_result.scalar() or 0

    # Stage 3: Billing Queue
    billing_result = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(
            base_filter,
            EventRow.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
            EventRow.zone_id.like("%BILLING%")
        ))
    )
    billing_count = billing_result.scalar() or 0

    # Stage 4: Purchase
    # This requires correlating billing visits with POS transactions.
    # We will get all billing entries and exits, and check POS.
    billing_events = await db.execute(
        select(EventRow.visitor_id, EventRow.event_type, EventRow.timestamp)
        .where(and_(
            base_filter,
            EventRow.event_type.in_(["ZONE_ENTER", "BILLING_QUEUE_JOIN", "ZONE_EXIT", "BILLING_QUEUE_ABANDON"]),
            EventRow.zone_id.like("%BILLING%")
        ))
        .order_by(EventRow.timestamp)
    )
    
    pos_transactions = await db.execute(
        select(POSRow.timestamp)
        .where(POSRow.store_id == store_id)
    )
    pos_ts = []
    for row in pos_transactions.all():
        pts = row[0]
        if pts.tzinfo is None:
            from datetime import timezone
            pts = pts.replace(tzinfo=timezone.utc)
        pos_ts.append(pts)

    sessions = {}
    for row in billing_events.all():
        vid, etype, ts = row
        if ts.tzinfo is None:
            from datetime import timezone
            ts = ts.replace(tzinfo=timezone.utc)
            
        if vid not in sessions:
            sessions[vid] = {"entry": None, "exit": None}
        if etype in ("ZONE_ENTER", "BILLING_QUEUE_JOIN"):
            if not sessions[vid]["entry"]:
                sessions[vid]["entry"] = ts
        elif etype in ("ZONE_EXIT", "BILLING_QUEUE_ABANDON"):
            sessions[vid]["exit"] = ts

    purchase_count = 0
    for vid, s in sessions.items():
        b_entry = s["entry"]
        if not b_entry:
            continue
        b_exit = s["exit"] or (b_entry + timedelta(seconds=300))
        converted = False
        for txn_ts in pos_ts:
            if b_entry <= txn_ts and b_exit >= (txn_ts - timedelta(seconds=300)):
                converted = True
                break
        if converted:
            purchase_count += 1

    counts = [entry_count, zone_count, billing_count, purchase_count]
    
    # Ensure monotonically decreasing (some logic anomalies might make zone > entry)
    for i in range(1, 4):
        counts[i] = min(counts[i], counts[i-1])

    names = ["Entry", "Zone Visit", "Billing Queue", "Purchase"]
    stages = []
    
    for i in range(4):
        c = counts[i]
        if i < 3:
            drop = 0.0
            if c > 0:
                drop = (c - counts[i+1]) / c * 100
        else:
            drop = 0.0
        stages.append(FunnelStage(name=names[i], count=c, drop_off_pct=round(drop, 1)))

    return FunnelResponse(store_id=store_id, stages=stages)
