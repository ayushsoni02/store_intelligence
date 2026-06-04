"""GET /stores/{store_id}/anomalies — detect operational anomalies."""

import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, distinct
from datetime import datetime, timezone, timedelta
from typing import List
from app.database import get_db, EventRow, POSRow
from pydantic import BaseModel

router = APIRouter(tags=["anomalies"])

class Anomaly(BaseModel):
    anomaly_id: str
    type: str
    severity: str
    detected_at: datetime
    description: str
    suggested_action: str
    metadata: dict

class AnomaliesResponse(BaseModel):
    store_id: str
    anomalies: List[Anomaly]

@router.get("/stores/{store_id}/anomalies", response_model=AnomaliesResponse)
async def get_anomalies(
    store_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnomaliesResponse:
    now = datetime.now(timezone.utc)
    anomalies = []

    # Get max timestamp in events as 'current' time for historical data
    ts_result = await db.execute(
        select(func.max(EventRow.timestamp))
        .where(EventRow.store_id == store_id)
    )
    last_event_ts = ts_result.scalar()
    
    # Only run anomalies if there is data
    if not last_event_ts:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Store {store_id} not found")
        
    if last_event_ts.tzinfo is None:
        last_event_ts = last_event_ts.replace(tzinfo=timezone.utc)
        
    current_time = last_event_ts

    # 1. BILLING_QUEUE_SPIKE
    queue_result = await db.execute(
        select(EventRow.queue_depth)
        .where(and_(
            EventRow.store_id == store_id,
            EventRow.event_type == "BILLING_QUEUE_JOIN",
            EventRow.queue_depth.isnot(None),
        ))
        .order_by(EventRow.timestamp.desc())
        .limit(1)
    )
    queue_row = queue_result.scalar_one_or_none()
    current_queue = queue_row or 0

    if current_queue > 3:
        sev = "CRITICAL" if current_queue > 6 else "WARN"
        anomalies.append(Anomaly(
            anomaly_id=str(uuid.uuid4()),
            type="BILLING_QUEUE_SPIKE",
            severity=sev,
            detected_at=now,
            description=f"Billing queue depth is currently {current_queue}.",
            suggested_action="Deploy additional billing staff immediately",
            metadata={"queue_depth": current_queue}
        ))

    # 2. CONVERSION_DROP
    # current hour conversion vs overall avg.
    # Overall:
    overall_uv = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(EventRow.store_id == store_id, EventRow.is_staff == False, EventRow.event_type == "ENTRY"))
    )
    ouv = overall_uv.scalar() or 0
    overall_tx = await db.execute(
        select(func.count(POSRow.transaction_id)).where(POSRow.store_id == store_id)
    )
    otx = overall_tx.scalar() or 0
    
    # Billing visitors
    overall_bv = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(EventRow.store_id == store_id, EventRow.is_staff == False, EventRow.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]), EventRow.zone_id.like("%BILLING%")))
    )
    obv = overall_bv.scalar() or 0
    
    avg_conv = min(obv, otx) / ouv if ouv > 0 else 0.0

    # Current hour:
    hour_ago = current_time - timedelta(hours=1)
    h_uv = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(EventRow.store_id == store_id, EventRow.is_staff == False, EventRow.event_type == "ENTRY", EventRow.timestamp >= hour_ago))
    )
    h_uv = h_uv.scalar() or 0
    
    h_tx = await db.execute(
        select(func.count(POSRow.transaction_id))
        .where(and_(POSRow.store_id == store_id, POSRow.timestamp >= hour_ago))
    )
    h_tx = h_tx.scalar() or 0
    
    h_bv = await db.execute(
        select(func.count(distinct(EventRow.visitor_id)))
        .where(and_(EventRow.store_id == store_id, EventRow.is_staff == False, EventRow.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]), EventRow.zone_id.like("%BILLING%"), EventRow.timestamp >= hour_ago))
    )
    h_bv = h_bv.scalar() or 0
    
    cur_conv = min(h_bv, h_tx) / h_uv if h_uv > 0 else 0.0

    if avg_conv > 0:
        if cur_conv < avg_conv * 0.5:
            sev = "CRITICAL"
        elif cur_conv < avg_conv * 0.7:
            sev = "WARN"
        else:
            sev = None
            
        if sev:
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                type="CONVERSION_DROP",
                severity=sev,
                detected_at=now,
                description=f"Current conversion {cur_conv*100:.1f}% dropped below average {avg_conv*100:.1f}%.",
                suggested_action="Review billing zone staffing and queue wait times",
                metadata={"current": cur_conv, "average": avg_conv}
            ))

    # 3. DEAD_ZONE
    thirty_mins_ago = current_time - timedelta(minutes=30)
    all_zones_res = await db.execute(
        select(EventRow.zone_id)
        .where(and_(EventRow.store_id == store_id, EventRow.zone_id.isnot(None)))
        .group_by(EventRow.zone_id)
    )
    all_zones = [r[0] for r in all_zones_res.all()]
    
    recent_zones_res = await db.execute(
        select(EventRow.zone_id)
        .where(and_(EventRow.store_id == store_id, EventRow.zone_id.isnot(None), EventRow.event_type == "ZONE_ENTER", EventRow.timestamp >= thirty_mins_ago))
        .group_by(EventRow.zone_id)
    )
    recent_zones = [r[0] for r in recent_zones_res.all()]
    
    for z in set(all_zones) - set(recent_zones):
        if "ENTRY" not in z: # exclude entry zones
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                type="DEAD_ZONE",
                severity="INFO",
                detected_at=now,
                description=f"Zone {z} had no visits in the last 30 minutes.",
                suggested_action=f"Check camera feed and zone signage for {z}",
                metadata={"zone_id": z}
            ))

    # 4. STALE_FEED
    wall_clock_now = datetime.now(timezone.utc)
    lag = (wall_clock_now - current_time).total_seconds()
    if lag > 600:
        sev = "CRITICAL" if lag > 1800 else "WARN"
        anomalies.append(Anomaly(
            anomaly_id=str(uuid.uuid4()),
            type="STALE_FEED",
            severity=sev,
            detected_at=now,
            description=f"Last event was {lag/60:.1f} minutes ago.",
            suggested_action="Check camera connectivity and pipeline health",
            metadata={"lag_seconds": lag}
        ))

    return AnomaliesResponse(store_id=store_id, anomalies=anomalies)
