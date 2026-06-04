"""GET /stores/{store_id}/heatmap — store zone heatmap."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, distinct
from datetime import datetime
from typing import List
from app.database import get_db, EventRow
from pydantic import BaseModel

router = APIRouter(tags=["heatmap"])

class ZoneHeatmap(BaseModel):
    zone_id: str
    visit_count: int
    avg_dwell_ms: int
    normalised_score: float
    data_confidence: str

class HeatmapResponse(BaseModel):
    store_id: str
    zones: List[ZoneHeatmap]
    generated_at: datetime

@router.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store_id: str,
    db: AsyncSession = Depends(get_db),
) -> HeatmapResponse:
    base_filter = and_(
        EventRow.store_id == store_id,
        EventRow.is_staff == False,
        EventRow.zone_id.isnot(None),
        ~EventRow.zone_id.like("%ENTRY%")
    )

    # Visit count per zone
    visits_result = await db.execute(
        select(EventRow.zone_id, func.count(distinct(EventRow.visitor_id)).label("visits"))
        .where(and_(base_filter, EventRow.event_type == "ZONE_ENTER"))
        .group_by(EventRow.zone_id)
    )
    zone_visits = {row.zone_id: row.visits for row in visits_result.all()}

    # Avg dwell per zone
    dwell_result = await db.execute(
        select(EventRow.zone_id, func.avg(EventRow.dwell_ms).label("avg_dwell"))
        .where(and_(base_filter, EventRow.event_type == "ZONE_DWELL"))
        .group_by(EventRow.zone_id)
    )
    zone_dwells = {row.zone_id: int(row.avg_dwell) for row in dwell_result.all()}

    # Combine
    all_zones = set(zone_visits.keys()) | set(zone_dwells.keys())
    
    max_visits = max(zone_visits.values()) if zone_visits else 0
    
    zones_out = []
    for z in all_zones:
        v = zone_visits.get(z, 0)
        d = zone_dwells.get(z, 0)
        norm = (v / max_visits * 100.0) if max_visits > 0 else 0.0
        conf = "LOW" if v < 20 else "HIGH"
        zones_out.append(ZoneHeatmap(
            zone_id=z,
            visit_count=v,
            avg_dwell_ms=d,
            normalised_score=round(norm, 1),
            data_confidence=conf
        ))
        
    return HeatmapResponse(
        store_id=store_id,
        zones=zones_out,
        generated_at=datetime.utcnow()
    )
