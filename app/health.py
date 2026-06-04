"""GET /health — simple liveness and readiness probe."""

import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func
from datetime import datetime, timezone
from app.database import get_db, EventRow
from pydantic import BaseModel
from typing import Dict, List

router = APIRouter(tags=["health"])

class HealthResponse(BaseModel):
    status: str
    version: str
    checked_at: datetime
    db_connected: bool
    last_event_per_store: Dict[str, str]
    stale_stores: List[str]
    total_events_ingested: int
    uptime_secs: float

START_TIME = time.perf_counter()

@router.get("/health", response_model=HealthResponse)
async def get_health(db: AsyncSession = Depends(get_db)):
    db_connected = True
    total_events = 0
    last_events = {}
    stale = []
    status = "healthy"
    now = datetime.now(timezone.utc)
    
    try:
        # Check DB
        count_res = await db.execute(text("SELECT COUNT(*) FROM events"))
        total_events = count_res.scalar() or 0
        
        # Last event per store
        stores_res = await db.execute(
            select(EventRow.store_id, func.max(EventRow.timestamp))
            .group_by(EventRow.store_id)
        )
        
        for row in stores_res.all():
            store_id, max_ts = row[0], row[1]
            if max_ts.tzinfo is None:
                max_ts = max_ts.replace(tzinfo=timezone.utc)
            last_events[store_id] = max_ts.isoformat()
            lag = (now - max_ts).total_seconds()
            if lag > 600:
                stale.append(store_id)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        db_connected = False
        status = "unhealthy"
        from fastapi import Response
        import json
        return Response(
            content=json.dumps({
                "status": status,
                "version": "1.0.0",
                "checked_at": now.isoformat(),
                "db_connected": False,
                "last_event_per_store": {},
                "stale_stores": [],
                "total_events_ingested": 0,
                "uptime_secs": round(time.perf_counter() - START_TIME, 1)
            }),
            status_code=503,
            media_type="application/json"
        )
        
    if db_connected and stale:
        status = "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        checked_at=now,
        db_connected=db_connected,
        last_event_per_store=last_events,
        stale_stores=stale,
        total_events_ingested=total_events,
        uptime_secs=round(time.perf_counter() - START_TIME, 1)
    )
