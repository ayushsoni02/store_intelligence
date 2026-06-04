"""
Server-Sent Events streaming endpoint.
Pushes live store metrics to connected dashboard clients
every 3 seconds.

Also exposes GET /dashboard to serve the HTML UI.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct, and_

from app.database import get_db, EventRow, POSRow

router = APIRouter(tags=["streaming"])

STORE_ID      = "PURPLLE_MUM_1076"
PUSH_INTERVAL = 3   # seconds between SSE pushes


# ── SSE event generator ────────────────────────────────────────────────────────

async def metrics_stream(
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted metric snapshots.
    Each yield is one complete SSE message.
    Runs until client disconnects.
    """
    while True:
        try:
            # Unique customer visitors
            uv = await db.execute(
                select(func.count(distinct(EventRow.visitor_id)))
                .where(and_(
                    EventRow.store_id  == STORE_ID,
                    EventRow.is_staff  == False,
                    EventRow.event_type == "ENTRY",
                ))
            )
            unique_visitors = uv.scalar() or 0

            # Current queue depth
            qd = await db.execute(
                select(EventRow.queue_depth)
                .where(and_(
                    EventRow.store_id   == STORE_ID,
                    EventRow.event_type == "BILLING_QUEUE_JOIN",
                    EventRow.queue_depth.isnot(None),
                ))
                .order_by(EventRow.timestamp.desc())
                .limit(1)
            )
            queue_depth = qd.scalar_one_or_none() or 0

            # Total events ingested (proxy for pipeline activity)
            te = await db.execute(
                select(func.count(EventRow.event_id))
                .where(EventRow.store_id == STORE_ID)
            )
            total_events = te.scalar() or 0

            # Conversion rate approximation
            billing = await db.execute(
                select(func.count(distinct(EventRow.visitor_id)))
                .where(and_(
                    EventRow.store_id  == STORE_ID,
                    EventRow.is_staff  == False,
                    EventRow.zone_id.like("%BILLING%"),
                ))
            )
            billing_count = billing.scalar() or 0
            conv_rate = round(
                billing_count / unique_visitors * 100
                if unique_visitors > 0 else 0.0, 1
            )

            # Latest event timestamp
            lt = await db.execute(
                select(func.max(EventRow.timestamp))
                .where(EventRow.store_id == STORE_ID)
            )
            last_ts = lt.scalar()
            last_ts_str = (
                last_ts.isoformat() if last_ts else "N/A"
            )

            # Event type breakdown
            et = await db.execute(
                select(EventRow.event_type,
                       func.count(EventRow.event_id))
                .where(EventRow.store_id == STORE_ID)
                .group_by(EventRow.event_type)
            )
            event_dist = {row[0]: row[1] for row in et.all()}

            payload = {
                "ts":             datetime.now(timezone.utc).isoformat(),
                "unique_visitors": unique_visitors,
                "queue_depth":    queue_depth,
                "total_events":   total_events,
                "conv_rate_pct":  conv_rate,
                "last_event_ts":  last_ts_str,
                "event_dist":     event_dist,
            }

            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        await asyncio.sleep(PUSH_INTERVAL)


# ── SSE endpoint ───────────────────────────────────────────────────────────────

@router.get("/stream/metrics")
async def stream_metrics(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    SSE endpoint. Browser connects once and receives metric
    pushes every 3 seconds indefinitely.
    Closes cleanly when client disconnects.
    """
    async def event_generator():
        async for chunk in metrics_stream(db):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Dashboard HTML endpoint ────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the live dashboard HTML."""
    html_path = Path("app/static/dashboard.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
