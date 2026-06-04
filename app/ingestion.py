"""POST /events/ingest — validate, deduplicate, store up to 500 events."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db, EventRow
from app.models import IngestRequest, IngestResponse, StoreEvent

router = APIRouter(tags=["ingestion"])


@router.post("/events/ingest", response_model=IngestResponse)
async def ingest_events(
    payload: IngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Accepts batch of up to 500 events.
    - Validates each event (Pydantic already ran on IngestRequest)
    - Deduplicates by event_id (idempotent)
    - Stores accepted events
    - Returns partial success on malformed events

    Idempotency: calling twice with same payload returns same counts.
    Duplicate events increment duplicate counter, not rejected counter.
    """
    accepted  = 0
    rejected  = 0
    duplicate = 0
    errors    = []

    for ev in payload.events:
        try:
            # Check for duplicate
            result = await db.execute(
                select(EventRow).where(EventRow.event_id == ev.event_id)
            )
            existing = result.scalar_one_or_none()

            if existing:
                duplicate += 1
                continue

            row = EventRow(
                event_id    = ev.event_id,
                store_id    = ev.store_id,
                camera_id   = ev.camera_id,
                visitor_id  = ev.visitor_id,
                event_type  = ev.event_type.value,
                timestamp   = ev.timestamp,
                zone_id     = ev.zone_id,
                dwell_ms    = ev.dwell_ms,
                is_staff    = ev.is_staff,
                confidence  = ev.confidence,
                queue_depth = ev.metadata.queue_depth,
                sku_zone    = ev.metadata.sku_zone,
                session_seq = ev.metadata.session_seq,
            )
            db.add(row)
            accepted += 1

        except Exception as e:
            rejected += 1
            errors.append({
                "event_id": getattr(ev, "event_id", "unknown"),
                "reason":   str(e),
            })

    if accepted > 0:
        try:
            await db.commit()
        except Exception as e:
            await db.rollback()
            raise HTTPException(status_code=503, detail={
                "error":   "database_unavailable",
                "message": "Failed to persist events",
                "detail":  str(e),
            })

    return IngestResponse(
        accepted=accepted,
        rejected=rejected,
        duplicate=duplicate,
        errors=errors,
    )
