"""FastAPI application entrypoint with structured logging middleware."""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db, seed_from_jsonl

# ── Structured logger ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
)
logger = logging.getLogger("store_intelligence")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, seed from JSONL if empty."""
    await init_db()
    seed_result = await seed_from_jsonl()
    if seed_result["skipped"]:
        logger.info('"DB already seeded, skipping"')
    else:
        logger.info(
            f'"DB seeded: {seed_result["events_seeded"]} events, '
            f'{seed_result["pos_seeded"]} POS transactions"'
        )
    yield
    # Shutdown: nothing to clean up for SQLite


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Store Intelligence API",
    version="1.0.0",
    description="Real-time retail store analytics from CCTV event stream",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Structured logging middleware ──────────────────────────────────────────────

@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    """
    Log every request with:
    trace_id, store_id, endpoint, latency_ms,
    event_count (for ingest), status_code
    """
    trace_id  = str(uuid.uuid4())[:8]
    start     = time.perf_counter()

    # Extract store_id from path if present
    path_parts = request.url.path.split("/")
    store_id   = "N/A"
    if "stores" in path_parts:
        idx = path_parts.index("stores")
        if idx + 1 < len(path_parts):
            store_id = path_parts[idx + 1]

    response   = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        f'"trace_id":"{trace_id}",'
        f'"store_id":"{store_id}",'
        f'"endpoint":"{request.url.path}",'
        f'"method":"{request.method}",'
        f'"latency_ms":{latency_ms},'
        f'"status_code":{response.status_code}'
    )
    return response


# ── Router mounts ──────────────────────────────────────────────────────────────

from app.ingestion  import router as ingest_router
from app.metrics    import router as metrics_router
from app.funnel     import router as funnel_router
from app.heatmap    import router as heatmap_router
from app.anomalies  import router as anomaly_router
from app.health     import router as health_router

app.include_router(ingest_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(heatmap_router)
app.include_router(anomaly_router)
app.include_router(health_router)
