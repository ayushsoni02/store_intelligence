# CHOICES.md — Engineering Decision Log

## Decision 1: Detection Model Selection

### Options Considered
- YOLOv8n (chosen)
- YOLOv8s / YOLOv8m
- RT-DETR
- MediaPipe Pose

### What AI Suggested
The AI initially suggested using YOLOv8m (medium) or RT-DETR for higher detection accuracy, noting that retail environments with heavy occlusion require stronger feature extraction architectures to maintain track continuity. It also suggested dropping any detections below a 0.45 confidence threshold to keep the downstream tracking clean.

### What We Chose and Why
We explicitly chose YOLOv8n (nano). While the medium model offers superior bounding box accuracy and RT-DETR handles severe occlusions better, the YOLOv8n model is the absolute fastest architecture that runs acceptably on standard CPU infrastructure. It automatically downloads weights on startup, requires zero external service configuration, and the ultralytics package provides the ByteTrack algorithm integrated seamlessly out of the box. The core accuracy trade-off is that the nano model frequently misses partial occlusions (e.g., when a person is mostly hidden behind a display rack). However, the massive gain in inference speed is overwhelmingly more critical for our requirement of processing 8 concurrent high-definition camera feeds in parallel on a CPU-bound scoring harness.

### Performance Observed
Detections at frame 300: 2/6/1/2 persons across cameras. Low-conf detections (0.10–0.24) flagged not dropped. Aspect ratio filter (>1.1) successfully removed implausible boxes.

---

## Decision 2: Event Schema Design

### The Core Problem
The event schema needs to support inherently different types of actions without overly fracturing the data model. For example, `ENTRY` and `EXIT` events do not strictly apply to a specific internal zone, so `zone_id` must be allowed to be null. Conversely, `ZONE_DWELL` events represent durations, while other events are instantaneous point-in-time actions where `dwell_ms` must be 0. We also need to embed metadata like `session_seq` to allow for proper time-series reconstruction during the funnel analysis. Finally, confidence values must be preserved to allow analysts to filter data quality at query time, rather than hard-dropping data at the edge.

### Options Considered
- Flat event with all fields required
- Typed union (EntryEvent | ZoneEvent | BillingEvent)
- Current schema: single type with optional fields + validators

### What AI Suggested
The AI explicitly suggested utilizing a strict Pydantic Typed Union approach (e.g., creating completely separate classes for `EntryEvent`, `ZoneEvent`, and `BillingEvent`). The AI reasoned that this would provide strict per-event-type validation guarantees and prevent invalid state combinations at the type level.

### What We Chose and Why
We rejected the Typed Union approach and chose a single, unified `StoreEvent` schema model that heavily utilizes Pydantic conditional validators (`@model_validator`). 
Benefits: This allows for a much simpler ingestion pipeline. There is only one DB table to manage, one single ingest endpoint schema to document, and one Pydantic model to serialize. 
Trade-off: We are forced to rely on implicit conventions (like `zone_id=null` for `ENTRY/EXIT`) which rely entirely on documentation and application-layer logic rather than strict type safety.

### UNKNOWN Direction — Key Sub-Decision
UNKNOWN direction events: AI suggested dropping them. We kept them at half-confidence (minimum 0.15). Rationale: 75/126 ENTRY events came from entry cameras with short trajectories. Dropping them would halve visitor count accuracy in the first 60 seconds of a clip.

---

## Decision 3: API Architecture — Storage Engine

### Options Considered
- SQLite + async SQLAlchemy (chosen)
- PostgreSQL + asyncpg
- In-memory dict + background persistence
- Redis (hot) + SQLite (cold)

### What AI Suggested
The AI strongly advocated for using PostgreSQL coupled with asyncpg. The AI's rationale was that a true relational database designed for high concurrency is an absolute requirement for a production-readiness submission, specifically citing its ability to handle thousands of concurrent writes during peak retail hours.

### What We Chose and Why
We chose SQLite configured with `aiosqlite` and SQLAlchemy's async engine. We utilized a seed-on-startup pattern where the `.db` file is completely gitignored and created entirely fresh on the first `docker compose up` execution by parsing `data/processed/all_events.jsonl`. 
Why we chose this: The absolute highest priority for this submission is that `docker compose up` must execute successfully with zero external service dependencies or complex networking. A single application container is vastly simpler to review and evaluate. Furthermore, since the scoring harness evaluates against a fixed dataset on a single machine, the high-concurrency benefits of PostgreSQL are entirely wasted.
Trade-off documented: SQLite uses a file-level write lock. Even in WAL mode, concurrent ingest operations will heavily bottleneck around ~100 requests per second. This is perfectly acceptable for a single-store demonstration but would utterly fail for a 40-store production rollout.

### Conversion Rate Approximation
Document the conscious trade-off: `/metrics` uses `min(billing_visitors, txn_count)` as a fast SQL approximation. Full 5-minute window correlation is in `pos_correlator.py` but runs as a batch job, not per-request. This is acceptable for the submission scope as it guarantees sub-10ms response times for the API layer while still providing directionally accurate intelligence.

---

## Decision 4: Live Dashboard — SSE vs WebSockets

### Options Considered
- Server-Sent Events / SSE (chosen)
- WebSockets
- HTTP polling every N seconds
- React + socket.io

### What AI Suggested
AI suggested WebSockets for bidirectional real-time communication. For a read-only dashboard, bidirectional capability is unnecessary overhead.

### What We Chose and Why
SSE via FastAPI StreamingResponse. One-way server→client push is all a metrics dashboard needs. SSE works through proxies without upgrade headers, uses native browser EventSource API (no library), and auto-reconnects built-in. Zero additional dependencies beyond what is already installed.

### Trade-off
SSE is HTTP/1.1 and subject to 6-connection-per-domain browser limit. WebSockets would be better for >6 concurrent dashboard tabs or for bidirectional control (e.g. sending alert acknowledgements from dashboard to API).

