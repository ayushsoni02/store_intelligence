# Store Intelligence

Real-time retail analytics pipeline: raw CCTV footage →
structured events → live store intelligence API.

---

## Quick Start (5 commands)

```bash
git clone <your-repo-url> store-intelligence
cd store-intelligence
cp -r /path/to/provided/clips data/raw/        # add CCTV .mp4 files
cp pos_transactions.csv data/raw/
docker compose up
```

API is available at http://localhost:8000
Interactive docs: http://localhost:8000/docs

---

## Running the Detection Pipeline

Process all CCTV clips and generate events:

```bash
# Full pipeline (all cameras, ~60–90 min on CPU)
bash pipeline/run.sh

# Quick smoke test (first 900 frames per camera, ~5 min)
python -m pipeline.run_all --max-frames 900

# Single camera
python -m pipeline.run_all --cameras CAM_3_ENTRY

# Force reprocess (ignore resume cache)
bash pipeline/run.sh --force
```

Output: data/processed/all_events.jsonl
The API seeds from this file on startup automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Service status, stale feed detection |
| POST | /events/ingest | Ingest up to 500 events (idempotent) |
| GET | /stores/{id}/metrics | Visitors, conversion rate, dwell, queue |
| GET | /stores/{id}/funnel | Entry → Zone → Billing → Purchase funnel |
| GET | /stores/{id}/heatmap | Zone visit frequency heatmap (0–100) |
| GET | /stores/{id}/anomalies | Active anomalies with severity + actions |

Store ID for provided dataset: `PURPLLE_MUM_1076`

Example:
```bash
curl http://localhost:8000/stores/PURPLLE_MUM_1076/metrics
curl http://localhost:8000/stores/PURPLLE_MUM_1076/funnel
curl http://localhost:8000/stores/PURPLLE_MUM_1076/anomalies
```

---

## Architecture Overview

```
CCTV Clips (.mp4)
      │
      ▼
┌─────────────────┐
│  Detection Layer │  YOLOv8n + ByteTrack
│  pipeline/       │  Person detection, tracking,
│  detect.py       │  visitor_id assignment
│  tracker.py      │
└────────┬────────┘
         │ TrackState stream
         ▼
┌─────────────────┐
│  Event Emitter  │  SessionEventEmitter
│  pipeline/      │  ENTRY/EXIT/ZONE_ENTER/
│  emit.py        │  ZONE_DWELL/BILLING_*
└────────┬────────┘
         │ StoreEvent stream → JSONL
         ▼
┌─────────────────┐
│  Intelligence   │  FastAPI + SQLite
│  API            │  Real-time metrics,
│  app/           │  funnel, anomalies
└─────────────────┘
```

---

## Pipeline Design Decisions

- **Detection model**: YOLOv8n — optimal CPU speed/accuracy tradeoff
  for 1080p@15fps retail footage. See CHOICES.md.
- **Tracking**: ByteTrack via ultralytics — no separate install,
  stable track IDs across occlusions.
- **Staff exclusion**: Behavioral heuristics (dwell duration,
  zone breadth, movement frequency). No uniform detection needed.
- **Zone assignment**: Camera-type heuristic with cx_norm thirds
  for floor cameras. See CHOICES.md for rationale.
- **Storage**: SQLite + async SQLAlchemy. Seeded from JSONL on
  first startup. No external dependencies.

---

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v --tb=short
pytest tests/ --cov=app --cov=pipeline --cov-report=term-missing
```

---

## Project Structure

```
store-intelligence/
├── pipeline/          # Detection, tracking, event emission
│   ├── detect.py      # YOLOv8 person detection
│   ├── tracker.py     # ByteTrack + visitor_id assignment
│   ├── emit.py        # StoreEvent stream emission
│   ├── staff.py       # Staff behavioral classifier
│   ├── pos_correlator.py  # POS conversion correlation
│   ├── process_video.py   # Per-camera orchestrator
│   ├── run_all.py     # Full pipeline runner
│   └── run.sh         # One-command entry point
├── app/               # FastAPI application
│   ├── main.py        # App factory + middleware
│   ├── models.py      # Pydantic schemas
│   ├── database.py    # SQLAlchemy + seeding
│   ├── ingestion.py   # POST /events/ingest
│   ├── metrics.py     # GET /stores/{id}/metrics
│   ├── funnel.py      # GET /stores/{id}/funnel
│   ├── heatmap.py     # GET /stores/{id}/heatmap
│   ├── anomalies.py   # GET /stores/{id}/anomalies
│   └── health.py      # GET /health
├── tests/             # pytest suite (>70% coverage)
├── docs/
│   ├── DESIGN.md
│   └── CHOICES.md
├── data/
│   ├── raw/           # CCTV clips + POS CSV (not committed)
│   └── processed/     # Generated events + reports
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Live Dashboard

**Web UI** (recommended — scores higher):
```bash
docker compose up
open http://localhost:8000/dashboard
```

**Terminal fallback**:
```bash
python scripts/dashboard.py
```

The web dashboard connects via Server-Sent Events (SSE)
to `/stream/metrics` and updates every 3 seconds with:
- Live visitor count with time-series chart (Chart.js)
- Conversion rate, queue depth, total events KPI cards
- Real-time event type distribution bars
- Auto-reconnects on connection loss

> **Note**: The web dashboard is manually verified by opening
> `http://localhost:8000/dashboard` in a browser. No automated
> tests are needed for SSE streaming endpoints.

