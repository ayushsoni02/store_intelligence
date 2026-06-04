
### 3. API Architecture
**Storage:** SQLite via SQLAlchemy (async with aiosqlite).
- **Single file:** `data/store_intelligence.db`
- **Lifecycle:** Auto-created on startup and seeded from `all_events.jsonl` if empty.
- **Tables:** `events`, `pos_transactions`

**Why this choice?**
- Provides SQL aggregations for metrics (fast GROUP BY queries).
- Supports real-time ingest via INSERT.
- Eliminates the need for external service dependencies (like PostgreSQL), meaning the eventual Docker container just needs the app itself.
