"""
POS correlation layer.

The problem statement defines conversion as:
  "A visitor who was in the billing zone in the 5-minute window
   BEFORE a transaction timestamp counts as a converted visitor."

This module:
  1. Loads POS transactions from CSV
  2. Loads visitor sessions from all_events.jsonl
  3. For each POS transaction, finds visitors who were in
     the billing zone in the 5 minutes preceding the transaction
  4. Marks those visitor_ids as "converted"
  5. Computes conversion rate = converted / total_customer_visitors
  6. Emits BILLING_QUEUE_ABANDON for billing zone visitors
     with no matching POS transaction in their window

Key constraint from spec:
  - Correlation is store-level (no customer_id in POS data)
  - Time window: billing_zone_exit_time to billing_zone_exit_time + 5min
  - A visitor counts as converted if ANY transaction in that window
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from app.models import StoreEvent, EventType

# ── Constants ──────────────────────────────────────────────────────────────────

CONVERSION_WINDOW_SECS = 300   # 5 minutes
POS_CSV_PATH           = "data/raw/pos_transactions.csv"
ALL_EVENTS_PATH        = "data/processed/all_events.jsonl"
BILLING_ZONE_SUFFIX    = "Z_BILLING_01"  # zone_id ends with this for billing


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class VisitorSession:
    visitor_id:        str
    is_staff:          bool
    entry_time:        Optional[datetime]
    exit_time:         Optional[datetime]
    billing_entry_time: Optional[datetime]
    billing_exit_time:  Optional[datetime]
    zones_visited:     set
    converted:         bool = False
    abandoned_billing: bool = False


@dataclass
class ConversionReport:
    store_id:            str
    window_start:        datetime
    window_end:          datetime
    total_visitors:      int    # excludes staff
    converted_visitors:  int
    conversion_rate:     float  # 0.0–1.0
    billing_visitors:    int    # reached billing zone
    billing_abandoned:   int    # left billing without purchase
    abandonment_rate:    float
    avg_dwell_ms:        dict   # zone_id → avg dwell in ms
    total_transactions:  int
    total_basket_value:  float


# ── Session builder ────────────────────────────────────────────────────────────

def build_visitor_sessions(
    events: list[StoreEvent],
) -> dict[str, VisitorSession]:
    """
    Reconstruct per-visitor sessions from event stream.
    One VisitorSession per unique visitor_id.
    Handles multiple ZONE_ENTER/EXIT events per visitor.
    """
    sessions: dict[str, VisitorSession] = {}

    for ev in sorted(events, key=lambda e: e.timestamp):
        vid = ev.visitor_id
        if vid not in sessions:
            sessions[vid] = VisitorSession(
                visitor_id=vid,
                is_staff=ev.is_staff,
                entry_time=None,
                exit_time=None,
                billing_entry_time=None,
                billing_exit_time=None,
                zones_visited=set(),
            )

        s = sessions[vid]

        # Always update is_staff to latest value (retroactive correction)
        if ev.is_staff:
            s.is_staff = True

        if ev.event_type == EventType.ENTRY:
            if s.entry_time is None:
                s.entry_time = ev.timestamp

        elif ev.event_type == EventType.EXIT:
            s.exit_time = ev.timestamp

        elif ev.event_type == EventType.ZONE_ENTER:
            if ev.zone_id:
                s.zones_visited.add(ev.zone_id)
            if ev.zone_id and BILLING_ZONE_SUFFIX in ev.zone_id:
                s.billing_entry_time = ev.timestamp

        elif ev.event_type in (EventType.ZONE_EXIT, EventType.BILLING_QUEUE_ABANDON):
            if ev.zone_id and BILLING_ZONE_SUFFIX in ev.zone_id:
                s.billing_exit_time = ev.timestamp

    return sessions


# ── POS loader ─────────────────────────────────────────────────────────────────

def load_pos_transactions(
    csv_path: str = POS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load POS transactions CSV.
    Returns DataFrame with columns:
      store_id, transaction_id, timestamp (datetime), basket_value_inr
    Filters to rows with valid timestamps only.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    if "order_date" in df.columns and "order_time" in df.columns:
        # e.g., 10-04-2026 12:15:05
        df["timestamp"] = pd.to_datetime(df["order_date"] + " " + df["order_time"], format="%d-%m-%Y %H:%M:%S", utc=True)
        df["basket_value_inr"] = df["total_amount"]
        df["transaction_id"] = df["order_id"]
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Core correlation ───────────────────────────────────────────────────────────

def correlate_conversions(
    sessions: dict[str, VisitorSession],
    pos_df: pd.DataFrame,
    window_secs: int = CONVERSION_WINDOW_SECS,
) -> dict[str, VisitorSession]:
    """
    For each POS transaction, find customer visitors who were
    in the billing zone in the preceding window_secs window.

    Matching rule (from spec):
      visitor was in billing zone at time T if:
        billing_entry_time <= transaction_timestamp
        AND (billing_exit_time is None OR
             billing_exit_time >= transaction_timestamp - window_secs)

    Marks matched sessions as converted=True.
    Returns updated sessions dict.
    """
    if pos_df.empty:
        return sessions

    pos_timestamps = pos_df["timestamp"].tolist()

    for session in sessions.values():
        if session.is_staff:
            continue
        if session.billing_entry_time is None:
            continue

        b_entry = session.billing_entry_time
        b_exit  = session.billing_exit_time or (
            b_entry + timedelta(seconds=window_secs)
        )

        # Check each transaction
        for txn_ts in pos_timestamps:
            txn_dt = txn_ts.to_pydatetime()
            window_start = txn_dt - timedelta(seconds=window_secs)

            # Visitor billing window overlaps with transaction window
            if b_entry <= txn_dt and b_exit >= window_start:
                session.converted = True
                break

    # Mark abandonments: was in billing but not converted
    for session in sessions.values():
        if session.is_staff:
            continue
        if session.billing_entry_time and not session.converted:
            session.abandoned_billing = True

    return sessions


# ── Report builder ─────────────────────────────────────────────────────────────

def compute_conversion_report(
    sessions:   dict[str, VisitorSession],
    pos_df:     pd.DataFrame,
    store_id:   str,
    events:     list[StoreEvent],
) -> ConversionReport:
    """
    Compute the full conversion report for a store.
    Excludes staff from all metrics.
    """
    customer_sessions = [
        s for s in sessions.values() if not s.is_staff
    ]

    total_visitors     = len(customer_sessions)
    converted          = sum(1 for s in customer_sessions if s.converted)
    billing_visitors   = sum(
        1 for s in customer_sessions if s.billing_entry_time
    )
    billing_abandoned  = sum(
        1 for s in customer_sessions if s.abandoned_billing
    )

    conversion_rate  = converted / total_visitors if total_visitors > 0 else 0.0
    abandonment_rate = (
        billing_abandoned / billing_visitors
        if billing_visitors > 0 else 0.0
    )

    # Avg dwell per zone from ZONE_DWELL events
    zone_dwell_totals: dict[str, list[int]] = {}
    for ev in events:
        if ev.event_type == EventType.ZONE_DWELL and ev.zone_id and not ev.is_staff:
            zone_dwell_totals.setdefault(ev.zone_id, []).append(ev.dwell_ms)
    avg_dwell_ms = {
        z: int(sum(v) / len(v))
        for z, v in zone_dwell_totals.items()
    }

    # Timestamp window
    timestamps   = [e.timestamp for e in events]
    window_start = min(timestamps) if timestamps else datetime.now(timezone.utc)
    window_end   = max(timestamps) if timestamps else datetime.now(timezone.utc)

    total_transactions = len(pos_df)
    total_basket_value = float(pos_df["basket_value_inr"].sum()) \
        if not pos_df.empty else 0.0

    return ConversionReport(
        store_id=store_id,
        window_start=window_start,
        window_end=window_end,
        total_visitors=total_visitors,
        converted_visitors=converted,
        conversion_rate=round(conversion_rate, 4),
        billing_visitors=billing_visitors,
        billing_abandoned=billing_abandoned,
        abandonment_rate=round(abandonment_rate, 4),
        avg_dwell_ms=avg_dwell_ms,
        total_transactions=total_transactions,
        total_basket_value=total_basket_value,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def run_pos_correlation(
    events_path: str = ALL_EVENTS_PATH,
    pos_csv:     str = POS_CSV_PATH,
    store_id:    str = "PURPLLE_MUM_1076",
) -> ConversionReport:
    """
    Full pipeline: load events + POS → correlate → report.
    Saves report to data/processed/conversion_report.json.
    """
    import json

    # Load events
    lines  = Path(events_path).read_text().strip().splitlines()
    events = [
        StoreEvent.model_validate_json(l)
        for l in lines if l.strip()
    ]

    # Load POS
    pos_df = load_pos_transactions(pos_csv)
    print(f"Loaded {len(events)} events, {len(pos_df)} POS transactions")

    # Build sessions
    sessions = build_visitor_sessions(events)
    print(f"Built {len(sessions)} visitor sessions")

    # Correlate
    sessions = correlate_conversions(sessions, pos_df)

    # Report
    report = compute_conversion_report(sessions, pos_df, store_id, events)

    # Save
    output = {
        "store_id":           report.store_id,
        "window_start":       report.window_start.isoformat(),
        "window_end":         report.window_end.isoformat(),
        "total_visitors":     report.total_visitors,
        "converted_visitors": report.converted_visitors,
        "conversion_rate":    report.conversion_rate,
        "billing_visitors":   report.billing_visitors,
        "billing_abandoned":  report.billing_abandoned,
        "abandonment_rate":   report.abandonment_rate,
        "avg_dwell_ms":       report.avg_dwell_ms,
        "total_transactions": report.total_transactions,
        "total_basket_value": report.total_basket_value,
    }
    Path("data/processed/conversion_report.json").write_text(
        json.dumps(output, indent=2)
    )
    print(f"Conversion report saved to data/processed/conversion_report.json")
    return report
