"""Event schema construction and emission."""
import uuid
from app.models import StoreEvent

def make_event_id() -> str:
    """Generate a new UUID v4 string."""
    return str(uuid.uuid4())

def validate_event(raw_dict: dict) -> tuple[bool, str]:
    """
    Attempt to parse raw_dict into StoreEvent.
    Returns (True, "") on success.
    Returns (False, error_message) on validation failure.
    Do not raise — always return a tuple.
    """
    mapped = {}
    
    et = raw_dict.get("event_type", "").lower()
    if et == "entry": mapped["event_type"] = "ENTRY"
    elif et == "exit": mapped["event_type"] = "EXIT"
    elif et == "zone_entered": mapped["event_type"] = "ZONE_ENTER"
    elif et == "zone_exited": mapped["event_type"] = "ZONE_EXIT"
    elif et == "queue_completed": mapped["event_type"] = "BILLING_QUEUE_JOIN"
    elif et == "queue_abandoned": mapped["event_type"] = "BILLING_QUEUE_ABANDON"
    else: mapped["event_type"] = et

    mapped["event_id"] = raw_dict.get("event_id") or raw_dict.get("queue_event_id") or make_event_id()
    mapped["visitor_id"] = str(raw_dict.get("id_token") or raw_dict.get("track_id") or "")
    mapped["store_id"] = raw_dict.get("store_code") or raw_dict.get("store_id") or ""
    mapped["camera_id"] = raw_dict.get("camera_id") or ""
    
    ts = raw_dict.get("event_timestamp") or raw_dict.get("event_time") or raw_dict.get("queue_join_ts") or ""
    if ts and not ts.endswith("Z"):
        ts += "Z"
    mapped["timestamp"] = ts
    
    mapped["zone_id"] = raw_dict.get("zone_id")
    mapped["dwell_ms"] = raw_dict.get("dwell_ms") or (int(raw_dict.get("wait_seconds", 0) * 1000) if raw_dict.get("wait_seconds") else 0)
    mapped["is_staff"] = raw_dict.get("is_staff", False)
    mapped["confidence"] = raw_dict.get("confidence", 0.9)
    
    mapped["metadata"] = {
        "queue_depth": raw_dict.get("queue_position_at_join", 1),
        "sku_zone": raw_dict.get("zone_type"),
        "session_seq": raw_dict.get("session_seq", 1)
    }

    try:
        StoreEvent.model_validate(mapped)
        raw_dict.update(mapped)  # For script output
        return True, ""
    except Exception as e:
        return False, str(e)

