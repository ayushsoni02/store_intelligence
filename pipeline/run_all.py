"""
Orchestrates process_video() across all cameras.
Handles arguments, POS timestamps, merging, and validation.
"""

import sys
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.process_video import process_video
from pipeline.detect import get_camera_id
from app.models import StoreEvent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", default=[])
    return parser.parse_args()

def load_pos_timestamps(path="data/raw/pos_transactions.csv") -> list[datetime]:
    timestamps = []
    if Path(path).exists():
        import csv
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Handle old schema or new schema
                ts_str = row.get("timestamp")
                if not ts_str and "order_date" in row and "order_time" in row:
                    ts_str = f"{row['order_date']} {row['order_time']}"
                    try:
                        # format: 10-04-2026 12:15:05
                        ts = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
                        ts = ts.replace(tzinfo=datetime.now(timezone.utc).tzinfo)
                        timestamps.append(ts)
                        continue
                    except ValueError:
                        pass
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        timestamps.append(ts)
                    except ValueError:
                        pass
    return sorted(timestamps)

def post_process_staff_flags(
    processed_dir: Path = Path("data/processed")
) -> dict[str, int]:
    """
    Fixes the mid-clip flush bug by ensuring that if a visitor was EVER 
    classified as staff (in the final flush), ALL their events are retroactively 
    marked as is_staff=True across the entire JSONL file.
    """
    from app.models import StoreEvent
    import json
    
    jsonl_files = sorted(processed_dir.glob("events_*.jsonl"))
    
    # Step 1: Find all visitor_ids that have at least one is_staff=True event
    staff_visitor_ids = set()
    for f in jsonl_files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            ev = StoreEvent.model_validate_json(line)
            if ev.is_staff:
                staff_visitor_ids.add(ev.visitor_id)
                
    # Step 2: Rewrite each JSONL file with corrected is_staff flags
    corrections_made: dict[str, int] = {}
    for f in jsonl_files:
        camera_id   = f.stem.replace("events_", "")
        lines       = [l for l in f.read_text().splitlines() if l.strip()]
        corrected   = []
        n_corrected = 0
        for line in lines:
            ev = StoreEvent.model_validate_json(line)
            if ev.visitor_id in staff_visitor_ids and not ev.is_staff:
                ev = ev.model_copy(update={"is_staff": True})
                n_corrected += 1
            corrected.append(ev.model_dump_json())
        f.write_text("\n".join(corrected) + "\n")
        corrections_made[camera_id] = n_corrected

    return corrections_made

def main():
    args = parse_args()
    
    mp4_files = glob.glob("data/raw/**/*.mp4", recursive=True) + glob.glob("data/raw/*.mp4")
    mp4_files = sorted(list(set(mp4_files)))
    
    if args.cameras:
        filtered = []
        for f in mp4_files:
            cam_id = get_camera_id(f)
            if cam_id in args.cameras:
                filtered.append(f)
        mp4_files = filtered

    if not mp4_files:
        print("ERROR: No matching video files found.")
        sys.exit(1)

    pos_ts = load_pos_timestamps()
    print(f"Loaded {len(pos_ts)} POS timestamps.")

    results = []
    for video_path in mp4_files:
        res = process_video(
            video_path=video_path,
            force=args.force,
            pos_timestamps=pos_ts,
            max_frames=args.max_frames,
        )
        results.append(res)
        
    print("\n" + "="*60)
    print("Running post-process staff flag correction...")
    corrections = post_process_staff_flags(Path("data/processed"))
    total_corrections = sum(corrections.values())
    print(f"  Corrected is_staff on {total_corrections} events")

    print("Merging all events into all_events.jsonl...")
    
    all_events = []
    for r in results:
        out_path = Path(r["output_path"])
        if out_path.exists():
            for line in open(out_path):
                line = line.strip()
                if line:
                    ev = StoreEvent.model_validate_json(line)
                    all_events.append(ev)
                    
    # Sort by timestamp ascending
    all_events.sort(key=lambda x: x.timestamp)
    
    final_out = Path("data/processed/all_events.jsonl")
    with open(final_out, "w") as f:
        for ev in all_events:
            f.write(ev.model_dump_json() + "\n")
            
    print(f"Wrote {len(all_events)} events to {final_out}")
    
    # Grand summary table
    print("\n" + "-"*60)
    print(f"{'Camera':<20} | {'Frames':<7} | {'Duration':<9} | {'Events':<7} | {'Visitors':<9} | {'Staff':<5}")
    print("-" * 60)
    for r in results:
        cam = r["camera_id"]
        frm = r.get("total_frames", "-")
        dur = f"{r.get('duration_secs', 0):.1f}s"
        evt = r.get("events_emitted", "-")
        vis = r.get("visitor_count", "-")
        stf = r.get("staff_count", "-")
        print(f"{cam:<20} | {frm:<7} | {dur:<9} | {evt:<7} | {vis:<9} | {stf:<5}")
    print("-" * 60)

    # Event type distribution
    dist = Counter(e.event_type.value for e in all_events)
    print("\nEvent Distribution:")
    for k, v in dist.items():
        print(f"  {k}: {v}")
        
    # Validate
    failures = 0
    for ev in all_events:
        try:
            StoreEvent.model_validate(ev.model_dump())
        except Exception as e:
            failures += 1
            
    print(f"\nValidation Failures: {failures}")
    
    # Save summary
    with open("data/processed/processing_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    if failures > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
