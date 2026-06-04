"""
Orchestrates process_video() across all cameras.
Handles arguments, POS timestamps, merging, and validation.
"""

import sys
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime
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
                ts_str = row.get("timestamp")
                if ts_str:
                    try:
                        # Assumes ISO format e.g. 2026-04-10T12:05:00Z
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        timestamps.append(ts)
                    except ValueError:
                        pass
    return sorted(timestamps)

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
