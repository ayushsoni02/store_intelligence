"""
Full video processing orchestrator.
Processes one video file end-to-end:
  frame loop → tracker → emitter → staff classifier → JSONL output

Called by run.sh for each camera.
Can also be imported and called programmatically.
"""

import cv2
import json
import sys
import signal
from pathlib import Path
from datetime import datetime, timezone

from pipeline.detect import get_camera_id, ENTRY_CAMERAS
from pipeline.tracker import PersonTracker
from pipeline.emit import SessionEventEmitter
from pipeline.staff import classify_staff_batch

# ── Constants ──────────────────────────────────────────────────────────────────

CLIP_START_TIME  = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
FPS              = 15.0
FLUSH_INTERVAL   = 500    # write events to disk every N frames
PROGRESS_INTERVAL = 300   # print progress every N frames
OUTPUT_DIR       = Path("data/processed")


# ── Flush open zones at end of video ──────────────────────────────────────────

def flush_open_zones(
    emitter: SessionEventEmitter,
    tracker: PersonTracker,
    final_frame: int,
) -> None:
    """
    At end of video, emit ZONE_EXIT for any visitor still in a zone,
    and EXIT for any track still active (not yet exited).

    This closes the 51-event ZONE_ENTER/EXIT gap identified in Phase 6.
    """
    # Force-exit all active tracks
    for track in list(tracker.active_tracks.values()):
        track.exited = True
        track.exit_frame = final_frame
        tracker.exited_tracks[track.track_id] = track

    tracker.active_tracks.clear()

    # Run one final process_frame to emit all pending exits
    emitter.process_frame(
        active_tracks=[],
        exited_tracks=list(tracker.exited_tracks.values()),
        frame_idx=final_frame,
    )


# ── Queue depth estimator ──────────────────────────────────────────────────────

def make_queue_depth_provider(tracker: PersonTracker):
    """
    Returns a callable (frame_idx) → int that estimates billing queue depth
    from the number of active tracks on a billing camera.

    Queue depth = active track count - 1 (subtract the person being served).
    Minimum 0.
    """
    def provider(frame_idx: int) -> int:
        count = len(tracker.active_tracks)
        return max(0, count - 1)
    return provider


# ── Main processing function ───────────────────────────────────────────────────

def process_video(
    video_path: str,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
    pos_timestamps: list[datetime] = None,
    max_frames: int = None,
) -> dict:
    """
    Process a single video file end-to-end.

    Returns summary dict:
      {camera_id, total_frames, events_emitted, staff_count,
       visitor_count, output_path, duration_secs}
    """
    pos_timestamps = pos_timestamps or []
    path           = Path(video_path)
    camera_id      = get_camera_id(str(path))
    output_path    = output_dir / f"events_{camera_id}.jsonl"

    # Resume support
    if output_path.exists() and not force:
        print(f"  [SKIP] {camera_id} — already processed ({output_path})")
        line_count = sum(1 for _ in open(output_path))
        return {
            "camera_id": camera_id,
            "skipped": True,
            "events_emitted": line_count,
            "output_path": str(output_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps   = cap.get(cv2.CAP_PROP_FPS) or FPS
    print(f"\n{'='*60}")
    print(f"Processing: {path.name}")
    print(f"Camera ID:  {camera_id}")
    print(f"Frames:     {total_frames} @ {actual_fps:.1f}fps")
    print(f"Output:     {output_path}")

    tracker = PersonTracker(
        camera_id=camera_id,
        fps=actual_fps,
    )

    emitter = SessionEventEmitter(
        camera_id=camera_id,
        fps=actual_fps,
        clip_start_time=CLIP_START_TIME,
        queue_depth_provider=make_queue_depth_provider(tracker),
    )

    # Graceful interrupt handler
    interrupted = False
    def _handle_interrupt(sig, frame):
        nonlocal interrupted
        print(f"\n  [INTERRUPT] Flushing {camera_id} events before exit...")
        interrupted = True
    signal.signal(signal.SIGINT, _handle_interrupt)

    frame_idx      = 0
    last_flush_idx = 0

    # Incremental writer
    outfile = open(output_path, "w")

    def flush_events_to_disk(up_to_idx: int) -> int:
        """Write newly emitted events to disk. Returns count written."""
        new_events = emitter.emitted_events[last_flush_idx:up_to_idx]
        for ev in new_events:
            outfile.write(ev.model_dump_json() + "\n")
        outfile.flush()
        return len(new_events)

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret or interrupted:
                break

            # Tracker update
            tracker.update(frame, frame_idx)

            # Emitter update
            emitter.process_frame(
                active_tracks=list(tracker.active_tracks.values()),
                exited_tracks=list(tracker.exited_tracks.values()),
                frame_idx=frame_idx,
            )

            # Progress reporting
            if frame_idx % PROGRESS_INTERVAL == 0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"  [frame {frame_idx:>6}/{total_frames}] "
                    f"{pct:5.1f}% | "
                    f"active={len(tracker.active_tracks)} | "
                    f"events={len(emitter.emitted_events)}"
                )

            # Incremental flush to disk
            if frame_idx - last_flush_idx >= FLUSH_INTERVAL:
                written = flush_events_to_disk(len(emitter.emitted_events))
                last_flush_idx = len(emitter.emitted_events)

            frame_idx += 1

    finally:
        cap.release()

    final_frame = frame_idx - 1

    # Flush open zones and remaining active tracks
    flush_open_zones(emitter, tracker, final_frame)

    # POS correlation for billing abandons
    emitter.flush_billing_abandons(
        pos_timestamps=pos_timestamps,
        flush_at_frame=final_frame,
    )

    # Run staff classification with REAL aspect ratios
    all_tracks = (
        list(tracker.active_tracks.values()) +
        list(tracker.exited_tracks.values())
    )
    zones_per_visitor   = emitter._zones_per_visitor
    aspects_per_visitor = {
        t.visitor_id: t.aspect_ratio_history
        for t in all_tracks
    }
    staff_results = classify_staff_batch(
        tracks=all_tracks,
        zones_per_visitor=zones_per_visitor,
        aspects_per_visitor=aspects_per_visitor,
        fps=actual_fps,
    )
    emitter._staff_cache = staff_results
    # Retroactively fix is_staff on emitted events
    corrected = []
    for ev in emitter.emitted_events:
        clf = staff_results.get(ev.visitor_id)
        if clf:
            corrected.append(ev.model_copy(update={"is_staff": clf.is_staff}))
        else:
            corrected.append(ev)
    emitter.emitted_events = corrected

    # Final flush of all remaining events
    flush_events_to_disk(len(emitter.emitted_events))
    last_flush_idx = len(emitter.emitted_events)
    outfile.close()

    staff_count   = sum(1 for c in staff_results.values() if c.is_staff)
    visitor_count = len(staff_results) - staff_count
    duration_secs = frame_idx / actual_fps

    print(f"  [DONE] {camera_id}")
    print(f"    Frames processed:  {frame_idx}")
    print(f"    Duration:          {duration_secs:.1f}s")
    print(f"    Events emitted:    {len(emitter.emitted_events)}")
    print(f"    Unique visitors:   {len(staff_results)}")
    print(f"    Staff classified:  {staff_count}")
    print(f"    Customers:         {visitor_count}")

    return {
        "camera_id":      camera_id,
        "skipped":        False,
        "total_frames":   frame_idx,
        "duration_secs":  duration_secs,
        "events_emitted": len(emitter.emitted_events),
        "staff_count":    staff_count,
        "visitor_count":  visitor_count,
        "output_path":    str(output_path),
    }
