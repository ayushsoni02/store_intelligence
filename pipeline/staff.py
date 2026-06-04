"""
Staff classifier: behavioral heuristics to identify store staff
vs customers from TrackState trajectory data.

No uniform color detection — uses only:
  - Track duration (staff stay all day)
  - Zone coverage breadth (staff visit all zones)
  - Movement frequency / direction reversals
  - Early presence at clip start
  - Aspect ratio consistency (upright, consistent gait)

Output: StaffClassification with is_staff bool + confidence + signals fired
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import numpy as np

from pipeline.tracker import TrackState

# ── Thresholds (tune these after seeing full-clip results) ─────────────────────

STAFF_DWELL_THRESHOLD_SECS  = 600.0   # 10 minutes continuous presence
STAFF_ZONE_THRESHOLD        = 3       # distinct zones visited
STAFF_MIN_ZONE_SECS         = 300.0   # must also be present 5+ min for zone signal
STAFF_REVERSAL_THRESHOLD    = 4.0     # direction reversals per minute
STAFF_EARLY_FRAME           = 45      # first 3 seconds at 15fps
STAFF_CLASSIFY_THRESHOLD    = 0.55    # combined confidence to call is_staff=True


# ── Output schema ──────────────────────────────────────────────────────────────

@dataclass
class StaffSignal:
    name: str
    fired: bool
    confidence: float
    detail: str   # human-readable explanation for .ai_history.log / DESIGN.md


@dataclass
class StaffClassification:
    visitor_id: str
    is_staff: bool
    combined_confidence: float
    signals: list[StaffSignal]
    reason: str   # single sentence summary for logging


# ── Individual signal evaluators ───────────────────────────────────────────────

def _signal_dwell_duration(
    track: TrackState, fps: float
) -> StaffSignal:
    """Signal 1: track visible for > 10 minutes continuously."""
    if not track.centroid_history:
        return StaffSignal("DWELL_DURATION", False, 0.0, "no centroid history")

    frames_visible = track.last_seen_frame - track.first_seen_frame
    secs_visible   = frames_visible / fps if fps > 0 else 0

    fired = secs_visible >= STAFF_DWELL_THRESHOLD_SECS
    conf  = 0.85 if fired else 0.0
    detail = f"{secs_visible:.1f}s visible (threshold={STAFF_DWELL_THRESHOLD_SECS}s)"
    return StaffSignal("DWELL_DURATION", fired, conf, detail)


def _signal_zone_breadth(
    track: TrackState,
    zones_visited: set[str],
    fps: float,
) -> StaffSignal:
    """Signal 2: visited >= 3 distinct zones AND present 5+ minutes."""
    frames_visible = track.last_seen_frame - track.first_seen_frame
    secs_visible   = frames_visible / fps if fps > 0 else 0

    n_zones = len(zones_visited)
    fired   = n_zones >= STAFF_ZONE_THRESHOLD and secs_visible >= STAFF_MIN_ZONE_SECS
    conf    = 0.75 if fired else 0.0
    detail  = (
        f"{n_zones} zones visited, {secs_visible:.1f}s present "
        f"(need >={STAFF_ZONE_THRESHOLD} zones AND >={STAFF_MIN_ZONE_SECS}s)"
    )
    return StaffSignal("ZONE_BREADTH", fired, conf, detail)


def _signal_movement_frequency(
    track: TrackState, fps: float
) -> StaffSignal:
    """
    Signal 3: count cy_norm direction reversals per minute.
    A reversal is when cy movement changes sign (up→down or down→up).
    """
    if len(track.centroid_history) < 10:
        return StaffSignal("MOVEMENT_FREQ", False, 0.0, "too few frames")

    cy_values  = [c[1] for c in track.centroid_history]
    deltas     = np.diff(cy_values)
    # Sign of each delta: +1 down, -1 up, 0 stationary
    signs      = np.sign(deltas)
    # Remove zeros
    signs      = signs[signs != 0]

    if len(signs) < 2:
        return StaffSignal("MOVEMENT_FREQ", False, 0.0, "insufficient movement")

    reversals  = int(np.sum(np.diff(signs) != 0))
    frames_visible = track.last_seen_frame - track.first_seen_frame
    secs_visible   = max(1.0, frames_visible / fps)
    reversals_per_min = (reversals / secs_visible) * 60.0

    fired  = reversals_per_min > STAFF_REVERSAL_THRESHOLD
    conf   = 0.65 if fired else 0.0
    detail = (
        f"{reversals} reversals in {secs_visible:.1f}s "
        f"= {reversals_per_min:.2f}/min (threshold={STAFF_REVERSAL_THRESHOLD})"
    )
    return StaffSignal("MOVEMENT_FREQ", fired, conf, detail)


def _signal_early_presence(track: TrackState) -> StaffSignal:
    """Signal 4: first seen in first 3 seconds of clip."""
    fired  = track.first_seen_frame <= STAFF_EARLY_FRAME
    conf   = 0.40 if fired else 0.0
    detail = f"first_seen_frame={track.first_seen_frame} (threshold<={STAFF_EARLY_FRAME})"
    return StaffSignal("EARLY_PRESENCE", fired, conf, detail)


def _signal_aspect_consistency(
    aspect_ratio_history: list[float],
) -> StaffSignal:
    """
    Signal 5: modifier — consistent upright aspect ratio across track.
    Returns fired=True if std_dev < 0.15 AND mean > 1.5.
    Used as a +0.1 confidence modifier, not standalone classifier.
    """
    if len(aspect_ratio_history) < 5:
        return StaffSignal("ASPECT_CONSISTENCY", False, 0.0, "too few samples")

    arr     = np.array(aspect_ratio_history)
    mean_ar = float(np.mean(arr))
    std_ar  = float(np.std(arr))

    fired   = std_ar < 0.15 and mean_ar > 1.5
    conf    = 0.10 if fired else 0.0   # modifier only
    detail  = f"mean_aspect={mean_ar:.3f}, std={std_ar:.3f}"
    return StaffSignal("ASPECT_CONSISTENCY", fired, conf, detail)


# ── Combination logic ──────────────────────────────────────────────────────────

def _combine_signals(signals: list[StaffSignal]) -> tuple[float, str]:
    """
    Combine signal confidences into a single score.

    Rules:
    - DWELL_DURATION alone is sufficient (>= 0.85 → is_staff)
    - ZONE_BREADTH alone is sufficient (>= 0.75 → is_staff)
    - MOVEMENT_FREQ + EARLY_PRESENCE together are sufficient
      (0.65 + 0.55 > threshold)
    - ASPECT_CONSISTENCY adds 0.10 to the max fired signal's confidence
    - Overall: take max of fired non-modifier signals + modifier bonus

    Returns (combined_confidence, reason_string)
    """
    signal_map   = {s.name: s for s in signals}
    fired        = [s for s in signals if s.fired and s.name != "ASPECT_CONSISTENCY"]
    modifier     = signal_map.get("ASPECT_CONSISTENCY")
    modifier_val = modifier.confidence if (modifier and modifier.fired) else 0.0

    if not fired:
        return 0.0, "no signals fired"

    # Strongest single signal
    max_signal = max(fired, key=lambda s: s.confidence)
    base_conf  = max_signal.confidence + modifier_val

    # If multiple signals fired, boost by 0.05 per additional signal
    if len(fired) > 1:
        base_conf += 0.05 * (len(fired) - 1)

    base_conf = min(1.0, base_conf)
    fired_names = [s.name for s in fired]
    reason = f"Signals fired: {fired_names}, combined_conf={base_conf:.3f}"
    return base_conf, reason


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_staff(
    track: TrackState,
    zones_visited: set[str],
    aspect_ratio_history: list[float],
    fps: float = 15.0,
) -> StaffClassification:
    """
    Classify a completed or long-running track as staff or customer.

    Args:
        track:                  TrackState from PersonTracker
        zones_visited:          set of zone_ids this visitor entered
        aspect_ratio_history:   list of bbox aspect ratios across track lifetime
        fps:                    video frame rate

    Returns:
        StaffClassification with is_staff, confidence, and signal breakdown
    """
    signals = [
        _signal_dwell_duration(track, fps),
        _signal_zone_breadth(track, zones_visited, fps),
        _signal_movement_frequency(track, fps),
        _signal_early_presence(track),
        _signal_aspect_consistency(aspect_ratio_history),
    ]

    combined_conf, reason = _combine_signals(signals)
    is_staff = combined_conf >= STAFF_CLASSIFY_THRESHOLD

    return StaffClassification(
        visitor_id=track.visitor_id,
        is_staff=is_staff,
        combined_confidence=round(combined_conf, 4),
        signals=signals,
        reason=reason,
    )


def classify_staff_batch(
    tracks: list[TrackState],
    zones_per_visitor: dict[str, set[str]],
    aspects_per_visitor: dict[str, list[float]],
    fps: float = 15.0,
) -> dict[str, StaffClassification]:
    """
    Classify a batch of tracks.
    Returns dict of visitor_id → StaffClassification.
    """
    return {
        track.visitor_id: classify_staff(
            track=track,
            zones_visited=zones_per_visitor.get(track.visitor_id, set()),
            aspect_ratio_history=aspects_per_visitor.get(track.visitor_id, []),
            fps=fps,
        )
        for track in tracks
    }
