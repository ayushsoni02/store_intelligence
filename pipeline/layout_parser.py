"""
Store layout parser: PNG floor plan → structured zone map JSON.
Strategy: OpenCV contour detection finds rectangular zones,
pytesseract OCR reads zone label text within each contour.
Fallback: if OCR confidence is low, zone is labelled ZONE_N.
"""

import cv2
import pytesseract
import json
import re
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


# ── Schema ────────────────────────────────────────────────────────────────────

class ZoneBounds(BaseModel):
    x_min: float   # as fraction of image width  (0.0–1.0)
    y_min: float   # as fraction of image height (0.0–1.0)
    x_max: float
    y_max: float

class Zone(BaseModel):
    zone_id: str
    sku_zone: str
    camera_ids: list[str]
    is_billing: bool
    is_entry: bool
    bounds: Optional[ZoneBounds] = None

class StoreLayout(BaseModel):
    store_id: str
    image_path: str
    zones: list[Zone]
    open_hours: Optional[str] = None
    raw_vlm_response: str    # repurposed: stores OCR debug text


# ── Keyword rules ─────────────────────────────────────────────────────────────
# These map OCR-extracted text fragments to semantic flags.
# Extend this dict as you discover more zone labels in your images.

BILLING_KEYWORDS  = {"billing", "checkout", "counter", "payment", "cash"}
ENTRY_KEYWORDS    = {"entry", "entrance", "exit", "door", "threshold"}
CAMERA_KEYWORDS   = {"cam", "camera"}

def _classify_zone(label: str) -> tuple[bool, bool]:
    """Return (is_billing, is_entry) based on label keywords."""
    lower = label.lower()
    is_billing = any(k in lower for k in BILLING_KEYWORDS)
    is_entry   = any(k in lower for k in ENTRY_KEYWORDS)
    return is_billing, is_entry

def _infer_cameras(zone_id: str, is_billing: bool, is_entry: bool) -> list[str]:
    """Assign likely camera IDs based on zone type."""
    if is_entry:
        return ["CAM_3_ENTRY"]
    if is_billing:
        return ["CAM_5_BILLING"]
    return ["CAM_1_ZONE", "CAM_2_ZONE"]


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """
    Preprocess a region-of-interest for better OCR accuracy.
    Steps: grayscale → resize 2x → threshold → dilate slightly.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(scaled, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def _ocr_roi(roi: np.ndarray) -> str:
    """Run tesseract on a preprocessed ROI. Return cleaned text."""
    processed = _preprocess_roi(roi)
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    text = pytesseract.image_to_string(processed, config=config)
    # Collapse whitespace, strip noise
    cleaned = re.sub(r"[^A-Za-z0-9 ]", "", text).strip()
    return cleaned.upper().replace(" ", "_")


# ── Contour-based zone detection ──────────────────────────────────────────────

def _find_zone_contours(img: np.ndarray) -> list[tuple[int,int,int,int]]:
    """
    Find rectangular zone boundaries in the floor plan.
    Returns list of (x, y, w, h) bounding boxes.
    Filters out: full-image border, tiny noise boxes, very thin lines.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area_fraction = (bw * bh) / (w * h)
        # Skip: too small (<1% image area), too large (>85%), too thin
        if area_fraction < 0.01 or area_fraction > 0.85:
            continue
        if bw < 30 or bh < 30:
            continue
        boxes.append((x, y, bw, bh))

    # Deduplicate heavily overlapping boxes (IoU > 0.7)
    return _dedup_boxes(boxes, iou_threshold=0.7)

def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2-ix)*(iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def _dedup_boxes(boxes: list, iou_threshold: float) -> list:
    kept = []
    for box in sorted(boxes, key=lambda b: b[2]*b[3], reverse=True):
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_layout(image_path: str, store_id: str) -> StoreLayout:
    """
    Parse a floor plan PNG into a StoreLayout.
    Pipeline:
      1. Load image with OpenCV
      2. Find zone contours
      3. OCR each contour ROI for zone label
      4. Apply keyword classification
      5. Fallback: if no contours found, create zones from
         full-image quadrant split (entry=top, floor=mid, billing=bottom)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Layout image not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")

    h, w = img.shape[:2]
    boxes = _find_zone_contours(img)
    ocr_debug_lines = [f"Image: {w}x{h}, Contours found: {len(boxes)}"]

    zones = []
    seen_labels = set()

    for i, (x, y, bw, bh) in enumerate(boxes):
        roi = img[y:y+bh, x:x+bw]
        label = _ocr_roi(roi)

        # Fallback label if OCR returns empty or very short result
        if len(label) < 2:
            label = f"ZONE_{i+1}"

        # Deduplicate identical labels
        if label in seen_labels:
            label = f"{label}_{i}"
        seen_labels.add(label)

        is_billing, is_entry = _classify_zone(label)
        cameras = _infer_cameras(label, is_billing, is_entry)

        ocr_debug_lines.append(
            f"  Box({x},{y},{bw},{bh}) → OCR='{label}' "
            f"billing={is_billing} entry={is_entry}"
        )

        zones.append(Zone(
            zone_id=label,
            sku_zone=label,
            camera_ids=cameras,
            is_billing=is_billing,
            is_entry=is_entry,
            bounds=ZoneBounds(
                x_min=round(x/w, 3),
                y_min=round(y/h, 3),
                x_max=round((x+bw)/w, 3),
                y_max=round((y+bh)/h, 3),
            )
        ))

    # ── Fallback: quadrant split if contour detection found nothing ──
    if not zones:
        ocr_debug_lines.append("WARNING: No contours found. Using quadrant fallback.")
        fallback_defs = [
            ("ENTRY",   False, True,  0.0, 0.0,  1.0, 0.25),
            ("FLOOR_A", False, False, 0.0, 0.25, 0.5, 0.75),
            ("FLOOR_B", False, False, 0.5, 0.25, 1.0, 0.75),
            ("BILLING", True,  False, 0.0, 0.75, 1.0, 1.0 ),
        ]
        for zid, bflag, eflag, xmn, ymn, xmx, ymx in fallback_defs:
            cameras = _infer_cameras(zid, bflag, eflag)
            zones.append(Zone(
                zone_id=zid, sku_zone=zid,
                camera_ids=cameras,
                is_billing=bflag, is_entry=eflag,
                bounds=ZoneBounds(x_min=xmn, y_min=ymn,
                                  x_max=xmx, y_max=ymx),
            ))

    return StoreLayout(
        store_id=store_id,
        image_path=str(path.absolute()),
        zones=zones,
        raw_vlm_response="\n".join(ocr_debug_lines),
    )


def save_layout(layout: StoreLayout,
                output_dir: str = "data/processed") -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{layout.store_id}_layout.json"
    out_path.write_text(layout.model_dump_json(indent=2))
    return str(out_path)


def load_layout(store_id: str,
                processed_dir: str = "data/processed") -> StoreLayout:
    path = Path(processed_dir) / f"{store_id}_layout.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No parsed layout for {store_id}. Run parse_layout() first."
        )
    return StoreLayout.model_validate_json(path.read_text())

def load_canonical_layout(
    store_id: str,
    canonical_path: str = "data/processed/canonical_zones.json"
) -> StoreLayout:
    """
    Load zone definitions from the hand-authored canonical_zones.json.
    This is the authoritative source for zone_ids used by the pipeline.
    Falls back to load_layout() if store_id not found in canonical file.
    """
    path = Path(canonical_path)
    if not path.exists():
        raise FileNotFoundError(f"Canonical zones file not found: {canonical_path}")

    data = json.loads(path.read_text())
    if store_id not in data:
        print(f"[WARN] {store_id} not in canonical_zones.json, falling back to OCR layout")
        return load_layout(store_id)

    entry = data[store_id]
    zones = [Zone(**z) for z in entry["zones"]]
    return StoreLayout(
        store_id=store_id,
        image_path="canonical",
        zones=zones,
        open_hours=entry.get("open_hours"),
        raw_vlm_response="Source: canonical_zones.json (hand-authored from sample_events.jsonl)"
    )

def get_zone_for_point(
    cx: float, cy: float, layout: StoreLayout
) -> Optional[str]:
    """
    Given a normalised centroid (cx, cy) in [0,1]x[0,1],
    return the zone_id of the first zone whose bounds contain it.
    Returns None if no zone matches or if all bounds are null.
    """
    for zone in layout.zones:
        if zone.bounds is None:
            continue
        b = zone.bounds
        if b.x_min <= cx <= b.x_max and b.y_min <= cy <= b.y_max:
            return zone.zone_id
    return None
