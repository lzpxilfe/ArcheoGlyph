# -*- coding: utf-8 -*-
"""
Compare a generated symbol against the reference silhouette.

QGIS-free. Filled styles are judged by mask overlap; stroke styles (Line,
Measured) are judged by whether the drawn ink follows the silhouette's
boundary band and stays inside the object - comparing thin strokes against a
*filled* mask, as the old check did, could never succeed.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

FILL_THRESHOLDS = {
    "fill": {"iou": 0.66, "recall": 0.80, "precision": 0.66},
    "strict": {"iou": 0.72, "recall": 0.84, "precision": 0.72},
}
STROKE_BAND_RATIO = 0.035      # band half-width as a fraction of the object's size
STROKE_MIN_COVERAGE = 0.45     # share of the boundary band that must carry ink
STROKE_MIN_CONTAINMENT = 0.55  # share of ink that must lie on/inside the object


def _bool(mask) -> np.ndarray:
    return np.asarray(mask) > 0


def overlap_scores(reference, prediction) -> Dict[str, float]:
    """IoU, recall and precision of ``prediction`` against ``reference``."""
    ref = _bool(reference)
    pred = _bool(prediction)
    ref_count = int(np.count_nonzero(ref))
    pred_count = int(np.count_nonzero(pred))
    inter = int(np.count_nonzero(ref & pred))
    union = int(np.count_nonzero(ref | pred))
    return {
        "iou": inter / union if union else 0.0,
        "recall": inter / ref_count if ref_count else 0.0,
        "precision": inter / pred_count if pred_count else 0.0,
        "reference_pixels": float(ref_count),
        "prediction_pixels": float(pred_count),
    }


def boundary_band(mask, width: Optional[int] = None) -> np.ndarray:
    """Ring of pixels straddling the silhouette edge (dilation minus erosion)."""
    m = (_bool(mask)).astype(np.uint8)
    if cv2 is None:
        return m
    if width is None:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return m
        extent = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        width = max(2, int(round(extent * STROKE_BAND_RATIO)))
    k = int(max(3, 2 * int(width) + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.subtract(cv2.dilate(m, kernel), cv2.erode(m, kernel))


def stroke_scores(reference, ink, band_width: Optional[int] = None) -> Dict[str, float]:
    """
    How well stroke geometry follows a filled reference silhouette.

    ``coverage``: share of the boundary band that has ink within the band width.
    ``containment``: share of ink lying on or inside the (slightly grown) object.
    """
    ref = (_bool(reference)).astype(np.uint8)
    ink_mask = (_bool(ink)).astype(np.uint8)
    if cv2 is None:
        return {"coverage": 0.0, "containment": 0.0, "ink_pixels": float(np.count_nonzero(ink_mask))}

    ys, xs = np.nonzero(ref)
    if len(xs) == 0:
        return {"coverage": 0.0, "containment": 0.0, "ink_pixels": float(np.count_nonzero(ink_mask))}
    extent = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    width = band_width if band_width is not None else max(2, int(round(extent * STROKE_BAND_RATIO)))
    k = int(max(3, 2 * int(width) + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    band = boundary_band(ref, width)
    grown_ink = cv2.dilate(ink_mask, kernel)
    band_count = int(np.count_nonzero(band))
    coverage = (
        int(np.count_nonzero((band > 0) & (grown_ink > 0))) / band_count if band_count else 0.0
    )

    inside = cv2.dilate(ref, kernel) > 0
    ink_count = int(np.count_nonzero(ink_mask))
    containment = int(np.count_nonzero((ink_mask > 0) & inside)) / ink_count if ink_count else 0.0
    return {
        "coverage": coverage,
        "containment": containment,
        "ink_pixels": float(ink_count),
        "band_pixels": float(band_count),
    }


def matches_reference(
    reference,
    prediction,
    stroke_style: bool = False,
    strict: bool = False,
) -> Tuple[bool, str]:
    """
    Decide whether a rendered symbol matches the reference silhouette.

    :param reference: filled silhouette mask (bool/uint8)
    :param prediction: painted pixels of the rendered symbol
    :param stroke_style: True for Line/Measured output (thin strokes)
    :return: (ok, reason) - reason is empty when ok
    """
    ref = _bool(reference)
    if int(np.count_nonzero(ref)) < 40:
        return True, ""  # nothing meaningful to compare against

    if stroke_style:
        scores = stroke_scores(ref, prediction)
        if scores["ink_pixels"] < 20:
            return False, "no strokes were drawn"
        ok = scores["coverage"] >= STROKE_MIN_COVERAGE and scores["containment"] >= STROKE_MIN_CONTAINMENT
        if ok:
            return True, ""
        return False, (
            f"strokes do not follow the reference outline "
            f"(coverage={scores['coverage']:.2f}, containment={scores['containment']:.2f})"
        )

    scores = overlap_scores(ref, prediction)
    if scores["prediction_pixels"] < 20:
        return False, "empty rendered geometry against reference silhouette"
    limits = FILL_THRESHOLDS["strict" if strict else "fill"]
    ok = (
        scores["iou"] >= limits["iou"]
        and scores["recall"] >= limits["recall"]
        and scores["precision"] >= limits["precision"]
    )
    if ok:
        return True, ""
    return False, (
        f"silhouette mismatch (IoU={scores['iou']:.2f}, recall={scores['recall']:.2f}, "
        f"precision={scores['precision']:.2f})"
    )


def mask_from_png(png_bytes: bytes, dark_threshold: int = 90) -> Optional[np.ndarray]:
    """Boolean mask of the dark area of a black-on-white silhouette PNG."""
    if cv2 is None or not png_bytes:
        return None
    buf = np.frombuffer(bytes(png_bytes), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            image = image[:, :, :3]
        else:
            alpha = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if alpha is not None:
            gray = np.where(alpha > 16, gray, 255).astype(np.uint8)
    else:
        gray = image
    return gray < int(dark_threshold)


def painted_mask_from_png(png_bytes: bytes) -> Optional[np.ndarray]:
    """Boolean mask of pixels a rendered symbol actually painted (not blank)."""
    if cv2 is None or not png_bytes:
        return None
    buf = np.frombuffer(bytes(png_bytes), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return image < 248
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        rgb = image[:, :, :3]
        near_white = np.all(rgb > 248, axis=2)
        return (alpha > 16) & ~(near_white & (alpha > 220))
    return ~np.all(image > 248, axis=2)
