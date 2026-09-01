# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Ink Centerline Module
====================================
Multi-scale ink-center evidence extraction for Auto Trace and AI-assisted generation.

Algorithm:
  1. Black Top-Hat per scale (9, 15, 31 px) - isolates narrow dark strokes from background
  2. Champion-response fusion (pixel-wise max across scales)
  3. Percentile thresholding + optional scikit-image / Zhang-Suen skeletonization
  4. Polyline tracing via connected-component chain walking
  5. Constraint image rendering for Gemini / HuggingFace guidance

Dependency ladder (graceful fallbacks):
  Best:  OpenCV + scikit-image + scipy
  Good:  OpenCV + scikit-image
  Ok:    OpenCV-only (Zhang-Suen numpy for thinning)
  Bare:  NumPy-only (pure-numpy morphology + Zhang-Suen)

QGIS is NOT imported here - this module is stand-alone.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None  # type: ignore

try:
    from skimage.morphology import skeletonize as _ski_skeletonize
except ImportError:
    _ski_skeletonize = None

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:
    _scipy_ndimage = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INK_SCALES: Tuple[int, ...] = (9, 15, 31)
INK_RESPONSE_PERCENTILE: float = 99.0
INK_MIN_NORMALIZED_RESPONSE: float = 0.04
INK_MIN_COMPONENT_SIZE: int = 5
INK_MAX_SPUR_LENGTH_PX: float = 2.0

CONSTRAINT_LINE_COLOR: Tuple[int, int, int] = (30, 30, 30)
CONSTRAINT_BG_COLOR: Tuple[int, int, int] = (245, 240, 230)
CONSTRAINT_LINE_THICKNESS: int = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray_uint8(bgr_or_gray: np.ndarray) -> np.ndarray:
    """Convert BGR or already-gray array to single-channel uint8."""
    arr = np.asarray(bgr_or_gray)
    if arr.ndim == 3:
        if _cv2 is not None:
            return _cv2.cvtColor(arr.astype(np.uint8), _cv2.COLOR_BGR2GRAY)
        rgb = arr[..., :3].astype(np.float32)
        gray = rgb[..., 2] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 0] * 0.114
        return np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _morph_close_numpy(img: np.ndarray, radius: int) -> np.ndarray:
    """Pure-NumPy morphological closing with a flat circular kernel."""
    d = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2) <= radius ** 2

    if _scipy_ndimage is not None:
        dilated = _scipy_ndimage.maximum_filter(img.astype(np.float32), footprint=kernel)
        closed = _scipy_ndimage.minimum_filter(dilated, footprint=kernel)
        return np.clip(closed, 0, 255).astype(np.uint8)

    pad = radius
    h, w = img.shape
    arr_f = img.astype(np.float32)
    padded = np.pad(arr_f, pad, mode="edge")
    dilated = np.full((h, w), -np.inf, dtype=np.float32)
    closed = np.full((h, w), np.inf, dtype=np.float32)
    for dr in range(d):
        for dc in range(d):
            if kernel[dr, dc]:
                patch = padded[dr:dr + h, dc:dc + w]
                dilated = np.maximum(dilated, patch)
    padded_dil = np.pad(dilated, pad, mode="edge")
    for dr in range(d):
        for dc in range(d):
            if kernel[dr, dc]:
                patch = padded_dil[dr:dr + h, dc:dc + w]
                closed = np.minimum(closed, patch)
    return np.clip(closed, 0, 255).astype(np.uint8)


def _black_tophat_single(gray: np.ndarray, radius: int) -> np.ndarray:
    """Black Top-Hat at one scale: closed(I, r) - I. Returns float32 >= 0."""
    if _cv2 is not None:
        kernel = _cv2.getStructuringElement(
            _cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        closed = _cv2.morphologyEx(gray, _cv2.MORPH_CLOSE, kernel)
        response = closed.astype(np.float32) - gray.astype(np.float32)
    else:
        closed = _morph_close_numpy(gray, radius)
        response = closed.astype(np.float32) - gray.astype(np.float32)
    return np.clip(response, 0.0, 255.0)


# ---------------------------------------------------------------------------
# Public API - Ink Score
# ---------------------------------------------------------------------------

def compute_ink_score(
    bgr_or_gray: np.ndarray,
    scales: Tuple[int, ...] = INK_SCALES,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute multi-scale Black Top-Hat ink center score.

    Returns float32 array in [0, 1] - higher means "darker narrow stroke here".
    """
    gray = _to_gray_uint8(bgr_or_gray)
    champion = np.zeros(gray.shape, dtype=np.float32)
    for r in scales:
        champion = np.maximum(champion, _black_tophat_single(gray, r))

    peak = float(np.max(champion))
    if peak < 1e-6:
        return np.zeros_like(champion)
    score = champion / peak

    if mask is not None:
        score = score * (mask > 0).astype(np.float32)
    return score


# ---------------------------------------------------------------------------
# Public API - Skeletonization
# ---------------------------------------------------------------------------

def _zhang_suen_numpy(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning in pure NumPy."""
    skel = binary.copy().astype(bool)
    if not skel.any():
        return skel

    def _neighbors(s):
        p = np.pad(s, 1, mode="constant", constant_values=False)
        return (
            p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2],
        )

    while True:
        changed = False
        ns = _neighbors(skel)
        nc = sum(n.astype(np.int8) for n in ns)
        tr = sum(
            (~c & f).astype(np.int8)
            for c, f in zip(ns, ns[1:] + ns[:1])
        )
        p2, _, p4, _, p6, _, p8, _ = ns
        rm = skel & (nc >= 2) & (nc <= 6) & (tr == 1) & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
        if rm.any():
            skel[rm] = False
            changed = True
        ns = _neighbors(skel)
        nc = sum(n.astype(np.int8) for n in ns)
        tr = sum(
            (~c & f).astype(np.int8)
            for c, f in zip(ns, ns[1:] + ns[:1])
        )
        p2, _, p4, _, p6, _, p8, _ = ns
        rm = skel & (nc >= 2) & (nc <= 6) & (tr == 1) & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
        if rm.any():
            skel[rm] = False
            changed = True
        if not changed:
            break
    return skel


def skeletonize_score(
    score: np.ndarray,
    percentile: float = INK_RESPONSE_PERCENTILE,
    min_response: float = INK_MIN_NORMALIZED_RESPONSE,
    min_component: int = INK_MIN_COMPONENT_SIZE,
) -> np.ndarray:
    """Threshold + skeletonize an ink score to a 1-pixel boolean centerline mask."""
    nonzero = score[score > 0]
    if nonzero.size == 0:
        return np.zeros(score.shape, dtype=bool)

    threshold = max(float(np.percentile(nonzero, percentile)), float(min_response))
    binary = (score >= threshold)

    if _ski_skeletonize is not None:
        skel = _ski_skeletonize(binary)
    elif _cv2 is not None:
        ximgproc = getattr(_cv2, "ximgproc", None)
        if ximgproc is not None and hasattr(ximgproc, "thinning"):
            skel = ximgproc.thinning(binary.astype(np.uint8) * 255) > 0
        else:
            skel = _zhang_suen_numpy(binary)
    else:
        skel = _zhang_suen_numpy(binary)

    # Remove tiny speckle components
    if _cv2 is not None:
        num, labels, stats, _ = _cv2.connectedComponentsWithStats(
            skel.astype(np.uint8), connectivity=8
        )
        keep = np.zeros(skel.shape, dtype=bool)
        for lbl in range(1, num):
            if stats[lbl, _cv2.CC_STAT_AREA] >= min_component:
                keep |= (labels == lbl)
        skel = keep
    elif _scipy_ndimage is not None:
        labeled, n_labels = _scipy_ndimage.label(skel)
        sizes = _scipy_ndimage.sum(skel, labeled, range(1, n_labels + 1))
        keep = np.zeros_like(skel, dtype=bool)
        for i, sz in enumerate(sizes):
            if sz >= min_component:
                keep |= (labeled == (i + 1))
        skel = keep

    return skel.astype(bool)


# ---------------------------------------------------------------------------
# Public API - Polyline Extraction
# ---------------------------------------------------------------------------

def _arc_length(pline: List[Tuple[int, int]]) -> float:
    total = 0.0
    for i in range(1, len(pline)):
        dx = pline[i][0] - pline[i - 1][0]
        dy = pline[i][1] - pline[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def _trace_component(mask: np.ndarray, start_r: int, start_c: int) -> List[Tuple[int, int]]:
    """Walk a 1-pixel skeleton component into an ordered polyline (col, row)."""
    coord_set = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    visited: set = set()
    path = []
    current = (start_r, start_c)
    path.append(current)
    visited.add(current)

    def _n8(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                n = (r + dr, c + dc)
                if n in coord_set and n not in visited:
                    yield n

    while True:
        r, c = current
        cands = list(_n8(r, c))
        if not cands:
            break
        if len(path) >= 2:
            pr, pc = path[-2]
            dr_prev, dc_prev = r - pr, c - pc
            cands.sort(
                key=lambda n: (n[0] - r) * dr_prev + (n[1] - c) * dc_prev,
                reverse=True,
            )
        current = cands[0]
        path.append(current)
        visited.add(current)

    return [(c, r) for r, c in path]


def extract_ink_polylines(
    bgr_or_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: Tuple[int, ...] = INK_SCALES,
    percentile: float = INK_RESPONSE_PERCENTILE,
    min_response: float = INK_MIN_NORMALIZED_RESPONSE,
    min_component: int = INK_MIN_COMPONENT_SIZE,
    min_arc_length: float = 6.0,
) -> List[List[Tuple[int, int]]]:
    """
    Full pipeline: image -> ink score -> skeleton -> polyline list.
    Returns list of [(x,y), ...] polylines - each is a factual ink centerline.
    """
    score = compute_ink_score(bgr_or_gray, scales=scales, mask=mask)
    skel = skeletonize_score(
        score, percentile=percentile,
        min_response=min_response, min_component=min_component,
    )
    if not skel.any():
        return []

    polylines: List[List[Tuple[int, int]]] = []

    if _cv2 is not None:
        num, labels, stats, _ = _cv2.connectedComponentsWithStats(
            skel.astype(np.uint8), connectivity=8
        )
        for lbl in range(1, num):
            if stats[lbl, _cv2.CC_STAT_AREA] < min_component:
                continue
            comp = (labels == lbl)
            rows, cols = np.where(comp)
            pline = _trace_component(comp, int(rows[0]), int(cols[0]))
            if _arc_length(pline) >= min_arc_length:
                polylines.append(pline)
    elif _scipy_ndimage is not None:
        labeled, n_labels = _scipy_ndimage.label(skel)
        for lbl in range(1, n_labels + 1):
            comp = (labeled == lbl)
            rows, cols = np.where(comp)
            if len(rows) < min_component:
                continue
            pline = _trace_component(comp, int(rows[0]), int(cols[0]))
            if _arc_length(pline) >= min_arc_length:
                polylines.append(pline)
    else:
        rows, cols = np.where(skel)
        if len(rows) > 0:
            pline = _trace_component(skel, int(rows[0]), int(cols[0]))
            if _arc_length(pline) >= min_arc_length:
                polylines.append(pline)

    return polylines


# ---------------------------------------------------------------------------
# Public API - AI Constraint Image
# ---------------------------------------------------------------------------

def render_ink_constraint_image(
    bgr_or_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: Tuple[int, ...] = INK_SCALES,
    line_color: Tuple[int, int, int] = CONSTRAINT_LINE_COLOR,
    bg_color: Tuple[int, int, int] = CONSTRAINT_BG_COLOR,
    line_thickness: int = CONSTRAINT_LINE_THICKNESS,
    min_arc_length: float = 6.0,
    silhouette_color: Optional[Tuple[int, int, int]] = (80, 80, 80),
    silhouette_thickness: int = 2,
) -> Optional[np.ndarray]:
    """
    Render Ink Centerline polylines onto a parchment-like canvas (BGR uint8).

    This image is passed to Gemini / HuggingFace as a "line constraint" so the AI
    respects factual stroke positions rather than inventing motifs.
    """
    try:
        h, w = bgr_or_gray.shape[:2]
        canvas = np.full((h, w, 3), bg_color[::-1], dtype=np.uint8)

        if mask is not None and silhouette_color is not None:
            if _cv2 is not None:
                contours, _ = _cv2.findContours(
                    mask.astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
                )
                _cv2.drawContours(canvas, contours, -1,
                                  color=silhouette_color[::-1],
                                  thickness=silhouette_thickness)
            else:
                padded = np.pad(mask > 0, 1, mode="constant", constant_values=False)
                boundary = (mask > 0) & ~(
                    padded[:-2, 1:-1] & padded[2:, 1:-1] &
                    padded[1:-1, :-2] & padded[1:-1, 2:]
                )
                canvas[boundary] = silhouette_color[::-1]

        polylines = extract_ink_polylines(
            bgr_or_gray, mask=mask, scales=scales, min_arc_length=min_arc_length
        )

        if _cv2 is not None:
            for pline in polylines:
                if len(pline) < 2:
                    continue
                pts = np.array(pline, dtype=np.int32).reshape(-1, 1, 2)
                _cv2.polylines(canvas, [pts], isClosed=False,
                               color=line_color[::-1],
                               thickness=line_thickness,
                               lineType=_cv2.LINE_AA)
        else:
            for pline in polylines:
                for px, py in pline:
                    if 0 <= py < h and 0 <= px < w:
                        canvas[py, px] = line_color[::-1]

        return canvas
    except Exception:
        return None


def render_ink_constraint_bytes(
    bgr_or_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: Tuple[int, ...] = INK_SCALES,
    fmt: str = "png",
    **kwargs,
) -> Optional[bytes]:
    """Like render_ink_constraint_image but returns PNG bytes for API payloads."""
    img = render_ink_constraint_image(bgr_or_gray, mask=mask, scales=scales, **kwargs)
    if img is None:
        return None
    if _cv2 is not None:
        ok, buf = _cv2.imencode(f".{fmt.lstrip('.')}", img)
        if ok:
            return bytes(buf)
    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ink_runtime_status() -> dict:
    """Describe which backend tiers are active."""
    morph = "opencv" if _cv2 is not None else (
        "scipy" if _scipy_ndimage is not None else "numpy"
    )
    thin = (
        "scikit-image" if _ski_skeletonize is not None else (
            "opencv-ximgproc"
            if (_cv2 is not None
                and getattr(_cv2, "ximgproc", None) is not None
                and hasattr(_cv2.ximgproc, "thinning"))
            else "zhang-suen-numpy"
        )
    )
    label = "opencv" if _cv2 is not None else (
        "scipy" if _scipy_ndimage is not None else "numpy-single"
    )
    return {
        "ok": True,
        "optimized": morph == "opencv" and thin in ("scikit-image", "opencv-ximgproc"),
        "morph_backend": morph,
        "thin_backend": thin,
        "label_backend": label,
        "message": f"Ink Centerline ready (morph: {morph}, thin: {thin}, label: {label})",
    }
