# -*- coding: utf-8 -*-
"""
ArchaeoGlyph - Ink Centerline (v2)
==================================
Multi-scale ink-centre evidence for line drawings, rubbings and incised or
painted strokes on artifact photos. QGIS-free (numpy + OpenCV, scikit-image
optional).

Pipeline
  1. Appearance sources: luminance plus the R, G and B channels, so
     isoluminant coloured strokes are still visible.
  2. Black top-hat at kernel sizes 9, 15 and 31 px per source, each response
     capped by a local-mean darkness term that suppresses the halo a top-hat
     produces beside *bright* strokes.
  3. Champion fusion (pixel-wise max over sources and scales).
  4. Tile-wise robust normalisation (128 px tiles, 16 px halo) so faint
     strokes in a quiet region survive a strong stroke elsewhere.
  5. Threshold, skeletonise, prune short spurs.
  6. Junction-aware tracing into polylines (segments split at junctions and
     re-joined only where the tangent continues).
  7. Helpers for AI guidance: a composed guide image and a compact text
     encoding of simplified polylines.

The design mirrors ArchaeoTrace's Ink v2 evidence; the implementation here is
independent and MIT licensed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None  # type: ignore

try:
    from skimage.morphology import skeletonize as _ski_skeletonize
except ImportError:  # pragma: no cover
    _ski_skeletonize = None

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:  # pragma: no cover
    _scipy_ndimage = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INK_SCALES: Tuple[int, ...] = (9, 15, 31)          # closing kernel sizes in source pixels
INK_DARK_CONTEXT: int = 31                         # local-mean window for the darkness cap
INK_TILE_SIZE: int = 128
INK_TILE_HALO: int = 16
INK_RESPONSE_PERCENTILE: float = 99.0
INK_MIN_NORMALIZED_RESPONSE: float = 0.04
INK_THRESHOLD: float = 0.35                        # on the tile-normalised score
INK_MIN_COMPONENT_SIZE: int = 5
INK_MAX_SPUR_LENGTH_PX: int = 6
INK_JOIN_MIN_COS: float = 0.70                     # tangent continuity needed to join across a junction
DRAWING_PROBE_MAX_SIDE: int = 400                  # working size for looks_like_drawing

GUIDE_BG_COLOR: Tuple[int, int, int] = (255, 255, 255)      # RGB
GUIDE_LINE_COLOR: Tuple[int, int, int] = (0, 0, 0)
GUIDE_SILHOUETTE_COLOR: Tuple[int, int, int] = (220, 30, 30)
GUIDE_LINE_THICKNESS: int = 3
GUIDE_SILHOUETTE_THICKNESS: int = 3

# Legacy names kept for callers of render_ink_constraint_image.
CONSTRAINT_LINE_COLOR = GUIDE_LINE_COLOR
CONSTRAINT_BG_COLOR = GUIDE_BG_COLOR
CONSTRAINT_LINE_THICKNESS = GUIDE_LINE_THICKNESS

Polyline = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Sources and morphology
# ---------------------------------------------------------------------------

def _require_cv2():
    if _cv2 is None:
        raise ImportError("OpenCV is required for the ink centerline module.")


def _sources(bgr_or_gray: np.ndarray) -> List[np.ndarray]:
    """Float32 [0, 1] appearance surfaces: luminance, then R, G, B."""
    arr = np.asarray(bgr_or_gray)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return [arr.astype(np.float32) / 255.0]
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    b = arr[:, :, 0].astype(np.float32) / 255.0
    g = arr[:, :, 1].astype(np.float32) / 255.0
    r = arr[:, :, 2].astype(np.float32) / 255.0
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return [luma, r, g, b]


def _closing(src: np.ndarray, size: int) -> np.ndarray:
    _require_cv2()
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (int(size), int(size)))
    return _cv2.morphologyEx(src, _cv2.MORPH_CLOSE, kernel, borderType=_cv2.BORDER_REPLICATE)


def _local_mean(src: np.ndarray, size: int) -> np.ndarray:
    _require_cv2()
    return _cv2.blur(src, (int(size), int(size)), borderType=_cv2.BORDER_REPLICATE)


def normalize_tiled(
    response: np.ndarray,
    tile: int = INK_TILE_SIZE,
    halo: int = INK_TILE_HALO,
    percentile: float = INK_RESPONSE_PERCENTILE,
) -> np.ndarray:
    """Divide each tile by the ``percentile`` of positive responses in tile+halo."""
    values = np.asarray(response, dtype=np.float32)
    h, w = values.shape
    out = np.zeros_like(values)
    eps = np.finfo(np.float32).eps
    for y0 in range(0, h, tile):
        y1 = min(h, y0 + tile)
        hy0, hy1 = max(0, y0 - halo), min(h, y1 + halo)
        for x0 in range(0, w, tile):
            x1 = min(w, x0 + tile)
            hx0, hx1 = max(0, x0 - halo), min(w, x1 + halo)
            neighborhood = values[hy0:hy1, hx0:hx1]
            positive = neighborhood[neighborhood > 0.0]
            if positive.size == 0:
                continue
            scale = max(float(np.percentile(positive, percentile)), eps)
            out[y0:y1, x0:x1] = np.clip(values[y0:y1, x0:x1] / scale, 0.0, 1.0)
    return out


def compute_ink_score(
    bgr_or_gray: np.ndarray,
    scales: Tuple[int, ...] = INK_SCALES,
    mask: Optional[np.ndarray] = None,
    tile: int = INK_TILE_SIZE,
) -> np.ndarray:
    """
    Multi-scale, colour-aware ink centre score in [0, 1] (float32).
    Higher means "a narrow dark stroke is centred here".
    """
    sources = _sources(bgr_or_gray)
    shape = sources[0].shape
    fused = np.zeros(shape, dtype=np.float32)
    for src in sources:
        dark_support = np.maximum(_local_mean(src, INK_DARK_CONTEXT) - src, 0.0)
        for size in scales:
            response = np.maximum(_closing(src, size) - src, 0.0)
            np.maximum(fused, np.minimum(response, dark_support), out=fused)
    if not np.any(fused > 0.0):
        return fused
    score = normalize_tiled(fused, tile=tile) if tile and tile > 0 else fused / float(fused.max())
    if mask is not None:
        score = score * (np.asarray(mask) > 0).astype(np.float32)
    return score.astype(np.float32)


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

def _zhang_suen_numpy(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning in pure NumPy (fallback when scikit-image is absent)."""
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
        for step in (0, 1):
            ns = _neighbors(skel)
            nc = sum(n.astype(np.int8) for n in ns)
            tr = sum((~c & f).astype(np.int8) for c, f in zip(ns, ns[1:] + ns[:1]))
            p2, _, p4, _, p6, _, p8, _ = ns
            if step == 0:
                cond = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                cond = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            rm = skel & (nc >= 2) & (nc <= 6) & (tr == 1) & cond
            if rm.any():
                skel[rm] = False
                changed = True
        if not changed:
            break
    return skel


def _thin(binary: np.ndarray) -> np.ndarray:
    if _ski_skeletonize is not None:
        return np.asarray(_ski_skeletonize(binary), dtype=bool)
    if _cv2 is not None:
        ximgproc = getattr(_cv2, "ximgproc", None)
        if ximgproc is not None and hasattr(ximgproc, "thinning"):
            return ximgproc.thinning(binary.astype(np.uint8) * 255) > 0
    return _zhang_suen_numpy(binary)


def _degree_map(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours for each skeleton pixel (0 elsewhere)."""
    _require_cv2()
    u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0
    counts = _cv2.filter2D(u8.astype(np.float32), -1, kernel, borderType=_cv2.BORDER_CONSTANT)
    return np.rint(counts).astype(np.int32) * u8


def _label(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    _require_cv2()
    num, labels = _cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(num), labels


def prune_spurs(skel: np.ndarray, max_len: int = INK_MAX_SPUR_LENGTH_PX) -> np.ndarray:
    """
    Remove dangling branches of at most ``max_len`` pixels that hang off a
    junction. Genuine line ends (segments not attached to a junction) and
    isolated short strokes are left alone.
    """
    skel = skel.astype(bool)
    if max_len <= 0 or not skel.any():
        return skel
    degree = _degree_map(skel)
    junction = degree >= 3
    if not junction.any():
        return skel
    segments = skel & ~junction
    num, labels = _label(segments)
    if num <= 1:
        return skel
    counts = np.bincount(labels.ravel(), minlength=num)
    # Does the segment touch a junction, and does it have a free end?
    _require_cv2()
    junction_dilated = _cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    seg_degree = _degree_map(segments)
    result = skel.copy()
    removed = False
    for lbl in range(1, num):
        if counts[lbl] > max_len:
            continue
        comp = labels == lbl
        touches_junction = bool(np.any(comp & junction_dilated))
        has_free_end = bool(np.any(comp & (seg_degree <= 1)))
        if touches_junction and has_free_end:
            result[comp] = False
            removed = True
    if removed:
        # Junction pixels left behind by a pruned spur are now redundant; re-thin.
        result = _thin(result)
    return result


def skeletonize_score(
    score: np.ndarray,
    threshold: float = INK_THRESHOLD,
    min_response: float = INK_MIN_NORMALIZED_RESPONSE,
    min_component: int = INK_MIN_COMPONENT_SIZE,
    max_spur: int = INK_MAX_SPUR_LENGTH_PX,
    percentile: Optional[float] = None,
) -> np.ndarray:
    """Threshold + thin + prune an ink score into a 1-pixel boolean skeleton."""
    score = np.asarray(score, dtype=np.float32)
    if not np.any(score > 0):
        return np.zeros(score.shape, dtype=bool)
    if percentile is not None:
        nonzero = score[score > 0]
        threshold = max(float(np.percentile(nonzero, percentile)), float(min_response))
    binary = score >= max(float(threshold), float(min_response))
    if not binary.any():
        return binary
    # Close 1-px gaps before thinning so strokes broken by noise stay connected.
    if _cv2 is not None:
        binary = _cv2.morphologyEx(binary.astype(np.uint8), _cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    skel = _thin(binary)
    skel = prune_spurs(skel, max_spur)
    if min_component > 1 and skel.any():
        num, labels = _label(skel)
        counts = np.bincount(labels.ravel(), minlength=num)
        small = counts < min_component
        small[0] = False
        skel[small[labels]] = False
    return skel.astype(bool)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

_N8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _walk(coords: set, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Order the pixels of a junction-free component starting at ``start``."""
    path = [start]
    visited = {start}
    current = start
    while True:
        r, c = current
        nxt = None
        for dr, dc in _N8:  # 4-neighbours first, then diagonals
            cand = (r + dr, c + dc)
            if cand in coords and cand not in visited:
                nxt = cand
                break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path


def _arc_length(pline: Sequence[Tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pline, pline[1:]))


def _end_direction(pline: Sequence[Tuple[int, int]], at_start: bool, span: int = 6) -> Tuple[float, float]:
    """Unit vector pointing *out of* the polyline at one end."""
    if len(pline) < 2:
        return (0.0, 0.0)
    if at_start:
        a, b = pline[min(span, len(pline) - 1)], pline[0]
    else:
        a, b = pline[max(0, len(pline) - 1 - span)], pline[-1]
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def trace_skeleton(
    skel: np.ndarray,
    min_arc_length: float = 6.0,
    join_min_cos: float = INK_JOIN_MIN_COS,
) -> List[Polyline]:
    """
    Convert a 1-px skeleton into ordered polylines of (x, y) tuples.

    Segments are cut at junction pixels, traced from an end point, extended
    to the junction they touch (so branches share that point), and joined
    across a junction only when the tangents continue (cos >= join_min_cos).
    """
    skel = np.asarray(skel, dtype=bool)
    if not skel.any():
        return []
    degree = _degree_map(skel)
    junction = degree >= 3
    segments = skel & ~junction
    num, labels = _label(segments)
    junction_pts = {(int(r), int(c)) for r, c in np.argwhere(junction)}

    # Group junction pixels into clusters (adjacent junction pixels are one node).
    jnum, jlabels = _label(junction) if junction.any() else (1, np.zeros_like(labels))
    jcenters: Dict[int, Tuple[float, float]] = {}
    for lbl in range(1, jnum):
        rc = np.argwhere(jlabels == lbl)
        jcenters[lbl] = (float(rc[:, 0].mean()), float(rc[:, 1].mean()))

    def _junction_label_near(r: int, c: int) -> int:
        for dr, dc in _N8:
            rr, cc = r + dr, c + dc
            if (rr, cc) in junction_pts:
                return int(jlabels[rr, cc])
        return 0

    polylines: List[List[Tuple[int, int]]] = []
    ends: List[Tuple[int, int]] = []  # (junction label at start, at end) per polyline
    seg_degree = _degree_map(segments)
    for lbl in range(1, num):
        rc = np.argwhere(labels == lbl)
        coords = {(int(r), int(c)) for r, c in rc}
        endpoints = [p for p in coords if seg_degree[p] <= 1]
        start = min(endpoints) if endpoints else min(coords)
        remaining = set(coords)
        while remaining:
            path = _walk(remaining, start)
            remaining -= set(path)
            j_start = _junction_label_near(*path[0])
            j_end = _junction_label_near(*path[-1])
            pts = [(c, r) for r, c in path]
            if j_start:
                cy, cx = jcenters[j_start]
                pts.insert(0, (int(round(cx)), int(round(cy))))
            if j_end:
                cy, cx = jcenters[j_end]
                pts.append((int(round(cx)), int(round(cy))))
            polylines.append(pts)
            ends.append((j_start, j_end))
            if remaining:
                leftover_ends = [p for p in remaining if seg_degree[p] <= 1]
                start = min(leftover_ends) if leftover_ends else min(remaining)

    # Join across junctions where the tangent continues.
    incident: Dict[int, List[Tuple[int, bool]]] = {}
    for idx, (js, je) in enumerate(ends):
        if js:
            incident.setdefault(js, []).append((idx, True))
        if je:
            incident.setdefault(je, []).append((idx, False))
    alive = [True] * len(polylines)
    parent = list(range(len(polylines)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for jl, members in incident.items():
        if len(members) < 2:
            continue
        pairs = []
        for i in range(len(members)):
            for k in range(i + 1, len(members)):
                (a, a_start), (b, b_start) = members[i], members[k]
                if _find(a) == _find(b):
                    continue
                da = _end_direction(polylines[a], a_start)
                db = _end_direction(polylines[b], b_start)
                cos = -(da[0] * db[0] + da[1] * db[1])  # opposite directions continue
                if cos >= join_min_cos:
                    pairs.append((cos, a, a_start, b, b_start))
        pairs.sort(reverse=True)
        used = set()
        for _, a, a_start, b, b_start in pairs:
            if a in used or b in used or not (alive[a] and alive[b]):
                continue
            ra, rb = _find(a), _find(b)
            if ra == rb:
                continue
            pa = polylines[ra] if not a_start else list(reversed(polylines[ra]))
            pb = polylines[rb] if b_start else list(reversed(polylines[rb]))
            polylines[ra] = pa + pb[1:]
            alive[rb] = False
            parent[rb] = ra
            used.add(a)
            used.add(b)

    out = [pl for i, pl in enumerate(polylines) if alive[i] and _arc_length(pl) >= min_arc_length]
    out.sort(key=_arc_length, reverse=True)
    return out


def extract_ink_polylines(
    bgr_or_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: Tuple[int, ...] = INK_SCALES,
    threshold: float = INK_THRESHOLD,
    min_response: float = INK_MIN_NORMALIZED_RESPONSE,
    min_component: int = INK_MIN_COMPONENT_SIZE,
    min_arc_length: float = 6.0,
    percentile: Optional[float] = None,
) -> List[Polyline]:
    """Full pipeline: image -> ink score -> skeleton -> polylines (longest first)."""
    score = compute_ink_score(bgr_or_gray, scales=scales, mask=mask)
    skel = skeletonize_score(
        score, threshold=threshold, min_response=min_response,
        min_component=min_component, percentile=percentile,
    )
    if not skel.any():
        return []
    return trace_skeleton(skel, min_arc_length=min_arc_length)


# ---------------------------------------------------------------------------
# Input-kind heuristics
# ---------------------------------------------------------------------------

def looks_like_drawing(bgr_or_gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[bool, Dict[str, float]]:
    """
    Decide whether an image is a line drawing / rubbing (thin dark strokes on a
    pale ground) rather than a photograph of an object.

    Returns (is_drawing, metrics). Signals: pale uniform border, few mid-tones,
    and dark pixels that are mostly *thin* (skeleton length / dark area high).
    """
    _require_cv2()
    arr = np.asarray(bgr_or_gray)
    # Evaluate at a fixed working size so stroke thickness (and therefore the
    # thin-ratio test) does not depend on how much the caller upscaled.
    h0, w0 = arr.shape[:2]
    scale = min(1.0, float(DRAWING_PROBE_MAX_SIDE) / float(max(h0, w0)))
    if scale < 1.0:
        arr = _cv2.resize(arr, (max(8, int(round(w0 * scale))), max(8, int(round(h0 * scale)))), interpolation=_cv2.INTER_AREA)
        if mask is not None:
            mask = _cv2.resize(np.asarray(mask).astype(np.uint8), (arr.shape[1], arr.shape[0]), interpolation=_cv2.INTER_NEAREST)
    gray = _cv2.cvtColor(arr, _cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr.astype(np.uint8)
    h, w = gray.shape[:2]
    b = max(4, min(h, w) // 30)
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    border_mean = float(border.mean())
    border_std = float(border.std())

    region = np.ones_like(gray, dtype=bool) if mask is None else (np.asarray(mask) > 0)
    if region.sum() < 100:
        region = np.ones_like(gray, dtype=bool)
    vals = gray[region]
    dark = vals < 110
    mid = (vals >= 110) & (vals < 190)
    dark_fraction = float(dark.mean())
    mid_fraction = float(mid.mean())

    dark_mask = ((gray < 110) & region).astype(np.uint8)
    dark_count = int(dark_mask.sum())
    if dark_count > 0:
        skel = _thin(dark_mask.astype(bool))
        thin_ratio = float(skel.sum()) / float(dark_count)
    else:
        thin_ratio = 0.0

    if arr.ndim == 3:
        sat = _cv2.cvtColor(arr, _cv2.COLOR_BGR2HSV)[:, :, 1][region]
        saturation = float(sat.mean())
    else:
        saturation = 0.0

    metrics = {
        "border_mean": border_mean, "border_std": border_std, "dark_fraction": dark_fraction,
        "mid_fraction": mid_fraction, "thin_ratio": thin_ratio, "saturation": saturation,
    }
    is_drawing = (
        border_mean >= 215.0 and border_std <= 30.0
        and 0.002 <= dark_fraction <= 0.45
        and thin_ratio >= 0.22
        and mid_fraction <= 0.15
        and saturation <= 60.0
    )
    return bool(is_drawing), metrics


# ---------------------------------------------------------------------------
# AI guidance helpers
# ---------------------------------------------------------------------------

def simplify_polyline(points: Sequence[Tuple[float, float]], epsilon: float = 1.5) -> List[Tuple[int, int]]:
    """Douglas-Peucker simplification (open polyline)."""
    if len(points) < 3:
        return [(int(p[0]), int(p[1])) for p in points]
    _require_cv2()
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = _cv2.approxPolyDP(arr, float(epsilon), False).reshape(-1, 2)
    return [(int(round(x)), int(round(y))) for x, y in approx]


def polylines_to_text(
    polylines: Sequence[Sequence[Tuple[float, float]]],
    max_lines: int = 40,
    epsilon: float = 1.5,
    scale: float = 1.0,
    max_points: int = 40,
) -> str:
    """
    Compact text encoding for prompts: one line per polyline,
    ``x,y x,y ...`` in (optionally rescaled) integer coordinates.
    """
    lines = []
    for pline in list(polylines)[: max(0, int(max_lines))]:
        simplified = simplify_polyline(pline, epsilon)
        if len(simplified) > max_points:
            step = max(1, len(simplified) // max_points)
            simplified = simplified[::step] + [simplified[-1]]
        lines.append(" ".join(f"{int(round(x * scale))},{int(round(y * scale))}" for x, y in simplified))
    return "\n".join(lines)


def compose_guide_image(
    bgr_or_gray: np.ndarray,
    mask: Optional[np.ndarray],
    polylines: Sequence[Sequence[Tuple[float, float]]],
    line_color: Tuple[int, int, int] = GUIDE_LINE_COLOR,
    bg_color: Tuple[int, int, int] = GUIDE_BG_COLOR,
    silhouette_color: Optional[Tuple[int, int, int]] = GUIDE_SILHOUETTE_COLOR,
    line_thickness: int = GUIDE_LINE_THICKNESS,
    silhouette_thickness: int = GUIDE_SILHOUETTE_THICKNESS,
) -> np.ndarray:
    """
    White canvas with the silhouette contour in red and the ink polylines in
    black (BGR uint8). Colours are RGB tuples.
    """
    _require_cv2()
    h, w = np.asarray(bgr_or_gray).shape[:2]
    canvas = np.full((h, w, 3), bg_color[::-1], dtype=np.uint8)
    if mask is not None and silhouette_color is not None:
        contours, _ = _cv2.findContours(
            (np.asarray(mask) > 0).astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
        )
        _cv2.drawContours(canvas, contours, -1, silhouette_color[::-1], int(silhouette_thickness), _cv2.LINE_AA)
    for pline in polylines:
        if len(pline) < 2:
            continue
        pts = np.asarray(pline, dtype=np.int32).reshape(-1, 1, 2)
        _cv2.polylines(canvas, [pts], False, line_color[::-1], int(line_thickness), _cv2.LINE_AA)
    return canvas


def render_ink_constraint_image(
    bgr_or_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: Tuple[int, ...] = INK_SCALES,
    line_color: Tuple[int, int, int] = GUIDE_LINE_COLOR,
    bg_color: Tuple[int, int, int] = GUIDE_BG_COLOR,
    line_thickness: int = GUIDE_LINE_THICKNESS,
    min_arc_length: float = 6.0,
    silhouette_color: Optional[Tuple[int, int, int]] = GUIDE_SILHOUETTE_COLOR,
    silhouette_thickness: int = GUIDE_SILHOUETTE_THICKNESS,
) -> Optional[np.ndarray]:
    """Extract ink polylines and render them as a guide image (BGR uint8) or None."""
    try:
        polylines = extract_ink_polylines(bgr_or_gray, mask=mask, scales=scales, min_arc_length=min_arc_length)
        return compose_guide_image(
            bgr_or_gray, mask, polylines, line_color=line_color, bg_color=bg_color,
            silhouette_color=silhouette_color, line_thickness=line_thickness,
            silhouette_thickness=silhouette_thickness,
        )
    except Exception:
        return None


def render_ink_constraint_bytes(
    bgr_or_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scales: Tuple[int, ...] = INK_SCALES,
    fmt: str = "png",
    **kwargs,
) -> Optional[bytes]:
    """Like render_ink_constraint_image but returns encoded bytes for API payloads."""
    img = render_ink_constraint_image(bgr_or_gray, mask=mask, scales=scales, **kwargs)
    if img is None or _cv2 is None:
        return None
    ok, buf = _cv2.imencode(f".{fmt.lstrip('.')}", img)
    return bytes(buf) if ok else None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ink_runtime_status() -> dict:
    """Describe which backend tiers are active."""
    ximgproc = getattr(_cv2, "ximgproc", None) if _cv2 is not None else None
    thin = (
        "scikit-image" if _ski_skeletonize is not None
        else ("opencv-ximgproc" if (ximgproc is not None and hasattr(ximgproc, "thinning")) else "zhang-suen-numpy")
    )
    return {
        "ok": _cv2 is not None,
        "optimized": _cv2 is not None and thin != "zhang-suen-numpy",
        "morph_backend": "opencv" if _cv2 is not None else "missing",
        "thin_backend": thin,
        "label_backend": "opencv" if _cv2 is not None else "missing",
        "message": f"Ink Centerline v2 ready (thin: {thin})" if _cv2 is not None else "OpenCV missing",
    }
