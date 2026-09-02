# -*- coding: utf-8 -*-
"""
Vectorise raster symbol images (AI output, painted templates) into SVG.

QGIS-free. Two strategies:

* ``vtracer`` when the optional package is installed - smooth Bezier curves.
* OpenCV contour tracing otherwise - colour quantisation followed by
  ``findContours`` with even-odd holes, which is always available because
  Auto Trace already requires OpenCV.

For stroke-oriented styles (Line, Measured) the image is reduced to ink
centrelines instead of filled regions, reusing ``ink_centerline``.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .autotrace.svg_builder import SVG_NS, smooth_closed_path

MIN_REGION_AREA_RATIO = 0.0008     # of the image area
MAX_PATHS = 60
DEFAULT_MAX_COLORS = 4
BACKGROUND_TOLERANCE = 26          # per-channel distance to the border colour


def vtracer_available() -> bool:
    return importlib.util.find_spec("vtracer") is not None


# ---------------------------------------------------------------------------
# Foreground detection
# ---------------------------------------------------------------------------

def foreground_mask(image: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Foreground mask for a generated symbol image.

    Alpha wins when informative; otherwise pixels close to the median border
    colour are treated as background (AI backends return opaque images on a
    flat ground).
    """
    h, w = image.shape[:2]
    if alpha is not None:
        coverage = float(np.count_nonzero(alpha > 128)) / float(max(1, alpha.size))
        if 0.002 < coverage < 0.985:
            return (alpha > 128).astype(np.uint8) * 255

    b = max(2, min(h, w) // 40)
    border = np.concatenate([
        image[:b].reshape(-1, 3), image[-b:].reshape(-1, 3),
        image[:, :b].reshape(-1, 3), image[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    distance = np.abs(image.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = (distance > BACKGROUND_TOLERANCE).astype(np.uint8) * 255
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def quantize(image: np.ndarray, mask: np.ndarray, max_colors: int = DEFAULT_MAX_COLORS, seed: int = 0):
    """
    Reduce the masked region to at most ``max_colors`` tones.

    :return: (label image with -1 outside the mask, list of BGR centres)
    """
    inside = mask > 0
    labels = np.full(mask.shape, -1, dtype=np.int32)
    pixels = image[inside].astype(np.float32)
    if pixels.size == 0:
        return labels, []
    k = int(max(1, min(int(max_colors), 8, len(np.unique(pixels, axis=0)))))
    if k == 1:
        labels[inside] = 0
        return labels, [np.median(pixels, axis=0)]
    cv2.setRNGSeed(int(seed))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 14, 1.0)
    _, flat, centers = cv2.kmeans(pixels, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    labels[inside] = flat.reshape(-1)
    return labels, [centers[i] for i in range(k)]


def _hex(bgr) -> str:
    b, g, r = (int(max(0, min(255, round(float(v))))) for v in bgr[:3])
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def _luma(bgr) -> float:
    b, g, r = (float(v) for v in bgr[:3])
    return 0.114 * b + 0.587 * g + 0.299 * r


# ---------------------------------------------------------------------------
# Contour tracing
# ---------------------------------------------------------------------------

def _region_paths(region: np.ndarray, min_area: float, smooth: bool) -> List[str]:
    """
    Path data for one colour region, outer contours plus their holes
    (even-odd fill rule keeps holes transparent).
    """
    contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    children: Dict[int, List[int]] = {}
    for idx, (_nxt, _prev, _child, parent) in enumerate(hierarchy):
        if parent >= 0:
            children.setdefault(parent, []).append(idx)

    paths: List[str] = []
    for idx, contour in enumerate(contours):
        if hierarchy[idx][3] >= 0:  # hole; emitted with its parent
            continue
        if abs(cv2.contourArea(contour)) < min_area:
            continue
        parts = []
        for member in [idx] + children.get(idx, []):
            c = contours[member]
            if abs(cv2.contourArea(c)) < min_area * 0.35:
                continue
            epsilon = max(0.8, 0.0035 * cv2.arcLength(c, True))
            approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            if smooth:
                parts.append(smooth_closed_path(approx.tolist(), corner_deg=34.0))
            else:
                pts = " L ".join(f"{int(x)},{int(y)}" for x, y in approx)
                parts.append(f"M {pts} Z")
        if parts:
            paths.append(" ".join(p for p in parts if p))
    return paths


def _vtracer_svg(png_bytes: bytes, max_colors: int) -> Optional[str]:
    """SVG from vtracer, or None when unavailable or it fails."""
    try:
        import vtracer

        svg = vtracer.convert_raw_image_to_svg(
            bytes(png_bytes),
            img_format="png",
            colormode="color",
            color_precision=max(1, min(8, int(max_colors).bit_length() + 2)),
            filter_speckle=6,
            mode="spline",
        )
        return svg if svg and "<svg" in svg else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_png(png_bytes: bytes):
    """Decode PNG bytes to (bgr, alpha|None)."""
    buf = np.frombuffer(bytes(png_bytes), dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), None
    if raw.shape[2] == 4:
        return np.ascontiguousarray(raw[:, :, :3]), np.ascontiguousarray(raw[:, :, 3])
    return np.ascontiguousarray(raw[:, :, :3]), None


def vectorize_image(
    image: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    max_colors: int = DEFAULT_MAX_COLORS,
    stroke_style: bool = False,
    smooth: bool = True,
    seed: int = 0,
) -> Tuple[Optional[str], List[str]]:
    """
    Vectorise a BGR image into SVG text.

    :param stroke_style: emit ink centrelines (Line / Measured) instead of
        filled colour regions.
    :return: (svg text or None, warnings)
    """
    warnings: List[str] = []
    if cv2 is None:
        return None, ["OpenCV is required to vectorise raster output."]

    h, w = image.shape[:2]
    mask = foreground_mask(image, alpha)
    if int(np.count_nonzero(mask)) < 40:
        return None, ["No symbol shape could be separated from the background."]

    header = f'<svg xmlns="{SVG_NS}" viewBox="0 0 {w} {h}">'

    if stroke_style:
        from .ink_centerline import extract_ink_polylines

        polylines = extract_ink_polylines(image, mask=mask, min_arc_length=max(6.0, 0.02 * min(h, w)))
        if not polylines:
            warnings.append("No strokes were found; falling back to filled regions.")
        else:
            parts = [header]
            outline = _region_paths(mask, float(h * w) * MIN_REGION_AREA_RATIO, smooth)
            for path in outline[:2]:
                parts.append(
                    f'<path d="{path}" fill="none" stroke="param(outline) #1e1a16" '
                    'stroke-width="param(outline-width) 2.2" stroke-linejoin="round" fill-rule="evenodd"/>'
                )
            for pline in polylines[:MAX_PATHS]:
                d = "M " + " L ".join(f"{int(x)},{int(y)}" for x, y in pline)
                parts.append(
                    f'<path d="{d}" fill="none" stroke="#1e1a16" stroke-opacity="0.85" '
                    'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>'
                )
            parts.append("</svg>")
            return "".join(parts), warnings

    labels, centers = quantize(image, mask, max_colors=max_colors, seed=seed)
    if not centers:
        return None, ["Colour quantisation produced no regions."]

    # Darkest tone last so outline-like detail paints on top of the body fill.
    order = sorted(range(len(centers)), key=lambda i: _luma(centers[i]), reverse=True)
    min_area = float(h * w) * MIN_REGION_AREA_RATIO
    parts = [header]
    emitted = 0
    body_hex = _hex(centers[order[0]])
    for rank, index in enumerate(order):
        region = ((labels == index).astype(np.uint8)) * 255
        for path in _region_paths(region, min_area, smooth):
            if emitted >= MAX_PATHS:
                warnings.append("Vectorised output was truncated to keep the symbol simple.")
                break
            colour = _hex(centers[index])
            fill = f"param(fill) {colour}" if rank == 0 else colour
            parts.append(f'<path d="{path}" fill="{fill}" fill-rule="evenodd" stroke="none"/>')
            emitted += 1
        if emitted >= MAX_PATHS:
            break

    if emitted == 0:
        return None, ["No region was large enough to vectorise."]

    outline = _region_paths(mask, min_area, smooth)
    for path in outline[:2]:
        parts.append(
            f'<path d="{path}" fill="none" stroke="param(outline) {_hex(np.array([0, 0, 0]))}" '
            'stroke-width="param(outline-width) 2.0" stroke-linejoin="round" fill-rule="evenodd"/>'
        )
    parts.append("</svg>")
    del body_hex
    return "".join(parts), warnings


def vectorize_png(
    png_bytes: bytes,
    max_colors: int = DEFAULT_MAX_COLORS,
    stroke_style: bool = False,
    prefer_vtracer: bool = True,
    seed: int = 0,
) -> Tuple[Optional[str], List[str]]:
    """
    Vectorise PNG bytes. Returns ``(svg, warnings)``; svg is None on failure so
    the caller can keep the raster.
    """
    if cv2 is None:
        return None, ["OpenCV is required to vectorise raster output."]
    image, alpha = decode_png(png_bytes)
    if image is None:
        return None, ["The generated image could not be decoded."]

    if prefer_vtracer and not stroke_style and vtracer_available():
        svg = _vtracer_svg(png_bytes, max_colors)
        if svg:
            return svg, []
    return vectorize_image(image, alpha, max_colors=max_colors, stroke_style=stroke_style, seed=seed)


def vectorize_result(result, style: str = "", max_colors: int = DEFAULT_MAX_COLORS):
    """
    Fill in ``result.svg`` from ``result.raster_png`` when the result is
    raster-only. Mutates and returns the SymbolResult.
    """
    if result is None or result.is_vector or not result.raster_png:
        return result
    stroke_style = any(token in str(style or "").lower() for token in ("line", "measured"))
    svg, warnings = vectorize_png(result.raster_png, max_colors=max_colors, stroke_style=stroke_style)
    for message in warnings:
        result.add_warning(message)
    if svg:
        from .autotrace.svg_builder import add_provenance, finalize_svg

        svg, info = finalize_svg(svg)
        result.meta.update(info)
        result.svg = add_provenance(svg, result.meta)
    else:
        result.add_warning("Kept the raster image: vectorisation did not produce a usable shape.")
    return result


def polylines_from_regions(mask: np.ndarray, min_area_ratio: float = MIN_REGION_AREA_RATIO) -> List[Sequence]:
    """Outer contours of ``mask`` as point lists (helper for callers/tests)."""
    if cv2 is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(mask.size) * min_area_ratio
    return [c.reshape(-1, 2).tolist() for c in contours if abs(cv2.contourArea(c)) >= min_area]
