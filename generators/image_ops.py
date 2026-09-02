# -*- coding: utf-8 -*-
"""
Pixel operations for AI symbol post-processing.

QGIS-free and vectorised. These are faithful ports of the per-pixel loops the
Hugging Face and Gemini backends used to run in pure Python: each generated
image went through four to six chained loops calling ``QImage.pixelColor`` once
per pixel, which costs tens of seconds at 1000x1000.

Convention: images are ``uint8`` arrays shaped ``(h, w, 3)`` in RGB order with a
separate ``(h, w)`` alpha array; masks are boolean ``(h, w)`` arrays.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

RGB = Tuple[int, int, int]

DEFAULT_MATERIAL_RGB: RGB = (88, 112, 92)
MASK_DARK_THRESHOLD = 90       # silhouette PNGs are black object on white ground
MIN_ALPHA = 8


# ---------------------------------------------------------------------------
# Small scalar helpers (shared with the callers so results match exactly)
# ---------------------------------------------------------------------------

def clamp_rgb(rgb: Sequence[float]) -> RGB:
    return tuple(max(0, min(255, int(v))) for v in rgb)


def blend_rgb(base_rgb: Sequence[float], mix_rgb: Sequence[float], mix_ratio: float = 0.35) -> RGB:
    """Blend two colours; truncating like the original int() arithmetic."""
    br, bg, bb = clamp_rgb(base_rgb)
    mr, mg, mb = clamp_rgb(mix_rgb)
    t = max(0.0, min(1.0, float(mix_ratio)))
    return (
        int((br * (1.0 - t)) + (mr * t)),
        int((bg * (1.0 - t)) + (mg * t)),
        int((bb * (1.0 - t)) + (mb * t)),
    )


def luma(rgb: Sequence[float]) -> float:
    return (0.299 * rgb[0]) + (0.587 * rgb[1]) + (0.114 * rgb[2])


def parse_hex_rgb(hex_color) -> Optional[RGB]:
    """Parse ``#RRGGBB`` into a tuple, or None when it is not a valid colour."""
    value = str(hex_color or "").strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return None


def _luma_array(rgb: np.ndarray) -> np.ndarray:
    channels = rgb.astype(np.float64)
    return 0.299 * channels[..., 0] + 0.587 * channels[..., 1] + 0.114 * channels[..., 2]


def _truncate(values: np.ndarray) -> np.ndarray:
    """int() semantics: truncate toward zero, then clamp to a byte."""
    return np.clip(values.astype(np.int64), 0, 255).astype(np.uint8)


def mask_inside(mask_rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the object area in a black-on-white silhouette image."""
    return np.all(mask_rgb[..., :3] < MASK_DARK_THRESHOLD, axis=-1)


# ---------------------------------------------------------------------------
# Reference colour and palette
# ---------------------------------------------------------------------------

def estimate_reference_rgb(ref_rgb: np.ndarray, ref_alpha: np.ndarray, inside: np.ndarray) -> RGB:
    """
    Average colour of the artifact in the reference photo.

    Saturated mid-tones are preferred; if too few qualify, every visible pixel
    inside the silhouette is averaged instead.
    """
    visible = inside & (ref_alpha >= MIN_ALPHA)
    if not visible.any():
        return DEFAULT_MATERIAL_RGB

    values = ref_rgb[visible].astype(np.int32)
    high = values.max(axis=1)
    low = values.min(axis=1)
    saturated = (high - low >= 12) & (high >= 28) & (high <= 245)

    selected = values[saturated]
    if len(selected) < 25:
        # The original kept the saturated running total and then added every
        # visible pixel on top of it, so the fallback averages both together.
        selected = np.concatenate([selected, values]) if len(selected) else values
    if len(selected) < 5:
        return DEFAULT_MATERIAL_RGB

    totals = selected.sum(axis=0)
    count = len(selected)
    return (int(totals[0] / count), int(totals[1] / count), int(totals[2] / count))


def extract_reference_palette(
    ref_rgb: np.ndarray,
    ref_alpha: np.ndarray,
    inside: np.ndarray,
    max_colors: int = 4,
    step: Optional[int] = None,
    min_distance: float = 24.0,
) -> List[RGB]:
    """
    Dominant material tones, binned in 32-level RGB cubes and ranked by pixel
    count, skipping tones too close to one already chosen.
    """
    h, w = inside.shape[:2]
    if step is None:
        step = 1 if (w * h) <= 180000 else 2

    sub_inside = inside[::step, ::step]
    sub_rgb = ref_rgb[::step, ::step]
    sub_alpha = ref_alpha[::step, ::step]

    visible = sub_inside & (sub_alpha >= MIN_ALPHA)
    if not visible.any():
        return []

    values = sub_rgb[visible].astype(np.int32)
    high = values.max(axis=1)
    low = values.min(axis=1)
    keep = (high - low >= 8) & (high >= 20) & (high <= 248)
    values = values[keep]
    if not len(values):
        return []

    keys = values >> 5
    flat_keys = (keys[:, 0].astype(np.int64) << 12) | (keys[:, 1].astype(np.int64) << 6) | keys[:, 2]
    unique, inverse, counts = np.unique(flat_keys, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(unique), 3), dtype=np.int64)
    np.add.at(sums, inverse, values)

    # Rank by pixel count, breaking ties by first appearance so the result
    # matches the original dict-insertion ordering rather than key order.
    first_seen = np.full(len(unique), len(flat_keys), dtype=np.int64)
    np.minimum.at(first_seen, inverse, np.arange(len(flat_keys), dtype=np.int64))
    order = np.lexsort((first_seen, -counts))
    palette: List[RGB] = []
    for index in order:
        count = max(1, int(counts[index]))
        rgb = (
            int(sums[index][0] / count),
            int(sums[index][1] / count),
            int(sums[index][2] / count),
        )
        if any(
            ((rgb[0] - ex[0]) ** 2 + (rgb[1] - ex[1]) ** 2 + (rgb[2] - ex[2]) ** 2) ** 0.5 < min_distance
            for ex in palette
        ):
            continue
        palette.append(rgb)
        if len(palette) >= int(max_colors):
            break
    return palette


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def estimate_texture_noise(rgb: np.ndarray, inside: np.ndarray) -> float:
    """
    Mean absolute 4-neighbour colour difference inside the silhouette: a proxy
    for painterly noise in a generated image.
    """
    h, w = inside.shape[:2]
    if w < 3 or h < 3:
        return 0.0

    values = rgb.astype(np.int32)
    core = values[1:-1, 1:-1]
    differences = (
        np.abs(core - values[1:-1, :-2]).sum(axis=-1)
        + np.abs(core - values[1:-1, 2:]).sum(axis=-1)
        + np.abs(core - values[:-2, 1:-1]).sum(axis=-1)
        + np.abs(core - values[2:, 1:-1]).sum(axis=-1)
    ) / 12.0

    selected = inside[1:-1, 1:-1]
    samples = int(np.count_nonzero(selected))
    if samples < 20:
        return 0.0
    return float(differences[selected].sum() / samples)


def estimate_luma_variance(rgb: np.ndarray, inside: np.ndarray) -> float:
    """Luminance variance inside the silhouette (flatness detector)."""
    count = int(np.count_nonzero(inside))
    if count < 20:
        return 0.0
    values = _luma_array(rgb)[inside]
    mean = float(values.sum() / count)
    mean_square = float((values * values).sum() / count)
    return max(0.0, mean_square - (mean * mean))


# ---------------------------------------------------------------------------
# Harmonisation
# ---------------------------------------------------------------------------

def typology_tones(base_rgb: Sequence[float], palette_rgb: Optional[Sequence[Sequence[float]]] = None):
    """
    Derive the shadow / mid / highlight / patina tones for the typology style,
    keeping them far enough apart to read as separate tone blocks.
    """
    base = clamp_rgb(base_rgb)
    palette = [clamp_rgb(rgb) for rgb in (palette_rgb or []) if rgb]
    if not palette:
        palette = [base]

    ordered = sorted(palette, key=luma, reverse=True)
    hi_seed = ordered[0]
    lo_seed = ordered[-1]
    mid_seed = ordered[1] if len(ordered) > 2 else base

    highlight = blend_rgb(base, hi_seed, 0.52)
    mid = blend_rgb(base, mid_seed, 0.42)
    shadow = blend_rgb(base, lo_seed, 0.56)

    if (luma(highlight) - luma(mid)) < 16.0:
        highlight = blend_rgb(mid, (255, 255, 255), 0.18)
    if (luma(mid) - luma(shadow)) < 16.0:
        shadow = blend_rgb(mid, (0, 0, 0), 0.22)
    if (luma(highlight) - luma(shadow)) < 32.0:
        highlight = blend_rgb(highlight, (255, 255, 255), 0.12)
        shadow = blend_rgb(shadow, (0, 0, 0), 0.12)

    patina = ordered[2] if len(ordered) > 2 else blend_rgb(mid, highlight, 0.34)
    return shadow, mid, highlight, patina


def _blend_arrays(base: np.ndarray, mix: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-pixel blend with int() truncation, matching blend_rgb."""
    t = t[..., None] if t.ndim == base.ndim - 1 else t
    return np.trunc(base * (1.0 - t) + mix * t)


def harmonize_typology(
    rgb: np.ndarray,
    alpha: np.ndarray,
    base_rgb: Sequence[float],
    palette_rgb: Optional[Sequence[Sequence[float]]] = None,
    preserve_ratio: float = 0.34,
):
    """
    Map the generated image onto two to three analogous tones plus a patina
    accent, keeping ``preserve_ratio`` of the original pixel.

    :return: (rgb, alpha) with alpha set to 255 where the pixel was visible
    """
    shadow, mid, highlight, patina = typology_tones(base_rgb, palette_rgb)
    preserve = max(0.10, min(0.62, float(preserve_ratio)))

    values = rgb.astype(np.float64)
    lum = _luma_array(values) / 255.0
    # Quantise to five steps to suppress micro-variation, as the original did.
    lum = np.round(np.clip(lum, 0.0, 1.0) * 5.0) / 5.0

    shadow_a = np.array(shadow, dtype=np.float64)
    mid_a = np.array(mid, dtype=np.float64)
    high_a = np.array(highlight, dtype=np.float64)
    patina_a = np.array(patina, dtype=np.float64)

    tone = np.empty_like(values)
    band_low = lum <= 0.25
    band_mid = (~band_low) & (lum <= 0.50)
    band_high = (~band_low) & (~band_mid) & (lum <= 0.78)
    band_top = lum > 0.78

    tone[band_low] = shadow_a
    if band_mid.any():
        t = ((lum[band_mid] - 0.25) / 0.25)[:, None]
        tone[band_mid] = np.trunc(shadow_a * (1.0 - t) + mid_a * t)
    if band_high.any():
        t = ((lum[band_high] - 0.50) / 0.28)[:, None]
        tone[band_high] = np.trunc(mid_a * (1.0 - t) + high_a * t)
    tone[band_top] = high_a

    channels = rgb.astype(np.int32)
    saturation = channels.max(axis=-1) - channels.min(axis=-1)
    patina_mix = np.clip((saturation - 12.0) / 180.0, 0.0, 0.22)[..., None]
    tone = np.trunc(tone * (1.0 - patina_mix) + patina_a * patina_mix)

    blended = np.trunc(tone * (1.0 - preserve) + values * preserve)

    visible = alpha >= MIN_ALPHA
    out_rgb = rgb.copy()
    out_rgb[visible] = _truncate(blended)[visible]
    out_alpha = alpha.copy()
    out_alpha[visible] = 255
    return out_rgb, out_alpha


def harmonize_colored(
    rgb: np.ndarray,
    alpha: np.ndarray,
    base_rgb: Sequence[float],
    flatten: bool = False,
    preserve_ratio: float = 0.18,
):
    """Pull the generated colours back toward the measured material colour."""
    base = np.array(clamp_rgb(base_rgb), dtype=np.float64)
    lum = _luma_array(rgb) / 255.0
    if flatten:
        lum = np.round(lum * 3.0) / 3.0
    shade = (0.58 + (0.64 * lum))[..., None]
    target = np.clip(np.trunc(base * shade), 0, 255)
    blended = np.trunc(target * (1.0 - preserve_ratio) + rgb.astype(np.float64) * preserve_ratio)

    visible = alpha >= MIN_ALPHA
    out_rgb = rgb.copy()
    out_rgb[visible] = _truncate(blended)[visible]
    out_alpha = alpha.copy()
    out_alpha[visible] = 255
    return out_rgb, out_alpha


def harmonize_mono(rgb: np.ndarray, alpha: np.ndarray, publication: bool = False):
    """Flatten to a stable monochrome ramp for Line and Measured styles."""
    lum = np.trunc(_luma_array(rgb)).astype(np.int32)
    if publication:
        value = np.where(lum < 135, 25, 70)
    else:
        value = np.trunc(20 + (lum * 0.35)).astype(np.int32)
    value = np.clip(value, 0, 255).astype(np.uint8)

    visible = alpha >= MIN_ALPHA
    out_rgb = rgb.copy()
    out_rgb[visible] = np.repeat(value[..., None], 3, axis=-1)[visible]
    out_alpha = alpha.copy()
    out_alpha[visible] = 255
    return out_rgb, out_alpha


def reference_tone_map(
    rgb: np.ndarray,
    alpha: np.ndarray,
    ref_rgb: np.ndarray,
    inside: np.ndarray,
    strength: float = 0.5,
):
    """
    Apply a coarse three-level lightness structure taken from the reference
    photo, so flat AI output regains measured highlights and shadows.
    """
    if not inside.any():
        return rgb, alpha
    ref_luma = _luma_array(ref_rgb)
    selected = ref_luma[inside]
    min_l = float(selected.min())
    max_l = float(selected.max())
    span = max_l - min_l
    if span < 6.0:
        return rgb, alpha

    s = max(0.0, min(1.0, float(strength)))
    norm = (ref_luma - min_l) / span
    tone = np.where(norm < 0.34, 0.90, np.where(norm < 0.68, 1.00, 1.10))[..., None]

    values = rgb.astype(np.float64)
    toned = np.clip(np.trunc(values * tone), 0, 255)
    blended = np.trunc(values * (1.0 - s) + toned * s)

    out_rgb = rgb.copy()
    out_rgb[inside] = _truncate(blended)[inside]
    out_alpha = alpha.copy()
    out_alpha[inside] = 255
    return out_rgb, out_alpha


def apply_silhouette(rgb: np.ndarray, inside: np.ndarray):
    """
    Keep only the pixels inside the silhouette, fully opaque; everything else
    becomes transparent.
    """
    out_rgb = np.zeros_like(rgb)
    out_alpha = np.zeros(inside.shape, dtype=np.uint8)
    out_rgb[inside] = rgb[inside]
    out_alpha[inside] = 255
    return out_rgb, out_alpha
