# -*- coding: utf-8 -*-
"""
Colour extraction and hex helpers for Auto Trace.

Auto-generated from the former ContourGenerator methods; QGIS-free.
"""

import cv2
import numpy as np


def hex_to_rgb(hex_color, fallback=(139, 69, 19)):
    """Convert hex color to RGB tuple."""
    value = str(hex_color or "").strip().lstrip("#")
    if len(value) != 6:
        return fallback
    try:
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )
    except Exception:
        return fallback


def rgb_to_hex(r, g, b):
    """Convert RGB values to hex color."""
    rr = max(0, min(255, int(r)))
    gg = max(0, min(255, int(g)))
    bb = max(0, min(255, int(b)))
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def blend_hex(base_hex, mix_hex, mix_ratio=0.35):
    """Blend two hex colors while keeping base dominance."""
    br, bg, bb = hex_to_rgb(base_hex)
    mr, mg, mb = hex_to_rgb(mix_hex, fallback=(br, bg, bb))
    t = max(0.0, min(1.0, float(mix_ratio)))
    r = (br * (1.0 - t)) + (mr * t)
    g = (bg * (1.0 - t)) + (mg * t)
    b = (bb * (1.0 - t)) + (mb * t)
    return rgb_to_hex(r, g, b)


def hex_luminance(hex_color):
    """Return perceived luminance for a hex color."""
    r, g, b = hex_to_rgb(hex_color)
    return (0.299 * r) + (0.587 * g) + (0.114 * b)


def hex_distance(color_a, color_b):
    """Euclidean RGB distance between two hex colors."""
    ar, ag, ab = hex_to_rgb(color_a)
    br, bg, bb = hex_to_rgb(color_b)
    return ((ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2) ** 0.5


def extract_material_palette(bgr_img, mask=None, max_colors=4):
    """
    Extract a compact material palette from masked object pixels.
    Returns colors sorted by prevalence while removing near-duplicates.
    """
    if bgr_img is None:
        return []

    try:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        valid_mask = np.ones_like(h, dtype=bool)
        if mask is not None:
            valid_mask = (mask > 0)

        color_mask = (s > 10) & (v > 24) & (v < 248) & valid_mask
        pixels = bgr_img[color_mask]
        if len(pixels) < 120:
            pixels = bgr_img[valid_mask]
        if len(pixels) < 6:
            return []

        sample_limit = 6500
        if len(pixels) > sample_limit:
            sampled_idx = np.linspace(
                0,
                len(pixels) - 1,
                num=sample_limit,
                dtype=np.int32,
            )
            pixels = pixels[sampled_idx]

        samples = np.float32(pixels)
        if len(samples) < 2:
            only = samples[0]
            return [rgb_to_hex(only[2], only[1], only[0])]

        n_colors = max(2, min(int(max_colors), 5, len(samples)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 14, 1.0)
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(
            samples,
            n_colors,
            None,
            criteria,
            8,
            cv2.KMEANS_PP_CENTERS,
        )

        counts = np.bincount(labels.reshape(-1), minlength=n_colors)
        ranked = sorted(range(n_colors), key=lambda i: int(counts[i]), reverse=True)

        palette = []
        for idx in ranked:
            center = centers[idx]
            hex_color = rgb_to_hex(center[2], center[1], center[0])
            if any(hex_distance(hex_color, existing) < 24.0 for existing in palette):
                continue
            palette.append(hex_color)
            if len(palette) >= int(max_colors):
                break

        return palette
    except Exception:
        return []


def darken_hex(hex_color, factor):
    """Darken a hex color by multiplying channels by factor [0..1]."""
    value = (hex_color or "#8B4513").strip().lstrip("#")
    if len(value) != 6:
        return "#333333"
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#333333"


def lighten_hex(hex_color, amount):
    """Lighten a hex color by blending toward white by amount [0..1]."""
    value = (hex_color or "#8B4513").strip().lstrip("#")
    if len(value) != 6:
        return "#d0d0d0"
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        a = max(0.0, min(1.0, float(amount)))
        r = int(r + ((255 - r) * a))
        g = int(g + ((255 - g) * a))
        b = int(b + ((255 - b) * a))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#d0d0d0"


def muted_hex(hex_color, keep=0.70):
    """Mute saturation by blending channels toward luminance."""
    value = (hex_color or "#8B4513").strip().lstrip("#")
    if len(value) != 6:
        return "#6f7c70"
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        lum = int((0.299 * r) + (0.587 * g) + (0.114 * b))
        k = max(0.0, min(1.0, float(keep)))
        r = int((r * k) + (lum * (1.0 - k)))
        g = int((g * k) + (lum * (1.0 - k)))
        b = int((b * k) + (lum * (1.0 - k)))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#6f7c70"


def extract_dominant_color(bgr_img, mask=None):
    """
    Representative object colour: the median of the eroded mask interior,
    ignoring specular highlights and deep shadows. Deterministic.
    """
    try:
        h, w = bgr_img.shape[:2]
        if mask is None:
            valid = np.ones((h, w), dtype=bool)
        else:
            k = max(3, int(round(min(h, w) * 0.02)) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            eroded = cv2.erode((mask > 0).astype(np.uint8) * 255, kernel)
            valid = eroded > 0
            if np.count_nonzero(valid) < 50:
                valid = mask > 0
        pixels = bgr_img[valid]
        if len(pixels) < 10:
            return "#8B4513"
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        mid = (hsv[:, 2] > 30) & (hsv[:, 2] < 235)
        if np.count_nonzero(mid) >= 50:
            pixels = pixels[mid]
        b, g, r = (int(round(float(v))) for v in np.median(pixels, axis=0))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    except Exception:
        return "#8B4513"
