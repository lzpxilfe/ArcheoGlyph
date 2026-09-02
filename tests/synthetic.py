# -*- coding: utf-8 -*-
"""
Synthetic image fixtures drawn with OpenCV so tests need no real photos.

All functions return uint8 numpy arrays in OpenCV's BGR (or BGRA) order.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def blank(size=400, color=(240, 240, 240)):
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = np.array(color, dtype=np.uint8)
    return canvas


def ellipse_blade(size=400, color=(60, 90, 140)):
    """Filled vertical ellipse resembling a blade silhouette on a light background."""
    img = blank(size)
    cv2.ellipse(img, (size // 2, size // 2), (size // 8, size // 3), 0, 0, 360, color, -1)
    return img


def blade_mask(size=400):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 8, size // 3), 0, 0, 360, 255, -1)
    return mask


def mirror_with_rings(size=400):
    """Bronze-ish disc with two dark concentric rings and a central boss."""
    img = blank(size, color=(235, 235, 235))
    c = (size // 2, size // 2)
    r = int(size * 0.4)
    cv2.circle(img, c, r, (70, 120, 160), -1)
    cv2.circle(img, c, int(r * 0.75), (40, 60, 90), 3)
    cv2.circle(img, c, int(r * 0.45), (40, 60, 90), 3)
    cv2.circle(img, c, int(r * 0.12), (40, 60, 90), -1)
    return img


def dark_flint_on_white(size=400):
    """Dark grey flint shape on white paper with a soft cast shadow to the lower right."""
    img = blank(size, color=(252, 252, 252))
    pts = np.array([
        [size * 0.50, size * 0.12],
        [size * 0.66, size * 0.40],
        [size * 0.60, size * 0.85],
        [size * 0.40, size * 0.85],
        [size * 0.34, size * 0.40],
    ], dtype=np.int32)
    shadow = np.zeros_like(img)
    cv2.fillPoly(shadow, [pts + np.array([12, 14])], (200, 200, 200))
    shadow = cv2.GaussianBlur(shadow, (0, 0), 6)
    img = np.where(shadow > 0, np.minimum(img, 255 - (255 - shadow) // 4), img).astype(np.uint8)
    cv2.fillPoly(img, [pts], (55, 55, 60))
    return img


def rgba_cutout(size=400):
    """Object with a fully transparent background; alpha is the exact mask."""
    bgr = ellipse_blade(size, color=(120, 80, 40))
    alpha = blade_mask(size)
    return np.dstack([bgr, alpha])


def red_stroke_on_gray(size=300):
    """Isoluminant red stroke on a mid-gray ground (invisible in luminance alone)."""
    img = blank(size, color=(120, 120, 120))
    cv2.line(img, (30, size // 2), (size - 30, size // 2), (60, 60, 200), 3, cv2.LINE_AA)
    return img


def single_dark_stroke(size=300, thickness=3):
    """One dark diagonal stroke on white. Returns (image, drawn_length_px)."""
    img = blank(size, color=(250, 250, 250))
    p0 = (40, 60)
    p1 = (size - 40, size - 60)
    cv2.line(img, p0, p1, (20, 20, 20), thickness, cv2.LINE_AA)
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return img, length


def y_junction(size=300, thickness=3):
    """Three strokes meeting at one junction (Y shape)."""
    img = blank(size, color=(250, 250, 250))
    c = (size // 2, size // 2)
    for end in ((size // 2, 30), (40, size - 40), (size - 40, size - 40)):
        cv2.line(img, c, end, (20, 20, 20), thickness, cv2.LINE_AA)
    return img


def line_drawing_sherd(size=400):
    """Thin black outline drawing of a rim sherd profile on white, like a report figure."""
    img = blank(size, color=(255, 255, 255))
    pts = np.array([
        [size * 0.30, size * 0.20], [size * 0.70, size * 0.20],
        [size * 0.72, size * 0.30], [size * 0.62, size * 0.80],
        [size * 0.38, size * 0.80], [size * 0.28, size * 0.30],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (int(size * 0.33), int(size * 0.32)), (int(size * 0.67), int(size * 0.32)), (0, 0, 0), 1, cv2.LINE_AA)
    return img


def encode_png(image):
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return bytes(buf)


def write_png(path, image):
    path = str(path)
    assert cv2.imwrite(path, image)
    return path
