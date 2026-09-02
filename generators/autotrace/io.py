# -*- coding: utf-8 -*-
"""
Image loading and analysis prescaling for Auto Trace (QGIS-free).

``load_image`` returns 8-bit BGR plus an optional alpha channel:
* EXIF orientation is honoured (OpenCV applies it for the colour decode).
* RGBA inputs keep their alpha, which later serves as an exact silhouette.
* Grayscale and 16-bit inputs are normalised to 8-bit BGR.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class LoadedImage:
    bgr: np.ndarray
    alpha: Optional[np.ndarray] = None
    path: str = ""

    @property
    def shape(self):
        return self.bgr.shape[:2]


def _to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr.astype(np.float32) / 257.0).round().astype(np.uint8)
    arr = arr.astype(np.float32)
    hi = float(arr.max()) if arr.size else 1.0
    scale = 255.0 / hi if hi > 1.0 else 255.0
    return np.clip(arr * scale, 0, 255).astype(np.uint8)


def decode_image(data, path=""):
    """Decode image bytes into a LoadedImage, or None when undecodable."""
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    alpha = None
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = _to_uint8(raw[:, :, 3])

    # IMREAD_COLOR applies EXIF orientation and always yields 8-bit BGR.
    color = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if color is None:
        if raw.ndim == 2:
            color = cv2.cvtColor(_to_uint8(raw), cv2.COLOR_GRAY2BGR)
        elif raw.shape[2] == 4:
            color = cv2.cvtColor(_to_uint8(raw), cv2.COLOR_BGRA2BGR)
        else:
            color = _to_uint8(raw[:, :, :3])
    if alpha is not None and alpha.shape[:2] != color.shape[:2]:
        alpha = None
    return LoadedImage(bgr=np.ascontiguousarray(color), alpha=alpha, path=str(path))


def load_image(path):
    with open(path, "rb") as stream:
        data = stream.read()
    return decode_image(data, path=path)


def resize_alpha(alpha, shape):
    """Resize an alpha channel to (h, w) with linear interpolation."""
    if alpha is None:
        return None
    h, w = shape[:2]
    if alpha.shape[:2] == (h, w):
        return alpha
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)


def adaptive_prescale(img, force_lowres_upscale=False, detail_fast=False):
    """
    Resize input image for contour analysis.
    - Downscale very large inputs for speed/stability.
    - Upscale low-resolution inputs to recover edge/motif geometry.
    - When force_lowres_upscale is enabled, upscale more aggressively.
    - detail_fast mode uses slightly lower scale targets to preserve speed.
    Returns (resized_img, scale_factor).
    """
    if img is None:
        return img, 1.0
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return img, 1.0

    max_side = float(max(h, w))
    min_side = float(min(h, w))
    scale = 1.0

    # Bound huge inputs.
    if max_side > 1600.0:
        scale = 1600.0 / max_side
    elif bool(force_lowres_upscale):
        # Explicit user opt-in for stronger low-res recovery.
        if detail_fast:
            if max_side < 1320.0:
                scale = min(3.2, 1360.0 / max_side)
            if min_side < 920.0:
                scale = max(scale, min(3.0, 940.0 / min_side))
            if min_side < 360.0:
                scale = max(scale, min(3.4, 760.0 / min_side))
        else:
            if max_side < 1440.0:
                scale = min(4.0, 1600.0 / max_side)
            if min_side < 1024.0:
                scale = max(scale, min(3.6, 1200.0 / min_side))
            if min_side < 360.0:
                scale = max(scale, min(4.2, 900.0 / min_side))
    # Default low-resolution catalog/screenshot handling.
    elif detail_fast:
        if max_side < 780.0:
            scale = min(2.8, 980.0 / max_side)
        elif min_side < 500.0:
            scale = min(2.3, 690.0 / min_side)
    elif max_side < 840.0:
        scale = min(3.2, 1080.0 / max_side)
    elif min_side < 520.0:
        scale = min(2.6, 760.0 / min_side)

    if abs(scale - 1.0) < 0.05:
        return img, 1.0

    new_w = max(2, int(round(float(w) * scale)))
    new_h = max(2, int(round(float(h) * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return resized, float(scale)
