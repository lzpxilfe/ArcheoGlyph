# -*- coding: utf-8 -*-
"""
Polyline geometry helpers for Auto Trace.

Auto-generated from the former ContourGenerator methods; QGIS-free.
"""

import math

import cv2
import numpy as np

from ...log import log_exception


def clamp(value, lower, upper):
    """Clamp numeric value into [lower, upper]."""
    return max(lower, min(upper, value))


def polyline_to_path(points):
    """Convert list of points to SVG polyline path."""
    if not points or len(points) < 2:
        return ""
    start = points[0]
    path = f"M {int(start[0])},{int(start[1])} "
    for pt in points[1:]:
        path += f"L {int(pt[0])},{int(pt[1])} "
    return path.strip()


def circle_path(cx, cy, radius, steps=72):
    """Build SVG path for a smooth circle-like outline."""
    r = max(2.0, float(radius))
    n = int(max(24, min(160, int(steps))))
    pts = []
    for i in range(n):
        t = (2.0 * np.pi * float(i)) / float(n)
        x = float(cx) + (r * np.cos(t))
        y = float(cy) + (r * np.sin(t))
        pts.append([x, y])
    if not pts:
        return ""
    path = f"M {pts[0][0]:.2f},{pts[0][1]:.2f} "
    for x, y in pts[1:]:
        path += f"L {x:.2f},{y:.2f} "
    path += "Z"
    return path


def circle_polyline(cx, cy, radius, steps=52):
    """Return closed polyline points approximating a circle."""
    r = max(2.0, float(radius))
    n = int(max(20, min(120, int(steps))))
    pts = []
    for i in range(n):
        t = (2.0 * np.pi * float(i)) / float(n)
        x = int(round(float(cx) + (r * np.cos(t))))
        y = int(round(float(cy) + (r * np.sin(t))))
        pts.append([x, y])
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def arc_polyline(cx, cy, radius, start_deg, end_deg, steps=16):
    """Return open arc polyline points (degrees)."""
    r = max(2.0, float(radius))
    n = int(max(6, min(96, int(steps))))
    a0 = np.deg2rad(float(start_deg))
    a1 = np.deg2rad(float(end_deg))
    pts = []
    for i in range(n):
        t = a0 + ((a1 - a0) * (float(i) / float(max(1, n - 1))))
        x = int(round(float(cx) + (r * np.cos(t))))
        y = int(round(float(cy) + (r * np.sin(t))))
        pts.append([x, y])
    return pts


def diamond_polyline(cx, cy, radius):
    """Create diamond-shaped frame polyline points."""
    try:
        r = float(max(1.0, radius))
        pts = [
            [int(round(cx)), int(round(cy - r))],
            [int(round(cx + r)), int(round(cy))],
            [int(round(cx)), int(round(cy + r))],
            [int(round(cx - r)), int(round(cy))],
            [int(round(cx)), int(round(cy - r))],
        ]
        return pts
    except Exception as e:
        log_exception("diamond_polyline", e)
        return []


def line_centroid_and_length(line):
    """Return centroid (x,y) and arc-length of a polyline."""
    if not line or len(line) < 2:
        return None, 0.0
    arr = np.asarray(line, dtype=np.float32)
    cx = float(np.mean(arr[:, 0]))
    cy = float(np.mean(arr[:, 1]))
    seg = np.diff(arr, axis=0)
    arc_len = float(np.sum(np.sqrt(np.sum(seg * seg, axis=1))))
    return (cx, cy), arc_len


def line_ring_likeness(line, cx, cy):
    """
    Return a ring-likeness score in [0,1] where 1 means near-concentric arc.
    """
    if not line or len(line) < 4:
        return 0.0
    arr = np.asarray(line, dtype=np.float32)
    dx = arr[:, 0] - float(cx)
    dy = arr[:, 1] - float(cy)
    rr = np.sqrt((dx * dx) + (dy * dy))
    r_mean = float(np.mean(rr)) if len(rr) > 0 else 0.0
    if r_mean <= 1e-6:
        return 0.0
    r_std = float(np.std(rr))
    radial_cv = r_std / r_mean
    # Near-ring lines have small radial variation from center.
    radial_score = max(0.0, min(1.0, 1.0 - (radial_cv / 0.18)))

    angles = np.unwrap(np.arctan2(dy, dx))
    ang_span = float(np.max(angles) - np.min(angles)) if len(angles) > 1 else 0.0
    # Long angular sweep indicates circular bands/arcs.
    sweep_score = max(0.0, min(1.0, ang_span / (np.pi * 0.80)))
    return max(0.0, min(1.0, (0.65 * radial_score) + (0.35 * sweep_score)))


def line_angle_span(line, cx, cy):
    """Return unwrapped angular span (radians) of a polyline around center."""
    if not line or len(line) < 3:
        return 0.0
    arr = np.asarray(line, dtype=np.float32)
    dx = arr[:, 0] - float(cx)
    dy = arr[:, 1] - float(cy)
    angles = np.unwrap(np.arctan2(dy, dx))
    if len(angles) < 2:
        return 0.0
    return float(np.max(angles) - np.min(angles))


def rotate_line_about_center(line, cx, cy, angle_rad):
    """Rotate polyline points around center."""
    if not line or len(line) < 2:
        return []
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    out = []
    for pt in line:
        x = float(pt[0]) - float(cx)
        y = float(pt[1]) - float(cy)
        rx = (x * c) - (y * s) + float(cx)
        ry = (x * s) + (y * c) + float(cy)
        out.append([int(round(rx)), int(round(ry))])
    return out


def merge_distinct_lines(base_lines,
    extra_lines,
    min_center_sep=6.0,
    max_lines=12,
    min_arc_len=2.0,
):
    """
    Merge line sets while avoiding near-duplicate center positions.
    Keeps insertion order and caps output size.
    """
    out = list(base_lines or [])
    target = int(max(0, int(max_lines)))
    if target <= 0:
        return []

    centers = []
    for line in out:
        center, _ = line_centroid_and_length(line)
        if center is not None:
            centers.append(center)
    if len(out) >= target:
        return out[:target]

    min_sep = max(1.0, float(min_center_sep))
    min_arc = max(0.0, float(min_arc_len))
    for line in extra_lines or []:
        if len(out) >= target:
            break
        center, arc_len = line_centroid_and_length(line)
        if center is None or arc_len < min_arc:
            continue
        if any((((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5) < min_sep for c in centers):
            continue
        out.append(line)
        centers.append(center)
    return out[:target]


def dedupe_lines(lines, min_points=4, max_lines=20):
    """Deduplicate and simplify candidate lines while keeping strongest strokes."""
    try:
        if not lines:
            return []
        ranked = []
        signatures = set()

        for line in lines:
            if line is None or len(line) < int(min_points):
                continue
            arr = np.asarray(line, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] < int(min_points):
                continue

            dif = np.diff(arr, axis=0)
            seg = np.sqrt((dif[:, 0] * dif[:, 0]) + (dif[:, 1] * dif[:, 1]))
            length = float(np.sum(seg))
            if length < 8.0:
                continue

            poly = arr.reshape(-1, 1, 2)
            eps = max(0.6, 0.012 * length)
            simp = cv2.approxPolyDP(poly, eps, False)
            if simp is not None and len(simp) >= int(min_points):
                arr = simp.reshape(-1, 2)

            start = arr[0]
            end = arr[-1]
            center = np.mean(arr, axis=0)
            sig = (
                int(round(start[0] / 3.0)),
                int(round(start[1] / 3.0)),
                int(round(end[0] / 3.0)),
                int(round(end[1] / 3.0)),
                int(round(center[0] / 4.0)),
                int(round(center[1] / 4.0)),
                int(round(length / 6.0)),
            )
            if sig in signatures:
                continue
            signatures.add(sig)
            ranked.append((length, [[int(round(p[0])), int(round(p[1]))] for p in arr]))

        ranked.sort(key=lambda item: item[0], reverse=True)
        keep = max(1, int(max_lines))
        return [item[1] for item in ranked[:keep]]
    except Exception as e:
        log_exception("dedupe_lines", e)
        if not lines:
            return []
        return lines[: max(1, int(max_lines))]


def remove_near_horizontal_lines(lines, ratio=0.35):
    """Drop near-horizontal lines to avoid crossbar artifacts in line style."""
    filtered = []
    for line in lines or []:
        if not line or len(line) < 2:
            continue
        xs = [int(pt[0]) for pt in line]
        ys = [int(pt[1]) for pt in line]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x >= 8 and span_y <= (float(span_x) * float(ratio)):
            continue
        filtered.append(line)
    return filtered


def smooth_1d(values, window=9):
    """Simple moving-average smoothing for 1D numeric arrays."""
    if values is None or len(values) == 0:
        return values
    w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    if len(values) < w:
        return values.astype(np.float32)
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def polyline_arc_length(points):
    """Total length of a polyline given as a sequence of (x, y) pairs."""
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    return total


def line_center(points):
    """Mean point of a polyline, or None when empty."""
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))
