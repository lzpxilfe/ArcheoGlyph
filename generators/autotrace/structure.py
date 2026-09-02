# -*- coding: utf-8 -*-
"""
Synthetic schematic structure lines (profile bands, spine, terminal bars, round bands).

Auto-generated from the former ContourGenerator methods; QGIS-free.
"""

import cv2
import numpy as np

from .geometry import smooth_1d


def estimate_spine_line(mask):
    """Estimate central spine line from mask when texture lines are weak."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return []

    top_y = int(np.min(ys))
    bot_y = int(np.max(ys))
    h = max(1, bot_y - top_y)
    step = max(2, h // 42)

    axis_ratio = 0.5
    points = []
    for y in range(top_y, bot_y + 1, step):
        row_xs = np.where(mask[y] > 0)[0]
        if len(row_xs) < 2:
            continue
        left = int(row_xs[0])
        right = int(row_xs[-1])
        width = right - left
        if width < 2:
            continue
        x_mid = int(left + (width * axis_ratio))
        x_mid = max(left + 1, min(right - 1, x_mid))
        points.append([x_mid, y])

    if len(points) < 6:
        return []
    return [points]


def estimate_profile_bands(mask, max_lines=3):
    """
    Estimate typological structural bands (rim/shoulder/base) from silhouette profile.
    This creates symbol-like interior cues without relying on image texture.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return []

    top_y = int(np.min(ys))
    bot_y = int(np.max(ys))
    h = max(1, bot_y - top_y + 1)
    w = mask.shape[1]

    widths = np.zeros((h,), dtype=np.float32)
    lefts = np.zeros((h,), dtype=np.int32)
    rights = np.zeros((h,), dtype=np.int32)
    valid = np.zeros((h,), dtype=bool)

    for y in range(top_y, bot_y + 1):
        row = np.where(mask[y] > 0)[0]
        idx = y - top_y
        if len(row) < 2:
            continue
        left = int(row[0])
        right = int(row[-1])
        width = right - left
        if width < 5:
            continue
        lefts[idx] = left
        rights[idx] = right
        widths[idx] = float(width)
        valid[idx] = True

    if int(np.count_nonzero(valid)) < 24:
        return []

    # Fill invalid rows by nearest valid width/edges.
    valid_ids = np.where(valid)[0]
    for i in range(h):
        if valid[i]:
            continue
        nearest = valid_ids[np.argmin(np.abs(valid_ids - i))]
        widths[i] = widths[nearest]
        lefts[i] = lefts[nearest]
        rights[i] = rights[nearest]

    smooth_w = smooth_1d(widths, window=max(7, h // 18))
    grad = np.gradient(smooth_w)
    curv = np.gradient(grad)

    y_min = int(h * 0.12)
    y_max = int(h * 0.90)
    if y_max <= y_min:
        return []

    candidates = []
    max_abs_curv = float(np.max(np.abs(curv[y_min:y_max + 1]))) if y_max > y_min else 0.0
    if max_abs_curv <= 1e-6:
        return []

    threshold = max_abs_curv * 0.23
    for i in range(y_min, y_max + 1):
        if abs(float(curv[i])) < threshold:
            continue
        ww = float(smooth_w[i])
        if ww < (w * 0.06):
            continue
        candidates.append((abs(float(curv[i])), i, float(curv[i]), ww))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    min_sep = max(8, int(h * 0.13))
    for _, idx, signed_curv, ww in candidates:
        if any(abs(idx - sidx) < min_sep for sidx, _, _ in selected):
            continue
        selected.append((idx, signed_curv, ww))
        if len(selected) >= max(1, int(max_lines)):
            break

    selected.sort(key=lambda item: item[0])
    lines = []
    for idx, signed_curv, ww in selected:
        y = top_y + idx
        x0 = int(lefts[idx])
        x1 = int(rights[idx])
        margin = max(2, int(ww * 0.10))
        x0 += margin
        x1 -= margin
        if x1 - x0 < 10:
            continue

        # Slight arced line to mimic catalog symbol conventions.
        arc = int(max(1, min(6, ww * 0.028)))
        direction = -1 if signed_curv > 0 else 1
        q1 = int(x0 + (x1 - x0) * 0.33)
        q2 = int(x0 + (x1 - x0) * 0.66)
        line = [
            [x0, y],
            [q1, y + (direction * arc)],
            [q2, y + (direction * arc)],
            [x1, y],
        ]
        lines.append(line)

    return lines


def estimate_round_bands(mask, max_lines=2):
    """
    Estimate concentric ring-like lines for circular artifacts (coins, seals).
    Avoids forcing horizontal bars through round silhouettes.
    """
    line_count = int(max(0, min(3, int(max_lines))))
    if line_count <= 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    main = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main))
    if area <= 80.0:
        return []

    perim = float(cv2.arcLength(main, True))
    if perim <= 1e-6:
        return []
    circularity = (4.0 * np.pi * area) / (perim * perim)
    x, y, w_box, h_box = cv2.boundingRect(main)
    aspect_balance = min(w_box, h_box) / max(1.0, float(max(w_box, h_box)))
    if circularity < 0.58 or aspect_balance < 0.62:
        return []

    (cx, cy), radius = cv2.minEnclosingCircle(main)
    if radius < 10.0:
        return []

    ratios = [0.76, 0.58, 0.42]
    lines = []
    point_count = 56
    for ratio in ratios[:line_count]:
        r = radius * ratio
        if r < 6.0:
            continue
        pts = []
        for i in range(point_count):
            t = (2.0 * np.pi * float(i)) / float(point_count)
            px = int(round(cx + (r * np.cos(t))))
            py = int(round(cy + (r * np.sin(t))))
            px = max(0, min(mask.shape[1] - 1, px))
            py = max(0, min(mask.shape[0] - 1, py))
            if mask[py, px] > 0:
                pts.append([px, py])
        if len(pts) >= 18:
            pts.append(pts[0])
            lines.append(pts)
    return lines


def estimate_terminal_bars(mask, max_lines=2):
    """
    Estimate short terminal bars near top/bottom extremes.
    These emulate typological marker conventions seen in catalog symbols.
    """
    target_lines = int(max(0, int(max_lines)))
    if target_lines == 0:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return []

    top_y = int(np.min(ys))
    bot_y = int(np.max(ys))
    h = max(1, bot_y - top_y + 1)
    if h < 20:
        return []

    axis_ratio = 0.5
    rows = []
    for y in (top_y + int(h * 0.06), bot_y - int(h * 0.08)):
        if y < 0 or y >= mask.shape[0]:
            continue
        row = np.where(mask[y] > 0)[0]
        if len(row) < 6:
            continue
        left = int(row[0])
        right = int(row[-1])
        width = right - left
        if width < 10:
            continue
        margin = int(max(2, width * 0.16))
        span_left = left + margin
        span_right = right - margin
        if span_right - span_left < 6:
            continue
        axis_x = int(left + (width * axis_ratio))
        half_len = max(3, int(width * 0.12))
        x0 = max(span_left, axis_x - half_len)
        x1 = min(span_right, axis_x + half_len)
        if x1 - x0 < 4:
            x0 = span_left
            x1 = span_right
        if x1 - x0 < 4:
            continue
        rows.append([[x0, y], [x1, y]])
        if len(rows) >= target_lines:
            break

    return rows[:target_lines]
