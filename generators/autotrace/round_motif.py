# -*- coding: utf-8 -*-
"""
Round-artifact motif extraction (annular, polar, relief, mirror signature).

Auto-generated from the former ContourGenerator methods; QGIS-free.
"""

import cv2
import numpy as np

from ...log import log_exception

from .enhance import adaptive_canny
from .geometry import arc_polyline, circle_polyline, dedupe_lines, diamond_polyline, line_angle_span, line_center, line_centroid_and_length, line_ring_likeness, merge_distinct_lines, polyline_arc_length, rotate_line_about_center


def extract_round_low_quality_lines(bgr_img, target_mask, main_contour, max_lines=10):
    """
    Relaxed fallback for low-quality round artifacts.
    Recovers short motif strokes from enhanced edges in an annular region.
    """
    try:
        if bgr_img is None or target_mask is None or main_contour is None:
            return []
        h, w = target_mask.shape[:2]
        if h < 20 or w < 20:
            return []
        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        if radius < 18.0:
            return []

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        bg_sigma = max(5.0, float(radius) * 0.08)
        bg = cv2.GaussianBlur(eq, (0, 0), bg_sigma)
        flat = cv2.subtract(eq, bg)
        flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)

        edges_eq = adaptive_canny(eq, mask=target_mask, low_floor=14, high_cap=172)
        edges_flat = adaptive_canny(flat, mask=target_mask, low_floor=10, high_cap=156)
        edges = cv2.bitwise_or(edges_eq, edges_flat)
        edges = cv2.bitwise_and(edges, target_mask)

        yy, xx = np.indices((h, w))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        annulus = ((dist >= (radius * 0.20)) & (dist <= (radius * 0.86))).astype(np.uint8) * 255
        edges = cv2.bitwise_and(edges, annulus)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            arc_len = float(cv2.arcLength(contour, False))
            if arc_len < 8.0:
                continue
            pts = contour.reshape(-1, 2)
            if pts.shape[0] < 4:
                continue

            rs = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            r_mean = float(np.mean(rs))
            r_std = float(np.std(rs))
            if r_mean <= (radius * 0.16) or r_mean >= (radius * 0.90):
                continue
            # Remove long ring-like segments.
            if r_std < 1.1 and arc_len > (radius * 0.22):
                continue

            score = arc_len * (1.0 + min(1.0, r_std / 4.5))
            line_pts = [[int(p[0]), int(p[1])] for p in pts.tolist()]
            candidates.append((score, line_pts))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in candidates[:max(1, int(max_lines))]]
    except Exception as e:
        log_exception("extract_round_low_quality_lines", e)
        return []


def round_line_center_coverage(lines, target_mask):
    """Return fraction of lines that fall in center/middle radius bands."""
    try:
        if target_mask is None:
            return 1.0
        ys, xs = np.where(target_mask > 0)
        if len(xs) < 20:
            return 1.0

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        radius = float(np.percentile(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2), 95))
        if radius < 6.0:
            return 1.0

        total = 0
        centerish = 0
        for line in lines or []:
            if not line or len(line) < 2:
                continue
            pts = np.asarray(line, dtype=np.float32)
            rs = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            r_mean = float(np.mean(rs))
            total += 1
            if r_mean <= (radius * 0.64):
                centerish += 1

        if total <= 0:
            return 0.0
        return float(centerish) / float(total)
    except Exception as e:
        log_exception("round_line_center_coverage", e)
        return 0.0


def round_line_inner_count(lines, target_mask, ratio=0.50):
    """Count how many line centers lie within inner-radius ratio."""
    try:
        if target_mask is None:
            return 0
        ys, xs = np.where(target_mask > 0)
        if len(xs) < 20:
            return 0

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        radius = float(np.percentile(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2), 95))
        if radius < 6.0:
            return 0

        limit = radius * float(max(0.1, min(0.95, ratio)))
        count = 0
        for line in lines or []:
            if not line or len(line) < 2:
                continue
            pts = np.asarray(line, dtype=np.float32)
            r_mean = float(np.mean(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)))
            if r_mean <= limit:
                count += 1
        return int(count)
    except Exception as e:
        log_exception("round_line_inner_count", e)
        return 0


def build_round_structural_lines(target_mask, main_contour, round_lines=None, max_lines=8):
    """
    Fast deterministic round-structure abstraction.
    Prioritizes readable archaeological structure over noisy motif chasing.
    """
    try:
        target = max(1, int(max_lines))
        out = []
        out = merge_distinct_lines(
            out,
            list(round_lines or [])[:2],
            min_center_sep=3.6,
            max_lines=target,
            min_arc_len=8.0,
        )

        if target_mask is None or main_contour is None:
            return out[:target]

        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        if radius < 10.0:
            return out[:target]

        ring_a = circle_polyline(cx, cy, radius * 0.62, steps=54)
        ring_b = circle_polyline(cx, cy, radius * 0.44, steps=48)
        knob = circle_polyline(cx, cy, radius * 0.14, steps=36)
        diamond = diamond_polyline(cx, cy, radius * 0.40)

        out = merge_distinct_lines(out, [ring_a, ring_b, diamond, knob], min_center_sep=2.0, max_lines=target, min_arc_len=6.0)

        stubs = []
        for deg in (36.0, 126.0, 216.0, 306.0):
            theta = np.deg2rad(deg)
            r0 = radius * 0.24
            r1 = radius * 0.52
            p0 = [int(round(cx + r0 * np.cos(theta))), int(round(cy + r0 * np.sin(theta)))]
            p1 = [int(round(cx + r1 * np.cos(theta))), int(round(cy + r1 * np.sin(theta)))]
            stubs.append([p0, p1])
        out = merge_distinct_lines(out, stubs, min_center_sep=2.2, max_lines=target, min_arc_len=4.0)
        return out[:target]
    except Exception as e:
        log_exception("build_round_structural_lines", e)
        return list(round_lines or [])[:max(1, int(max_lines))]


def prefer_round_inner_lines(lines, target_mask, max_lines=10, inner_ratio=0.56, min_inner=3):
    """Prioritize center/middle motif lines over outer-ring fragments."""
    try:
        target = max(1, int(max_lines))
        if target_mask is None:
            return list(lines or [])[:target]

        ys, xs = np.where(target_mask > 0)
        if len(xs) < 20:
            return list(lines or [])[:target]

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        radius = float(np.percentile(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2), 95))
        if radius < 6.0:
            return list(lines or [])[:target]

        inner_limit = radius * float(max(0.12, min(0.90, inner_ratio)))
        inner_items = []
        outer_items = []
        for line in lines or []:
            if not line or len(line) < 2:
                continue
            arc_len = polyline_arc_length(line)
            if arc_len < 4.0:
                continue
            center = line_center(line)
            if center is None:
                continue
            r = ((float(center[0]) - cx) ** 2 + (float(center[1]) - cy) ** 2) ** 0.5
            item = (line, float(arc_len), float(r))
            if r <= inner_limit:
                inner_items.append(item)
            else:
                outer_items.append(item)

        inner_items.sort(key=lambda item: (item[1], -item[2]), reverse=True)
        outer_items.sort(key=lambda item: item[1], reverse=True)

        selected = []
        selected = merge_distinct_lines(
            selected,
            [item[0] for item in inner_items],
            min_center_sep=2.0,
            max_lines=target,
            min_arc_len=4.5,
        )

        if len(selected) < int(max(0, min_inner)) and inner_items:
            selected = merge_distinct_lines(
                selected,
                [item[0] for item in inner_items],
                min_center_sep=1.2,
                max_lines=target,
                min_arc_len=4.0,
            )

        if len(selected) < target:
            selected = merge_distinct_lines(
                selected,
                [item[0] for item in outer_items],
                min_center_sep=2.6,
                max_lines=target,
                min_arc_len=6.0,
            )

        return selected[:target]
    except Exception as e:
        log_exception("prefer_round_inner_lines", e)
        return list(lines or [])[:max(1, int(max_lines))]


def extract_round_center_fallback_lines(bgr_img, target_mask, main_contour, max_lines=10):
    """
    Center-biased motif fallback for low-quality round artifacts.
    Extracts short curved strokes from inner/middle zones.
    """
    try:
        if bgr_img is None or target_mask is None or main_contour is None:
            return []

        h, w = target_mask.shape[:2]
        if h < 20 or w < 20:
            return []

        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        if radius < 18.0:
            return []

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        eq = clahe.apply(gray)

        bg_sigma = max(4.0, float(radius) * 0.07)
        bg = cv2.GaussianBlur(eq, (0, 0), bg_sigma)
        flat = cv2.subtract(eq, bg)
        flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)

        edges_eq = adaptive_canny(eq, mask=target_mask, low_floor=10, high_cap=148)
        edges_flat = adaptive_canny(flat, mask=target_mask, low_floor=8, high_cap=136)
        blackhat = cv2.morphologyEx(
            eq,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        mix = cv2.addWeighted(flat, 0.72, blackhat, 0.58, 0)
        edges_mix = adaptive_canny(mix, mask=target_mask, low_floor=8, high_cap=132)
        edges = cv2.bitwise_or(cv2.bitwise_or(edges_eq, edges_flat), edges_mix)
        edges = cv2.bitwise_and(edges, target_mask)

        yy, xx = np.indices((h, w))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Prefer center-to-middle motif zones; avoid outer ring dominance.
        center_band = (
            (dist >= (radius * 0.12)) &
            (dist <= (radius * 0.58))
        ).astype(np.uint8) * 255
        edges = cv2.bitwise_and(edges, center_band)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            arc_len = float(cv2.arcLength(contour, False))
            if arc_len < 6.0:
                continue

            pts = contour.reshape(-1, 2)
            if pts.shape[0] < 4:
                continue

            rs = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            r_mean = float(np.mean(rs))
            r_std = float(np.std(rs))
            if r_mean < (radius * 0.08) or r_mean > (radius * 0.62):
                continue
            # suppress ring-like arcs
            if r_std < 1.2 and arc_len > (radius * 0.20):
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 3 and bh < 3:
                continue

            inner_bias = max(0.0, 1.0 - (r_mean / max(1.0, radius)))
            score = (arc_len * (1.0 + min(1.0, r_std / 4.0))) + (inner_bias * 8.0)
            line_pts = [[int(p[0]), int(p[1])] for p in pts.tolist()]
            candidates.append((score, line_pts))

        candidates.sort(key=lambda item: item[0], reverse=True)
        out = [line for _, line in candidates[:max(1, int(max_lines))]]
        return out
    except Exception as e:
        log_exception("extract_round_center_fallback_lines", e)
        return []


def extract_round_unwrap_lines(bgr_img, target_mask, main_contour, max_lines=12):
    """
    Polar-unwrapped motif extraction for circular artifacts.
    Uses radial geometry only (no semantic class assumptions).
    """
    try:
        if bgr_img is None or target_mask is None or main_contour is None:
            return []

        h, w = target_mask.shape[:2]
        if h < 24 or w < 24:
            return []

        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        if radius < 20.0:
            return []

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.0)

        g_f = gray.astype(np.float32) + 1.0
        illum = cv2.GaussianBlur(g_f, (0, 0), max(10.0, float(radius) * 0.10))
        illum = np.maximum(illum, 1.0)
        flat = np.clip((g_f / illum) * 146.0, 0, 255).astype(np.uint8)

        angle_bins = int(max(420, min(1280, round(radius * 8.0))))
        radial_bins = int(max(120, min(420, round(radius * 0.95))))
        max_radius_used = float(radius * 0.94)

        polar = cv2.warpPolar(
            flat,
            (radial_bins, angle_bins),
            (float(cx), float(cy)),
            max_radius_used,
            cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
        )
        if polar is None or polar.size == 0:
            return []

        polar_bg = cv2.GaussianBlur(polar, (0, 0), 4.6)
        polar_hp = cv2.subtract(polar, polar_bg)
        polar_hp = cv2.normalize(polar_hp, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        polar_eq = clahe.apply(polar_hp)

        edges = adaptive_canny(polar_eq, mask=None, low_floor=8, high_cap=142)
        if edges is None:
            return []

        rr = np.tile(np.arange(radial_bins, dtype=np.float32), (angle_bins, 1))
        valid_band = (
            (rr >= (radial_bins * 0.16))
            & (rr <= (radial_bins * 0.86))
        )
        edges[~valid_band] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []

        candidates = []
        for contour in contours:
            arc_len = float(cv2.arcLength(contour, False))
            if arc_len < 16.0:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 3 or bh < 3:
                continue

            # Suppress long near-constant-radius ring fragments.
            if bh > (angle_bins * 0.22) and bw < max(3, int(radial_bins * 0.035)):
                continue

            pts = contour.reshape(-1, 2)
            if pts.shape[0] < 4:
                continue

            step = max(1, int(len(pts) / 120))
            line = []
            for p in pts[::step]:
                rho = (float(p[0]) / max(1.0, float(radial_bins - 1))) * max_radius_used
                theta = (float(p[1]) / max(1.0, float(angle_bins - 1))) * (2.0 * np.pi)
                px = int(round(cx + (rho * np.cos(theta))))
                py = int(round(cy + (rho * np.sin(theta))))
                if px < 0 or py < 0 or px >= w or py >= h:
                    continue
                if target_mask[py, px] <= 0:
                    continue
                line.append([px, py])

            if len(line) < 4:
                continue

            arr = np.asarray(line, dtype=np.float32)
            rs = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
            r_mean = float(np.mean(rs))
            r_std = float(np.std(rs))
            if r_mean < (radius * 0.10) or r_mean > (radius * 0.88):
                continue
            if r_std < 0.9 and arc_len > (radius * 0.22):
                continue

            inner_bias = max(0.0, 1.0 - (r_mean / max(1.0, radius)))
            score = arc_len * (1.0 + min(1.0, r_std / 4.5)) + (inner_bias * 7.0)
            candidates.append((score, line))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in candidates[:max(1, int(max_lines))]]
    except Exception as e:
        log_exception("extract_round_unwrap_lines", e)
        return []


def extract_round_mirror_signature_lines(bgr_img, mask, main_contour, max_lines=10):
    """
    Mirror-specific structural fallback for low-resolution round artifacts.
    Builds stable geometry (boss + diamond frame + ring arcs) when motif extraction fails.
    """
    try:
        limit = int(max(0, int(max_lines)))
        if limit <= 0:
            return []
        if bgr_img is None or mask is None or main_contour is None:
            return []

        h, w = mask.shape[:2]
        if h < 30 or w < 30:
            return []

        ys, xs = np.where(mask > 0)
        if len(xs) < 100:
            return []
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        r_ref = max(14.0, 0.5 * float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))))
        if r_ref < 16.0:
            return []

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
        clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8, 8))
        eq = clahe.apply(gray)

        g_f = eq.astype(np.float32) + 1.0
        illum = cv2.GaussianBlur(g_f, (0, 0), max(8.0, 0.10 * r_ref))
        illum = np.maximum(illum, 1.0)
        flat = np.clip((g_f / illum) * 145.0, 0, 255).astype(np.uint8)

        gx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        yy, xx = np.indices((h, w))
        rr = np.sqrt(((xx.astype(np.float32) - cx) ** 2) + ((yy.astype(np.float32) - cy) ** 2))

        r_min = max(6, int(round(0.16 * r_ref)))
        r_max = max(r_min + 2, int(round(0.94 * r_ref)))
        radii = []
        scores = []
        fg = (mask > 0)

        for r in range(r_min, r_max + 1, 2):
            band = ((rr >= (float(r) - 1.6)) & (rr <= (float(r) + 1.6)) & fg)
            count = int(np.count_nonzero(band))
            if count < 120:
                continue
            vals = grad[band]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            radial_ratio = float(r) / max(1.0, float(r_ref))
            prior = 1.0
            if radial_ratio < 0.14 or radial_ratio > 0.97:
                prior = 0.50
            scores.append((mean_v + (0.38 * std_v)) * prior)
            radii.append(float(r))

        if len(radii) < 5:
            return []

        arr = np.asarray(scores, dtype=np.float32)
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel = kernel / float(np.sum(kernel))
        smoothed = np.convolve(arr, kernel, mode="same")

        peak_items = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] <= smoothed[i - 1] or smoothed[i] < smoothed[i + 1]:
                continue
            r_val = float(radii[i])
            ratio = r_val / max(1.0, r_ref)
            keep = 1.0 - min(1.0, abs(ratio - 0.58))
            peak_items.append((float(smoothed[i]) * (0.68 + (0.45 * keep)), r_val))

        peak_items.sort(key=lambda item: item[0], reverse=True)
        picked_r = []
        min_sep = max(5.0, 0.075 * r_ref)
        for _, r_val in peak_items:
            if any(abs(r_val - p) < min_sep for p in picked_r):
                continue
            picked_r.append(r_val)
            if len(picked_r) >= 4:
                break
        if not picked_r:
            return []

        picked_r.sort()
        lines = []

        boss_r = estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
        if boss_r <= 0.0:
            boss_r = max(4.0, 0.16 * r_ref)
        lines.append(circle_polyline(cx, cy, boss_r, steps=42))

        target_ratio = 0.34
        diamond_r = min(picked_r, key=lambda rv: abs((rv / max(1.0, r_ref)) - target_ratio))
        diamond_r = float(np.clip(diamond_r, 0.24 * r_ref, 0.46 * r_ref))
        d0 = int(round(diamond_r))
        d1 = int(round(max(6.0, diamond_r * 0.78)))
        diamond_outer = [
            [int(round(cx)), int(round(cy - d0))],
            [int(round(cx + d0)), int(round(cy))],
            [int(round(cx)), int(round(cy + d0))],
            [int(round(cx - d0)), int(round(cy))],
            [int(round(cx)), int(round(cy - d0))],
        ]
        diamond_inner = [
            [int(round(cx)), int(round(cy - d1))],
            [int(round(cx + d1)), int(round(cy))],
            [int(round(cx)), int(round(cy + d1))],
            [int(round(cx - d1)), int(round(cy))],
            [int(round(cx)), int(round(cy - d1))],
        ]
        lines.append(diamond_outer)
        lines.append(diamond_inner)

        if len(picked_r) >= 3:
            ring_r = [picked_r[1], picked_r[2], picked_r[-1]]
        else:
            ring_r = picked_r[-2:] if len(picked_r) > 1 else picked_r[:]
        ring_r = ring_r[:3]

        arc_templates = [(20.0, 92.0), (140.0, 212.0), (260.0, 332.0)]
        for idx, rv in enumerate(ring_r):
            arc_steps = int(max(10, min(22, round((rv / max(1.0, r_ref)) * 20.0))))
            rotate = float((idx % 2) * 10.0)
            for a0, a1 in arc_templates:
                lines.append(
                    arc_polyline(
                        cx,
                        cy,
                        rv,
                        a0 + rotate,
                        a1 + rotate,
                        steps=arc_steps,
                    )
                )

        arm = float(max(boss_r + 2.0, min(diamond_r * 0.82, 0.40 * r_ref)))
        for ang_deg in (45.0, 135.0, 225.0, 315.0):
            a = np.deg2rad(ang_deg)
            p0 = [int(round(cx + (boss_r * np.cos(a)))), int(round(cy + (boss_r * np.sin(a))))]
            p1 = [int(round(cx + (arm * np.cos(a)))), int(round(cy + (arm * np.sin(a))))]
            lines.append([p0, p1])

        cleaned = dedupe_lines(lines, min_points=2, max_lines=max(4, limit))
        return cleaned[:max(1, limit)]
    except Exception as e:
        log_exception("extract_round_mirror_signature_lines", e)
        return []


def regularize_round_publication_lines(lines, mask, max_lines=14):
    """
    Simplify and de-noise measured round motif lines into readable publication geometry.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0:
        return []
    if not lines:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return list(lines[:limit])
    x0 = int(np.min(xs))
    x1 = int(np.max(xs))
    y0 = int(np.min(ys))
    y1 = int(np.max(ys))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx_ref = float(np.mean(xs))
    cy_ref = float(np.mean(ys))
    r_ref = max(12.0, 0.5 * float(max(bw, bh)))
    min_arc = max(10.0, 0.034 * float(min(bw, bh)))
    min_sep = max(6.0, 0.070 * float(min(bw, bh)))
    angle_bin_count = 12
    per_bin_limit = 1

    candidates = []
    for line in lines:
        arr = np.asarray(line, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue

        closed = bool(arr.shape[0] >= 3 and np.array_equal(arr[0], arr[-1]))
        curve = arr.reshape(-1, 1, 2).astype(np.float32)
        arc_len = float(cv2.arcLength(curve, closed))
        if arc_len < min_arc:
            continue

        eps = max(0.8, 0.012 * arc_len)
        approx = cv2.approxPolyDP(curve, eps, closed).reshape(-1, 2)
        if approx.shape[0] < (3 if closed else 2):
            continue

        approx[:, 0] = np.clip(approx[:, 0], 0, mask.shape[1] - 1)
        approx[:, 1] = np.clip(approx[:, 1], 0, mask.shape[0] - 1)

        cx = float(np.mean(approx[:, 0]))
        cy = float(np.mean(approx[:, 1]))
        if cx <= (x0 + 2) or cx >= (x1 - 2) or cy <= (y0 + 2) or cy >= (y1 - 2):
            continue

        inside = 0
        for px, py in approx:
            if mask[int(py), int(px)] > 0:
                inside += 1
        if inside / float(max(1, len(approx))) < 0.84:
            continue

        d_norm = (((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2) ** 0.5) / max(1e-6, r_ref)
        # Exclude center-boss region and very outer rim noise.
        if d_norm < 0.14 or d_norm > 0.92:
            continue

        out_line = approx.astype(int).tolist()
        ring_like = line_ring_likeness(out_line, cx_ref, cy_ref)
        if ring_like >= 0.97 and d_norm > 0.42:
            continue
        if not closed:
            sx, sy = float(out_line[0][0]), float(out_line[0][1])
            ex, ey = float(out_line[-1][0]), float(out_line[-1][1])
            chord = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            tortuosity = arc_len / max(1.0, chord)
            if tortuosity > 2.35 and len(out_line) <= 12:
                continue

        angle = float(np.arctan2(cy - cy_ref, cx - cx_ref))
        angle_bin = int(((angle + np.pi) / (2.0 * np.pi)) * angle_bin_count) % angle_bin_count
        band_score = max(0.30, 1.0 - (abs(d_norm - 0.50) / 0.62))
        complexity = min(1.0, float(len(out_line)) / 20.0)
        score = (0.40 * band_score) + (0.42 * complexity) + (0.18 * (1.0 - min(1.0, ring_like)))
        if closed and out_line[0] != out_line[-1]:
            out_line.append(out_line[0])

        candidates.append((score, angle_bin, (cx, cy), out_line))

    if not candidates:
        return list(lines[:limit])

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    centers = []
    used_bins = {}
    for _, angle_bin, center, out_line in candidates:
        used_count = int(used_bins.get(angle_bin, 0))
        if used_count >= per_bin_limit:
            continue
        if any((((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5) < min_sep for c in centers):
            continue
        selected.append(out_line)
        centers.append(center)
        used_bins[angle_bin] = used_count + 1
        if len(selected) >= limit:
            break

    if not selected:
        return list(lines[:limit])
    return selected[:limit]


def round_line_angular_coverage(lines, cx, cy, bins=12):
    """Return angular bin coverage ratio for round motif lines."""
    if not lines:
        return 0.0
    bin_count = int(max(4, int(bins)))
    used = set()
    for line in lines:
        center, arc_len = line_centroid_and_length(line)
        if center is None or arc_len < 4.0:
            continue
        theta = float(np.arctan2(center[1] - cy, center[0] - cx))
        b = int(((theta + np.pi) / (2.0 * np.pi)) * bin_count) % bin_count
        used.add(b)
    return float(len(used)) / float(bin_count)


def augment_round_rotational_symmetry(lines, mask, desired_lines=12):
    """
    If round motif lines are one-sided (lighting bias), augment by rotational copies.
    This stabilizes measured symbols for circular artifacts.
    """
    target = int(max(0, int(desired_lines)))
    if target <= 0:
        return []
    base_lines = list(lines or [])
    if len(base_lines) == 0:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 60:
        return base_lines[:target]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(10.0, 0.5 * float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))))

    coverage = round_line_angular_coverage(base_lines, cx, cy, bins=12)
    if coverage >= 0.58 and len(base_lines) >= min(target, 8):
        return base_lines[:target]

    # Sort by length (longer motif lines are more stable for rotational augmentation).
    line_items = []
    for line in base_lines:
        center, arc_len = line_centroid_and_length(line)
        if center is None or arc_len < 7.0:
            continue
        line_items.append((arc_len, line))
    if not line_items:
        return base_lines[:target]
    line_items.sort(key=lambda item: item[0], reverse=True)

    out = [item[1] for item in line_items[:max(1, min(6, len(line_items)))]]
    out = out[:target]
    centers = []
    for line in out:
        c0, _ = line_centroid_and_length(line)
        if c0 is not None:
            centers.append(c0)
    min_sep = max(3.8, 0.045 * float(min(mask.shape[0], mask.shape[1])))

    # 4-way rotational copies preserve common mirror motif repetition.
    angles = [np.pi * 0.5, np.pi, np.pi * 1.5]
    for _, line in line_items:
        if len(out) >= target:
            break
        for ang in angles:
            if len(out) >= target:
                break
            rot = rotate_line_about_center(line, cx, cy, ang)
            if len(rot) < 2:
                continue

            arr = np.asarray(rot, dtype=np.int32)
            arr[:, 0] = np.clip(arr[:, 0], 0, mask.shape[1] - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, mask.shape[0] - 1)

            # Keep only lines mostly inside silhouette and out of center boss zone.
            inside = 0
            for px, py in arr:
                if mask[int(py), int(px)] > 0:
                    inside += 1
            if inside / float(max(1, len(arr))) < 0.88:
                continue

            center, arc_len = line_centroid_and_length(arr.tolist())
            if center is None or arc_len < 7.0:
                continue
            d_norm = (((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
            if d_norm < 0.34 or d_norm > 0.93:
                continue
            if any((((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5) < min_sep for c in centers):
                continue

            out.append(arr.tolist())
            centers.append(center)

    return out[:target]


def round_ring_line_ratio(lines, mask):
    """Estimate how many lines are ring-like for round artifacts."""
    if not lines:
        return 0.0
    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return 0.0
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    ring_count = 0
    valid_count = 0
    for line in lines:
        _, arc_len = line_centroid_and_length(line)
        if arc_len < 6.0:
            continue
        valid_count += 1
        ring_like = line_ring_likeness(line, cx, cy)
        angle_span = line_angle_span(line, cx, cy)
        if ring_like >= 0.90 and angle_span >= (np.pi * 0.70):
            ring_count += 1
    if valid_count <= 0:
        return 0.0
    return float(ring_count) / float(valid_count)


def needs_round_mirror_rescue(lines, mask, strict=True):
    """
    Decide whether low-quality round artifact output is too weak/noisy and
    should be replaced by mirror-structured fallback geometry.
    """
    if not lines:
        return True

    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return len(lines) < 4

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    center_cov = round_line_center_coverage(lines, mask)
    inner_count = round_line_inner_count(lines, mask, ratio=0.52)
    angular_cov = round_line_angular_coverage(lines, cx, cy, bins=14)
    ring_ratio = round_ring_line_ratio(lines, mask)

    if strict:
        return (
            len(lines) < 6
            or center_cov < 0.44
            or inner_count < 4
            or angular_cov < 0.40
            or ring_ratio > 0.54
        )

    return (
        len(lines) < 5
        or center_cov < 0.34
        or inner_count < 3
        or angular_cov < 0.32
        or ring_ratio > 0.66
    )


def suppress_round_ring_lines(lines, mask, max_ring_lines=1):
    """
    Keep motif-like lines for measured round artifacts and cap concentric bands.
    """
    if not lines:
        return []
    keep_ring = max(0, int(max_ring_lines))
    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return list(lines)

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(10.0, 0.5 * float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))))

    ring_items = []
    motif_items = []
    for line in lines:
        center, arc_len = line_centroid_and_length(line)
        if center is None or arc_len < 4.0:
            continue
        d_norm = (((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
        ring_like = line_ring_likeness(line, cx, cy)
        angle_span = line_angle_span(line, cx, cy)
        is_ring = ring_like >= 0.90 and angle_span >= (np.pi * 0.80) and d_norm >= 0.20
        motif_score = arc_len * (1.0 - (0.45 * ring_like)) * max(0.20, 1.0 - abs(d_norm - 0.60))
        if is_ring:
            ring_items.append((motif_score, line))
        else:
            motif_items.append((motif_score, line))

    motif_items.sort(key=lambda item: item[0], reverse=True)
    ring_items.sort(key=lambda item: item[0], reverse=True)

    out = [item[1] for item in motif_items]
    if keep_ring > 0 and ring_items:
        out.extend(item[1] for item in ring_items[:keep_ring])

    if not out:
        all_items = motif_items + ring_items
        all_items.sort(key=lambda item: item[0], reverse=True)
        out = [item[1] for item in all_items[:max(1, keep_ring)]]
    return out


def estimate_round_angular_motif_markers(bgr_img, mask, max_lines=12):
    """
    Sector-based fallback for measured round artifacts.
    Uses angular sectors over an annulus to recover interior motif islands when
    ring-like edges dominate.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 90:
        return []
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(10.0, 0.5 * float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))))

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur_sigma = max(1.6, float(min(mask.shape[0], mask.shape[1])) * 0.008)
    low = cv2.GaussianBlur(enhanced, (0, 0), blur_sigma)
    high = cv2.absdiff(enhanced, low)
    lap = cv2.convertScaleAbs(cv2.Laplacian(enhanced, cv2.CV_16S, ksize=3))
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    detail = cv2.addWeighted(high, 0.64, lap, 0.52, 0)
    detail = cv2.addWeighted(detail, 0.74, grad, 0.30, 0)

    _, otsu_bin = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_bin = cv2.adaptiveThreshold(
        detail,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        27,
        -3,
    )
    edges = cv2.Canny(enhanced, 24, 92)
    fused = cv2.bitwise_or(otsu_bin, adaptive_bin)
    fused = cv2.bitwise_or(fused, edges)

    h, w = mask.shape[:2]
    yy, xx = np.indices((h, w))
    rr = np.sqrt(((xx.astype(np.float32) - cx) ** 2) + ((yy.astype(np.float32) - cy) ** 2))
    boss_r = estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
    inner_ratio = 0.24
    if boss_r > 0.0:
        inner_ratio = max(inner_ratio, min(0.50, (boss_r / max(1e-6, r_ref)) * 1.15))
    annulus = ((rr >= (inner_ratio * r_ref)) & (rr <= (0.93 * r_ref))).astype(np.uint8) * 255
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    fused = cv2.bitwise_and(fused, fused, mask=interior)
    fused = cv2.bitwise_and(fused, annulus)

    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if mask_contours:
        boundary = np.zeros_like(mask)
        cv2.drawContours(boundary, [max(mask_contours, key=cv2.contourArea)], -1, 255, thickness=4)
        fused[boundary > 0] = 0

    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(fused, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    min_area = max(7.0, float((r_ref * r_ref) * 0.0010))
    max_area = float((r_ref * r_ref) * 0.11)
    min_arc = max(8.0, 0.028 * float(min(h, w)))
    sector_count = max(10, min(24, int(limit * 2)))
    min_sep = max(3.2, 0.040 * float(min(h, w)))

    candidates = []
    for contour in contours:
        area = float(abs(cv2.contourArea(contour)))
        if area < min_area or area > max_area:
            continue
        arc_len = float(cv2.arcLength(contour, True))
        if arc_len < min_arc:
            continue

        epsilon = max(1.0, 0.0090 * arc_len)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue

        arr = np.asarray(approx, dtype=np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)

        center = np.mean(arr, axis=0)
        lx = float(center[0])
        ly = float(center[1])
        ix = int(max(0, min(w - 1, int(round(lx)))))
        iy = int(max(0, min(h - 1, int(round(ly)))))
        if mask[iy, ix] == 0:
            continue

        d_norm = (((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
        if d_norm < (inner_ratio * 0.80) or d_norm > 0.95:
            continue

        inside_ratio = float(np.mean(mask[arr[:, 1], arr[:, 0]] > 0))
        if inside_ratio < 0.86:
            continue

        line = arr.tolist()
        if line[0] != line[-1]:
            line.append(line[0])
        ring_like = line_ring_likeness(line, cx, cy)
        angle_span = line_angle_span(line, cx, cy)
        if ring_like >= 0.92 and angle_span >= (np.pi * 0.75):
            continue

        theta = float(np.arctan2(ly - cy, lx - cx))
        sector_idx = int(((theta + np.pi) / (2.0 * np.pi)) * sector_count) % sector_count
        complexity = min(1.0, float(len(line)) / 20.0)
        band_score = max(0.22, 1.0 - abs(d_norm - 0.58))
        score = (1.6 * area) + (0.60 * arc_len) + (9.0 * band_score) + (7.0 * complexity) - (9.0 * ring_like)
        candidates.append((score, sector_idx, (lx, ly), line))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    centers = []
    used_sector = {}
    for _, sector_idx, center, line in candidates:
        if int(used_sector.get(sector_idx, 0)) >= 1:
            continue
        if any((((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5) < min_sep for c in centers):
            continue
        selected.append(line)
        centers.append(center)
        used_sector[sector_idx] = int(used_sector.get(sector_idx, 0)) + 1
        if len(selected) >= limit:
            break

    return selected[:limit]


def extract_round_motif_lines(bgr_img, mask, main_contour, max_lines=24):
    """
    Extract additional non-ring motif lines for circular artifacts.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0:
        return []

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Top-hat emphasizes local embossed details in coin-like surfaces.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    fused = cv2.addWeighted(enhanced, 0.70, tophat, 1.30, 0)

    edges = cv2.Canny(fused, 24, 96)
    interior_mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, edges, mask=interior_mask)

    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, [main_contour], -1, 255, thickness=6)
    edges[boundary > 0] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    line_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not line_contours:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return []
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(8.0, 0.5 * float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))))
    min_dim = min(mask.shape[0], mask.shape[1])
    min_len = max(6.0, float(min_dim) * 0.012)
    max_len = float(max(mask.shape[0], mask.shape[1])) * 1.20

    candidates = []
    for contour in line_contours:
        arc_len = float(cv2.arcLength(contour, False))
        if arc_len < min_len or arc_len > max_len:
            continue

        epsilon = 0.0025 * arc_len
        approx = cv2.approxPolyDP(contour, epsilon, False)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 2:
            continue

        center = np.mean(pts, axis=0).astype(int)
        if not (0 <= center[0] < mask.shape[1] and 0 <= center[1] < mask.shape[0]):
            continue
        if mask[center[1], center[0]] == 0:
            continue

        d = ((float(center[0]) - cx) ** 2 + (float(center[1]) - cy) ** 2) ** 0.5
        d_norm = d / max(1e-6, r_ref)
        if d_norm > 0.95:
            continue

        ring_like = line_ring_likeness(pts.tolist(), cx, cy)
        if ring_like >= 0.90 and d_norm > 0.35:
            continue

        motif_weight = 1.0 - (0.86 * ring_like)
        center_weight = max(0.20, 1.0 - (0.45 * d_norm))
        score = arc_len * motif_weight * center_weight
        candidates.append((score, pts.tolist()))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:limit]]


def extract_round_center_motif_lines(bgr_img, mask, main_contour, max_lines=12):
    """
    Extract motif lines from the inner-mid band of round artifacts.
    Helps preserve central measured motifs around the boss.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return []
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(10.0, 0.5 * float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))))

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    lap = cv2.convertScaleAbs(cv2.Laplacian(enhanced, cv2.CV_16S, ksize=3))
    edge = cv2.Canny(enhanced, 22, 88)
    fused = cv2.bitwise_or(lap, edge)

    h, w = mask.shape[:2]
    yy, xx = np.indices((h, w))
    rr = np.sqrt(((xx.astype(np.float32) - cx) ** 2) + ((yy.astype(np.float32) - cy) ** 2))
    annulus = ((rr >= (0.18 * r_ref)) & (rr <= (0.66 * r_ref))).astype(np.uint8) * 255
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    fused = cv2.bitwise_and(fused, fused, mask=interior)
    fused = cv2.bitwise_and(fused, annulus)
    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, [main_contour], -1, 255, thickness=5)
    fused[boundary > 0] = 0

    contours, _ = cv2.findContours(fused, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    min_len = max(9.0, 0.024 * float(min(h, w)))
    max_len = float(max(h, w)) * 1.10
    candidates = []
    for c in contours:
        arc_len = float(cv2.arcLength(c, False))
        if arc_len < min_len or arc_len > max_len:
            continue

        epsilon = 0.0065 * arc_len
        approx = cv2.approxPolyDP(c, epsilon, False).reshape(-1, 2)
        if approx.shape[0] < 2:
            continue

        center = np.mean(approx, axis=0)
        lx = float(center[0])
        ly = float(center[1])
        d_norm = (((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
        if d_norm < 0.16 or d_norm > 0.70:
            continue

        arr = np.asarray(approx, dtype=np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        inside = 0
        for px, py in arr:
            if mask[int(py), int(px)] > 0:
                inside += 1
        if inside / float(max(1, len(arr))) < 0.82:
            continue

        line = arr.tolist()
        ring_like = line_ring_likeness(line, cx, cy)
        if ring_like >= 0.96 and d_norm > 0.28:
            continue

        band_score = max(0.20, 1.0 - abs(d_norm - 0.40))
        score = arc_len * (1.0 - (0.32 * ring_like)) * band_score
        candidates.append((score, line))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:limit]]


def estimate_round_boss_radius(gray_img, mask, cx, cy, r_ref):
    """
    Estimate central boss radius in round artifacts by radial edge response.
    Returns radius in pixels; 0.0 when no reliable boss boundary is found.
    """
    try:
        if r_ref <= 12.0:
            return 0.0
        h, w = gray_img.shape[:2]
        blur = cv2.GaussianBlur(gray_img, (0, 0), 1.1)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad_u8 = cv2.convertScaleAbs(grad)

        yy, xx = np.indices((h, w))
        rr = np.sqrt(((xx.astype(np.float32) - float(cx)) ** 2) + ((yy.astype(np.float32) - float(cy)) ** 2))
        valid_mask = (mask > 0).astype(np.uint8)
        if np.count_nonzero(valid_mask) < 80:
            return 0.0

        r_min = max(4, int(round(0.10 * r_ref)))
        r_max = max(r_min + 2, int(round(0.56 * r_ref)))
        if r_max <= r_min:
            return 0.0

        best_r = 0.0
        best_score = 0.0
        for r in range(r_min, r_max + 1):
            band = np.where((rr >= (float(r) - 1.6)) & (rr <= (float(r) + 1.6)), 255, 0).astype(np.uint8)
            band = cv2.bitwise_and(band, valid_mask)
            count = int(np.count_nonzero(band))
            if count < 90:
                continue
            mean_grad = float(np.mean(grad_u8[band > 0]))
            ratio = float(r) / max(1.0, float(r_ref))
            # Favor boss candidates in typical ratio range.
            ratio_prior = max(0.0, 1.0 - (abs(ratio - 0.24) / 0.24))
            score = (0.72 * mean_grad) + (28.0 * ratio_prior)
            if score > best_score:
                best_score = score
                best_r = float(r)

        if best_score < 18.0:
            return 0.0
        return float(best_r)
    except Exception as e:
        log_exception("estimate_round_boss_radius", e)
        return 0.0


def extract_round_relief_lines(bgr_img, mask, main_contour, max_lines=24):
    """
    Fallback extractor for embossed motifs on round artifacts (e.g. bronze mirrors).
    Targets non-uniform inner relief features when generic motif extraction is sparse.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return []

    x0 = int(np.min(xs))
    x1 = int(np.max(xs))
    y0 = int(np.min(ys))
    y1 = int(np.max(ys))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(10.0, 0.5 * float(max(bw, bh)))

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_u8 = cv2.convertScaleAbs(grad)
    grad_bin = cv2.adaptiveThreshold(
        grad_u8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -2,
    )

    edges = cv2.Canny(enhanced, 24, 92)
    detail_bin = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        6,
    )
    fused = cv2.bitwise_or(grad_bin, edges)
    fused = cv2.bitwise_or(fused, detail_bin)

    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    h, w = mask.shape[:2]
    yy, xx = np.indices((h, w))
    rr = np.sqrt(((xx.astype(np.float32) - cx) ** 2) + ((yy.astype(np.float32) - cy) ** 2))
    boss_r = estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
    inner_ratio = 0.34
    if boss_r > 0.0:
        inner_ratio = max(inner_ratio, min(0.56, (boss_r / max(1e-6, r_ref)) * 1.24))
    annulus = ((rr >= (inner_ratio * r_ref)) & (rr <= (0.93 * r_ref))).astype(np.uint8) * 255

    fused = cv2.bitwise_and(fused, fused, mask=interior)
    fused = cv2.bitwise_and(fused, annulus)
    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, [main_contour], -1, 255, thickness=4)
    fused[boundary > 0] = 0

    line_contours, _ = cv2.findContours(fused, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not line_contours:
        return []

    min_len = max(12.0, float(min(bw, bh)) * 0.028)
    max_len = float(max(bw, bh)) * 1.30
    min_area = max(8.0, float(bw * bh) * 0.00060)
    selected_items = []
    for contour in line_contours:
        arc_len = float(cv2.arcLength(contour, False))
        if arc_len < min_len or arc_len > max_len:
            continue
        area = float(abs(cv2.contourArea(contour)))
        if area < min_area and arc_len < (min_len * 1.55):
            continue

        epsilon = 0.0060 * arc_len
        approx = cv2.approxPolyDP(contour, epsilon, False)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 2:
            continue

        center = np.mean(pts, axis=0)
        lx = float(center[0])
        ly = float(center[1])
        d_norm = (((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
        if d_norm > 0.95:
            continue

        arr = np.asarray(pts, dtype=np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        inside = 0
        for px, py in arr:
            if mask[int(py), int(px)] > 0:
                inside += 1
        if inside / float(max(1, len(arr))) < 0.78:
            continue

        ring_like = line_ring_likeness(arr.tolist(), cx, cy)
        ang_span = line_angle_span(arr.tolist(), cx, cy)
        if ring_like >= 0.96 and ang_span >= (np.pi * 1.55) and d_norm > 0.28:
            continue

        complexity = min(1.0, float(pts.shape[0]) / 18.0)
        band_score = max(0.18, 1.0 - abs(d_norm - 0.62))
        area_score = min(1.0, area / max(1.0, 0.02 * float(bw * bh)))
        score = (
            arc_len * (0.55 + (0.45 * complexity)) * band_score * (1.0 - (0.30 * ring_like))
        ) + (24.0 * area_score)
        selected_items.append((score, arr.tolist()))

    if not selected_items:
        return []

    selected_items.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in selected_items[:limit]]


def extract_round_relief_region_lines(bgr_img, mask, main_contour, max_lines=18):
    """
    Extract closed relief-region contours for round artifacts.
    This complements edge-only lines when motifs appear as low-contrast patches.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return []

    x0 = int(np.min(xs))
    x1 = int(np.max(xs))
    y0 = int(np.min(ys))
    y1 = int(np.max(ys))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(10.0, 0.5 * float(max(bw, bh)))

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur_sigma = max(2.0, float(min(bw, bh)) * 0.018)
    low = cv2.GaussianBlur(enhanced, (0, 0), blur_sigma)
    high = cv2.absdiff(enhanced, low)
    lap = cv2.convertScaleAbs(cv2.Laplacian(enhanced, cv2.CV_16S, ksize=3))
    mix = cv2.addWeighted(high, 0.75, lap, 0.70, 0)

    _, otsu_bin = cv2.threshold(mix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_bin = cv2.adaptiveThreshold(
        mix,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        27,
        -3,
    )
    fused = cv2.bitwise_or(otsu_bin, adaptive_bin)

    h, w = mask.shape[:2]
    yy, xx = np.indices((h, w))
    rr = np.sqrt(((xx.astype(np.float32) - cx) ** 2) + ((yy.astype(np.float32) - cy) ** 2))
    boss_r = estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
    inner_ratio = 0.36
    if boss_r > 0.0:
        inner_ratio = max(inner_ratio, min(0.58, (boss_r / max(1e-6, r_ref)) * 1.28))
    annulus = ((rr >= (inner_ratio * r_ref)) & (rr <= (0.94 * r_ref))).astype(np.uint8) * 255
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    fused = cv2.bitwise_and(fused, fused, mask=interior)
    fused = cv2.bitwise_and(fused, annulus)
    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    fused = cv2.morphologyEx(
        fused,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, [main_contour], -1, 255, thickness=4)
    fused[boundary > 0] = 0

    contours, _ = cv2.findContours(fused, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    min_area = max(10.0, float(bw * bh) * 0.00090)
    max_area = float(bw * bh) * 0.12
    min_arc = max(14.0, float(min(bw, bh)) * 0.036)

    candidates = []
    for contour in contours:
        area = float(abs(cv2.contourArea(contour)))
        if area < min_area or area > max_area:
            continue
        arc_len = float(cv2.arcLength(contour, True))
        if arc_len < min_arc:
            continue

        epsilon = max(1.2, 0.0095 * arc_len)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue

        center = np.mean(pts, axis=0)
        lx = float(center[0])
        ly = float(center[1])
        d_norm = (((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
        if d_norm > 0.96:
            continue

        arr = np.asarray(pts, dtype=np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
        inside = 0
        for px, py in arr:
            if mask[int(py), int(px)] > 0:
                inside += 1
        if inside / float(max(1, len(arr))) < 0.84:
            continue

        line = arr.tolist()
        line.append(line[0])
        ring_like = line_ring_likeness(line, cx, cy)
        ang_span = line_angle_span(line, cx, cy)
        if ring_like >= 0.95 and ang_span >= (np.pi * 1.35) and d_norm > 0.25:
            continue
        compactness = area / max(1e-6, arc_len * arc_len)
        if compactness < 0.0024:
            continue

        complexity = min(1.0, float(len(line)) / 22.0)
        band_score = max(0.25, 1.0 - abs(d_norm - 0.62))
        score = (area * 2.0) + (arc_len * (0.42 + (0.40 * complexity))) + (12.0 * band_score) - (7.0 * ring_like)
        candidates.append((score, line))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:limit]]


def polar_unwrap(image, cx, cy, max_radius, n_theta, n_rad):
    """
    Polar unwrap returning an array indexed as [radius_bin, angle_bin].

    cv2.warpPolar's dsize is (width, height) with width = radius axis and
    height = angle axis, so the result is transposed here to keep the
    radius-major convention used by the motif extractors.
    """
    polar = cv2.warpPolar(
        image,
        (int(n_rad), int(n_theta)),
        (float(cx), float(cy)),
        float(max_radius),
        cv2.WARP_POLAR_LINEAR,
    )
    if polar is None or polar.size == 0:
        return None
    return np.ascontiguousarray(polar.T)


def extract_round_polar_motif_lines(bgr_img, mask, main_contour, max_lines=16):
    """
    Extract inner motifs for round artifacts using polar-unwrapped detail analysis.
    This reduces ring bias by filtering long angular runs that represent concentric bands.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 120:
        return []

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(12.0, 0.5 * float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))))
    if r_ref < 14.0:
        return []

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    boss_r = estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
    inner_r = 0.24 * r_ref
    if boss_r > 0.0:
        inner_r = max(inner_r, min(0.54 * r_ref, boss_r * 1.12))
    outer_r = 0.93 * r_ref
    if (outer_r - inner_r) < 8.0:
        return []

    polar_max = max(6.0, 0.98 * r_ref)
    n_theta = int(max(320, min(840, round(5.4 * r_ref))))
    n_rad = int(max(110, min(460, round(1.25 * r_ref))))
    try:
        polar = polar_unwrap(enhanced, cx, cy, polar_max, n_theta, n_rad)
    except Exception as e:
        log_exception("extract_round_polar_motif_lines", e)
        return []
    if polar is None or polar.size == 0:
        return []

    r0_idx = int(max(0, min(n_rad - 2, round((inner_r / polar_max) * (n_rad - 1)))))
    r1_idx = int(max(r0_idx + 1, min(n_rad - 1, round((outer_r / polar_max) * (n_rad - 1)))))
    if r1_idx <= r0_idx:
        return []

    polar_crop = polar[r0_idx:r1_idx + 1, :]
    if polar_crop.size == 0:
        return []

    # Enhance local relief in polar space.
    blur = cv2.GaussianBlur(polar_crop, (0, 0), sigmaX=3.2, sigmaY=1.2)
    highpass = cv2.absdiff(polar_crop, blur.astype(np.uint8))
    grad_r = cv2.convertScaleAbs(cv2.Sobel(polar_crop, cv2.CV_16S, 0, 1, ksize=3))
    grad_t = cv2.convertScaleAbs(cv2.Sobel(polar_crop, cv2.CV_16S, 1, 0, ksize=3))
    fused = cv2.addWeighted(highpass, 0.58, grad_r, 0.44, 0)
    fused = cv2.addWeighted(fused, 0.84, grad_t, 0.22, 0)

    _, otsu_bin = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_bin = cv2.adaptiveThreshold(
        fused,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        29,
        -2,
    )
    edges = cv2.Canny(polar_crop, 26, 96)
    polar_bin = cv2.bitwise_or(otsu_bin, adaptive_bin)
    polar_bin = cv2.bitwise_or(polar_bin, edges)
    polar_bin = cv2.morphologyEx(
        polar_bin,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1,
    )
    polar_bin = cv2.morphologyEx(
        polar_bin,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(polar_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    crop_h, crop_w = polar_bin.shape[:2]
    min_area = max(7.0, float(crop_h * crop_w) * 0.00020)
    max_area = float(crop_h * crop_w) * 0.28
    min_arc = max(8.0, 0.030 * float(min(mask.shape[0], mask.shape[1])))
    sector_count = max(10, min(24, limit * 2))
    min_sep = max(3.0, 0.040 * float(min(mask.shape[0], mask.shape[1])))

    candidates = []
    h_img, w_img = mask.shape[:2]
    for contour in contours:
        area = float(abs(cv2.contourArea(contour)))
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        # Reject long angular runs with tiny radial span (concentric band artifacts).
        if bw >= int(0.32 * crop_w) and bh <= max(3, int(0.06 * crop_h)):
            continue
        if bw >= int(0.52 * crop_w) and bh <= max(5, int(0.10 * crop_h)):
            continue

        arc_len = float(cv2.arcLength(contour, True))
        if arc_len < min_arc:
            continue

        eps = max(1.0, 0.010 * arc_len)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue

        cart_pts = []
        inside_hits = 0
        for px, py in approx:
            gx = int(np.clip(px, 0, crop_w - 1))
            gy = int(np.clip(py, 0, crop_h - 1))
            theta = (float(gx) / float(max(1, n_theta - 1))) * (2.0 * np.pi)
            global_r_idx = int(np.clip(gy + r0_idx, 0, n_rad - 1))
            radius = (float(global_r_idx) / float(max(1, n_rad - 1))) * polar_max
            x_img = int(round(cx + (radius * np.cos(theta))))
            y_img = int(round(cy + (radius * np.sin(theta))))
            x_img = int(np.clip(x_img, 0, w_img - 1))
            y_img = int(np.clip(y_img, 0, h_img - 1))
            cart_pts.append([x_img, y_img])
            if mask[y_img, x_img] > 0:
                inside_hits += 1

        if len(cart_pts) < 3:
            continue
        inside_ratio = inside_hits / float(max(1, len(cart_pts)))
        if inside_ratio < 0.84:
            continue

        if cart_pts[0] != cart_pts[-1]:
            cart_pts.append(cart_pts[0])

        center, cart_arc = line_centroid_and_length(cart_pts)
        if center is None or cart_arc < 8.0:
            continue
        d_norm = (((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
        if d_norm < 0.18 or d_norm > 0.95:
            continue

        ring_like = line_ring_likeness(cart_pts, cx, cy)
        ang_span = line_angle_span(cart_pts, cx, cy)
        if ring_like >= 0.90 and ang_span >= (np.pi * 0.62) and d_norm > 0.24:
            continue

        theta_c = float(np.arctan2(center[1] - cy, center[0] - cx))
        sector_idx = int(((theta_c + np.pi) / (2.0 * np.pi)) * sector_count) % sector_count
        complexity = min(1.0, float(len(cart_pts)) / 20.0)
        band_score = max(0.25, 1.0 - abs(d_norm - 0.60))
        score = (1.3 * area) + (0.55 * cart_arc) + (8.0 * complexity) + (10.0 * band_score) - (10.0 * ring_like)
        candidates.append((score, sector_idx, center, cart_pts))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    centers = []
    used_sector = {}
    for _, sector_idx, center, line in candidates:
        if int(used_sector.get(sector_idx, 0)) >= 1:
            continue
        if any((((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5) < min_sep for c in centers):
            continue
        selected.append(line)
        centers.append(center)
        used_sector[sector_idx] = int(used_sector.get(sector_idx, 0)) + 1
        if len(selected) >= limit:
            break

    return selected[:limit]


def select_round_inner_motif_lines(lines, mask, max_lines=4, prefer_outer=False):
    """
    Select internal motif lines for round artifacts while suppressing border noise.
    """
    limit = int(max(0, int(max_lines)))
    if limit <= 0 or not lines:
        return []

    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return []

    x0 = int(np.min(xs))
    x1 = int(np.max(xs))
    y0 = int(np.min(ys))
    y1 = int(np.max(ys))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    r_ref = max(12.0, 0.5 * float(max(bw, bh)))

    edge_margin_ratio = 0.02 if prefer_outer else 0.04
    margin_x = max(2, int(bw * edge_margin_ratio))
    margin_y = max(2, int(bh * edge_margin_ratio))
    min_len = max(6.0, float(min(bw, bh)) * 0.03)
    max_len = float(max(bw, bh)) * 1.20

    candidates = []
    for line in lines:
        center, arc_len = line_centroid_and_length(line)
        if center is None or arc_len < min_len or arc_len > max_len:
            continue
        lx, ly = center

        # Keep features away from silhouette edge.
        if lx <= (x0 + margin_x) or lx >= (x1 - margin_x):
            continue
        if ly <= (y0 + margin_y) or ly >= (y1 - margin_y):
            continue

        d = ((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5
        d_norm = d / r_ref
        if d_norm > (0.97 if prefer_outer else 0.92):
            continue

        ring_like = line_ring_likeness(line, cx, cy)
        if prefer_outer:
            # Measured round artifacts should keep relief motifs across mid/outer bands.
            if ring_like >= 0.94 and d_norm > 0.30:
                continue
            length_score = min(1.0, arc_len / max(1.0, 0.16 * float(max(bw, bh))))
            center_score = max(0.0, 1.0 - d_norm)
            band_score = max(0.20, 1.0 - (abs(d_norm - 0.50) / 0.62))
            motif_weight = 1.0 - (0.48 * ring_like)
            score = ((0.32 * center_score) + (0.43 * length_score) + (0.25 * band_score)) * motif_weight
            if score < 0.06:
                continue
        else:
            # Drop lines that look like circular bands unless they are very central.
            if ring_like >= 0.82 and d_norm > 0.45:
                continue
            length_score = min(1.0, arc_len / max(1.0, 0.18 * float(max(bw, bh))))
            center_score = max(0.0, 1.0 - d_norm)
            motif_weight = 1.0 - (0.72 * ring_like)
            score = ((0.45 * center_score) + (0.55 * length_score)) * motif_weight
            if score < 0.08:
                continue
        if score <= 0.0:
            continue
        candidates.append((score, (lx, ly), d_norm, line))

    if not candidates:
        backup = []
        for line in lines:
            center, arc_len = line_centroid_and_length(line)
            if center is None or arc_len < 6.0:
                continue
            lx, ly = center
            if lx <= (x0 + 2) or lx >= (x1 - 2) or ly <= (y0 + 2) or ly >= (y1 - 2):
                continue
            arr = np.asarray(line, dtype=np.int32)
            arr[:, 0] = np.clip(arr[:, 0], 0, mask.shape[1] - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, mask.shape[0] - 1)
            inside = 0
            for px, py in arr:
                if mask[int(py), int(px)] > 0:
                    inside += 1
            if inside / float(max(1, len(arr))) < 0.80:
                continue
            ring_like = line_ring_likeness(line, cx, cy)
            d_norm = (((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
            if prefer_outer:
                if ring_like >= 0.95 and d_norm > 0.28:
                    continue
                backup_score = arc_len * (1.0 - (0.42 * ring_like)) * max(0.25, 1.0 - abs(d_norm - 0.50))
            else:
                if ring_like >= 0.88 and d_norm > 0.50:
                    continue
                backup_score = arc_len * (1.0 - (0.58 * ring_like))
            backup.append((backup_score, (lx, ly), d_norm, line))
        if not backup:
            return []
        backup.sort(key=lambda item: item[0], reverse=True)
        candidates = [(float(item[0]), item[1], item[2], item[3]) for item in backup]

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    selected_centers = []
    min_sep = (
        max(4.0, 0.05 * float(min(bw, bh)))
        if prefer_outer
        else max(5.0, 0.07 * float(min(bw, bh)))
    )
    used_angle_bins = {}
    angle_bin_count = 16 if prefer_outer else 12
    per_bin_limit = 2 if prefer_outer else 1

    for _, center, d_norm, line in candidates:
        if any((((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5) < min_sep for c in selected_centers):
            continue
        if prefer_outer and d_norm >= 0.24:
            theta = float(np.arctan2(center[1] - cy, center[0] - cx))
            angle_bin = int(((theta + np.pi) / (2.0 * np.pi)) * angle_bin_count) % angle_bin_count
            used_count = int(used_angle_bins.get(angle_bin, 0))
            if used_count >= per_bin_limit:
                continue
            used_angle_bins[angle_bin] = used_count + 1
        selected.append(line)
        selected_centers.append(center)
        if len(selected) >= limit:
            break
    return selected
