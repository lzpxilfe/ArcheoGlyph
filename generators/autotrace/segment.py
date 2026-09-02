# -*- coding: utf-8 -*-
"""
Silhouette mask extraction (OpenCV backend, GrabCut refinement, component selection).

Auto-generated from the former ContourGenerator methods; QGIS-free.
"""

import importlib.util

import cv2
import numpy as np


def get_mask_opencv(bgr_img):
    """
    OpenCV silhouette extraction with shadow suppression.
    """
    h, w = bgr_img.shape[:2]
    if h < 8 or w < 8:
        return np.zeros((h, w), dtype=np.uint8)

    blurred = cv2.GaussianBlur(bgr_img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    _, s, _ = cv2.split(hsv)

    # Estimate background from border strips.
    b = max(6, min(h, w) // 28)
    border_pixels = np.concatenate([
        lab[:b, :, :].reshape(-1, 3),
        lab[-b:, :, :].reshape(-1, 3),
        lab[:, :b, :].reshape(-1, 3),
        lab[:, -b:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border_pixels, axis=0)

    lab_f = lab.astype(np.float32)
    bg_f = bg.astype(np.float32)

    # Chroma distance (a,b) is much less sensitive to lighting/shadow than full Lab distance.
    delta_ab = np.linalg.norm(lab_f[:, :, 1:3] - bg_f[1:3], axis=2)
    delta_l = np.abs(lab_f[:, :, 0] - bg_f[0])

    ab_scale = max(6.0, float(np.percentile(delta_ab, 99.0)))
    l_scale = max(8.0, float(np.percentile(delta_l, 99.0)))
    ab_u8 = np.clip((delta_ab / ab_scale) * 255.0, 0, 255).astype(np.uint8)
    l_u8 = np.clip((delta_l / l_scale) * 255.0, 0, 255).astype(np.uint8)

    _, chroma_mask = cv2.threshold(ab_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, light_mask = cv2.threshold(l_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_mask = cv2.threshold(s, 16, 255, cv2.THRESH_BINARY)[1]

    target_mask = cv2.bitwise_or(chroma_mask, cv2.bitwise_and(light_mask, sat_mask))

    # If object is near-gray, relax to include luminance edges as fallback.
    min_fg = int(h * w * 0.008)
    if np.count_nonzero(target_mask) < min_fg:
        target_mask = cv2.bitwise_or(target_mask, light_mask)

    shadow_like = (
        (s < 22)
        & (delta_ab < float(np.percentile(delta_ab, 58.0)))
        & (delta_l > float(np.percentile(delta_l, 72.0)))
    )
    target_mask[shadow_like] = 0

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # White-background fallback:
    # many museum/reference photos have bright uniform background, where
    # non-white thresholding is often more stable than chroma/luma split.
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    border_gray = np.concatenate([
        gray[:b, :].reshape(-1),
        gray[-b:, :].reshape(-1),
        gray[:, :b].reshape(-1),
        gray[:, -b:].reshape(-1),
    ], axis=0)
    circle_mask = detect_center_circle_mask(gray)
    if float(np.mean(border_gray)) >= 180.0 and float(np.std(border_gray)) <= 36.0:
        white_fg = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)[1]
        white_fg = _split_off_cast_shadow(gray, s, white_fg)
        white_fg = cv2.morphologyEx(white_fg, cv2.MORPH_OPEN, kernel3, iterations=1)
        white_fg = cv2.morphologyEx(white_fg, cv2.MORPH_CLOSE, kernel5, iterations=2)
        white_fg = select_primary_component(white_fg)
        if np.count_nonzero(white_fg) >= int(h * w * 0.01):
            combined = cv2.bitwise_or(target_mask, white_fg)
            target_mask = select_primary_component(combined)

    if circle_mask is not None and np.count_nonzero(circle_mask) >= int(h * w * 0.04):
        metrics = mask_shape_metrics(target_mask)
        circle_area = float(np.count_nonzero(circle_mask))
        overlap = float(np.count_nonzero(cv2.bitwise_and(target_mask, circle_mask))) / float(
            max(1.0, circle_area)
        )
        area_ratio = float(metrics["area"]) / max(1.0, circle_area)
        round_candidate = (
            metrics["aspect_balance"] >= 0.70
            and metrics["fill_ratio"] <= 0.95
            and (metrics["circularity"] >= 0.50 or overlap >= 0.64)
        )
        if round_candidate:
            # Clamp to the detected circle only when the mask is clearly not the
            # object itself (background chunk, border leak); a compact oval that
            # is merely larger than its inscribed circle keeps its true outline.
            should_clamp = (
                metrics["touches_border"] or
                overlap < 0.58 or
                (area_ratio > 1.30 and metrics["circularity"] < 0.60)
            )
            if should_clamp:
                clamped = cv2.bitwise_and(cv2.bitwise_or(target_mask, circle_mask), circle_mask)
                if np.count_nonzero(clamped) >= int(circle_area * 0.42):
                    target_mask = clamped

    refined = refine_with_grabcut(blurred, target_mask)
    if refined is not None:
        target_mask = refined

    target_mask = select_primary_component(target_mask)
    # If selected mask still looks like a border-attached background chunk,
    # retry with a center-rectangle GrabCut pass.
    fg_ratio = float(np.count_nonzero(target_mask)) / float(max(1, h * w))
    if mask_touches_border(target_mask) and fg_ratio > 0.45:
        center_fallback = get_mask_center_grabcut(blurred)
        if center_fallback is not None:
            target_mask = center_fallback

    # Fallback: when current mask is blob-like, try recovering a tall/slender object
    # directly from image intensity (useful for daggers/spears on bright background).
    slender_candidate = recover_tall_component_from_image(blurred)
    if slender_candidate is not None:
        current_features = mask_bbox_features(target_mask)
        candidate_features = mask_bbox_features(slender_candidate)
        score_current = mask_selection_score(blurred, target_mask)
        score_candidate = mask_selection_score(blurred, slender_candidate)

        choose_candidate = False
        if candidate_features["tall_ratio"] >= 2.8 and current_features["tall_ratio"] < 2.1:
            choose_candidate = True
        if candidate_features["tall_ratio"] >= (current_features["tall_ratio"] * 1.35) and score_candidate >= (score_current - 0.04):
            choose_candidate = True
        if score_candidate >= (score_current + 0.08):
            choose_candidate = True

        if choose_candidate:
            target_mask = slender_candidate

    target_mask = smooth_mask_edges(target_mask)
    return target_mask


def _split_off_cast_shadow(gray, saturation, nonwhite_mask):
    """
    On a white/paper background the non-white region often contains the
    object *and* its soft cast shadow. When the region splits into a dark
    mode (object) and a light, unsaturated mode (shadow), keep only the dark
    mode. Returns the possibly reduced mask.
    """
    region = nonwhite_mask > 0
    count = int(np.count_nonzero(region))
    if count < 400:
        return nonwhite_mask
    values = gray[region].reshape(-1, 1)
    otsu_t, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = region & (gray < otsu_t)
    light = region & ~dark
    dark_count = int(np.count_nonzero(dark))
    light_count = int(np.count_nonzero(light))
    if dark_count < int(0.25 * count) or light_count < int(0.08 * count):
        return nonwhite_mask
    light_gray = float(np.mean(gray[light]))
    light_sat = float(np.mean(saturation[light]))
    dark_gray = float(np.mean(gray[dark]))
    # Shadow signature: light part is pale and grey, clearly separated from the object tone.
    if light_sat <= 28.0 and light_gray >= 150.0 and (light_gray - dark_gray) >= 60.0:
        return (dark.astype(np.uint8)) * 255
    return nonwhite_mask


def select_primary_component(mask):
    """Keep the best foreground component by size + center + compactness score."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask

    # Recover tall/slender artifacts (e.g. daggers) when large blobs dominate.
    slender = recover_slender_component(mask)
    if slender is not None:
        return slender

    cx_ref = w * 0.5
    cy_ref = h * 0.5
    candidate_items = []

    min_area = max(120.0, (h * w) * 0.002)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + (cw * 0.5)
        cy = y + (ch * 0.5)
        d = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2) ** 0.5
        d_norm = d / max(1.0, (w * w + h * h) ** 0.5)

        perim = float(cv2.arcLength(c, True))
        circularity = 0.0
        if perim > 1e-6:
            circularity = (4.0 * np.pi * area) / (perim * perim)
        circularity = max(0.0, min(1.0, circularity))

        fill_ratio = area / max(1.0, float(cw * ch))
        fill_ratio = max(0.0, min(1.0, fill_ratio))

        tall = ch / max(1.0, float(cw))
        tall_norm = min(1.0, tall / 2.4)

        touches_border = (
            x <= 1 or y <= 1 or (x + cw) >= (w - 1) or (y + ch) >= (h - 1)
        )

        score = area * (1.0 - min(0.95, d_norm))
        score *= (0.56 + (0.22 * fill_ratio) + (0.12 * circularity) + (0.10 * tall_norm))
        candidate_items.append((score, touches_border, c))

    if not candidate_items:
        best = max(contours, key=cv2.contourArea)
    else:
        non_border = [item for item in candidate_items if not item[1]]
        pool = non_border if non_border else candidate_items
        best = max(pool, key=lambda item: item[0])[2]

    out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(out, [best], -1, 255, thickness=cv2.FILLED)
    return out


def recover_slender_component(mask):
    """
    Try to recover a tall slender foreground component when the main mask is blob-like.
    Prevents swords/daggers from being swallowed by large background chunks.
    """
    h, w = mask.shape[:2]
    total = float(max(1, h * w))
    min_area = max(40.0, total * 0.0012)
    cx_ref = w * 0.5
    cy_ref = h * 0.5

    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
    ]
    for kernel in kernels:
        eroded = cv2.erode(mask, kernel, iterations=1)
        if np.count_nonzero(eroded) < int(total * 0.0008):
            continue

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 2 or ch < 2:
                continue

            tall = ch / max(1.0, float(cw))
            if tall < 2.4:
                continue
            fill_ratio = area / max(1.0, float(cw * ch))
            if fill_ratio > 0.82:
                continue

            cx = x + (cw * 0.5)
            cy = y + (ch * 0.5)
            d = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2) ** 0.5
            d_norm = d / max(1.0, (w * w + h * h) ** 0.5)
            if d_norm > 0.34:
                continue

            score = area * (1.0 - min(0.92, d_norm))
            score *= (0.55 + (0.45 * min(1.0, tall / 4.2)))
            score *= (1.10 - (0.55 * min(1.0, fill_ratio)))
            candidates.append((score, c))

        if not candidates:
            continue

        best = max(candidates, key=lambda item: item[0])[1]
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [best], -1, 255, thickness=cv2.FILLED)

        grow = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        out = cv2.bitwise_and(grow, mask)
        out = smooth_mask_edges(out)

        chk_contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not chk_contours:
            continue
        chk = max(chk_contours, key=cv2.contourArea)
        chk_area = float(cv2.contourArea(chk))
        x, y, cw, ch = cv2.boundingRect(chk)
        tall = ch / max(1.0, float(cw))
        if chk_area >= (total * 0.0018) and tall >= 1.9:
            return out

    return None


def mask_shape_metrics(mask):
    """Return simple shape metrics for the dominant foreground component."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"area": 0.0, "circularity": 0.0, "touches_border": False}
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    circularity = (4.0 * np.pi * area) / (perim * perim) if perim > 1e-6 else 0.0
    x, y, cw, ch = cv2.boundingRect(c)
    touches_border = (x <= 1 or y <= 1 or (x + cw) >= (w - 1) or (y + ch) >= (h - 1))
    aspect_balance = min(cw, ch) / max(1.0, float(max(cw, ch)))
    fill_ratio = area / max(1.0, float(cw * ch))
    return {
        "area": area,
        "circularity": max(0.0, min(1.0, circularity)),
        "touches_border": bool(touches_border),
        "aspect_balance": max(0.0, min(1.0, aspect_balance)),
        "fill_ratio": max(0.0, min(1.0, fill_ratio)),
    }


def mask_bbox_features(mask):
    """Return bbox-derived features for dominant component."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {
            "area_ratio": 0.0,
            "tall_ratio": 0.0,
            "fill_ratio": 0.0,
            "center_dist_norm": 1.0,
        }
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    x, y, cw, ch = cv2.boundingRect(c)
    bbox_area = max(1.0, float(cw * ch))
    cx = x + (cw * 0.5)
    cy = y + (ch * 0.5)
    d = ((cx - (w * 0.5)) ** 2 + (cy - (h * 0.5)) ** 2) ** 0.5
    d_norm = d / max(1.0, (w * w + h * h) ** 0.5)
    return {
        "area_ratio": area / max(1.0, float(h * w)),
        "tall_ratio": float(ch) / max(1.0, float(cw)),
        "fill_ratio": area / bbox_area,
        "center_dist_norm": d_norm,
    }


def recover_tall_component_from_image(bgr_img):
    """
    Recover a tall/slender centered component directly from image intensities.
    Useful when initial mask is a broad blob but object is dagger-like.
    """
    try:
        h, w = bgr_img.shape[:2]
        total = float(max(1, h * w))
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        bin_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        bin_adapt = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            6,
        )
        mask = cv2.bitwise_or(bin_otsu, bin_adapt)

        # Canny-derived fill candidate (helps when object edges are clearer than tone).
        med = float(np.median(blur))
        lo = int(max(16, 0.66 * med))
        hi = int(min(220, 1.33 * med))
        edges = cv2.Canny(blur, lo, hi)
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, ek, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ek, iterations=2)
        edge_fill = np.zeros((h, w), dtype=np.uint8)
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for ec in edge_contours:
            ea = float(cv2.contourArea(ec))
            if ea < max(40.0, total * 0.0009) or ea > (total * 0.65):
                continue
            cv2.drawContours(edge_fill, [ec], -1, 255, thickness=cv2.FILLED)
        if np.count_nonzero(edge_fill) > 0:
            mask = cv2.bitwise_or(mask, edge_fill)

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

        # Prefer centered band to suppress side/background blobs.
        band = np.zeros((h, w), dtype=np.uint8)
        x0 = int(max(0, w * 0.20))
        x1 = int(min(w, w * 0.80))
        band[:, x0:x1] = 255
        mask = cv2.bitwise_and(mask, band)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        cx_ref = w * 0.5
        cy_ref = h * 0.5
        min_area = max(60.0, total * 0.0015)
        max_area = total * 0.35
        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 2 or ch < 2:
                continue

            tall = float(ch) / max(1.0, float(cw))
            if tall < 2.5:
                continue

            fill_ratio = area / max(1.0, float(cw * ch))
            if fill_ratio > 0.86:
                continue

            cx = x + (cw * 0.5)
            cy = y + (ch * 0.5)
            d = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2) ** 0.5
            d_norm = d / max(1.0, (w * w + h * h) ** 0.5)
            if d_norm > 0.36:
                continue

            score = area * (1.0 - min(0.95, d_norm))
            score *= (0.60 + (0.40 * min(1.0, tall / 5.0)))
            score *= (1.08 - (0.55 * min(1.0, fill_ratio)))
            candidates.append((score, c))

        if not candidates:
            return None

        best = max(candidates, key=lambda item: item[0])[1]
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [best], -1, 255, thickness=cv2.FILLED)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k5, iterations=1)
        out = smooth_mask_edges(out)
        return select_primary_component(out)
    except Exception:
        return None


def mask_selection_score(bgr_img, mask):
    """
    Compute a backend-agnostic mask quality score.
    Higher is better for selecting between SAM and OpenCV masks in auto mode.
    """
    if mask is None or mask.size == 0:
        return -1.0

    h, w = mask.shape[:2]
    total = float(max(1, h * w))
    fg = float(np.count_nonzero(mask))
    if fg < max(80.0, total * 0.0015):
        return -1.0

    area_ratio = fg / total
    if area_ratio <= 0.004 or area_ratio >= 0.92:
        return -0.5

    metrics = mask_shape_metrics(mask)

    # Area plausibility: broad enough for mirrors, but penalize near-full-frame blobs.
    if area_ratio <= 0.06:
        area_score = 0.45 + (0.55 * ((area_ratio - 0.004) / max(1e-6, 0.056)))
    elif area_ratio <= 0.58:
        area_score = 1.0
    else:
        area_score = max(0.0, 1.0 - ((area_ratio - 0.58) / 0.30))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + (cw * 0.5)
        cy = y + (ch * 0.5)
        d = ((cx - (w * 0.5)) ** 2 + (cy - (h * 0.5)) ** 2) ** 0.5
        d_norm = d / max(1.0, (w * w + h * h) ** 0.5)
        center_score = max(0.0, 1.0 - (d_norm / 0.62))
    else:
        center_score = 0.0

    border_score = 0.0 if metrics["touches_border"] else 1.0

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 48, 142)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k3)
    boundary_count = int(np.count_nonzero(boundary))
    if boundary_count < 40:
        edge_score = 0.0
    else:
        overlap = float(np.count_nonzero(cv2.bitwise_and(edges, boundary))) / float(boundary_count)
        edge_score = max(0.0, min(1.0, overlap / 0.42))

    score = (
        (0.46 * edge_score) +
        (0.24 * center_score) +
        (0.18 * area_score) +
        (0.12 * border_score)
    )

    # Penalize suspicious near-rectangular full masks.
    if metrics["fill_ratio"] >= 0.96 and metrics["aspect_balance"] >= 0.58:
        score -= 0.22
    if area_ratio > 0.78 and metrics["touches_border"]:
        score -= 0.35

    return float(score)


def detect_center_circle_mask(gray_img):
    """Detect a dominant near-center circle and return it as a binary mask."""
    try:
        h, w = gray_img.shape[:2]
        min_r = int(max(8, min(h, w) * 0.16))
        max_r = int(max(min_r + 2, min(h, w) * 0.52))
        if max_r <= min_r:
            return None

        eq = cv2.equalizeHist(gray_img)
        blur = cv2.GaussianBlur(eq, (7, 7), 1.4)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(min(h, w) * 0.25)),
            param1=110,
            param2=30,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(20, int(min(h, w) * 0.25)),
                param1=100,
                param2=24,
                minRadius=min_r,
                maxRadius=max_r,
            )
        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)
        cx_ref = w * 0.5
        cy_ref = h * 0.5
        max_d = max(1.0, (w * w + h * h) ** 0.5)

        best = None
        best_score = -1.0
        for x, y, r in circles:
            if r < min_r or r > max_r:
                continue
            d = ((x - cx_ref) ** 2 + (y - cy_ref) ** 2) ** 0.5
            center_score = 1.0 - min(1.0, d / max_d)
            radius_score = float(r - min_r) / float(max(1, max_r - min_r))
            score = (0.72 * center_score) + (0.28 * radius_score)
            if score > best_score:
                best_score = score
                best = (x, y, r)

        if best is None:
            return None

        edges = cv2.Canny(blur, 52, 146)
        x, y, r = best
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring, (int(x), int(y)), int(r), 255, thickness=2)
        ring_pixels = int(np.count_nonzero(ring))
        if ring_pixels <= 0:
            return None
        edge_support = float(np.count_nonzero(cv2.bitwise_and(edges, ring))) / float(ring_pixels)
        if edge_support < 0.08:
            return None

        out = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(out, (int(x), int(y)), int(r), 255, thickness=-1)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        return out
    except Exception:
        return None


def mask_touches_border(mask, border=2):
    """Return True when foreground pixels touch any image border band."""
    if mask is None or mask.size == 0:
        return False
    b = int(max(1, border))
    if np.any(mask[:b, :] > 0):
        return True
    if np.any(mask[-b:, :] > 0):
        return True
    if np.any(mask[:, :b] > 0):
        return True
    if np.any(mask[:, -b:] > 0):
        return True
    return False


GRABCUT_MAX_SIDE = 360


def _grabcut_scaled(bgr_img, gc_mask=None, rect=None, iterations=3):
    """
    Run cv2.grabCut on a downscaled copy (max side GRABCUT_MAX_SIDE) and
    return the foreground mask at full resolution. GrabCut cost grows with
    pixel count, so this turns a 10-25 s call on a 1000 px analysis image into
    well under a second with no visible loss at symbol scale.
    """
    h, w = bgr_img.shape[:2]
    scale = min(1.0, float(GRABCUT_MAX_SIDE) / float(max(h, w)))
    if scale < 1.0:
        sw = max(8, int(round(w * scale)))
        sh = max(8, int(round(h * scale)))
        small = cv2.resize(bgr_img, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = None
        if gc_mask is not None:
            small_mask = cv2.resize(gc_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small_rect = None
        if rect is not None:
            x, y, rw, rh = rect
            small_rect = (
                int(round(x * scale)), int(round(y * scale)),
                max(2, int(round(rw * scale))), max(2, int(round(rh * scale))),
            )
    else:
        small, small_mask, small_rect = bgr_img, gc_mask, rect

    if small_mask is None:
        small_mask = np.zeros(small.shape[:2], dtype=np.uint8)
        mode = cv2.GC_INIT_WITH_RECT
    else:
        mode = cv2.GC_INIT_WITH_MASK
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, small_mask, small_rect, bgd_model, fgd_model, int(iterations), mode)
    fg_small = np.where(
        (small_mask == cv2.GC_FGD) | (small_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    if scale < 1.0:
        fg = cv2.resize(fg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.where(fg > 127, 255, 0).astype(np.uint8)
    return fg_small


def get_mask_center_grabcut(bgr_img):
    """
    Fallback mask extraction seeded by a central rectangle.
    Helps when border/background chunks dominate the initial mask.
    """
    try:
        h, w = bgr_img.shape[:2]
        if h < 8 or w < 8:
            return None

        x = int(max(1, w * 0.08))
        y = int(max(1, h * 0.06))
        rw = int(max(4, w * 0.84))
        rh = int(max(4, h * 0.88))

        fg = _grabcut_scaled(bgr_img, rect=(x, y, rw, rh), iterations=4)
        if np.count_nonzero(fg) < 120:
            return None

        fg = select_primary_component(fg)
        if np.count_nonzero(fg) < 120:
            return None
        return smooth_mask_edges(fg)
    except Exception:
        return None


def refine_with_grabcut(bgr_img, init_mask):
    """Refine foreground/background split with GrabCut when available."""
    try:
        h, w = init_mask.shape[:2]
        if np.count_nonzero(init_mask) < 120:
            return None

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        border = max(6, min(h, w) // 24)
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD
        gc_mask[init_mask > 0] = cv2.GC_PR_FGD

        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sure_fg = cv2.erode(init_mask, kernel5, iterations=1)
        sure_bg = cv2.bitwise_not(cv2.dilate(init_mask, kernel5, iterations=2))
        gc_mask[sure_fg > 0] = cv2.GC_FGD
        gc_mask[sure_bg > 0] = cv2.GC_BGD

        fg = _grabcut_scaled(bgr_img, gc_mask=gc_mask, iterations=3)
        if np.count_nonzero(fg) < 120:
            return None
        return smooth_mask_edges(fg)
    except Exception:
        return None


def auto_upright(bgr_img, mask):
    """
    Slightly rotate tall objects to upright orientation.
    Avoids small camera-tilt artifacts in Auto Trace outputs.
    """
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return bgr_img, mask

        main = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(main)
        if h_box <= (w_box * 1.12):
            return bgr_img, mask

        h, w = mask.shape[:2]
        margin_left = x
        margin_top = y
        margin_right = w - (x + w_box)
        margin_bottom = h - (y + h_box)
        min_margin = min(margin_left, margin_top, margin_right, margin_bottom)
        if min_margin < max(6, int(max(w_box, h_box) * 0.02)):
            return bgr_img, mask

        pts = main.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 10:
            return bgr_img, mask

        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, np.argmax(evals)]
        angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
        target = 90.0 if angle >= 0.0 else -90.0
        delta = angle - target

        if abs(delta) < 1.8 or abs(delta) > 10.0:
            return bgr_img, mask

        m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), -delta, 1.0)
        rot_bgr = cv2.warpAffine(
            bgr_img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        rot_mask = cv2.warpAffine(
            mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        rot_mask = select_primary_component(rot_mask)
        rot_mask = smooth_mask_edges(rot_mask)
        return rot_bgr, rot_mask
    except Exception:
        return bgr_img, mask


def smooth_mask_edges(mask):
    """Smooth jagged edges and fill tiny holes in binary mask."""
    if mask is None:
        return None

    blurred = cv2.GaussianBlur(mask, (0, 0), 1.05)
    smoothed = cv2.threshold(blurred, 116, 255, cv2.THRESH_BINARY)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel3, iterations=1)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # Fill enclosed holes so highlights do not punch through silhouette.
    # Pad by one pixel so the flood always starts on background, even when the
    # object touches the top-left corner of the frame.
    h, w = smoothed.shape[:2]
    padded = cv2.copyMakeBorder(smoothed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(padded)[1:-1, 1:-1]
    return cv2.bitwise_or(smoothed, holes)


# ---------------------------------------------------------------------------
# Alpha channel, ONNX salient-object model and backend selection
# ---------------------------------------------------------------------------

MASK_BACKENDS = ("auto", "opencv", "onnx", "sam")


def alpha_to_mask(alpha, threshold=128):
    """
    Turn an informative alpha channel into a silhouette mask.
    Returns None when alpha is absent or effectively opaque/empty.
    """
    if alpha is None:
        return None
    a = np.asarray(alpha)
    if a.ndim != 2 or a.size == 0:
        return None
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32) * (255.0 / max(1.0, float(a.max()))), 0, 255).astype(np.uint8)
    coverage = float(np.count_nonzero(a > threshold)) / float(a.size)
    if coverage < 0.002 or coverage > 0.985:
        return None
    mask = ((a > threshold).astype(np.uint8)) * 255
    mask = select_primary_component(mask)
    return smooth_mask_edges(mask)


def onnx_available():
    return importlib.util.find_spec("onnxruntime") is not None


_ONNX_SESSIONS = {}


class OnnxSalientBackend:
    """
    Salient-object segmentation with an ONNX model (ISNet / U²-Net family).
    ``spec`` is a ModelSpec from model_store; ``model_path`` the verified file.
    """

    def __init__(self, model_path, spec):
        self.model_path = str(model_path)
        self.spec = spec

    def _session(self):
        session = _ONNX_SESSIONS.get(self.model_path)
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.log_severity_level = 3
            session = ort.InferenceSession(
                self.model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            _ONNX_SESSIONS[self.model_path] = session
        return session

    def probability(self, bgr_img):
        """Foreground probability map in [0, 1] at the input resolution."""
        session = self._session()
        size = int(self.spec.input_size)
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = (x - np.array(self.spec.mean, dtype=np.float32)) / np.array(self.spec.std, dtype=np.float32)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])
        name = session.get_inputs()[0].name
        out = session.run(None, {name: x})[0]
        pred = np.asarray(out, dtype=np.float32)
        while pred.ndim > 2:
            pred = pred[0]
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / max(1e-6, hi - lo)
        h, w = bgr_img.shape[:2]
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

    def get_mask(self, bgr_img):
        prob = self.probability(bgr_img)
        mask = (prob > 0.5).astype(np.uint8) * 255
        if np.count_nonzero(mask) < 40:
            return None
        mask = select_primary_component(mask)
        return smooth_mask_edges(mask)


def _score(bgr_img, mask):
    try:
        return float(mask_selection_score(bgr_img, mask))
    except Exception:
        return 0.0


def _choose_between(bgr_img, model_mask, cv_mask, strict=False):
    """Pick between a model-derived mask and the OpenCV mask by selection score."""
    if model_mask is None:
        return cv_mask
    if cv_mask is None:
        return model_mask if (not strict or _score(bgr_img, model_mask) >= 0.22) else None

    score_model = _score(bgr_img, model_mask)
    score_cv = _score(bgr_img, cv_mask)
    feat_model = mask_bbox_features(model_mask)
    feat_cv = mask_bbox_features(cv_mask)

    # Guard: models occasionally pick a tiny centred fragment on reflective objects.
    tiny_model = (
        feat_model["area_ratio"] < 0.018
        and feat_cv["area_ratio"] >= (feat_model["area_ratio"] * 2.2)
    )
    if tiny_model and score_cv >= (score_model - 0.06):
        return cv_mask
    if score_model >= (score_cv + 0.08):
        return model_mask
    if score_cv >= (score_model + (0.03 if strict else 0.04)):
        return cv_mask
    return model_mask if score_model >= score_cv else cv_mask


def normalize_backend(backend):
    key = str(backend or "auto").strip().lower()
    return key if key in MASK_BACKENDS else "auto"


def select_mask(bgr_img, backend="auto", alpha=None, onnx_fn=None, sam_fn=None):
    """
    Produce the silhouette mask for ``bgr_img``.

    Priority: an informative alpha channel wins outright; then the requested
    backend. ``onnx_fn`` / ``sam_fn`` are callables ``bgr -> mask|None``
    supplied by the caller (they own model loading and settings).
    In ``auto`` mode the ONNX model is preferred over SAM when both exist, and
    every model result is cross-checked against the OpenCV mask.
    """
    mask = alpha_to_mask(alpha)
    if mask is not None:
        return mask

    backend = normalize_backend(backend)
    if backend == "opencv":
        return get_mask_opencv(bgr_img)

    model_fn = None
    if backend in ("onnx", "auto") and onnx_fn is not None:
        model_fn = onnx_fn
    if model_fn is None and backend in ("sam", "auto") and sam_fn is not None:
        model_fn = sam_fn
    if model_fn is None:
        return get_mask_opencv(bgr_img)

    try:
        model_mask = model_fn(bgr_img)
    except Exception:
        model_mask = None
    cv_mask = get_mask_opencv(bgr_img)
    chosen = _choose_between(bgr_img, model_mask, cv_mask, strict=(backend != "auto"))
    return chosen if chosen is not None else cv_mask
