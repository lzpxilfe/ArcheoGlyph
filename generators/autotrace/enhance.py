# -*- coding: utf-8 -*-
"""
Detail enhancement, adaptive Canny and generic internal-line extraction.

Auto-generated from the former ContourGenerator methods; QGIS-free.
"""

import cv2
import numpy as np

from ...log import log_exception

from .geometry import dedupe_lines


def estimate_masked_edge_density(bgr_img, mask):
    """Estimate edge density inside foreground mask."""
    try:
        if bgr_img is None or mask is None:
            return 0.0
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 48, 138)
        masked_edges = cv2.bitwise_and(edges, mask)
        fg_pixels = int(np.count_nonzero(mask))
        if fg_pixels <= 0:
            return 0.0
        return float(np.count_nonzero(masked_edges)) / float(fg_pixels)
    except Exception as e:
        log_exception("estimate_masked_edge_density", e)
        return 0.0


def prepare_detail_source(bgr_img, mask, boost=False):
    """
    Build a detail-enhanced image for internal line extraction.
    Applies denoise + local contrast + unsharp on low-quality inputs.
    """
    try:
        if bgr_img is None:
            return bgr_img
        if not boost:
            return bgr_img

        work = cv2.bilateralFilter(bgr_img, 7, 55, 55)
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))

        # Illumination flattening: suppress broad shading so relief edges survive.
        l_f = l_chan.astype(np.float32) + 1.0
        illum_sigma = max(8.0, float(min(work.shape[0], work.shape[1])) * 0.06)
        illum = cv2.GaussianBlur(l_f, (0, 0), illum_sigma)
        illum = np.maximum(illum, 1.0)
        l_flat = np.clip((l_f / illum) * 148.0, 0, 255).astype(np.uint8)
        l_eq = clahe.apply(l_flat)
        l_mix = cv2.addWeighted(l_eq, 0.72, l_chan, 0.28, 0)

        merged = cv2.merge((l_mix, a_chan, b_chan))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.05)
        sharp = cv2.addWeighted(enhanced, 1.42, blur, -0.42, 0)
        gray_sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        hp = cv2.addWeighted(
            gray_sharp,
            1.58,
            cv2.GaussianBlur(gray_sharp, (0, 0), 1.15),
            -0.58,
            0,
        )
        edge_map = adaptive_canny(hp, mask=mask, low_floor=16, high_cap=176)
        if edge_map is not None:
            edge_map = cv2.dilate(edge_map, np.ones((2, 2), np.uint8), iterations=1)
            edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
            sharp = cv2.addWeighted(sharp, 0.86, edge_rgb, 0.24, 0)

        if mask is None:
            return sharp

        out = bgr_img.copy()
        out[mask > 0] = sharp[mask > 0]
        return out
    except Exception as e:
        log_exception("prepare_detail_source", e)
        return bgr_img


def adaptive_canny(gray_img, mask=None, low_floor=12, high_cap=180):
    """Run Canny with adaptive thresholds from masked luminance distribution."""
    try:
        if gray_img is None:
            return None
        samples = gray_img
        if mask is not None:
            samples = gray_img[mask > 0]
            if samples is None or len(samples) < 20:
                samples = gray_img.reshape(-1)
        else:
            samples = gray_img.reshape(-1)

        med = float(np.median(samples))
        low = int(max(low_floor, min(high_cap - 6, 0.68 * med)))
        high = int(max(low + 6, min(high_cap, 1.36 * med)))
        return cv2.Canny(gray_img, low, high)
    except Exception as e:
        log_exception("adaptive_canny", e)
        return cv2.Canny(gray_img, int(low_floor), int(max(low_floor + 8, high_cap)))


def low_quality_variants(detail_bgr, base_bgr, mask):
    """Build additional enhanced views for low-quality line extraction."""
    variants = []
    for src in (detail_bgr, base_bgr):
        if src is None:
            continue
        try:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
            eq = clahe.apply(gray)

            hp = cv2.addWeighted(
                eq,
                1.70,
                cv2.GaussianBlur(eq, (0, 0), 1.25),
                -0.70,
                0,
            )
            edges = adaptive_canny(hp, mask=mask, low_floor=14, high_cap=170)
            if edges is not None:
                edges = cv2.morphologyEx(
                    edges,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                variants.append(cv2.addWeighted(src, 0.82, edge_rgb, 0.28, 0))

            variants.append(cv2.cvtColor(hp, cv2.COLOR_GRAY2BGR))
        except Exception as e:
            log_exception("low_quality_variants", e)
            continue
    return variants[:4]


def extract_internal_lines(bgr_img, mask, main_contour):
    """
    Extract internal feature lines inside artifact silhouette.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 36, 110)

    interior_mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, edges, mask=interior_mask)

    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, [main_contour], -1, 255, thickness=4)
    edges[boundary > 0] = 0

    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    line_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    min_dim = min(bgr_img.shape[0], bgr_img.shape[1])
    min_len = max(20, int(min_dim * 0.035))

    line_items = []
    for contour in line_contours:
        arc_len = cv2.arcLength(contour, False)
        if arc_len < min_len:
            continue

        epsilon = 0.003 * arc_len
        approx = cv2.approxPolyDP(contour, epsilon, False)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 2:
            continue

        center = np.mean(pts, axis=0).astype(int)
        if not (0 <= center[0] < mask.shape[1] and 0 <= center[1] < mask.shape[0]):
            continue
        if mask[center[1], center[0]] == 0:
            continue

        line_items.append((arc_len, pts.tolist()))

    line_items.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in line_items[:72]]


def extract_annular_relief_lines(bgr_img, target_mask, main_contour, max_lines=14):
    """
    Extract ring/annular motif strokes from round relief artifacts (e.g., mirrors).
    This targets concentric decoration zones that are often lost in low-res inputs.
    """
    try:
        if bgr_img is None or target_mask is None or main_contour is None:
            return []
        h, w = target_mask.shape[:2]
        if h < 24 or w < 24:
            return []

        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        if radius < 22.0:
            return []

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        g_f = gray.astype(np.float32) + 1.0
        illum = cv2.GaussianBlur(g_f, (0, 0), max(6.0, float(radius) * 0.16))
        flat = np.clip((g_f / np.maximum(illum, 1.0)) * 144.0, 0, 255).astype(np.uint8)

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
        annulus = np.zeros((h, w), dtype=np.uint8)
        annulus[(dist >= (0.24 * radius)) & (dist <= (0.93 * radius))] = 255
        annulus = cv2.bitwise_and(annulus, target_mask)
        if int(np.count_nonzero(annulus)) < 120:
            return []

        edges_a = adaptive_canny(flat, mask=annulus, low_floor=14, high_cap=170)
        sobel_x = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobel_x, sobel_y)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        _, edges_b = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edges = cv2.bitwise_or(edges_a, edges_b)
        edges = cv2.bitwise_and(edges, annulus)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []

        candidates = []
        max_collect = max(2, int(max_lines) * 2)
        min_arc = max(14.0, float(radius) * 0.12)
        radial_band_limit = float(radius) * 0.18

        for cnt in sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True):
            if len(candidates) >= max_collect:
                break
            if cnt is None or len(cnt) < 10:
                continue
            arc_len = float(cv2.arcLength(cnt, False))
            if arc_len < min_arc:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)
            radial = np.sqrt((pts[:, 0] - float(cx)) ** 2 + (pts[:, 1] - float(cy)) ** 2)
            if float(np.ptp(radial)) > radial_band_limit:
                continue

            eps = max(0.8, 0.008 * arc_len)
            simp = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, False)
            if simp is None or len(simp) < 4:
                continue
            line = [[int(round(p[0][0])), int(round(p[0][1]))] for p in simp]
            candidates.append(line)

        return dedupe_lines(candidates, min_points=4, max_lines=max_lines)
    except Exception as e:
        log_exception("extract_annular_relief_lines", e)
        return []
