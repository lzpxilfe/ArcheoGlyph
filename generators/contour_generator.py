# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Contour Generator (Auto Trace)
Extracts silhouette and factual internal feature lines from input images.
"""

import os

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None
from qgis.PyQt.QtCore import QSettings

from .style_utils import STYLE_LINE, STYLE_MEASURED, normalize_style


class ContourGenerator:
    """
    Generates SVG contours from images using OpenCV.
    """

    def __init__(self):
        self.settings = QSettings()
        self._sam_model = None
        self._sam_cache_key = None

    def _load_image(self, image_path):
        """Load image from path with cv2.imdecode."""
        with open(image_path, "rb") as stream:
            data = bytearray(stream.read())
        array = np.asarray(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

    def generate(self, image_path, style=None, color=None, symmetry=False):
        """
        Generate contour SVG from image.

        :param image_path: path to source image
        :param style: style name
        :param color: optional fixed color (hex)
        :param symmetry: optional mirror symmetry
        :return: SVG string
        """
        if cv2 is None or np is None:
            raise ImportError(
                "OpenCV and NumPy are required for Auto Trace. "
                "Please install via 'pip install opencv-python-headless numpy'."
            )

        img = self._load_image(image_path)
        if img is None:
            raise ValueError("Failed to load image.")

        original_h, original_w = img.shape[:2]
        max_dim = 1500

        processing_img = img
        if max(original_h, original_w) > max_dim:
            scale_factor = max_dim / max(original_h, original_w)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            processing_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(processing_img.shape) == 4:
            processing_bgr = cv2.cvtColor(processing_img, cv2.COLOR_BGRA2BGR)
        else:
            processing_bgr = processing_img

        target_mask = self._get_mask(processing_bgr)
        processing_bgr, target_mask = self._auto_upright(processing_bgr, target_mask)

        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'

        final_color = color if color else self._extract_dominant_color(processing_bgr, target_mask)

        main_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.0016 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)

        style_key = normalize_style(style)
        is_publication = style_key == STYLE_MEASURED
        is_line_drawing = style_key == STYLE_LINE
        is_mono = is_line_drawing or is_publication

        svg_w = processing_bgr.shape[1]
        svg_h = processing_bgr.shape[0]
        svg_output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}">']

        path_data = ""
        if len(approx) > 2:
            points = approx.reshape(-1, 2)
            final_points = points.tolist()

            if symmetry:
                top_pt = min(points, key=lambda p: p[1])
                bottom_pt = max(points, key=lambda p: p[1])
                axis_x = (top_pt[0] + bottom_pt[0]) / 2
                left_contour = [pt for pt in points if pt[0] < axis_x]

                if len(left_contour) >= 3:
                    left_sorted = sorted(left_contour, key=lambda p: p[1])
                    right_side = []
                    for pt in reversed(left_sorted):
                        reflected_x = int(axis_x + (axis_x - pt[0]))
                        right_side.append([reflected_x, int(pt[1])])
                    final_points = [[int(pt[0]), int(pt[1])] for pt in left_sorted] + right_side
                    final_points.append(final_points[0])

            if len(final_points) > 2:
                start = final_points[0]
                path_data = f"M {start[0]},{start[1]} "
                for pt in final_points[1:]:
                    path_data += f"L {pt[0]},{pt[1]} "
                path_data += "Z"

        internal_lines = self._extract_internal_lines(processing_bgr, target_mask, main_contour)
        internal_lines.extend(self._estimate_spine_line(target_mask))

        if is_mono:
            if is_publication:
                outline_width = "1.6"
                detail_width = "1.0"
                detail_dash = ' stroke-dasharray="1.2 2.2"'
                detail_opacity = "0.7"
            else:
                outline_width = "2.2"
                detail_width = "1.25"
                detail_dash = ""
                detail_opacity = "0.8"

            svg_output.append(
                f'<path d="{path_data}" fill="none" stroke="#000000" stroke-width="{outline_width}" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
            )
            for line in internal_lines:
                line_path = self._polyline_to_path(line)
                if line_path:
                    svg_output.append(
                        f'<path d="{line_path}" fill="none" stroke="#000000" stroke-opacity="{detail_opacity}" '
                        f'stroke-width="{detail_width}"{detail_dash} stroke-linecap="round" stroke-linejoin="round"/>'
                    )
        else:
            outline_color = self._darken_hex(final_color, 0.58)
            detail_color = self._darken_hex(final_color, 0.42)
            svg_output.append(
                f'<path d="{path_data}" fill="{final_color}" fill-opacity="1.0" stroke="{outline_color}" '
                'stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>'
            )
            for line in internal_lines:
                line_path = self._polyline_to_path(line)
                if line_path:
                    svg_output.append(
                        f'<path d="{line_path}" fill="none" stroke="{detail_color}" stroke-opacity="0.72" '
                        'stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round"/>'
                    )

        svg_output.append("</svg>")
        return "".join(svg_output)

    def _polyline_to_path(self, points):
        """Convert list of points to SVG polyline path."""
        if not points or len(points) < 2:
            return ""
        start = points[0]
        path = f"M {int(start[0])},{int(start[1])} "
        for pt in points[1:]:
            path += f"L {int(pt[0])},{int(pt[1])} "
        return path.strip()

    def _extract_internal_lines(self, bgr_img, mask, main_contour):
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
        return [item[1] for item in line_items[:18]]

    def _estimate_spine_line(self, mask):
        """Estimate central spine line from mask when texture lines are weak."""
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            return []

        top_y = int(np.min(ys))
        bot_y = int(np.max(ys))
        h = max(1, bot_y - top_y)
        step = max(2, h // 42)

        points = []
        for y in range(top_y, bot_y + 1, step):
            row_xs = np.where(mask[y] > 0)[0]
            if len(row_xs) < 2:
                continue
            x_mid = int((int(row_xs[0]) + int(row_xs[-1])) / 2)
            points.append([x_mid, y])

        if len(points) < 6:
            return []
        return [points]

    def _darken_hex(self, hex_color, factor):
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

    def get_silhouette_bytes(self, image_path):
        """
        Generate black/white silhouette PNG bytes (object=black, bg=white).
        """
        if cv2 is None or np is None:
            return None

        img = self._load_image(image_path)
        if img is None:
            return None

        original_h, original_w = img.shape[:2]
        max_dim = 1000

        processing_img = img
        if max(original_h, original_w) > max_dim:
            scale_factor = max_dim / max(original_h, original_w)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            processing_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(processing_img.shape) == 4:
            bgr = cv2.cvtColor(processing_img, cv2.COLOR_BGRA2BGR)
        else:
            bgr = processing_img

        target_mask = self._get_mask(bgr)

        out_img = np.ones((bgr.shape[0], bgr.shape[1], 3), dtype=np.uint8) * 255
        out_img[target_mask == 255] = [0, 0, 0]

        success, encoded_img = cv2.imencode('.png', out_img)
        if success:
            return encoded_img.tobytes()
        return None

    def _get_mask(self, bgr_img):
        """Internal helper: produce silhouette mask using selected backend."""
        backend = str(self.settings.value('ArcheoGlyph/mask_backend', 'opencv')).strip().lower()
        if backend == "sam":
            sam_mask = self._get_mask_sam(bgr_img)
            if sam_mask is not None:
                return sam_mask
        return self._get_mask_opencv(bgr_img)

    def _get_mask_opencv(self, bgr_img):
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

        refined = self._refine_with_grabcut(blurred, target_mask)
        if refined is not None:
            target_mask = refined

        target_mask = self._select_primary_component(target_mask)
        target_mask = self._smooth_mask_edges(target_mask)
        return target_mask

    def _select_primary_component(self, mask):
        """Keep the best foreground component by size + center + tallness score."""
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return mask

        cx_ref = w * 0.5
        cy_ref = h * 0.5
        best = None
        best_score = -1.0

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
            tall = ch / max(1.0, float(cw))
            tall_norm = min(1.0, tall / 2.2)
            score = area * (1.0 - min(0.95, d_norm)) * (0.6 + 0.4 * tall_norm)
            if score > best_score:
                best_score = score
                best = c

        if best is None:
            best = max(contours, key=cv2.contourArea)

        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [best], -1, 255, thickness=cv2.FILLED)
        return out

    def _refine_with_grabcut(self, bgr_img, init_mask):
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

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr_img, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

            fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)
            if np.count_nonzero(fg) < 120:
                return None
            return self._smooth_mask_edges(fg)
        except Exception:
            return None

    def _auto_upright(self, bgr_img, mask):
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
            rot_mask = self._select_primary_component(rot_mask)
            rot_mask = self._smooth_mask_edges(rot_mask)
            return rot_bgr, rot_mask
        except Exception:
            return bgr_img, mask

    def _smooth_mask_edges(self, mask):
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
        h, w = smoothed.shape[:2]
        flood = smoothed.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        return cv2.bitwise_or(smoothed, holes)

    def _get_mask_sam(self, bgr_img):
        """
        Optional SAM backend for users who installed segment-anything and checkpoint.
        Falls back to OpenCV if unavailable.
        """
        checkpoint = str(self.settings.value('ArcheoGlyph/sam_checkpoint_path', '')).strip()
        model_type = str(self.settings.value('ArcheoGlyph/sam_model_type', 'vit_b')).strip() or "vit_b"
        if not checkpoint:
            return None

        if not os.path.exists(checkpoint):
            return None

        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except Exception:
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = (checkpoint, model_type, device)

        try:
            if self._sam_model is None or self._sam_cache_key != cache_key:
                if model_type not in sam_model_registry:
                    return None
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device=device)
                self._sam_model = sam
                self._sam_cache_key = cache_key

            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            mask_generator = SamAutomaticMaskGenerator(
                self._sam_model,
                points_per_side=24,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                min_mask_region_area=400,
            )
            masks = mask_generator.generate(rgb_img)
            if not masks:
                return None

            h, w = bgr_img.shape[:2]
            cx, cy = w * 0.5, h * 0.5
            best_mask = None
            best_score = -1.0

            for item in masks:
                seg = item.get("segmentation")
                area = float(item.get("area", 0))
                if seg is None or area <= 0:
                    continue

                ys, xs = np.where(seg)
                if len(xs) == 0:
                    continue

                if area < (h * w * 0.015):
                    continue

                mx = float(np.mean(xs))
                my = float(np.mean(ys))
                center_dist = ((mx - cx) ** 2 + (my - cy) ** 2) / max(1.0, (w * w + h * h))

                bbox = item.get("bbox", [0, 0, w, h])
                bbox_area = max(1.0, float(bbox[2]) * float(bbox[3]))
                fill_ratio = area / bbox_area

                score = area * (1.0 - min(center_dist, 0.95)) * (0.65 + 0.35 * min(fill_ratio, 1.0))
                if score > best_score:
                    best_score = score
                    best_mask = seg

            if best_mask is None:
                return None

            target_mask = np.zeros((h, w), dtype=np.uint8)
            target_mask[best_mask] = 255

            kernel = np.ones((3, 3), np.uint8)
            target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            return target_mask
        except Exception:
            return None

    def _extract_dominant_color(self, bgr_img, mask=None):
        """
        Extract dominant color while prioritizing colorful pixels.
        """
        try:
            hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            valid_mask = np.ones_like(h, dtype=bool)
            if mask is not None:
                valid_mask = (mask > 0)

            color_mask = (s > 15) & (v > 40) & (v < 240) & valid_mask
            pixels = bgr_img[color_mask]

            if len(pixels) < 50:
                pixels = bgr_img[valid_mask]
            if len(pixels) < 10:
                return "#8B4513"

            pixels = np.float32(pixels)
            n_colors = 3 if len(pixels) >= 3 else 1

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
            _, _, centers = cv2.kmeans(
                pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            best_color = centers[0]
            max_sat = -1
            for center in centers:
                c_uint8 = np.uint8([[center]])
                c_hsv = cv2.cvtColor(c_uint8, cv2.COLOR_BGR2HSV)[0][0]
                if c_hsv[1] > max_sat:
                    max_sat = c_hsv[1]
                    best_color = center

            dom_color = best_color.astype(int)
            return "#{:02x}{:02x}{:02x}".format(dom_color[2], dom_color[1], dom_color[0])
        except Exception:
            return "#8B4513"
