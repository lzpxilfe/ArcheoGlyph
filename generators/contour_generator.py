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

from .style_utils import STYLE_LINE, STYLE_MEASURED, STYLE_TYPOLOGY, normalize_style
from .style_control_utils import (
    STYLE_CONTROL_EXAGGERATION,
    STYLE_CONTROL_FACTUALITY,
    STYLE_CONTROL_SYMBOLIC_LOOSENESS,
    resolve_style_controls,
)


class ContourGenerator:
    """
    Generates SVG contours from images using OpenCV.
    """

    def __init__(self):
        self.settings = QSettings()
        self._sam_model = None
        self._sam_cache_key = None
        self._sam_hf_generator = None
        self._sam_hf_cache_key = None

    def _load_image(self, image_path):
        """Load image from path with cv2.imdecode."""
        with open(image_path, "rb") as stream:
            data = bytearray(stream.read())
        array = np.asarray(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

    def generate(
        self,
        image_path,
        style=None,
        color=None,
        symmetry=False,
        factuality=None,
        symbolic_looseness=None,
        exaggeration=None,
    ):
        """
        Generate contour SVG from image.

        :param image_path: path to source image
        :param style: style name
        :param color: optional fixed color (hex)
        :param symmetry: optional mirror symmetry
        :param factuality: 0..100, higher keeps measured/documentary detail
        :param symbolic_looseness: 0..100, higher simplifies toward symbolic output
        :param exaggeration: 0..100, higher strengthens stylization emphasis
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

        processing_img, _analysis_scale = self._adaptive_prescale(img)

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

        style_key = normalize_style(style)
        is_typology = style_key == STYLE_TYPOLOGY
        is_publication = style_key == STYLE_MEASURED
        is_line_drawing = style_key == STYLE_LINE
        is_mono = is_line_drawing or is_publication
        controls = resolve_style_controls(
            settings=self.settings,
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        factuality_v = controls[STYLE_CONTROL_FACTUALITY] / 100.0
        symbolic_v = controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS] / 100.0
        exaggeration_v = controls[STYLE_CONTROL_EXAGGERATION] / 100.0
        if is_publication:
            # Measured style should remain documentation-first even when user sliders are high.
            symbolic_v = min(symbolic_v, 0.45)
            exaggeration_v = min(exaggeration_v, 0.35)
        profile_count = int(round(self._clamp((0.8 + (2.6 * symbolic_v) + (1.2 * exaggeration_v) - (1.2 * factuality_v)), 0.0, 4.0)))
        terminal_count = int(round(self._clamp((0.2 + (2.0 * symbolic_v) + (1.4 * exaggeration_v) - (0.9 * factuality_v)), 0.0, 4.0)))
        texture_count = int(round(self._clamp((2.0 + (13.0 * factuality_v) - (8.0 * symbolic_v) - (5.0 * exaggeration_v)), 0.0, 18.0)))
        line_detail_count = int(round(self._clamp((1.0 + (9.0 * factuality_v) - (6.0 * symbolic_v) - (4.0 * exaggeration_v)), 0.0, 12.0)))

        main_contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(main_contour))
        contour_perimeter = float(cv2.arcLength(main_contour, True))
        contour_circularity = 0.0
        if contour_perimeter > 1e-6:
            contour_circularity = (4.0 * np.pi * contour_area) / (contour_perimeter * contour_perimeter)
        _, _, w_box, h_box = cv2.boundingRect(main_contour)
        aspect_balance = min(w_box, h_box) / max(1.0, float(max(w_box, h_box)))
        bbox_fill_ratio = contour_area / max(1.0, float(w_box * h_box))
        is_roundish = (
            contour_circularity >= 0.70 and
            aspect_balance >= 0.78 and
            bbox_fill_ratio <= 0.90
        )
        if is_roundish:
            hull = cv2.convexHull(main_contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = contour_area / max(1.0, hull_area)
            if solidity < 0.88:
                main_contour = hull
                contour_area = float(cv2.contourArea(main_contour))
                contour_perimeter = float(cv2.arcLength(main_contour, True))

        if is_typology:
            base_epsilon = 0.0026
        else:
            base_epsilon = 0.0014
        if is_roundish:
            base_epsilon *= 0.72
        epsilon_factor = base_epsilon + (0.0018 * symbolic_v) + (0.0012 * exaggeration_v) - (0.0009 * factuality_v)
        epsilon_factor = self._clamp(epsilon_factor, 0.0008, 0.0052)
        epsilon = epsilon_factor * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)

        svg_w = processing_bgr.shape[1]
        svg_h = processing_bgr.shape[0]
        svg_output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}">']

        path_data = ""
        if len(approx) > 2:
            if is_roundish and not symmetry:
                (cx_round, cy_round), r_round = cv2.minEnclosingCircle(main_contour)
                path_data = self._circle_path(cx_round, cy_round, r_round, steps=88)
            else:
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

        profile_lines = self._estimate_profile_bands(target_mask, max_lines=max(1, profile_count))
        round_lines = self._estimate_round_bands(
            target_mask,
            max_lines=max(0, min(2, profile_count + 1)),
        ) if is_roundish else []
        spine_lines = self._estimate_spine_line(target_mask)
        terminal_target = terminal_count if is_typology else 2
        terminal_lines = self._estimate_terminal_bars(
            target_mask,
            max_lines=terminal_target,
        )
        texture_lines = self._extract_internal_lines(processing_bgr, target_mask, main_contour)
        round_motif_limit = int(round(self._clamp(
            (2.0 + (8.0 * factuality_v) - (3.0 * symbolic_v) - (2.0 * exaggeration_v)),
            0.0,
            10.0,
        )))
        round_motif_select_limit = round_motif_limit
        if is_roundish and is_publication:
            # Round measured drawings (e.g. bronze mirrors) need richer motif capture.
            round_motif_select_limit = max(
                round_motif_limit,
                max(7, min(11, texture_count + 4)),
            )
        round_motif_lines = self._select_round_inner_motif_lines(
            texture_lines + self._extract_round_motif_lines(
                processing_bgr,
                target_mask,
                main_contour,
                max_lines=max(18, round_motif_select_limit * 3),
            ),
            target_mask,
            max_lines=round_motif_select_limit,
            prefer_outer=(is_roundish and is_publication),
        ) if is_roundish else []
        round_relief_lines = self._extract_round_relief_lines(
            processing_bgr,
            target_mask,
            main_contour,
            max_lines=max(10, round_motif_select_limit * 3),
        ) if (is_roundish and is_publication) else []
        round_relief_region_lines = self._extract_round_relief_region_lines(
            processing_bgr,
            target_mask,
            main_contour,
            max_lines=max(8, round_motif_select_limit * 2),
        ) if (is_roundish and is_publication) else []
        round_polar_motif_lines = self._extract_round_polar_motif_lines(
            processing_bgr,
            target_mask,
            main_contour,
            max_lines=max(8, round_motif_select_limit * 2),
        ) if (is_roundish and is_publication) else []
        round_center_motif_lines = self._extract_round_center_motif_lines(
            processing_bgr,
            target_mask,
            main_contour,
            max_lines=max(6, round_motif_select_limit),
        ) if (is_roundish and is_publication) else []

        if is_typology:
            if is_roundish:
                internal_lines = round_lines[:1]
                if round_motif_lines:
                    internal_lines += round_motif_lines[:max(2, min(5, round_motif_limit))]
                if terminal_count > 0:
                    internal_lines += terminal_lines[:1]
            else:
                internal_lines = profile_lines[:profile_count] + spine_lines[:1] + terminal_lines[:terminal_count]
        elif is_publication:
            if is_roundish:
                # For round artifacts, prefer motif lines over forced center spine.
                internal_lines = []
                motif_target = max(7, min(11, round_motif_select_limit + 1))
                prefer_region = len(round_relief_region_lines) >= 4
                motif_lines = []
                candidate_pool = []
                if prefer_region:
                    candidate_pool = list(round_polar_motif_lines)
                    candidate_pool += list(round_relief_region_lines)
                    candidate_pool += list(round_center_motif_lines)
                    candidate_pool += list(round_motif_lines[:max(2, motif_target // 4)])
                else:
                    candidate_pool = (
                        list(round_polar_motif_lines)
                        + list(round_center_motif_lines)
                        + list(round_motif_lines)
                        + list(round_relief_lines)
                        + list(round_relief_region_lines)
                    )
                if candidate_pool:
                    motif_lines = self._select_round_inner_motif_lines(
                        candidate_pool,
                        target_mask,
                        max_lines=max(round_motif_select_limit + 2, 8),
                        prefer_outer=True,
                    )
                if len(motif_lines) < 2 and candidate_pool:
                    motif_lines = candidate_pool
                if motif_lines:
                    internal_lines += motif_lines[:max(4, motif_target // 2)]
                # Always backfill with region/relief candidates to meet motif density target.
                if round_polar_motif_lines:
                    internal_lines = self._merge_distinct_lines(
                        internal_lines,
                        round_polar_motif_lines,
                        min_center_sep=2.8,
                        max_lines=motif_target,
                        min_arc_len=6.0,
                    )
                if round_relief_region_lines:
                    internal_lines = self._merge_distinct_lines(
                        internal_lines,
                        round_relief_region_lines,
                        min_center_sep=3.2,
                        max_lines=motif_target,
                        min_arc_len=8.0,
                    )
                if (not prefer_region) and round_relief_lines:
                    internal_lines = self._merge_distinct_lines(
                        internal_lines,
                        round_relief_lines,
                        min_center_sep=3.2,
                        max_lines=motif_target,
                        min_arc_len=8.0,
                    )
                if round_motif_lines:
                    internal_lines = self._merge_distinct_lines(
                        internal_lines,
                        round_motif_lines,
                        min_center_sep=3.0,
                        max_lines=motif_target,
                        min_arc_len=7.0,
                    )
                internal_lines = self._regularize_round_publication_lines(
                    internal_lines,
                    target_mask,
                    max_lines=motif_target,
                )
                internal_lines = self._suppress_round_ring_lines(
                    internal_lines,
                    target_mask,
                    max_ring_lines=0,
                )
                internal_lines = self._augment_round_rotational_symmetry(
                    internal_lines,
                    target_mask,
                    desired_lines=max(5, motif_target - 1),
                )
                ys_round, xs_round = np.where(target_mask > 0)
                if len(xs_round) > 50:
                    cx_round = float(np.mean(xs_round))
                    cy_round = float(np.mean(ys_round))
                    angular_cov = self._round_line_angular_coverage(internal_lines, cx_round, cy_round, bins=12)
                    ring_ratio = self._round_ring_line_ratio(internal_lines, target_mask)
                else:
                    angular_cov = 1.0
                    ring_ratio = 0.0
                if len(internal_lines) < 4 or angular_cov < 0.34 or ring_ratio > 0.58:
                    angular_markers = self._estimate_round_angular_motif_markers(
                        processing_bgr,
                        target_mask,
                        max_lines=max(8, motif_target),
                    )
                    internal_lines = self._merge_distinct_lines(
                        internal_lines,
                        angular_markers,
                        min_center_sep=2.8,
                        max_lines=motif_target,
                        min_arc_len=6.0,
                    )
                    internal_lines = self._merge_distinct_lines(
                        internal_lines,
                        round_polar_motif_lines,
                        min_center_sep=2.8,
                        max_lines=motif_target,
                        min_arc_len=6.0,
                    )
                    internal_lines = self._regularize_round_publication_lines(
                        internal_lines,
                        target_mask,
                        max_lines=motif_target,
                    )
                    internal_lines = self._suppress_round_ring_lines(
                        internal_lines,
                        target_mask,
                        max_ring_lines=1,
                    )
                if round_lines and len(internal_lines) < 5:
                    anchor = round_lines[1] if len(round_lines) > 1 else round_lines[0]
                    internal_lines = [anchor] + internal_lines
                    internal_lines = internal_lines[:max(4, motif_target)]
                # Keep one circular band only as fallback when motif capture is weak.
                if len(internal_lines) < 2 and round_lines:
                    internal_lines += round_lines[:1]
            else:
                # Publication mode keeps factual texture hints plus structural cues.
                publication_profile = max(0, min(2, profile_count))
                internal_lines = texture_lines[:texture_count] + profile_lines[:publication_profile] + spine_lines[:1]
        elif is_line_drawing:
            if is_roundish:
                # Round line-drawing should not inject a vertical center seam.
                if round_motif_lines:
                    internal_lines = round_motif_lines[:max(2, line_detail_count + 1)]
                else:
                    line_lines = self._remove_near_horizontal_lines(texture_lines[:max(10, line_detail_count * 2)])
                    internal_lines = line_lines[:max(1, line_detail_count)] if line_detail_count > 0 else []
                if not internal_lines and round_lines:
                    internal_lines = round_lines[:1]
            else:
                # Line mode removes horizontal bars and keeps only vertical/diagonal factual cues.
                line_lines = self._remove_near_horizontal_lines(texture_lines[:max(6, line_detail_count)] + spine_lines[:1])
                internal_lines = line_lines[:max(1, line_detail_count)] if line_detail_count > 0 else []
        else:
            # Colored mode: symbolic structural lines only (avoid painterly/noisy interiors).
            if is_roundish:
                # Circular artifacts (e.g. coins) should avoid forced vertical spine lines.
                internal_lines = round_lines[:1]
                if round_motif_lines:
                    internal_lines += round_motif_lines[:round_motif_limit]
                elif factuality_v >= 0.72 and texture_count > 0 and not internal_lines:
                    internal_lines += self._remove_near_horizontal_lines(texture_lines)[:1]
            else:
                internal_lines = profile_lines[:max(1, profile_count)] + spine_lines[:1]
                if factuality_v >= 0.7 and symbolic_v <= 0.4 and texture_count > 0:
                    internal_lines += self._remove_near_horizontal_lines(texture_lines)[:2]

        if is_typology:
            base_color = self._muted_hex(final_color, keep=0.66)
            outline_color = self._darken_hex(base_color, 0.50)
            structure_color = self._darken_hex(base_color, 0.64)
            shade_color = self._darken_hex(base_color, 0.78)
            highlight_color = self._lighten_hex(base_color, 0.18)

            svg_output.append(
                f'<path d="{path_data}" fill="{base_color}" fill-opacity="1.0" stroke="{outline_color}" '
                'stroke-width="2.35" stroke-linecap="round" stroke-linejoin="round"/>'
            )

            for line in profile_lines[:3]:
                line_path = self._polyline_to_path(line)
                if not line_path:
                    continue
                svg_output.append(
                    f'<path d="{line_path}" fill="none" stroke="{shade_color}" stroke-opacity="0.44" '
                    'stroke-width="3.0" stroke-linecap="round" stroke-linejoin="round"/>'
                )
                svg_output.append(
                    f'<path d="{line_path}" fill="none" stroke="{structure_color}" stroke-opacity="0.88" '
                    'stroke-width="1.10" stroke-linecap="round" stroke-linejoin="round"/>'
                )

            for line in spine_lines[:1]:
                line_path = self._polyline_to_path(line)
                if not line_path:
                    continue
                svg_output.append(
                    f'<path d="{line_path}" fill="none" stroke="{highlight_color}" stroke-opacity="0.42" '
                    'stroke-width="1.80" stroke-linecap="round" stroke-linejoin="round"/>'
                )
                svg_output.append(
                    f'<path d="{line_path}" fill="none" stroke="{structure_color}" stroke-opacity="0.85" '
                    'stroke-width="1.00" stroke-linecap="round" stroke-linejoin="round"/>'
                )

            for line in terminal_lines[:terminal_count]:
                line_path = self._polyline_to_path(line)
                if not line_path:
                    continue
                svg_output.append(
                    f'<path d="{line_path}" fill="none" stroke="{structure_color}" stroke-opacity="0.90" '
                    'stroke-width="1.20" stroke-linecap="round" stroke-linejoin="round"/>'
                )
        elif is_mono:
            if is_publication:
                outline_width = "1.8"
                detail_width = "1.35" if is_roundish else "1.0"
                detail_dash = "" if is_roundish else ' stroke-dasharray="1.2 2.2"'
                detail_opacity = "0.94" if is_roundish else "0.7"
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

    def _clamp(self, value, lower, upper):
        """Clamp numeric value into [lower, upper]."""
        return max(lower, min(upper, value))

    def _adaptive_prescale(self, img):
        """
        Resize input image for contour analysis.
        - Downscale very large inputs for speed/stability.
        - Upscale low-resolution inputs to recover edge/motif geometry.
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
        # Low-resolution catalog/screenshot inputs need analysis upscaling.
        elif max_side < 720.0:
            scale = min(3.0, 960.0 / max_side)
        elif min_side < 420.0:
            scale = min(2.2, 640.0 / min_side)

        if abs(scale - 1.0) < 0.05:
            return img, 1.0

        new_w = max(2, int(round(float(w) * scale)))
        new_h = max(2, int(round(float(h) * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        return resized, float(scale)

    def _remove_near_horizontal_lines(self, lines, ratio=0.35):
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

    def _merge_distinct_lines(
        self,
        base_lines,
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
            center, _ = self._line_centroid_and_length(line)
            if center is not None:
                centers.append(center)
        if len(out) >= target:
            return out[:target]

        min_sep = max(1.0, float(min_center_sep))
        min_arc = max(0.0, float(min_arc_len))
        for line in extra_lines or []:
            if len(out) >= target:
                break
            center, arc_len = self._line_centroid_and_length(line)
            if center is None or arc_len < min_arc:
                continue
            if any((((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5) < min_sep for c in centers):
                continue
            out.append(line)
            centers.append(center)
        return out[:target]

    def _regularize_round_publication_lines(self, lines, mask, max_lines=14):
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
            ring_like = self._line_ring_likeness(out_line, cx_ref, cy_ref)
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

    def _round_line_angular_coverage(self, lines, cx, cy, bins=12):
        """Return angular bin coverage ratio for round motif lines."""
        if not lines:
            return 0.0
        bin_count = int(max(4, int(bins)))
        used = set()
        for line in lines:
            center, arc_len = self._line_centroid_and_length(line)
            if center is None or arc_len < 4.0:
                continue
            theta = float(np.arctan2(center[1] - cy, center[0] - cx))
            b = int(((theta + np.pi) / (2.0 * np.pi)) * bin_count) % bin_count
            used.add(b)
        return float(len(used)) / float(bin_count)

    def _rotate_line_about_center(self, line, cx, cy, angle_rad):
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

    def _augment_round_rotational_symmetry(self, lines, mask, desired_lines=12):
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

        coverage = self._round_line_angular_coverage(base_lines, cx, cy, bins=12)
        if coverage >= 0.58 and len(base_lines) >= min(target, 8):
            return base_lines[:target]

        # Sort by length (longer motif lines are more stable for rotational augmentation).
        line_items = []
        for line in base_lines:
            center, arc_len = self._line_centroid_and_length(line)
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
            c0, _ = self._line_centroid_and_length(line)
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
                rot = self._rotate_line_about_center(line, cx, cy, ang)
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

                center, arc_len = self._line_centroid_and_length(arr.tolist())
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

    def _round_ring_line_ratio(self, lines, mask):
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
            _, arc_len = self._line_centroid_and_length(line)
            if arc_len < 6.0:
                continue
            valid_count += 1
            ring_like = self._line_ring_likeness(line, cx, cy)
            angle_span = self._line_angle_span(line, cx, cy)
            if ring_like >= 0.90 and angle_span >= (np.pi * 0.70):
                ring_count += 1
        if valid_count <= 0:
            return 0.0
        return float(ring_count) / float(valid_count)

    def _suppress_round_ring_lines(self, lines, mask, max_ring_lines=1):
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
            center, arc_len = self._line_centroid_and_length(line)
            if center is None or arc_len < 4.0:
                continue
            d_norm = (((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
            ring_like = self._line_ring_likeness(line, cx, cy)
            angle_span = self._line_angle_span(line, cx, cy)
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

    def _estimate_round_angular_motif_markers(self, bgr_img, mask, max_lines=12):
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
        boss_r = self._estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
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
            ring_like = self._line_ring_likeness(line, cx, cy)
            angle_span = self._line_angle_span(line, cx, cy)
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

    def _polyline_to_path(self, points):
        """Convert list of points to SVG polyline path."""
        if not points or len(points) < 2:
            return ""
        start = points[0]
        path = f"M {int(start[0])},{int(start[1])} "
        for pt in points[1:]:
            path += f"L {int(pt[0])},{int(pt[1])} "
        return path.strip()

    def _circle_path(self, cx, cy, radius, steps=72):
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
        return [item[1] for item in line_items[:72]]

    def _extract_round_motif_lines(self, bgr_img, mask, main_contour, max_lines=24):
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

            ring_like = self._line_ring_likeness(pts.tolist(), cx, cy)
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

    def _extract_round_center_motif_lines(self, bgr_img, mask, main_contour, max_lines=12):
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
            ring_like = self._line_ring_likeness(line, cx, cy)
            if ring_like >= 0.96 and d_norm > 0.28:
                continue

            band_score = max(0.20, 1.0 - abs(d_norm - 0.40))
            score = arc_len * (1.0 - (0.32 * ring_like)) * band_score
            candidates.append((score, line))

        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in candidates[:limit]]

    def _estimate_round_boss_radius(self, gray_img, mask, cx, cy, r_ref):
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
        except Exception:
            return 0.0

    def _extract_round_relief_lines(self, bgr_img, mask, main_contour, max_lines=24):
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
        boss_r = self._estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
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

            ring_like = self._line_ring_likeness(arr.tolist(), cx, cy)
            ang_span = self._line_angle_span(arr.tolist(), cx, cy)
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

    def _extract_round_relief_region_lines(self, bgr_img, mask, main_contour, max_lines=18):
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
        boss_r = self._estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
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
            ring_like = self._line_ring_likeness(line, cx, cy)
            ang_span = self._line_angle_span(line, cx, cy)
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

    def _extract_round_polar_motif_lines(self, bgr_img, mask, main_contour, max_lines=16):
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

        boss_r = self._estimate_round_boss_radius(gray, mask, cx, cy, r_ref)
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
            polar = cv2.warpPolar(
                enhanced,
                (n_theta, n_rad),
                (cx, cy),
                polar_max,
                cv2.WARP_POLAR_LINEAR,
            )
        except Exception:
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
        blur = cv2.GaussianBlur(polar_crop, (0, 0), 3.2, 1.2)
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

            center, cart_arc = self._line_centroid_and_length(cart_pts)
            if center is None or cart_arc < 8.0:
                continue
            d_norm = (((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5) / max(1e-6, r_ref)
            if d_norm < 0.18 or d_norm > 0.95:
                continue

            ring_like = self._line_ring_likeness(cart_pts, cx, cy)
            ang_span = self._line_angle_span(cart_pts, cx, cy)
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

    def _estimate_spine_line(self, mask):
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

    def _smooth_1d(self, values, window=9):
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

    def _estimate_profile_bands(self, mask, max_lines=3):
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

        smooth_w = self._smooth_1d(widths, window=max(7, h // 18))
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

    def _estimate_round_bands(self, mask, max_lines=2):
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

    def _line_centroid_and_length(self, line):
        """Return centroid (x,y) and arc-length of a polyline."""
        if not line or len(line) < 2:
            return None, 0.0
        arr = np.asarray(line, dtype=np.float32)
        cx = float(np.mean(arr[:, 0]))
        cy = float(np.mean(arr[:, 1]))
        seg = np.diff(arr, axis=0)
        arc_len = float(np.sum(np.sqrt(np.sum(seg * seg, axis=1))))
        return (cx, cy), arc_len

    def _line_ring_likeness(self, line, cx, cy):
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

    def _line_angle_span(self, line, cx, cy):
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

    def _select_round_inner_motif_lines(self, lines, mask, max_lines=4, prefer_outer=False):
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
            center, arc_len = self._line_centroid_and_length(line)
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

            ring_like = self._line_ring_likeness(line, cx, cy)
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
                center, arc_len = self._line_centroid_and_length(line)
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
                ring_like = self._line_ring_likeness(line, cx, cy)
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

    def _estimate_terminal_bars(self, mask, max_lines=2):
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

    def _lighten_hex(self, hex_color, amount):
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

    def _muted_hex(self, hex_color, keep=0.70):
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
        backend = str(self.settings.value('ArcheoGlyph/mask_backend', 'auto')).strip().lower()
        if backend == "sam":
            sam_mask = self._get_mask_sam(bgr_img)
            cv_mask = self._get_mask_opencv(bgr_img)

            if sam_mask is None:
                return cv_mask
            if cv_mask is None:
                sam_score = self._mask_selection_score(bgr_img, sam_mask)
                return sam_mask if sam_score >= 0.22 else self._get_mask_opencv(bgr_img)

            score_sam = self._mask_selection_score(bgr_img, sam_mask)
            score_cv = self._mask_selection_score(bgr_img, cv_mask)
            feat_sam = self._mask_bbox_features(sam_mask)
            feat_cv = self._mask_bbox_features(cv_mask)

            # Guard: SAM occasionally picks tiny centered fragments on reflective round artifacts.
            tiny_sam = (
                feat_sam["area_ratio"] < 0.018
                and feat_cv["area_ratio"] >= (feat_sam["area_ratio"] * 2.2)
            )

            circle_pref_cv = False
            try:
                gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                circle_mask = self._detect_center_circle_mask(gray)
                if circle_mask is not None:
                    circle_area = float(np.count_nonzero(circle_mask))
                    if circle_area >= float(max(1, bgr_img.shape[0] * bgr_img.shape[1])) * 0.04:
                        overlap_sam = float(np.count_nonzero(cv2.bitwise_and(sam_mask, circle_mask))) / max(1.0, circle_area)
                        overlap_cv = float(np.count_nonzero(cv2.bitwise_and(cv_mask, circle_mask))) / max(1.0, circle_area)
                        if overlap_sam < 0.18 and overlap_cv >= (overlap_sam + 0.16):
                            circle_pref_cv = True
            except Exception:
                circle_pref_cv = False

            if tiny_sam and score_cv >= (score_sam - 0.06):
                return cv_mask
            if circle_pref_cv and score_cv >= (score_sam - 0.08):
                return cv_mask
            if score_sam >= (score_cv + 0.08):
                return sam_mask
            if score_cv >= (score_sam + 0.03):
                return cv_mask
            return sam_mask if score_sam >= score_cv else cv_mask

        if backend == "auto":
            cv_mask = self._get_mask_opencv(bgr_img)
            sam_mask = self._get_mask_sam(bgr_img)

            if sam_mask is None:
                return cv_mask
            if cv_mask is None:
                return sam_mask

            score_cv = self._mask_selection_score(bgr_img, cv_mask)
            score_sam = self._mask_selection_score(bgr_img, sam_mask)
            if score_sam >= (score_cv + 0.08):
                return sam_mask
            if score_cv >= (score_sam + 0.04):
                return cv_mask
            return sam_mask if score_sam >= score_cv else cv_mask

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
        circle_mask = self._detect_center_circle_mask(gray)
        if float(np.mean(border_gray)) >= 180.0 and float(np.std(border_gray)) <= 36.0:
            white_fg = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY_INV)[1]
            white_fg = cv2.morphologyEx(white_fg, cv2.MORPH_OPEN, kernel3, iterations=1)
            white_fg = cv2.morphologyEx(white_fg, cv2.MORPH_CLOSE, kernel5, iterations=2)
            white_fg = self._select_primary_component(white_fg)
            if np.count_nonzero(white_fg) >= int(h * w * 0.01):
                combined = cv2.bitwise_or(target_mask, white_fg)
                target_mask = self._select_primary_component(combined)

        if circle_mask is not None and np.count_nonzero(circle_mask) >= int(h * w * 0.04):
            metrics = self._mask_shape_metrics(target_mask)
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
                should_clamp = (
                    metrics["touches_border"] or
                    overlap < 0.58 or
                    area_ratio > 1.30
                )
                if should_clamp:
                    clamped = cv2.bitwise_and(cv2.bitwise_or(target_mask, circle_mask), circle_mask)
                    if np.count_nonzero(clamped) >= int(circle_area * 0.42):
                        target_mask = clamped

        refined = self._refine_with_grabcut(blurred, target_mask)
        if refined is not None:
            target_mask = refined

        target_mask = self._select_primary_component(target_mask)
        # If selected mask still looks like a border-attached background chunk,
        # retry with a center-rectangle GrabCut pass.
        fg_ratio = float(np.count_nonzero(target_mask)) / float(max(1, h * w))
        if self._mask_touches_border(target_mask) and fg_ratio > 0.45:
            center_fallback = self._get_mask_center_grabcut(blurred)
            if center_fallback is not None:
                target_mask = center_fallback

        # Fallback: when current mask is blob-like, try recovering a tall/slender object
        # directly from image intensity (useful for daggers/spears on bright background).
        slender_candidate = self._recover_tall_component_from_image(blurred)
        if slender_candidate is not None:
            current_features = self._mask_bbox_features(target_mask)
            candidate_features = self._mask_bbox_features(slender_candidate)
            score_current = self._mask_selection_score(blurred, target_mask)
            score_candidate = self._mask_selection_score(blurred, slender_candidate)

            choose_candidate = False
            if candidate_features["tall_ratio"] >= 2.8 and current_features["tall_ratio"] < 2.1:
                choose_candidate = True
            if candidate_features["tall_ratio"] >= (current_features["tall_ratio"] * 1.35) and score_candidate >= (score_current - 0.04):
                choose_candidate = True
            if score_candidate >= (score_current + 0.08):
                choose_candidate = True

            if choose_candidate:
                target_mask = slender_candidate

        target_mask = self._smooth_mask_edges(target_mask)
        return target_mask

    def _select_primary_component(self, mask):
        """Keep the best foreground component by size + center + compactness score."""
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return mask

        # Recover tall/slender artifacts (e.g. daggers) when large blobs dominate.
        slender = self._recover_slender_component(mask)
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

    def _recover_slender_component(self, mask):
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
            out = self._smooth_mask_edges(out)

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

    def _mask_shape_metrics(self, mask):
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

    def _mask_bbox_features(self, mask):
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

    def _recover_tall_component_from_image(self, bgr_img):
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
            out = self._smooth_mask_edges(out)
            return self._select_primary_component(out)
        except Exception:
            return None

    def _mask_selection_score(self, bgr_img, mask):
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

        metrics = self._mask_shape_metrics(mask)

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

    def _detect_center_circle_mask(self, gray_img):
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

    def _mask_touches_border(self, mask, border=2):
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

    def _get_mask_center_grabcut(self, bgr_img):
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

            gc_mask = np.zeros((h, w), dtype=np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                bgr_img,
                gc_mask,
                (x, y, rw, rh),
                bgd_model,
                fgd_model,
                4,
                cv2.GC_INIT_WITH_RECT,
            )

            fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
            if np.count_nonzero(fg) < 120:
                return None

            fg = self._select_primary_component(fg)
            if np.count_nonzero(fg) < 120:
                return None
            return self._smooth_mask_edges(fg)
        except Exception:
            return None

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
        Optional SAM backend.
        Supports:
        - SAM1 (segment-anything + local checkpoint)
        - SAM2.1/SAM3 (transformers mask-generation via HF model ID)
        """
        model_type = str(self.settings.value('ArcheoGlyph/sam_model_type', 'vit_b')).strip() or "vit_b"
        if model_type.lower().startswith("hf:"):
            model_id = model_type[3:].strip()
            if not model_id:
                model_id = "facebook/sam2.1-hiera-small"
            return self._get_mask_sam_hf(bgr_img, model_id)

        checkpoint = str(self.settings.value('ArcheoGlyph/sam_checkpoint_path', '')).strip()
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
            target_mask = self._select_primary_component(target_mask)
            target_mask = self._smooth_mask_edges(target_mask)
            return target_mask
        except Exception:
            return None

    def _get_mask_sam_hf(self, bgr_img, model_id):
        """
        HF/Transformers SAM2.1/SAM3 path.
        Expects model_id like 'facebook/sam2.1-hiera-small' or 'facebook/sam3-hiera-large'.
        """
        model_id = (model_id or "").strip()
        if not model_id:
            return None

        try:
            import torch
            from PIL import Image
            from transformers import pipeline
        except Exception:
            return None

        device = 0 if torch.cuda.is_available() else -1
        cache_key = (model_id, device)
        try:
            if self._sam_hf_generator is None or self._sam_hf_cache_key != cache_key:
                self._sam_hf_generator = pipeline(
                    task="mask-generation",
                    model=model_id,
                    device=device,
                )
                self._sam_hf_cache_key = cache_key
        except Exception:
            self._sam_hf_generator = None
            self._sam_hf_cache_key = None
            return None

        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            output = self._sam_hf_generator(image)
        except Exception:
            return None

        # Pipeline usually returns a list with one dict per image.
        payload = output
        if isinstance(output, list) and output:
            payload = output[0]
        if not isinstance(payload, dict):
            return None

        raw_masks = payload.get("masks", None)
        raw_scores = payload.get("scores", None)

        def _as_mask_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                if hasattr(value, "detach"):
                    arr_v = value.detach().cpu().numpy()
                else:
                    arr_v = np.asarray(value)
            except Exception:
                return []
            if arr_v is None:
                return []
            if arr_v.ndim == 2:
                return [arr_v]
            if arr_v.ndim >= 3:
                return [arr_v[i] for i in range(arr_v.shape[0])]
            return []

        def _as_score_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                out = []
                for sv in value:
                    try:
                        out.append(float(sv))
                    except Exception:
                        continue
                return out
            try:
                if hasattr(value, "detach"):
                    arr_s = value.detach().cpu().numpy()
                else:
                    arr_s = np.asarray(value)
            except Exception:
                return []
            if arr_s is None:
                return []
            arr_s = np.asarray(arr_s).reshape(-1)
            out = []
            for sv in arr_s:
                try:
                    out.append(float(sv))
                except Exception:
                    continue
            return out

        masks = _as_mask_list(raw_masks)
        scores = _as_score_list(raw_scores)
        if len(masks) == 0:
            return None

        h, w = bgr_img.shape[:2]
        total = float(max(1, h * w))
        circle_mask = None
        circle_area = 0.0
        try:
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            circle_mask = self._detect_center_circle_mask(gray)
            if circle_mask is not None:
                circle_area = float(np.count_nonzero(circle_mask))
                if circle_area < (total * 0.04):
                    circle_mask = None
                    circle_area = 0.0
        except Exception:
            circle_mask = None
            circle_area = 0.0

        best = None
        best_score = -1.0

        for idx, mask_item in enumerate(masks):
            arr = None
            try:
                if hasattr(mask_item, "detach"):
                    arr = mask_item.detach().cpu().numpy()
                else:
                    arr = np.asarray(mask_item)
            except Exception:
                continue

            if arr is None:
                continue
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            if arr.ndim != 2:
                continue

            if arr.shape[0] != h or arr.shape[1] != w:
                arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

            if arr.dtype == np.bool_:
                m = arr.astype(np.uint8) * 255
            else:
                thr = 0.5 if float(np.max(arr)) <= 1.0 else 127.0
                m = (arr > thr).astype(np.uint8) * 255

            if int(np.count_nonzero(m)) < max(80, int(h * w * 0.002)):
                continue

            m = self._select_primary_component(m)
            m = self._smooth_mask_edges(m)

            m_area = float(np.count_nonzero(m))
            if circle_mask is not None and circle_area > 0.0:
                overlap_circle = float(np.count_nonzero(cv2.bitwise_and(m, circle_mask))) / max(1.0, circle_area)
                # Reject tiny center fragments when a dominant round silhouette is detected.
                if m_area < (circle_area * 0.12) and overlap_circle < 0.20:
                    continue
            else:
                overlap_circle = 0.0

            quality = self._mask_selection_score(bgr_img, m)
            if idx < len(scores):
                try:
                    quality += (0.06 * float(scores[idx]))
                except Exception:
                    pass
            if circle_mask is not None and circle_area > 0.0:
                area_ratio_to_circle = m_area / max(1.0, circle_area)
                area_match = max(0.0, 1.0 - abs(area_ratio_to_circle - 1.0))
                quality += (0.14 * overlap_circle) + (0.08 * area_match)
            if quality > best_score:
                best_score = quality
                best = m

        return best

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
