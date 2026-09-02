# -*- coding: utf-8 -*-
"""
Auto Trace pipeline: image -> silhouette -> internal lines -> SVG (QGIS-free).

This is the former ContourGenerator.generate body with settings access
replaced by AutoTraceOptions and mask extraction delegated to the caller.
"""

import cv2
import numpy as np

from ..ink_centerline import extract_ink_polylines, looks_like_drawing, simplify_polyline
from ..style_control_utils import (
    STYLE_CONTROL_EXAGGERATION,
    STYLE_CONTROL_FACTUALITY,
    STYLE_CONTROL_SYMBOLIC_LOOSENESS,
    resolve_style_controls,
)
from ..style_utils import (
    STYLE_LINE,
    STYLE_MEASURED,
    STYLE_TYPOLOGY,
    is_legend_style,
    normalize_style,
)
from .colors import (
    blend_hex,
    darken_hex,
    extract_dominant_color,
    extract_material_palette,
    hex_luminance,
    lighten_hex,
    muted_hex,
)
from .enhance import (
    estimate_masked_edge_density,
    prepare_detail_source,
)
from .geometry import (
    circle_path,
    clamp,
    merge_distinct_lines,
    polyline_to_path,
    remove_near_horizontal_lines,
)
from .io import (
    adaptive_prescale,
)
from .lines import (
    extract_internal_lines_multisource,
)
from .round_motif import (
    augment_round_rotational_symmetry,
    build_round_structural_lines,
    estimate_round_angular_motif_markers,
    extract_round_center_fallback_lines,
    extract_round_center_motif_lines,
    extract_round_low_quality_lines,
    extract_round_mirror_signature_lines,
    extract_round_motif_lines,
    extract_round_polar_motif_lines,
    extract_round_relief_lines,
    extract_round_relief_region_lines,
    extract_round_unwrap_lines,
    needs_round_mirror_rescue,
    prefer_round_inner_lines,
    regularize_round_publication_lines,
    round_line_angular_coverage,
    round_line_center_coverage,
    round_line_inner_count,
    round_ring_line_ratio,
    select_round_inner_motif_lines,
    suppress_round_ring_lines,
)
from .segment import (
    auto_upright,
)
from .structure import (
    estimate_profile_bands,
    estimate_round_bands,
    estimate_spine_line,
    estimate_terminal_bars,
)


def run_autotrace(bgr, options, mask_provider):
    """
    Full Auto Trace pipeline on an 8-bit BGR image.

    :param bgr: source image (EXIF-corrected, no alpha)
    :param options: AutoTraceOptions
    :param mask_provider: callable ``processing_bgr -> uint8 mask`` (the
        caller owns backends, caching and alpha handling)
    :return: SVG string in analysis-pixel coordinates
    """
    options = options.normalized()
    style = options.style
    color = options.color
    symmetry = bool(options.symmetry)
    detail_mode_key = options.detail_mode
    round_strategy_key = options.round_strategy
    detail_fast = detail_mode_key == "fast"
    # Image-first mode prioritizes responsiveness.
    if round_strategy_key == "image_first":
        detail_fast = True
    synthetic = bool(options.synthetic_structure)

    processing_bgr, _analysis_scale = adaptive_prescale(
        bgr,
        force_lowres_upscale=bool(options.force_lowres_upscale),
        detail_fast=detail_fast,
    )
    target_mask = mask_provider(processing_bgr)
    if target_mask is None:
        target_mask = np.zeros(processing_bgr.shape[:2], dtype=np.uint8)
    processing_bgr, target_mask = auto_upright(processing_bgr, target_mask)
    edge_density = estimate_masked_edge_density(processing_bgr, target_mask)
    if detail_fast:
        low_quality_input = (
            min(processing_bgr.shape[0], processing_bgr.shape[1]) < 520
            or edge_density < 0.028
        )
    else:
        low_quality_input = (
            min(processing_bgr.shape[0], processing_bgr.shape[1]) < 560
            or edge_density < 0.031
        )
    detail_bgr = prepare_detail_source(
        processing_bgr,
        target_mask,
        boost=low_quality_input,
    )

    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'

    final_color = color if color else extract_dominant_color(processing_bgr, target_mask)
    material_palette = []
    if not color:
        material_palette = extract_material_palette(
            processing_bgr,
            target_mask,
            max_colors=4,
        )

    style_key = normalize_style(style)
    legend_mode = is_legend_style(style)
    is_typology = style_key == STYLE_TYPOLOGY and (not legend_mode)
    is_publication = style_key == STYLE_MEASURED
    is_line_drawing = style_key == STYLE_LINE
    is_mono = is_line_drawing or is_publication
    controls = resolve_style_controls(
        settings=None,
        factuality=options.factuality,
        symbolic_looseness=options.symbolic_looseness,
        exaggeration=options.exaggeration,
    )
    factuality_v = controls[STYLE_CONTROL_FACTUALITY] / 100.0
    symbolic_v = controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS] / 100.0
    exaggeration_v = controls[STYLE_CONTROL_EXAGGERATION] / 100.0
    if legend_mode:
        # Simple-symbol output should stay stable, simple, and map-readable.
        factuality_v = max(factuality_v, 0.78)
        symbolic_v = min(symbolic_v, 0.30)
        exaggeration_v = min(exaggeration_v, 0.22)
    if is_publication:
        # Measured style should remain documentation-first even when user sliders are high.
        symbolic_v = min(symbolic_v, 0.45)
        exaggeration_v = min(exaggeration_v, 0.35)
    profile_count = int(round(clamp((0.8 + (2.6 * symbolic_v) + (1.2 * exaggeration_v) - (1.2 * factuality_v)), 0.0, 4.0)))
    terminal_count = int(round(clamp((0.2 + (2.0 * symbolic_v) + (1.4 * exaggeration_v) - (0.9 * factuality_v)), 0.0, 4.0)))
    texture_count = int(round(clamp((2.0 + (13.0 * factuality_v) - (8.0 * symbolic_v) - (5.0 * exaggeration_v)), 0.0, 18.0)))
    line_detail_count = int(round(clamp((1.0 + (9.0 * factuality_v) - (6.0 * symbolic_v) - (4.0 * exaggeration_v)), 0.0, 12.0)))
    if legend_mode:
        profile_count = 1
        terminal_count = 1
        texture_count = 0
        line_detail_count = 1

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
    if legend_mode and is_roundish:
        profile_count = 0
        terminal_count = 0
    solidity = 1.0
    if is_roundish:
        hull = cv2.convexHull(main_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / max(1.0, hull_area)
    # Replace the traced outline by a perfect circle only when the object
    # really is one; ovals, chipped coins and rings keep their true contour.
    use_circle_outline = bool(is_roundish and contour_circularity >= 0.90 and solidity >= 0.95)
    # Schematic template lines for round artifacts are opt-in.
    fast_round_structural = bool(
        synthetic and
        is_roundish and
        is_publication and
        low_quality_input and
        factuality_v >= 0.72 and
        symbolic_v <= 0.48
    )

    if is_typology:
        base_epsilon = 0.0026
    else:
        base_epsilon = 0.0014
    if is_roundish:
        base_epsilon *= 0.72
    epsilon_factor = base_epsilon + (0.0018 * symbolic_v) + (0.0012 * exaggeration_v) - (0.0009 * factuality_v)
    if legend_mode:
        epsilon_factor += 0.0011
        epsilon_factor = clamp(epsilon_factor, 0.0012, 0.0068)
    else:
        epsilon_factor = clamp(epsilon_factor, 0.0008, 0.0052)
    epsilon = epsilon_factor * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)

    svg_w = processing_bgr.shape[1]
    svg_h = processing_bgr.shape[0]
    svg_output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}">']

    path_data = ""
    if len(approx) > 2:
        if use_circle_outline and not symmetry:
            (cx_round, cy_round), r_round = cv2.minEnclosingCircle(main_contour)
            path_data = circle_path(cx_round, cy_round, r_round, steps=88)
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

    profile_lines = estimate_profile_bands(target_mask, max_lines=max(1, profile_count))
    round_lines = estimate_round_bands(
        target_mask,
        max_lines=max(0, min(2, profile_count + 1)),
    ) if is_roundish else []
    spine_lines = estimate_spine_line(target_mask)
    terminal_target = terminal_count if is_typology else 2
    terminal_lines = estimate_terminal_bars(
        target_mask,
        max_lines=terminal_target,
    )
    if not synthetic:
        # Factual default: no invented profile bands, spine, terminal bars or rings.
        profile_lines = []
        round_lines = []
        spine_lines = []
        terminal_lines = []
    # ---- Input kind: line drawing / rubbing vs. photograph -------------------
    is_drawing = False
    if options.input_kind == "drawing":
        is_drawing = True
    elif options.input_kind == "auto":
        try:
            is_drawing, _drawing_metrics = looks_like_drawing(processing_bgr, target_mask)
        except Exception:
            is_drawing = False
    ink_lines = []
    if is_drawing or is_mono:
        try:
            erode_px = max(2, int(round(0.015 * min(target_mask.shape[:2]))))
            ink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
            ink_mask = cv2.erode(target_mask, ink_kernel)
            ink_source = processing_bgr if is_drawing else detail_bgr
            ink_lines = [
                [[int(x), int(y)] for x, y in simplify_polyline(pline, epsilon=1.2)]
                for pline in extract_ink_polylines(
                    ink_source,
                    mask=ink_mask,
                    min_arc_length=max(6.0, 0.02 * float(min(target_mask.shape[:2]))),
                )
            ]
        except Exception:
            ink_lines = []
    skip_round_motifs = is_drawing

    texture_lines = [] if (fast_round_structural or legend_mode or is_drawing) else extract_internal_lines_multisource(
        detail_bgr=detail_bgr,
        base_bgr=processing_bgr,
        target_mask=target_mask,
        main_contour=main_contour,
        low_quality=low_quality_input,
        is_roundish=is_roundish,
        detail_mode=detail_mode_key,
    )
    if is_drawing:
        texture_lines = list(ink_lines)
    elif is_mono and len(ink_lines) >= 3:
        # Photographs: true stroke centrelines beat double-edged Canny contours.
        texture_lines = list(ink_lines)
    round_motif_limit = int(round(clamp(
        (2.0 + (8.0 * factuality_v) - (3.0 * symbolic_v) - (2.0 * exaggeration_v)),
        0.0,
        10.0,
    )))
    round_motif_select_limit = round_motif_limit
    if is_roundish and low_quality_input:
        round_motif_limit = max(round_motif_limit, 14)
        round_motif_select_limit = max(round_motif_select_limit, 12)
    if detail_fast:
        round_motif_limit = min(round_motif_limit, 9)
        round_motif_select_limit = min(round_motif_select_limit, 8)
    if is_roundish and is_publication:
        # Round measured drawings (e.g. bronze mirrors) need richer motif capture.
        round_motif_select_limit = max(
            round_motif_limit,
            max(7, min(11, texture_count + 4)),
        )
    if fast_round_structural or skip_round_motifs:
        round_motif_lines = []
        round_relief_lines = []
        round_relief_region_lines = []
        round_polar_motif_lines = []
        round_center_motif_lines = []
    else:
        round_motif_lines = select_round_inner_motif_lines(
            texture_lines + extract_round_motif_lines(
                detail_bgr,
                target_mask,
                main_contour,
                max_lines=max(18, round_motif_select_limit * 3),
            ),
            target_mask,
            max_lines=round_motif_select_limit,
            prefer_outer=(is_roundish and is_publication and (not low_quality_input)),
        ) if is_roundish else []
        round_relief_lines = extract_round_relief_lines(
            detail_bgr,
            target_mask,
            main_contour,
            max_lines=max(10, round_motif_select_limit * 3),
        ) if (is_roundish and is_publication) else []
        round_relief_region_lines = extract_round_relief_region_lines(
            detail_bgr,
            target_mask,
            main_contour,
            max_lines=max(8, round_motif_select_limit * 2),
        ) if (is_roundish and is_publication) else []
        round_polar_motif_lines = extract_round_polar_motif_lines(
            detail_bgr,
            target_mask,
            main_contour,
            max_lines=max(8, round_motif_select_limit * 2),
        ) if (is_roundish and is_publication) else []
        round_center_motif_lines = extract_round_center_motif_lines(
            detail_bgr,
            target_mask,
            main_contour,
            max_lines=max(6, round_motif_select_limit),
        ) if (is_roundish and is_publication) else []

    if legend_mode:
        if is_roundish:
            internal_lines = round_motif_lines[:1] if round_motif_lines else round_lines[:1]
        else:
            internal_lines = profile_lines[:1] + spine_lines[:1]
            if terminal_count > 0:
                internal_lines += terminal_lines[:1]
    elif is_typology:
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
            if fast_round_structural:
                motif_target = max(6, min(9, round_motif_select_limit + 1))
                internal_lines = build_round_structural_lines(
                    target_mask=target_mask,
                    main_contour=main_contour,
                    round_lines=round_lines,
                    max_lines=motif_target,
                )
                center_fallback = extract_round_center_fallback_lines(
                    detail_bgr,
                    target_mask,
                    main_contour,
                    max_lines=max(4, motif_target - 2),
                )
                internal_lines = merge_distinct_lines(
                    internal_lines,
                    center_fallback,
                    min_center_sep=2.2,
                    max_lines=motif_target,
                    min_arc_len=5.0,
                )
                internal_lines = prefer_round_inner_lines(
                    internal_lines,
                    target_mask,
                    max_lines=motif_target,
                    inner_ratio=0.58,
                    min_inner=4,
                )
            else:
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
                    motif_lines = select_round_inner_motif_lines(
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
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        round_polar_motif_lines,
                        min_center_sep=2.8,
                        max_lines=motif_target,
                        min_arc_len=6.0,
                    )
                if round_relief_region_lines:
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        round_relief_region_lines,
                        min_center_sep=3.2,
                        max_lines=motif_target,
                        min_arc_len=8.0,
                    )
                if (not prefer_region) and round_relief_lines:
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        round_relief_lines,
                        min_center_sep=3.2,
                        max_lines=motif_target,
                        min_arc_len=8.0,
                    )
                if round_motif_lines:
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        round_motif_lines,
                        min_center_sep=3.0,
                        max_lines=motif_target,
                        min_arc_len=7.0,
                    )
                internal_lines = regularize_round_publication_lines(
                    internal_lines,
                    target_mask,
                    max_lines=motif_target,
                )
                internal_lines = suppress_round_ring_lines(
                    internal_lines,
                    target_mask,
                    max_ring_lines=0,
                )
                internal_lines = augment_round_rotational_symmetry(
                    internal_lines,
                    target_mask,
                    desired_lines=max(5, motif_target - 1),
                )
                ys_round, xs_round = np.where(target_mask > 0)
                if len(xs_round) > 50:
                    cx_round = float(np.mean(xs_round))
                    cy_round = float(np.mean(ys_round))
                    angular_cov = round_line_angular_coverage(internal_lines, cx_round, cy_round, bins=12)
                    ring_ratio = round_ring_line_ratio(internal_lines, target_mask)
                else:
                    angular_cov = 1.0
                    ring_ratio = 0.0
                if len(internal_lines) < 4 or angular_cov < 0.34 or ring_ratio > 0.58:
                    angular_markers = estimate_round_angular_motif_markers(
                        detail_bgr,
                        target_mask,
                        max_lines=max(8, motif_target),
                    )
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        angular_markers,
                        min_center_sep=2.8,
                        max_lines=motif_target,
                        min_arc_len=6.0,
                    )
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        round_polar_motif_lines,
                        min_center_sep=2.8,
                        max_lines=motif_target,
                        min_arc_len=6.0,
                    )
                    internal_lines = regularize_round_publication_lines(
                        internal_lines,
                        target_mask,
                        max_lines=motif_target,
                    )
                    internal_lines = suppress_round_ring_lines(
                        internal_lines,
                        target_mask,
                        max_ring_lines=1,
                    )
                if low_quality_input and len(internal_lines) < max(4, motif_target // 2):
                    low_quality_lines = extract_round_low_quality_lines(
                        detail_bgr,
                        target_mask,
                        main_contour,
                        max_lines=max(8, motif_target + 2),
                    )
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        low_quality_lines,
                        min_center_sep=2.6,
                        max_lines=motif_target,
                        min_arc_len=6.0,
                    )
                    internal_lines = regularize_round_publication_lines(
                        internal_lines,
                        target_mask,
                        max_lines=motif_target,
                    )
                center_coverage = round_line_center_coverage(internal_lines, target_mask)
                inner_line_count = round_line_inner_count(internal_lines, target_mask, ratio=0.50)
                if low_quality_input and (center_coverage < 0.42 or inner_line_count < 3):
                    center_fallback = extract_round_center_fallback_lines(
                        detail_bgr,
                        target_mask,
                        main_contour,
                        max_lines=max(8, motif_target),
                    )
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        center_fallback,
                        min_center_sep=2.2,
                        max_lines=motif_target,
                        min_arc_len=5.0,
                    )
                    internal_lines = regularize_round_publication_lines(
                        internal_lines,
                        target_mask,
                        max_lines=motif_target,
                    )
                    internal_lines = suppress_round_ring_lines(
                        internal_lines,
                        target_mask,
                        max_ring_lines=0,
                    )
                if low_quality_input:
                    unwrap_lines = extract_round_unwrap_lines(
                        detail_bgr,
                        target_mask,
                        main_contour,
                        max_lines=max(8, motif_target + 2),
                    )
                    internal_lines = merge_distinct_lines(
                        internal_lines,
                        unwrap_lines,
                        min_center_sep=2.2,
                        max_lines=motif_target,
                        min_arc_len=5.0,
                    )
                    internal_lines = prefer_round_inner_lines(
                        list(internal_lines) + list(round_center_motif_lines) + list(round_polar_motif_lines),
                        target_mask,
                        max_lines=motif_target,
                        inner_ratio=0.56,
                        min_inner=4,
                    )
                if round_lines and len(internal_lines) < 5:
                    anchor = round_lines[1] if len(round_lines) > 1 else round_lines[0]
                    internal_lines = [anchor] + internal_lines
                    internal_lines = internal_lines[:max(4, motif_target)]
                # Keep one circular band only as fallback when motif capture is weak.
                if len(internal_lines) < 2 and round_lines:
                    internal_lines += round_lines[:1]
                if round_strategy_key == "structure_first":
                    signature_need = True
                elif round_strategy_key == "hybrid":
                    signature_need = (
                        low_quality_input
                        and needs_round_mirror_rescue(
                            internal_lines,
                            target_mask,
                            strict=(not detail_fast),
                        )
                    )
                else:
                    # Image-first: only rescue when extraction is clearly broken.
                    signature_need = (
                        low_quality_input
                        and needs_round_mirror_rescue(
                            internal_lines,
                            target_mask,
                            strict=True,
                        )
                        and len(internal_lines) < max(4, motif_target - 2)
                    )
                if signature_need and synthetic:
                    mirror_signature = extract_round_mirror_signature_lines(
                        detail_bgr,
                        target_mask,
                        main_contour,
                        max_lines=max(8, motif_target + 1),
                    )
                    if mirror_signature:
                        if round_strategy_key == "image_first":
                            internal_lines = merge_distinct_lines(
                                internal_lines,
                                mirror_signature,
                                min_center_sep=2.4,
                                max_lines=max(8, motif_target + 1),
                                min_arc_len=4.0,
                            )
                        else:
                            seed_lines = []
                            if round_center_motif_lines:
                                seed_lines += round_center_motif_lines[:2]
                            if round_polar_motif_lines:
                                seed_lines += round_polar_motif_lines[:2]
                            if round_motif_lines:
                                seed_lines += round_motif_lines[:2]
                            if internal_lines:
                                seed_lines += internal_lines[:2]
                            internal_lines = merge_distinct_lines(
                                mirror_signature,
                                seed_lines,
                                min_center_sep=2.0,
                                max_lines=max(8, motif_target + 1),
                                min_arc_len=3.0,
                            )
                            internal_lines = regularize_round_publication_lines(
                                internal_lines,
                                target_mask,
                                max_lines=max(8, motif_target + 1),
                            )
                            internal_lines = suppress_round_ring_lines(
                                internal_lines,
                                target_mask,
                                max_ring_lines=1,
                            )
                        internal_lines = prefer_round_inner_lines(
                            internal_lines,
                            target_mask,
                            max_lines=max(8, motif_target + 1),
                            inner_ratio=0.58,
                            min_inner=4,
                        )
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
                line_lines = remove_near_horizontal_lines(texture_lines[:max(10, line_detail_count * 2)])
                internal_lines = line_lines[:max(1, line_detail_count)] if line_detail_count > 0 else []
            if not internal_lines and round_lines:
                internal_lines = round_lines[:1]
        else:
            # Line mode removes horizontal bars and keeps only vertical/diagonal factual cues.
            line_lines = remove_near_horizontal_lines(texture_lines[:max(6, line_detail_count)] + spine_lines[:1])
            internal_lines = line_lines[:max(1, line_detail_count)] if line_detail_count > 0 else []
    else:
        # Colored mode: symbolic structural lines only (avoid painterly/noisy interiors).
        if is_roundish:
            # Circular artifacts (e.g. coins) should avoid forced vertical spine lines.
            internal_lines = round_lines[:1]
            if round_motif_lines:
                internal_lines += round_motif_lines[:round_motif_limit]
            elif factuality_v >= 0.72 and texture_count > 0 and not internal_lines:
                internal_lines += remove_near_horizontal_lines(texture_lines)[:1]
        else:
            internal_lines = profile_lines[:max(1, profile_count)] + spine_lines[:1]
            if factuality_v >= 0.7 and symbolic_v <= 0.4 and texture_count > 0:
                internal_lines += remove_near_horizontal_lines(texture_lines)[:2]

    if is_drawing:
        # Drawings: the ink strokes *are* the content; keep them (longest first).
        drawing_limit = 80 if is_mono else max(3, line_detail_count + 2)
        internal_lines = [list(pl) for pl in ink_lines[:drawing_limit]]

    if is_typology:
        palette_seeds = list(material_palette[:4]) if material_palette else [final_color]
        harmonized_tones = []
        for idx, seed in enumerate(palette_seeds):
            mix_ratio = 0.34 if idx < 2 else 0.28
            tone = blend_hex(final_color, seed, mix_ratio)
            harmonized_tones.append(muted_hex(tone, keep=0.80))

        if not harmonized_tones:
            harmonized_tones.append(muted_hex(final_color, keep=0.66))
        while len(harmonized_tones) < 3:
            if len(harmonized_tones) == 1:
                harmonized_tones.append(lighten_hex(harmonized_tones[0], 0.16))
            else:
                harmonized_tones.append(darken_hex(harmonized_tones[0], 0.84))

        ordered_tones = sorted(
            harmonized_tones[:3],
            key=lambda c: hex_luminance(c),
            reverse=True,
        )
        warm_highlight_color = ordered_tones[0]
        base_color = ordered_tones[1]
        deep_shadow_color = ordered_tones[2]
        hi_luma = hex_luminance(warm_highlight_color)
        mid_luma = hex_luminance(base_color)
        lo_luma = hex_luminance(deep_shadow_color)
        if (hi_luma - mid_luma) < 16.0:
            warm_highlight_color = lighten_hex(base_color, 0.20)
        if (mid_luma - lo_luma) < 16.0:
            deep_shadow_color = darken_hex(base_color, 0.78)
        if (hex_luminance(warm_highlight_color) - hex_luminance(deep_shadow_color)) < 34.0:
            warm_highlight_color = lighten_hex(warm_highlight_color, 0.10)
            deep_shadow_color = darken_hex(deep_shadow_color, 0.90)
        patina_tone = (
            harmonized_tones[3]
            if len(harmonized_tones) > 3
            else blend_hex(base_color, warm_highlight_color, 0.30)
        )
        patina_tone = muted_hex(patina_tone, keep=0.84)

        outline_color = darken_hex(base_color, 0.56)
        structure_color = darken_hex(blend_hex(base_color, deep_shadow_color, 0.42), 0.74)
        shade_color = darken_hex(deep_shadow_color, 0.90)
        highlight_color = lighten_hex(blend_hex(base_color, warm_highlight_color, 0.58), 0.10)

        svg_output.append(
            "<defs>"
            f'<linearGradient id="agTypologyBase" x1="20%" y1="8%" x2="84%" y2="94%">'
            f'<stop offset="0%" stop-color="{warm_highlight_color}" stop-opacity="1"/>'
            f'<stop offset="55%" stop-color="{base_color}" stop-opacity="1"/>'
            f'<stop offset="100%" stop-color="{deep_shadow_color}" stop-opacity="1"/>'
            "</linearGradient>"
            f'<radialGradient id="agTypologyHighlight" cx="30%" cy="24%" r="64%">'
            f'<stop offset="0%" stop-color="{highlight_color}" stop-opacity="1"/>'
            f'<stop offset="100%" stop-color="{base_color}" stop-opacity="0"/>'
            "</radialGradient>"
            f'<radialGradient id="agTypologyPatina" cx="66%" cy="70%" r="58%">'
            f'<stop offset="0%" stop-color="{patina_tone}" stop-opacity="1"/>'
            f'<stop offset="100%" stop-color="{base_color}" stop-opacity="0"/>'
            "</radialGradient>"
            f'<linearGradient id="agTypologyShadow" x1="44%" y1="0%" x2="58%" y2="100%">'
            f'<stop offset="0%" stop-color="{base_color}" stop-opacity="0"/>'
            f'<stop offset="100%" stop-color="{shade_color}" stop-opacity="1"/>'
            "</linearGradient>"
            "</defs>"
        )

        svg_output.append(
            f'<path d="{path_data}" fill="url(#agTypologyBase)" fill-opacity="1.0" stroke="none" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        svg_output.append(
            f'<path d="{path_data}" fill="url(#agTypologyHighlight)" fill-opacity="0.30" stroke="none" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        svg_output.append(
            f'<path d="{path_data}" fill="url(#agTypologyPatina)" fill-opacity="0.28" stroke="none" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        svg_output.append(
            f'<path d="{path_data}" fill="url(#agTypologyShadow)" fill-opacity="0.30" stroke="none" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        svg_output.append(
            f'<path d="{path_data}" fill="none" stroke="{outline_color}" '
            'stroke-width="2.35" stroke-linecap="round" stroke-linejoin="round"/>'
        )

        for line in profile_lines[:3]:
            line_path = polyline_to_path(line)
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
            line_path = polyline_to_path(line)
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
            line_path = polyline_to_path(line)
            if not line_path:
                continue
            svg_output.append(
                f'<path d="{line_path}" fill="none" stroke="{structure_color}" stroke-opacity="0.90" '
                'stroke-width="1.20" stroke-linecap="round" stroke-linejoin="round"/>'
            )
    elif is_mono:
        if is_publication:
            outline_width = 1.8
            detail_width = 1.35 if is_roundish else 1.0
            detail_dash = "" if is_roundish else ' stroke-dasharray="1.2 2.2"'
            detail_opacity = 0.94 if is_roundish else 0.7
            mono_base = muted_hex(final_color, keep=0.16 if is_roundish else 0.12)
            outline_color = "#111111"
            detail_color = darken_hex(mono_base, 0.62)
            detail_under_color = lighten_hex(mono_base, 0.12)
            detail_under_opacity = 0.34 if is_roundish else 0.22
        else:
            outline_width = 2.2
            detail_width = 1.25
            detail_dash = ""
            detail_opacity = 0.8
            mono_base = muted_hex(final_color, keep=0.10)
            outline_color = "#111111"
            detail_color = darken_hex(mono_base, 0.68)
            detail_under_color = lighten_hex(mono_base, 0.10)
            detail_under_opacity = 0.18

        svg_output.append(
            f'<path d="{path_data}" fill="none" stroke="{outline_color}" stroke-width="{outline_width:.2f}" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for line in internal_lines:
            line_path = polyline_to_path(line)
            if line_path:
                svg_output.append(
                    f'<path d="{line_path}" fill="none" stroke="{detail_under_color}" stroke-opacity="{detail_under_opacity:.2f}" '
                    f'stroke-width="{(detail_width + 0.48):.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
                )
                svg_output.append(
                    f'<path d="{line_path}" fill="none" stroke="{detail_color}" stroke-opacity="{detail_opacity:.2f}" '
                    f'stroke-width="{detail_width:.2f}"{detail_dash} stroke-linecap="round" stroke-linejoin="round"/>'
                )
    else:
        if legend_mode:
            # Simple Symbol style: two-tone fill + bold outline + minimal structural linework.
            fill_color = muted_hex(final_color, keep=0.78)
            simple_light = lighten_hex(fill_color, 0.16)
            simple_dark = darken_hex(fill_color, 0.84)
            simple_glow = lighten_hex(fill_color, 0.26)
            outline_color = darken_hex(fill_color, 0.56)
            detail_color = darken_hex(fill_color, 0.70)
            fill_opacity = 0.90 if is_roundish else 0.94

            svg_output.append(
                "<defs>"
                f'<linearGradient id="agSimpleBase" x1="20%" y1="12%" x2="82%" y2="92%">'
                f'<stop offset="0%" stop-color="{simple_light}" stop-opacity="1"/>'
                f'<stop offset="62%" stop-color="{fill_color}" stop-opacity="1"/>'
                f'<stop offset="100%" stop-color="{simple_dark}" stop-opacity="1"/>'
                "</linearGradient>"
                f'<radialGradient id="agSimpleGlow" cx="28%" cy="22%" r="56%">'
                f'<stop offset="0%" stop-color="{simple_glow}" stop-opacity="1"/>'
                f'<stop offset="100%" stop-color="{fill_color}" stop-opacity="0"/>'
                "</radialGradient>"
                "</defs>"
            )
            svg_output.append(
                f'<path d="{path_data}" fill="url(#agSimpleBase)" fill-opacity="{fill_opacity:.2f}" stroke="none" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
            )
            svg_output.append(
                f'<path d="{path_data}" fill="url(#agSimpleGlow)" fill-opacity="0.20" stroke="none" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
            )
            svg_output.append(
                f'<path d="{path_data}" fill="none" stroke="{outline_color}" '
                'stroke-width="2.60" stroke-linecap="round" stroke-linejoin="round"/>'
            )

            for line in internal_lines[:2]:
                line_path = polyline_to_path(line)
                if line_path:
                    svg_output.append(
                        f'<path d="{line_path}" fill="none" stroke="{detail_color}" stroke-opacity="0.86" '
                        'stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round"/>'
                    )
        else:
            # Colored style: avoid flat single-color mass; keep subtle layered tones.
            fill_color = muted_hex(final_color, keep=0.72)
            outline_color = darken_hex(final_color, 0.58)
            detail_color = darken_hex(final_color, 0.42)
            accent_color = lighten_hex(final_color, 0.14)
            deep_fill_color = darken_hex(fill_color, 0.88)
            glow_color = lighten_hex(fill_color, 0.22)
            fill_opacity = 0.62 if is_roundish else 0.72
            svg_output.append(
                "<defs>"
                f'<linearGradient id="agColoredBase" x1="18%" y1="12%" x2="82%" y2="92%">'
                f'<stop offset="0%" stop-color="{glow_color}" stop-opacity="1"/>'
                f'<stop offset="62%" stop-color="{fill_color}" stop-opacity="1"/>'
                f'<stop offset="100%" stop-color="{deep_fill_color}" stop-opacity="1"/>'
                "</linearGradient>"
                f'<radialGradient id="agColoredGlow" cx="28%" cy="22%" r="58%">'
                f'<stop offset="0%" stop-color="{accent_color}" stop-opacity="1"/>'
                f'<stop offset="100%" stop-color="{fill_color}" stop-opacity="0"/>'
                "</radialGradient>"
                "</defs>"
            )
            svg_output.append(
                f'<path d="{path_data}" fill="url(#agColoredBase)" fill-opacity="{fill_opacity:.2f}" stroke="none" '
                'stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>'
            )
            svg_output.append(
                f'<path d="{path_data}" fill="url(#agColoredGlow)" fill-opacity="0.20" stroke="none" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
            )
            svg_output.append(
                f'<path d="{path_data}" fill="none" stroke="{outline_color}" '
                'stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>'
            )
            accent_lines = (round_lines[:1] if is_roundish else profile_lines[:1])
            for line in accent_lines:
                line_path = polyline_to_path(line)
                if line_path:
                    svg_output.append(
                        f'<path d="{line_path}" fill="none" stroke="{accent_color}" stroke-opacity="0.36" '
                        'stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>'
                    )
            for line in internal_lines:
                line_path = polyline_to_path(line)
                if line_path:
                    svg_output.append(
                        f'<path d="{line_path}" fill="none" stroke="{detail_color}" stroke-opacity="0.72" '
                        'stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round"/>'
                    )

    svg_output.append("</svg>")
    return "".join(svg_output)