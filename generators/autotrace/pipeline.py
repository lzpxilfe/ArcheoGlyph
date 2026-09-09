# -*- coding: utf-8 -*-
"""
Auto Trace pipeline: image -> silhouette -> internal lines -> SVG (QGIS-free).

This is the former ContourGenerator.generate body with settings access
replaced by AutoTraceOptions and mask extraction delegated to the caller.
"""

import re

import cv2
import numpy as np

from ...log import log
from ..ink_centerline import extract_ink_polylines, looks_like_drawing, simplify_polyline
from .stroke_font import text_extent, text_polylines
from .svg_builder import HOUSE_DETAIL_RATIO, HOUSE_OUTLINE_RATIO, smooth_closed_path
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
    INCISED,
    MODELLED,
    estimate_masked_edge_density,
    prepare_detail_source,
    relief_ink_sheet,
)
from .geometry import (
    MAX_INTERIOR_MARKS,
    circle_path,
    clamp,
    keep_marks_that_read,
    keep_marks_within_ink_budget,
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
from .feature_symmetry import centre_disagrees, vote_for_centre
from .round_motif import (
    FRAME_MIN_SCORE,
    find_rotational_frame,
    reading_is_stable,
    fold_rotational_motif,
    replay_rotational_motif,
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
    looks_like_a_vessel,
)


#: How much darker than its fill a silhouette stroke is drawn.
#:
#: The drawn catalogue uses ``QColor.darker(140)`` in template_generator._pen,
#: which divides each channel by 1.4. This was a flat "#111111" here, and in
#: QGIS that hardly showed - the fallback is replaced by whatever outline
#: colour the user picked. Outside QGIS it showed a great deal: in the
#: preview sheets, in documentation and in any plain SVG viewer, a traced
#: artefact came out drawn in near-black next to a catalogue drawn in its own
#: muted colour, and read as much heavier than it is at the same width.
HOUSE_OUTLINE_DARKEN = 1.0 / 1.4

#: How much of its own tile a traced symbol's interior covers in ink, taken
#: from the drawn catalogue this has to sit beside: over its 188 symbols the
#: median covers 27 percent and the busiest 1.76 times that. Past the ceiling
#: the weight is scaled to bring the drawing back to the median.
INTERIOR_INK_MEDIAN = 0.27
INTERIOR_INK_CEILING = 1.76 * INTERIOR_INK_MEDIAN


#: How tall a typology code is set, as a share of the artefact's longer side.
TYPE_CODE_HEIGHT = 0.16

#: How much of the artefact's width a code may span before it is set smaller.
TYPE_CODE_WIDTH = 0.80

#: The gap between the artefact and a code that would not fit inside it.
TYPE_CODE_GAP = 0.05

#: The smallest a code may be set, as a share of the artefact's longer side.
#: A three-by-five glyph needs about five legend pixels of cap height to read,
#: and a symbol is shown at 64 of them. Below this the code goes under the
#: artefact at full size instead of inside it as a squint: a slender dagger is
#: narrow enough that fitting five characters across it produced exactly that.
TYPE_CODE_MIN_HEIGHT = 0.09

_STROKE_WIDTH_RE = re.compile(r'stroke-width="([\d.]+)"')


def _fits_inside(mask, x, y, width, height):
    """Whether a box is wholly inside the silhouette, sampled on a grid."""
    if mask is None or not (width > 0 and height > 0):
        return False
    rows, cols = mask.shape[:2]
    for u in range(5):
        for v in range(3):
            px = int(round(x + (width * u / 4.0)))
            py = int(round(y + (height * v / 2.0)))
            if not (0 <= px < cols and 0 <= py < rows) or mask[py, px] == 0:
                return False
    return True


def _type_code_paths(code, drawn, mask, bounds, color):
    """
    A typology code, cut in stroke_font and set into the symbol.

    Inside the silhouette when it fits there, and underneath it when it does
    not - a blade has no room for five characters across it, and a caption
    under the drawing is what an archaeological plate does anyway.

    The stroke is half the heaviest already in the file, because svg_builder
    scales that heaviest one to the house outline weight: half of it is one
    grid unit, which is one legend pixel, which is the floor below which the
    code could not be read at all.
    """
    bx, by, bw, bh = (float(v) for v in bounds)
    side = max(bw, bh)
    if not (side > 0):
        return []

    height = side * TYPE_CODE_HEIGHT
    width, _ = text_extent(code, height)
    if width <= 0:
        return []

    # Inside the artefact, if it can be set there without shrinking past the
    # point of being readable.
    top = None
    inside_height = height
    if width > bw * TYPE_CODE_WIDTH:
        inside_height = height * (bw * TYPE_CODE_WIDTH) / width
    if inside_height >= side * TYPE_CODE_MIN_HEIGHT:
        inside_width, _ = text_extent(code, inside_height)
        left = bx + (bw - inside_width) / 2.0
        for step in range(9):
            candidate = by + bh - (inside_height * 1.35) - (step * inside_height * 0.5)
            if candidate < by:
                break
            if _fits_inside(mask, left, candidate, inside_width, inside_height):
                top, height, width = candidate, inside_height, inside_width
                break

    if top is None:
        # Underneath it, then, which is what a plate does anyway. Here the
        # code may span the whole symbol rather than the artefact's own width.
        if width > side:
            height *= side / width
            width, _ = text_extent(code, height)
        left = bx + (bw - width) / 2.0
        top = by + bh + (side * TYPE_CODE_GAP)

    heaviest = max([float(m) for chunk in drawn
                    for m in _STROKE_WIDTH_RE.findall(chunk)] or [2.0])
    stroke = heaviest * (HOUSE_DETAIL_RATIO / HOUSE_OUTLINE_RATIO)
    ink = darken_hex(color, 0.45)

    paths = []
    for line in text_polylines(code, height, origin=(left, top)):
        line_path = polyline_to_path([[int(round(x)), int(round(y))]
                                      for x, y in line])
        if line_path:
            paths.append(
                f'<path d="{line_path}" fill="none" stroke="{ink}" '
                f'stroke-width="{stroke:.2f}" stroke-linecap="round" '
                'stroke-linejoin="round"/>')
    return paths


def run_autotrace(bgr, options, mask_provider, relief=None):
    """
    Full Auto Trace pipeline on an 8-bit BGR image.

    :param bgr: source image (EXIF-corrected, no alpha)
    :param options: AutoTraceOptions
    :param mask_provider: callable ``processing_bgr -> uint8 mask`` (the
        caller owns backends, caching and alpha handling)
    :param relief: optional single-channel relief map, the same size as
        ``bgr``, built from a stack of differently lit photographs. The
        silhouette still comes from the photograph - a relief map has no
        clean outline - and only the decoration is read from this.
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
    circle_iou = 0.0
    if is_roundish:
        (ccx, ccy), cr = cv2.minEnclosingCircle(main_contour)
        circle_canvas = np.zeros(target_mask.shape[:2], dtype=np.uint8)
        cv2.circle(circle_canvas, (int(round(ccx)), int(round(ccy))), int(round(cr)), 255, -1)
        contour_canvas = np.zeros_like(circle_canvas)
        cv2.drawContours(contour_canvas, [main_contour], -1, 255, -1)
        inter = float(np.count_nonzero(cv2.bitwise_and(circle_canvas, contour_canvas)))
        union = float(np.count_nonzero(cv2.bitwise_or(circle_canvas, contour_canvas)))
        circle_iou = inter / max(1.0, union)
    use_circle_outline = bool(
        is_roundish and (circle_iou >= 0.94 or (contour_circularity >= 0.90 and solidity >= 0.95))
    )
    # Reading relief as a rubbing assumes a flat decorated face turned towards
    # the camera. is_roundish is too loose for that: it admits a comb-pattern
    # jar, whose shading is the curve of its own body rather than ornament,
    # and tracing that covered the pot in speckle. The two roof tile ends and
    # the mirror fill their enclosing circle to 0.968 and above; the jar
    # reaches 0.714 and a ground stone tool 0.630, so the same 0.94 the
    # outline test already uses separates them with room to spare.
    is_flat_faced_disc = bool(is_roundish and circle_iou >= 0.94)
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
                # Smooth curves between gentle vertices, hard corners kept.
                path_data = smooth_closed_path(final_points, corner_deg=38.0)

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
    # A vessel keeps its rim and shoulder whatever the sliders say. These come
    # from the outline changing curvature, not from the photograph - they are
    # the pot's shape, and showing it is the drawing convention for a pot.
    # What is *not* claimed here is its decoration: six attempts to read a
    # decorated zone out of these photographs all failed, and drawing the
    # whole surface instead was covering the pots in speckle.
    #
    # Held apart from profile_lines above, which the styles spend on their own
    # slider budgets: these are added after the style has chosen, so they
    # cannot be double-counted and cannot be dropped by a branch that strips
    # horizontals (Line does, deliberately - but a rim is not a stray bar).
    vessel_bands = []
    if not is_drawing and looks_like_a_vessel(target_mask):
        vessel_bands = estimate_profile_bands(target_mask, max_lines=2)[:2]
        if vessel_bands:
            log(f"Drawing {len(vessel_bands)} structural bands on this vessel "
                f"- its rim and shoulder, not its decoration.")
    ink_lines = []
    relief_sheet = None
    # A flat decorated face is traced whatever the style asked for. Simple
    # Symbol does not draw traced ink, but it still has to know the artefact
    # is decorated: without this a lotus roof tile end and a plain disc came
    # out as the same grey circle, which is the largest thing this set was
    # getting wrong.
    if is_drawing or is_mono or is_flat_faced_disc:
        try:
            erode_px = max(2, int(round(0.015 * min(target_mask.shape[:2]))))
            ink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
            ink_mask = cv2.erode(target_mask, ink_kernel)
            floor = max(6.0, 0.02 * float(min(target_mask.shape[:2])))

            def _trace(source):
                return [
                    [[int(x), int(y)] for x, y in simplify_polyline(pline, epsilon=1.2)]
                    for pline in extract_ink_polylines(
                        source, mask=ink_mask, min_arc_length=floor)
                ]

            if is_flat_faced_disc and not is_drawing:
                # A round artefact's decoration is shallow relief, and reading
                # it straight off the photograph gave lighting, not ornament -
                # on a lotus roof tile end, a diagonal stripe across the face
                # where its lit and shadowed halves met. Turning the relief
                # into a rubbing first hands this the kind of input it is
                # already good at.
                #
                # Both readings are traced and merged rather than one being
                # chosen, because the choice cannot be made from the
                # photograph: incised and modelled decoration have the same
                # mean mark width (1.79 and 1.70 percent of the artefact on
                # the two tiles), and five attempts to separate them by
                # measurement all failed. Merging is better than either alone
                # on all three discs - the lotus gains its rim and bead rings,
                # the dragon its coil, the mirror keeps both its rim lines.
                _rx, _ry, _rw, _rh = cv2.boundingRect(main_contour)
                face_radius = max(_rw, _rh) / 2.0
                relief_sheet = relief_ink_sheet(processing_bgr, target_mask,
                                                face_radius, reading=INCISED)
                ink_lines = merge_distinct_lines(
                    _trace(relief_sheet),
                    _trace(relief_ink_sheet(processing_bgr, target_mask,
                                            face_radius, reading=MODELLED)),
                    min_center_sep=max(3.0, max(_rw, _rh) * 0.012),
                    max_lines=400,
                    min_arc_len=max(_rw, _rh) * 0.03,
                )
            else:
                ink_source = (processing_bgr if is_drawing else detail_bgr).copy()
                if not is_drawing:
                    # Flatten the background to the object tone so the
                    # silhouette edge itself does not read as a dark stroke.
                    inside = target_mask > 0
                    if inside.any():
                        fill_value = np.median(
                            ink_source[inside].reshape(-1, 3), axis=0).astype(np.uint8)
                        ink_source[~inside] = fill_value
                ink_lines = _trace(ink_source)
        except Exception:
            ink_lines = []

    # Where a find has more readable traced marks than the busiest drawn
    # symbol it *is* decorated, whatever the style then chooses to draw. Read
    # here, off the traced ink, so every style gets the same verdict about the
    # same artefact - the per-style selection below decides what to draw, not
    # what is there, and Simple Symbol discards the traced ink entirely.
    traced_marks = []
    _tx, _ty, _tw, _th = cv2.boundingRect(main_contour)
    if ink_lines and not is_drawing:
        readable_ink = keep_marks_that_read(ink_lines, float(max(_tw, _th)),
                                            max_marks=None)
        if len(readable_ink) > MAX_INTERIOR_MARKS:
            traced_marks = readable_ink
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

    # A round artefact whose identity is its decoration - a mirror, a roof
    # tile end - is a bare disc without it, and every such disc is every
    # other one. Where the decoration genuinely repeats, fold the sectors
    # together and stamp the agreed shape back around the face.
    folded_motif_lines = []
    if is_roundish:
        # Shallow relief is height, and one photograph has none of it. Where
        # the caller supplied a stack lit from several directions, read the
        # decoration off that instead: what stays the same as the lamp moves
        # is a stain, and what changes is the relief.
        motif_source = relief
        if motif_source is not None and motif_source.shape[:2] != target_mask.shape[:2]:
            motif_source = cv2.resize(motif_source,
                                      (target_mask.shape[1], target_mask.shape[0]),
                                      interpolation=cv2.INTER_AREA)
        if motif_source is None:
            motif_source = cv2.cvtColor(processing_bgr, cv2.COLOR_BGR2GRAY)
        frame = find_rotational_frame(motif_source, target_mask)
        if frame is not None and frame.score >= FRAME_MIN_SCORE \
                and not reading_is_stable(motif_source, frame):
            # A score above the gate is not enough. Push the frame as far as
            # a drawn repeat can be pushed and ask again: a reading in the
            # middle of its basin holds, one on the edge of it does not, and
            # the next crop would land somewhere else and draw a different
            # artefact.
            log("The repeat on this round artefact changes when the frame is "
                f"nudged (best {frame.folds}-fold at {frame.score:.3f}); "
                f"drawing it plain.")
        elif frame is not None and frame.score >= FRAME_MIN_SCORE:
            # Last check before decoration is committed to someone's artefact,
            # and only here because it is the expensive one: ask a published
            # method that finds the centre a completely different way whether
            # it agrees. Silence from it is not disagreement - a plain disc
            # has nothing to match - so only an actual conflict refuses.
            voted = vote_for_centre(motif_source, target_mask,
                                    max(frame.a, frame.b))
            if centre_disagrees(frame, voted, max(frame.a, frame.b)):
                log("Two methods put this artefact's decorated face in "
                    f"different places - fitted ({frame.cx:.0f},{frame.cy:.0f}), "
                    f"feature vote ({voted[0]:.0f},{voted[1]:.0f}) - so the "
                    f"{frame.folds}-fold reading is not trusted; drawing it plain.")
            else:
                folded_motif_lines = replay_rotational_motif(
                    fold_rotational_motif(motif_source, frame), frame)
        elif frame is not None:
            # Saying nothing here would be the ONNX fallback trap again: the
            # symbol comes out a plain disc and nothing says why.
            log("No repeating motif found on this round artefact "
                f"(best {frame.folds}-fold scored {frame.score:.3f}, "
                f"under {FRAME_MIN_SCORE}); drawing it plain.")

    if legend_mode:
        if is_roundish:
            if folded_motif_lines:
                # All of it or none: see replay_rotational_motif.
                internal_lines = list(folded_motif_lines)
            else:
                if traced_marks:
                    # A decorated disc has to look decorated: a lotus roof
                    # tile end and a plain one were coming out as the same
                    # grey circle. Its own traced ornament beats the one ring
                    # the motif reader offers - that ring is a guess about
                    # where a repeat might be, and these are the marks that
                    # are actually on the face.
                    internal_lines = traced_marks[:MAX_INTERIOR_MARKS]
                else:
                    internal_lines = (round_motif_lines[:1] if round_motif_lines
                                      else round_lines[:1])
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

    if is_drawing and not (legend_mode and folded_motif_lines):
        # Drawings: the ink strokes *are* the content; keep them (longest first).
        # The exception is a repeating motif at legend size. Eighty ink strokes
        # are right for a plate and grey mush on a 64px marker, and a rubbing
        # of a mirror is the one input where the fold-and-replay reading is
        # both reliable and exactly what the marker needs - it scores twice
        # the gate where a photograph of the same object scores a fifth of it.
        drawing_limit = 80 if is_mono else max(3, line_detail_count + 2)
        internal_lines = [list(pl) for pl in ink_lines[:drawing_limit]]
    elif is_mono and is_roundish and ink_lines and not (legend_mode and folded_motif_lines):
        if relief_sheet is not None:
            # The ink was traced from a rubbing of this artefact's own relief,
            # so it is the content, exactly as it is for a real rubbing above -
            # and the region extractors it used to be merged with are reading
            # the photograph's shading. On a lotus roof tile end the merge kept
            # a diagonal band spanning three quarters of the face, which was
            # the boundary between its lit and shadowed halves, in front of the
            # petal ring.
            internal_lines = [list(pl) for pl in
                              ink_lines[:80 if is_mono else max(3, line_detail_count + 2)]]
        else:
            # Round photographs with no relief reading: real strokes (rings,
            # incised motifs) come first, motif-extractor candidates only fill
            # the remaining budget. A folded motif at legend size is the
            # exception - merging raw strokes in front of it pushed the eight
            # replayed petals down to two.
            ink_cap = max(6, texture_count)
            internal_lines = merge_distinct_lines(
                [list(pl) for pl in ink_lines[:ink_cap]],
                list(internal_lines),
                min_center_sep=3.0,
                max_lines=max(8, ink_cap + 4),
                min_arc_len=6.0,
            )

    # Everything above chose *which* marks say what this artefact is. This
    # asks whether they can be seen at the size the symbol is used, which is
    # a separate question and the one that was going unasked: the styles were
    # emitting twenty-four interior marks inside a round artefact, half of
    # them two or three pixels across on a 64px legend marker.
    #
    # A folded motif is exempt from the count: it is stamped once per fold and
    # trimming it would leave the face decorated round part of its turn and
    # bare for the rest, which reads as damage (see replay_rotational_motif).
    # A drawing is exempt too - there the ink strokes are the content, not an
    # inference about it - but its specks still go, because a speck is
    # unreadable whatever drew it.
    if vessel_bands:
        internal_lines = list(vessel_bands) + list(internal_lines)

    if internal_lines:
        _mx, _my, _mw, _mh = cv2.boundingRect(main_contour)
        artefact_extent = float(max(_mw, _mh))
        if legend_mode and folded_motif_lines:
            pass
        else:
            # The count is exempt wherever the strokes are the content rather
            # than an inference about it: a real drawing, and a photograph
            # whose ink was traced from its own relief. Eleven marks is the
            # busiest drawn *legend symbol*, and Line and Measured are
            # documentation plates, not markers - capping them there cut a
            # lotus rosette of 125 strokes down to ten arcs. The size floor
            # still applies to both: a speck is unreadable whatever drew it.
            strokes_are_content = is_drawing or relief_sheet is not None
            if strokes_are_content and is_mono:
                # A documentation plate keeps the whole drawing. The legend
                # floor cut a rosette of four hundred traced curves down to
                # sixty-four, and what it removed were the short pieces
                # joining the long ones - so the petal outlines came out as
                # dashes. A marker still gets the floor, below.
                pass
            else:
                internal_lines = keep_marks_that_read(
                    internal_lines, artefact_extent,
                    max_marks=None if strokes_are_content else MAX_INTERIOR_MARKS)

    # A stroke weight chosen for a symbol with five marks is far too heavy for
    # a plate with four hundred traced curves - they merge into blobs. So the
    # detail weight is scaled down until the interior ink lands inside the
    # budget the drawn catalogue keeps: over its 188 symbols the median covers
    # 27 percent of its tile and the busiest 1.76 times that. Measured before
    # this, the two roof tile ends came out at 49 and 62 percent.
    #
    # Only the ceiling is applied. The floor is not: the catalogue's symbols
    # are filled shapes and a traced line drawing legitimately carries less
    # ink, so raising a sparse drawing to meet it would thicken artefacts that
    # already read correctly.
    drawn_length = 0.0
    for line in internal_lines:
        drawn_length += sum(
            float(np.hypot(line[i + 1][0] - line[i][0],
                           line[i + 1][1] - line[i][1]))
            for i in range(len(line) - 1))
    _mx, _my, _mw, _mh = cv2.boundingRect(main_contour)
    symbol_extent = float(max(_mw, _mh))

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
            outline_color = darken_hex(final_color, HOUSE_OUTLINE_DARKEN)
            detail_color = darken_hex(mono_base, 0.62)
            detail_under_color = lighten_hex(mono_base, 0.12)
            detail_under_opacity = 0.34 if is_roundish else 0.22
        else:
            outline_width = 2.2
            detail_width = 1.25
            detail_dash = ""
            detail_opacity = 0.8
            mono_base = muted_hex(final_color, keep=0.10)
            outline_color = darken_hex(final_color, HOUSE_OUTLINE_DARKEN)
            detail_color = darken_hex(mono_base, 0.68)
            detail_under_color = lighten_hex(mono_base, 0.10)
            detail_under_opacity = 0.18

        # svg_builder scales the heaviest stroke in the file to the house
        # outline weight, so what a detail stroke actually ends up as is that
        # weight times its share of the heaviest - and each line is drawn
        # twice, the halo being the wider of the two.
        def _ink_share(width, length):
            halo = float(width) + 0.48
            heaviest = max(float(outline_width), halo)
            if not (heaviest > 0) or not (symbol_extent > 0):
                return 0.0
            painted = (HOUSE_OUTLINE_RATIO * symbol_extent
                       * (halo + float(width)) / heaviest)
            return (float(length) * painted) / (symbol_extent * symbol_extent)

        if drawn_length > 0 and symbol_extent > 0 and outline_width > 0:
            share = _ink_share(detail_width, drawn_length)
            if share > INTERIOR_INK_CEILING:
                # Out of band, so bring it to the middle of the set rather than
                # to its loudest edge: the ceiling is where the catalogue's
                # *busiest* symbol sits, and a plate of four hundred traced
                # curves is not entitled to that on the grounds of being busy.
                #
                # The overspend is paid in weight first and marks after, and
                # the weight stops at the legend floor. Paying it all in weight
                # was the first attempt and it drew the two roof tile ends at
                # 0.53 and 0.71 of a grid unit - a unit is a legend pixel, so
                # the ornament went under the size the legend can show. What
                # cannot be seen is not worth keeping, so past the floor the
                # bill is settled by dropping marks, longest first.
                was_share = share
                was_width, was_count = detail_width, len(internal_lines)
                floor_width = (float(outline_width)
                               * HOUSE_DETAIL_RATIO / HOUSE_OUTLINE_RATIO)
                if detail_width > floor_width:
                    lighter, heavier = floor_width, float(detail_width)
                    for _ in range(40):
                        middle = (lighter + heavier) / 2.0
                        if _ink_share(middle, drawn_length) > INTERIOR_INK_MEDIAN:
                            heavier = middle
                        else:
                            lighter = middle
                    detail_width = heavier
                share = _ink_share(detail_width, drawn_length)
                if share > INTERIOR_INK_CEILING:
                    allowed = drawn_length * (INTERIOR_INK_MEDIAN / share)
                    internal_lines, drawn_length = keep_marks_within_ink_budget(
                        internal_lines, allowed)
                    share = _ink_share(detail_width, drawn_length)
                log(f"Interior ink came to {was_share * 100:.0f}% of this symbol "
                    f"against the catalogue's {INTERIOR_INK_CEILING * 100:.0f}%; "
                    f"drew {len(internal_lines)} of its {was_count} traced curves "
                    f"at {detail_width / was_width:.2f} times the weight, "
                    f"landing at {share * 100:.0f}%.")
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

            # Two marks is right for a silhouette that carries its own
            # meaning, and wrong for one whose meaning is the decoration: a
            # sixteen-line eight-fold motif came out as two stray slivers.
            # A folded motif is drawn whole or not at all.
            if folded_motif_lines:
                simple_detail_cap = len(folded_motif_lines)
            elif traced_marks:
                simple_detail_cap = MAX_INTERIOR_MARKS
            else:
                simple_detail_cap = 2
            for line in internal_lines[:simple_detail_cap]:
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

    if options.type_code:
        svg_output.extend(
            _type_code_paths(options.type_code, svg_output, target_mask,
                             cv2.boundingRect(main_contour), final_color))

    svg_output.append("</svg>")
    return "".join(svg_output)