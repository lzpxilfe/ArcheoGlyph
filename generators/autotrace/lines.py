# -*- coding: utf-8 -*-
"""
Multi-source internal line extraction combining enhance and round_motif.

Auto-generated from the former ContourGenerator methods; QGIS-free.
"""


from .enhance import extract_annular_relief_lines, extract_internal_lines, low_quality_variants
from .geometry import dedupe_lines
from .round_motif import extract_round_low_quality_lines


def extract_internal_lines_multisource(detail_bgr,
    base_bgr,
    target_mask,
    main_contour,
    low_quality=False,
    is_roundish=False,
    detail_mode="precise",
):
    """
    Extract internal lines from multiple enhanced views and merge.
    Designed to recover motif strokes from low-resolution or shaded inputs.
    """
    mode = str(detail_mode or "precise").strip().lower()
    fast_mode = mode == "fast"
    base_lines = extract_internal_lines(detail_bgr, target_mask, main_contour)
    if not low_quality:
        cap = 14 if fast_mode else 22
        return dedupe_lines(base_lines, min_points=4, max_lines=cap)

    merged = list(base_lines) if base_lines else []
    variants = low_quality_variants(detail_bgr, base_bgr, target_mask)
    if fast_mode:
        variants = variants[:1]
        per_variant_cap = 6
    else:
        per_variant_cap = 12 if is_roundish else 8
    for variant in variants:
        extra = extract_internal_lines(variant, target_mask, main_contour)
        if extra:
            merged.extend(extra[:per_variant_cap])

    if is_roundish:
        relaxed = extract_round_low_quality_lines(
            detail_bgr,
            target_mask,
            main_contour,
            max_lines=10 if fast_mode else 18,
        )
        if relaxed:
            merged.extend(relaxed)

        if not fast_mode:
            annular = extract_annular_relief_lines(
                detail_bgr,
                target_mask,
                main_contour,
                max_lines=14,
            )
            if annular:
                merged.extend(annular)

    if fast_mode:
        max_keep = 16 if is_roundish else 14
    else:
        max_keep = 30 if is_roundish else 22
    return dedupe_lines(merged, min_points=4, max_lines=max_keep)
