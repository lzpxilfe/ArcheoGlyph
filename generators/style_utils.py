# -*- coding: utf-8 -*-
"""
Shared style constants and normalization helpers.
"""

STYLE_COLORED = "Colored"
STYLE_TYPOLOGY = "Typology"
STYLE_LEGEND = "Simple Symbol"
STYLE_LINE = "Line"
STYLE_MEASURED = "Measured"

# UI-facing styles: keep only the consolidated set.
STYLE_OPTIONS = [
    STYLE_LEGEND,
    STYLE_LINE,
    STYLE_MEASURED,
]


def normalize_style(style):
    """
    Normalize any style label into one of the canonical style constants.
    Consolidated behavior:
    - "simple symbol" and legacy color/typology labels map to typology key.
    - line/measured remain distinct.
    """
    low = str(style or "").strip().lower()

    if "measured" in low or "publication" in low:
        return STYLE_MEASURED
    if "line" in low:
        return STYLE_LINE
    if (
        "legend" in low
        or "simple symbol" in low
        or "typology" in low
        or "catalog" in low
        or "symbolic" in low
        or "colored" in low
        or "color" in low
    ):
        return STYLE_TYPOLOGY
    return STYLE_TYPOLOGY


def is_legend_style(style):
    """Return True when style should use simple-symbol rendering in Auto Trace."""
    low = str(style or "").strip().lower()
    return (
        "legend" in low
        or "simple symbol" in low
        or "typology" in low
        or "catalog" in low
        or "symbolic" in low
        or "colored" in low
        or "color" in low
    )
