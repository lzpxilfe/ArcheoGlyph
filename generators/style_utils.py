# -*- coding: utf-8 -*-
"""
Shared style constants and normalization helpers.
"""

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
    return STYLE_TYPOLOGY


def is_legend_style(style):
    """
    Whether Auto Trace should render this style as a simple symbol.

    Anything that is not Line or Measured, including an empty or unrecognised
    label. It used to be a second keyword list that did not quite match
    normalize_style's, and the gap between them - "" matched neither - selected
    a fourth renderer nobody could reach from the UI, which drew a silhouette
    with no interior lines at all.
    """
    return normalize_style(style) == STYLE_TYPOLOGY
