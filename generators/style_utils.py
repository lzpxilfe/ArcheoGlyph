# -*- coding: utf-8 -*-
"""
Shared style constants and normalization helpers.
"""

STYLE_COLORED = "Colored"
STYLE_TYPOLOGY = "Typology"
STYLE_LINE = "Line"
STYLE_MEASURED = "Measured"

STYLE_OPTIONS = [
    STYLE_COLORED,
    STYLE_TYPOLOGY,
    STYLE_LINE,
    STYLE_MEASURED,
]


def normalize_style(style):
    """
    Normalize any style label into one of the canonical style constants.
    """
    text = str(style or "").strip()
    low = text.lower()

    if "measured" in low or "publication" in low:
        return STYLE_MEASURED
    if "typology" in low or "catalog" in low or "symbolic" in low:
        return STYLE_TYPOLOGY
    if "line" in low:
        return STYLE_LINE
    return STYLE_COLORED
