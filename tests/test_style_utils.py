from archeoglyph.generators.style_utils import (
    STYLE_LINE,
    STYLE_MEASURED,
    STYLE_TYPOLOGY,
    is_legend_style,
    normalize_style,
)


def test_normalize_style_maps_labels():
    assert normalize_style("Measured") == STYLE_MEASURED
    assert normalize_style("Line") == STYLE_LINE
    assert normalize_style("Simple Symbol") == STYLE_TYPOLOGY
    assert normalize_style("Colored") == STYLE_TYPOLOGY
    assert normalize_style(None) == STYLE_TYPOLOGY


def test_is_legend_style():
    assert is_legend_style("Simple Symbol")
    assert is_legend_style("Typology")
    assert not is_legend_style("Line")
    assert not is_legend_style("Measured")
