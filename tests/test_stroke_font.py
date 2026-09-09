# -*- coding: utf-8 -*-
"""The stroke font that writes a typology code into a symbol."""

import pytest

from archeoglyph.generators.autotrace import stroke_font as sf


def test_every_character_a_typology_code_uses_has_strokes():
    """
    A code is an identifier: if one character of it cannot be drawn, the code
    means something else. So the whole alphanumeric set has to be cut, not
    only the letters the examples happen to use.
    """
    for character in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-./()":
        assert sf.supported(character), character
        lines = sf.text_polylines(character, 20.0)
        assert lines, f"{character!r} is claimed as supported but draws nothing"
        for line in lines:
            assert len(line) >= 2, f"{character!r} has a stroke with one point"

    for character in "abcdefghijklmnopqrstuvwxyz":
        assert sf.supported(character), character
        assert sf.text_polylines(character, 20.0), character


def test_case_survives_because_case_is_meaning():
    """
    In a typology code IIa is not IIA. A three-by-five cell has no room for
    real ascenders, so lower case is cut as small capitals - which still has
    to come out visibly different, or the distinction is lost in the symbol.
    """
    upper = sf.text_polylines("A", 20.0)
    lower = sf.text_polylines("a", 20.0)
    assert upper != lower, "upper and lower case drew the same glyph"

    def _height(lines):
        ys = [pt[1] for line in lines for pt in line]
        return max(ys) - min(ys)

    assert _height(lower) < _height(upper) * 0.85
    # Both sit on the same baseline, so a mixed code reads as one line of type.
    assert max(pt[1] for line in lower for pt in line) == pytest.approx(
        max(pt[1] for line in upper for pt in line), abs=0.001)


def test_a_character_this_font_cannot_draw_is_left_out():
    """
    Not substituted, not boxed: a Korean syllable or a symbol nobody cut has
    no glyph, and inventing one would put a mark in the symbol that means
    nothing. The rest of the code still sets.
    """
    assert not sf.supported("가")
    assert sf.text_polylines("가", 20.0) == []

    plain = sf.text_polylines("IIa", 20.0)
    with_junk = sf.text_polylines("II가a", 20.0)
    assert len(with_junk) == len(plain), "the unknown character took up space"


def test_the_extent_matches_what_is_drawn():
    """The caller places the code from text_extent, so it has to be true."""
    for code in ("IIa2b", "III-2", "Aa1", "0"):
        width, height = sf.text_extent(code, 30.0)
        lines = sf.text_polylines(code, 30.0, origin=(7.0, 11.0))
        xs = [pt[0] for line in lines for pt in line]
        ys = [pt[1] for line in lines for pt in line]
        assert min(xs) == pytest.approx(7.0, abs=0.001)
        assert max(xs) == pytest.approx(7.0 + width, abs=0.001)
        assert max(ys) == pytest.approx(11.0 + height, abs=0.001)

    assert sf.text_extent("", 30.0) == (0.0, 0.0)
    assert sf.text_extent("가", 30.0) == (0.0, 0.0)
