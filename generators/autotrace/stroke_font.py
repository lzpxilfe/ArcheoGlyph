# -*- coding: utf-8 -*-
"""
A stroke font, for writing a typology code into a symbol.

Drawn as polylines rather than as SVG ``<text>``, for three reasons that are
all about this project rather than about typography:

* ``svg_sanitize.ALLOWED_TAGS`` has no ``text``, so a text element would be
  stripped out of any SVG that passes through the sanitiser.
* ``svg_builder.geometry_bbox`` measures geometry, so a text element would sit
  outside the box the symbol is cropped and squared to.
* QGIS renders an SVG marker through Qt, which substitutes fonts. A symbol has
  to look the same on every machine that opens the project.

And one that is about the material: every symbol in this plugin is original
vector geometry drawn in code, which is what lets it be published at all.

The glyphs are cut on a three-by-five grid - x in 0..2, y in 0..4, y down -
which is the smallest grid that carries the whole alphabet legibly. Lower case
is drawn as small capitals sitting on the baseline: a three-by-five cell has no
room for real ascenders and descenders, and in a typology code the case is
meaning, so ``IIa`` has to look different from ``IIA`` even when both are cut
from the same glyph.
"""

#: Glyph cell, in grid units.
CELL_W = 2.0
CELL_H = 4.0

#: Pen travel from one glyph's left edge to the next. Monospaced: a code is
#: read character by character, not as a word.
ADVANCE = 3.0

#: Small-capital height, as a share of the capital height. Low enough to read
#: as lower case at a glance, high enough to stay legible at legend size.
SMALL_CAPS = 0.66

_GLYPHS = {
    "0": [[(0, 0), (2, 0), (2, 4), (0, 4), (0, 0)]],
    "1": [[(0, 1), (1, 0), (1, 4)], [(0, 4), (2, 4)]],
    "2": [[(0, 0), (2, 0), (2, 2), (0, 2), (0, 4), (2, 4)]],
    "3": [[(0, 0), (2, 0), (2, 4), (0, 4)], [(0, 2), (2, 2)]],
    "4": [[(0, 0), (0, 2), (2, 2)], [(2, 0), (2, 4)]],
    "5": [[(2, 0), (0, 0), (0, 2), (2, 2), (2, 4), (0, 4)]],
    "6": [[(2, 0), (0, 0), (0, 4), (2, 4), (2, 2), (0, 2)]],
    "7": [[(0, 0), (2, 0), (2, 4)]],
    "8": [[(0, 0), (2, 0), (2, 4), (0, 4), (0, 0)], [(0, 2), (2, 2)]],
    "9": [[(2, 2), (0, 2), (0, 0), (2, 0), (2, 4)]],
    "A": [[(0, 4), (0, 1), (1, 0), (2, 1), (2, 4)], [(0, 2), (2, 2)]],
    "B": [[(0, 4), (0, 0), (2, 0), (2, 2), (0, 2)], [(2, 2), (2, 4), (0, 4)]],
    "C": [[(2, 0), (0, 0), (0, 4), (2, 4)]],
    "D": [[(0, 0), (1, 0), (2, 1), (2, 3), (1, 4), (0, 4), (0, 0)]],
    "E": [[(2, 0), (0, 0), (0, 4), (2, 4)], [(0, 2), (1, 2)]],
    "F": [[(2, 0), (0, 0), (0, 4)], [(0, 2), (1, 2)]],
    "G": [[(2, 0), (0, 0), (0, 4), (2, 4), (2, 2), (1, 2)]],
    "H": [[(0, 0), (0, 4)], [(2, 0), (2, 4)], [(0, 2), (2, 2)]],
    "I": [[(0, 0), (2, 0)], [(1, 0), (1, 4)], [(0, 4), (2, 4)]],
    "J": [[(0, 0), (2, 0)], [(2, 0), (2, 3), (1, 4), (0, 3)]],
    "K": [[(0, 0), (0, 4)], [(2, 0), (0, 2), (2, 4)]],
    "L": [[(0, 0), (0, 4), (2, 4)]],
    "M": [[(0, 4), (0, 0), (1, 2), (2, 0), (2, 4)]],
    "N": [[(0, 4), (0, 0), (2, 4), (2, 0)]],
    "O": [[(0, 0), (2, 0), (2, 4), (0, 4), (0, 0)]],
    "P": [[(0, 4), (0, 0), (2, 0), (2, 2), (0, 2)]],
    "Q": [[(0, 0), (2, 0), (2, 4), (0, 4), (0, 0)], [(1, 3), (2, 4)]],
    "R": [[(0, 4), (0, 0), (2, 0), (2, 2), (0, 2)], [(1, 2), (2, 4)]],
    "S": [[(2, 0), (0, 0), (0, 2), (2, 2), (2, 4), (0, 4)]],
    "T": [[(0, 0), (2, 0)], [(1, 0), (1, 4)]],
    "U": [[(0, 0), (0, 4), (2, 4), (2, 0)]],
    "V": [[(0, 0), (1, 4), (2, 0)]],
    "W": [[(0, 0), (0, 4), (1, 2), (2, 4), (2, 0)]],
    "X": [[(0, 0), (2, 4)], [(2, 0), (0, 4)]],
    "Y": [[(0, 0), (1, 2), (2, 0)], [(1, 2), (1, 4)]],
    "Z": [[(0, 0), (2, 0), (0, 4), (2, 4)]],
    "-": [[(0, 2), (2, 2)]],
    ".": [[(1, 3.6), (1, 4)]],
    "/": [[(2, 0), (0, 4)]],
    "(": [[(2, 0), (1, 1), (1, 3), (2, 4)]],
    ")": [[(0, 0), (1, 1), (1, 3), (0, 4)]],
    " ": [],
}


def supported(character):
    """Whether this stroke font can draw ``character``."""
    if not character:
        return False
    return character in _GLYPHS or character.upper() in _GLYPHS


def _glyph_strokes(character):
    """The strokes for ``character``, or [] when it travels without marking."""
    glyph = _GLYPHS.get(character)
    if glyph is None:
        glyph = _GLYPHS.get(character.upper())
    return glyph or []


def _glyph_scale(character):
    """1.0 for a capital or a digit, SMALL_CAPS for the lower case cut from it."""
    return 1.0 if character in _GLYPHS else SMALL_CAPS


def _inked(text):
    """``(index within the pen's travel, character)`` for every drawn glyph."""
    return list(enumerate(ch for ch in (text or "") if supported(ch)))


def text_extent(text, height):
    """
    ``(width, height)`` of the ink ``text`` lays down at capital height
    ``height``, measured from the first mark to the last.

    Two things make this narrower than the pen's travel. A small capital is
    narrower than a capital, so a code ending in lower case stops short of the
    advance. And a space travels without marking: counting it as a glyph cell
    reported ``"A "`` as 50 units wide where the ink is 20, and reserved a box
    for ``" A"`` thirty units left of where the glyph lands. The caller centres
    the code on this number and tests that box against the silhouette, so it
    has to be the ink.
    """
    marks = [(i, ch) for i, ch in _inked(text) if _glyph_strokes(ch)]
    if not marks:
        return (0.0, 0.0)
    unit = float(height) / CELL_H
    first, _ = marks[0]
    last, last_ch = marks[-1]
    width = (ADVANCE * (last - first) + CELL_W * _glyph_scale(last_ch)) * unit
    return (width, float(height))


def text_polylines(text, height, origin=(0.0, 0.0)):
    """
    ``text`` as polylines, capital height ``height``, top-left at ``origin``.

    Characters this font cannot draw are skipped without advancing the pen -
    a typology code is a short identifier, and inventing a box for a character
    nobody can read says less than leaving it out.
    """
    if not text or not (height > 0):
        return []
    unit = float(height) / CELL_H
    ox, oy = (float(origin[0]), float(origin[1]))
    baseline = oy + float(height)

    marks = [i for i, ch in _inked(text) if _glyph_strokes(ch)]
    if not marks:
        return []
    # The origin is where the ink starts, so leading spaces do not shift the
    # code away from the box the caller reserved for it.
    lines = []
    pen = -ADVANCE * unit * marks[0]
    for character in text:
        glyph = _GLYPHS.get(character)
        if glyph is None:
            glyph = _GLYPHS.get(character.upper())
        if glyph is None:
            continue
        scale = _glyph_scale(character)
        # Small capitals sit on the baseline rather than hanging from the cap
        # line, so a mixed code reads as one line of type.
        top = baseline - (float(height) * scale)
        for stroke in glyph:
            lines.append([[ox + pen + (x * unit * scale),
                           top + (y * unit * scale)] for x, y in stroke])
        pen += ADVANCE * unit
    return lines
