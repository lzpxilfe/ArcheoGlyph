"""
Every template is actually painted here, against a recording stand-in for
QPainter.

QGIS is not available, so the real drawing cannot be rasterised. What can be
checked is what the drawing code does: that it runs at all, that it puts
something on the canvas, that the geometry stays inside the 256px square, and
that tone comes from opacity rather than a second colour.

The stand-ins come from scripts/qt_recorder.py, the same ones
scripts/render_templates.py replays as SVG - so what these tests check and
what the preview draws can never drift apart.
"""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import qt_recorder as qr  # noqa: E402

from archeoglyph.generators import template_generator as tg  # noqa: E402
from archeoglyph.generators.template_generator import TemplateGenerator  # noqa: E402

SIZE = 256
# Strokes are drawn centred on the path, so a couple of pixels of overhang is
# normal. Anything beyond this is a coordinate mistake, not a stroke width.
TOLERANCE = 14
BASE_RGB = qr.BASE_RGB
FakeColor = qr.Color

@pytest.fixture
def painter(monkeypatch):
    """Point the template module's Qt names at the recording stand-ins."""
    qr.install(monkeypatch, tg)
    return qr.Painter()


def _paint(painter, name):
    generator = TemplateGenerator.__new__(TemplateGenerator)
    generator._paint_template(painter, name, FakeColor(139, 69, 19), SIZE)
    return painter


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_every_template_paints_inside_the_canvas(painter, name):
    _paint(painter, name)

    assert len(painter.calls) > 0, f"{name} drew nothing"
    assert painter.points(), f"{name} produced no geometry"

    outside = [
        (x, y) for x, y in painter.points()
        if not (-TOLERANCE <= x <= SIZE + TOLERANCE)
        or not (-TOLERANCE <= y <= SIZE + TOLERANCE)
        or math.isnan(x) or math.isnan(y)
    ]
    assert not outside, f"{name} draws outside the {SIZE}px canvas at {outside[:4]}"


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_every_template_fills_a_usable_share_of_the_canvas(painter, name):
    """
    A symbol that occupies a small corner of the tile is unreadable once it is
    scaled down to a 5-10 mm marker.
    """
    _paint(painter, name)
    xs = [x for x, _ in painter.points()]
    ys = [y for _, y in painter.points()]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    assert max(width, height) >= SIZE * 0.45, (
        f"{name} only spans {width:.0f}x{height:.0f} of {SIZE}px"
    )


def test_unknown_template_falls_back_to_a_shape(painter):
    _paint(painter, "Not A Real Template")
    assert len(painter.calls) > 0


BASE_RGB = (139, 69, 19)


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_fill_tones_are_opacity_not_a_different_colour(painter, name):
    """
    QGIS replaces every `param(fill)` with one colour, so a drawing that gets
    its light and dark areas from different fill colours collapses into a flat
    silhouette the moment the user recolours it. Tone has to come from the
    alpha channel, which survives as per-element fill-opacity.
    """
    _paint(painter, name)

    offenders = {
        brush._rgb for brush in painter.brushes
        if isinstance(brush, FakeColor) and brush._rgb != BASE_RGB
    }
    # Pure white and pure black are the conventional "knock out" and "ink"
    # fills and are not meant to follow the symbol colour.
    offenders -= {(255, 255, 255), (0, 0, 0)}
    assert not offenders, (
        f"{name} fills with colours other than the symbol colour {offenders}; "
        "vary the alpha instead so QGIS recolouring keeps the tones"
    )


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_the_first_filled_shape_uses_the_symbol_colour(painter, name):
    """
    svg_builder.parametrize records the first solid fill in document order as
    the symbol's fallback colour, and symbol_manager hands that colour to the
    QGIS marker. A template whose first *drawn* shape carries some other
    colour would give the whole symbol the wrong fallback.

    The brush in effect at each draw is what reaches the SVG, so that is what
    is recorded here rather than every setBrush call.
    """
    _paint(painter, name)
    first = next(
        (brush for brush in painter.fills() if isinstance(brush, FakeColor)), None
    )
    assert first is None or first._rgb == BASE_RGB, (
        f"{name} first fills with {first._rgb} instead of the symbol colour {BASE_RGB}"
    )


def test_no_template_fills_with_a_fully_transparent_colour():
    """
    A transparent colour brush still emits a solid `fill` attribute that the
    parametriser counts as real, so it can become the symbol's fallback
    colour. Qt.NoBrush emits fill="none" and is skipped.
    """
    import inspect

    source = inspect.getsource(tg)
    assert "QColor(255, 255, 255, 0)" not in source, (
        "use Qt.NoBrush for an unfilled shape, not a fully transparent colour"
    )


# A symbol whose identity IS repetition - scales, piled stones, a posthole
# grid - needs more marks than one that is a silhouette. Everything else has
# to stay under the cap, with the reason recorded here.
MARK_CAP = 10
REPETITION_IS_THE_TYPE = {
    "Lamellar Armour": "the field of laced scales is what names it",
    "Kofun (Tsumiishizuka)": "a cairn is its piled stones",
    "Keyhole Tomb (Tsumishizuka)": "a cairn is its piled stones",
    "Stone-mounded Wooden Chamber Tomb": "the stone pile over the chamber",
    "Stone-lined Tomb": "the walling stones around the chamber",
    "Gold Crown": "three uprights, each with its own branching arms",
    "Comb-pattern Pottery": "the comb impressions are the ware",
    "Midden / Shell Mound": "a shell mound is a mass of shells",
    "Charcoal Kiln": "the charcoal inside is the point",
    "Raised-floor Building": "the building survives only as its posthole grid",
    "Plate Armour": "the rivets down each plate",
    "Dry Field": "ridge and furrow is the feature",
}


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_a_symbol_carries_only_the_marks_it_needs(painter, name):
    """
    Detail is what kills a map marker.

    The symbols this catalogue is measured against carry one to three marks:
    a silhouette, and at most the one thing that separates the type from its
    neighbours. Everything beyond that turns to grey at 5-10 mm - which is the
    size these are drawn for. A template that needs more must say why.
    """
    _paint(painter, name)
    marks = len(painter.calls)
    if name in REPETITION_IS_THE_TYPE:
        return
    assert marks <= MARK_CAP, (
        f"{name} draws {marks} marks. Reduce it to the silhouette plus what "
        f"distinguishes the type, or add it to REPETITION_IS_THE_TYPE with a "
        f"reason."
    )


def test_the_repetition_allowlist_has_no_stale_entries(painter):
    """An entry for a symbol that no longer needs it hides a real regression."""
    stale = []
    for name in sorted(REPETITION_IS_THE_TYPE):
        assert name in TemplateGenerator.TEMPLATE_INFO, f"{name} is not a template"
        recorder = qr.Painter()
        _paint(recorder, name)
        if len(recorder.calls) <= MARK_CAP:
            stale.append(f"{name} is down to {len(recorder.calls)} marks")
    assert not stale, "\n".join(stale)
