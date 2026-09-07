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

from archeoglyph.generators import icon_grid  # noqa: E402

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
    from archeoglyph.generators import icon_grid

    qr.install(monkeypatch, tg, icon_grid)
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
    "Comb-pattern Pottery": "the comb impressions are the ware",
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


# Templates rebuilt on the icon grid. The set is the conversion's progress
# report: adding a name here without moving the drawing onto the grid fails.
GRID_NATIVE = sorted(
    [name for name in TemplateGenerator.TEMPLATE_INFO
     if name.startswith(("Bronze Dagger (", "Projectile Point (",
                         "Kofun (", "Keyhole Tomb (", "Dolmen ("))]
    + ["Comb-pattern Pottery", "Plain Coarse Pottery", "Red Burnished Pottery",
       "Black Burnished Long-necked Jar", "Soft Grey Pottery (Wajil)",
       "Hard Grey Stoneware (Gyeongjil)", "Mounted Dish (Gobae)",
       "Storage Jar (Ho)", "Steamer (Siru)", "Celadon", "Buncheong Ware",
       "White Porcelain", "Onggi Jar",
       "Stone Cist Tomb", "Stone-lined Tomb", "Wooden Coffin Tomb",
       "Wooden Chamber Tomb", "Jar Coffin Tomb",
       "Stone-mounded Wooden Chamber Tomb", "Corridor-style Stone Chamber Tomb",
       "Earthen Mounded Tomb", "Ditch-encircled Tomb", "Earthen Pit Tomb",
       "Pit Dwelling (Round)", "Pit Dwelling (Square)",
       "Pit Dwelling (Protruding Entrance)", "Pit Dwelling (Twin-room)",
       "Raised-floor Building", "Cooking Stove (Kamado)", "Ondol Heating Flue",
       "Pottery Kiln", "Roof Tile Kiln", "Iron Smelting Feature",
       "Charcoal Kiln", "Paddy Field", "Dry Field", "Earthen Rampart Fortress",
       "Stone Rampart Fortress", "Mountain Fortress", "Palisade",
       "Encircling Ditch", "Beacon Station", "Water Collection Basin",
       "Handaxe", "Chopper", "Tanged Point", "Microblade Core",
       "Polished Stone Dagger", "Semi-lunar Stone Knife", "Stone Hoe",
       "Grinding Slab and Muller", "Stone Arrowhead", "Net Sinker",
       "Coarse-lined Bronze Mirror", "Fine-lined Bronze Mirror",
       "Bronze Rattle", "Bronze Bell", "Iron Sword", "Iron Spearhead",
       "Iron Arrowhead", "Iron Axe", "Iron Ard", "Iron Sickle", "Plate Armour",
       "Lamellar Armour", "Horse Bit", "Stirrup", "Iron Ingot",
       "Comma-shaped Jade (Gogok)", "Tubular Jade Bead (Gwanok)",
       "Glass Bead", "Gold Earring", "Gold Crown", "Belt Fitting Set",
       "Wooden Document Slip (Mokgan)", "Round Roof-end Tile",
       "Eaves Roof Tile", "Floor Brick", "Inkstone", "Clay Figurine",
       "Ridge-end Roof Ornament (Chimi)", "Building Foundation Stone",
       "North Arrow (Map Standard)", "Scale Bar (Map Standard)",
       "Harris Matrix Context", "Stratigraphic Unit", "Survey Point",
       "Find Spot", "Trench", "Datum Point", "Photo Point", "Grid Corner",
       "Sample Location", "Ash Layer", "Excavation Area", "Test Pit",
       "Pottery", "Stone Tool", "Arrowhead", "Scraper", "Bronze Artifact",
       "Iron Artifact", "Chisel", "Ornament", "Bead", "Bracelet / Ring",
       "Coin", "Seal / Stamp", "Spindle Whorl", "Bone Tool", "Needle / Pin",
       "Animal Remains", "Weapon", "Blade",
       "Pottery Rim Sherd (Section)", "Pottery Base Sherd (Section)",
       "Pottery Body Sherd (Section)", "Bronze Sword", "Bronze Dagger-axe",
       "Bronze Spear", "Fortress / Castle", "Gate", "Tower",
       "Dwelling / House", "Workshop", "Temple / Shrine", "Tomb",
       "Mound / Barrow", "Kiln / Furnace", "Well", "Wall / Rampart",
       "Pit", "Storage Pit", "Posthole", "Road / Pavement", "Bridge",
       "Human Remains", "Skeleton", "Burial", "Cremation Burial",
       "Hearth / Fire Pit", "Burnt Area", "Midden / Shell Mound",
       "Ditch / Moat", "Canal / Water Channel", "Stone Alignment", "Dolmen",
       "Rock Art", "Standing Stone", "Terrace"]
)


@pytest.mark.parametrize("name", GRID_NATIVE)
def test_grid_native_templates_stay_on_the_grid(painter, name):
    """
    Every coordinate must land on the half-unit grid.

    Freehand coordinates are why 188 symbols had subtly different wall angles,
    margins and centres - the thing that stopped them looking like one hand
    drew them. Snapping is what a design grid actually is, so it is checked
    rather than trusted.
    """
    _paint(painter, name)
    step = SIZE / (icon_grid.UNITS * 2)      # half a unit, in pixels
    off = sorted({
        (x, y) for x, y in painter.points()
        if abs(x / step - round(x / step)) > 1e-6
        or abs(y / step - round(y / step)) > 1e-6
    })
    assert not off, f"{name} has {len(off)} coordinates off the grid, e.g. {off[:3]}"


@pytest.mark.parametrize("name", GRID_NATIVE)
def test_grid_native_templates_respect_the_safe_area(painter, name):
    """
    Artwork lives inside the safe area, so the whole set shares a margin.

    Half the outline sits outside the path, so the tolerance is that half
    stroke and no more.
    """
    _paint(painter, name)
    unit = SIZE / icon_grid.UNITS
    slack = icon_grid.OUTLINE * unit / 2.0
    low = icon_grid.MARGIN * unit - slack
    high = SIZE - low
    outside = [(x, y) for x, y in painter.points() if not (low <= x <= high and low <= y <= high)]
    assert not outside, (
        f"{name} draws outside the {icon_grid.MARGIN}-unit safe area at {outside[:3]}"
    )


# ── the house style: weight contrast, and corners that turn ──────────────

def _anchor_rings(path):
    """
    The corner points of a recorded path, per closed run.

    Control points are not corners, so a quad contributes only its endpoint;
    otherwise every curve would look like a chain of sharp turns.
    """
    rings, current, pos = [], [], (0.0, 0.0)
    for command in getattr(path, "commands", ()):
        head = command[0]
        if head == "M":
            if len(current) > 2:
                rings.append(current)
            pos = (command[1], command[2])
            current = [pos]
        elif head == "L":
            pos = (command[1], command[2])
            current.append(pos)
        elif head == "Q":
            pos = (command[3], command[4])
            current.append(pos)
        elif head == "C":
            pos = (command[5], command[6])
            current.append(pos)
        elif head == "close":
            if len(current) > 2:
                rings.append(current)
            current = [pos]
    if len(current) > 2:
        rings.append(current)
    return rings


#: The floor for a corner in a filled silhouette. Anything sharper is a spike
#: rather than a point, and a set of spikes is what makes a catalogue look
#: hostile at marker size. Measured only across edges long enough to have a
#: direction - a 2-unit chamfer is not a corner.
MIN_CORNER_DEGREES = 40.0
MIN_EDGE_PX = 10.0


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_no_silhouette_comes_to_a_spike(painter, name):
    """
    Corners turn; they do not come to a point.

    Grid.poly and the faceted Grid.symmetric cut every corner back and turn it
    through a quad, so this holds by construction - a north arrow that used to
    close at 19 degrees now noses over. A regression here means a drawing
    built its outline by hand instead of going through the grid.
    """
    _paint(painter, name)
    for kind, payload, brush, pen, clip in painter.calls:
        if not isinstance(brush, FakeColor):
            continue
        for ring in _anchor_rings(payload):
            count = len(ring)
            for index in range(count):
                before, corner, after = (
                    ring[index - 1], ring[index], ring[(index + 1) % count],
                )
                first = (before[0] - corner[0], before[1] - corner[1])
                second = (after[0] - corner[0], after[1] - corner[1])
                one = math.hypot(*first)
                two = math.hypot(*second)
                if one < MIN_EDGE_PX or two < MIN_EDGE_PX:
                    continue
                cosine = (first[0] * second[0] + first[1] * second[1]) / (one * two)
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                assert angle >= MIN_CORNER_DEGREES, (
                    f"{name} has a {angle:.0f} degree corner at "
                    f"({corner[0]:.0f}, {corner[1]:.0f}); build the outline "
                    f"with Grid.poly or Grid.symmetric so it turns"
                )


def test_the_outline_is_at_least_twice_its_internal_lines():
    """
    The reader has to be able to tell the silhouette from the detail inside
    it. At 3 units against 2 the two weights were close enough to read as one,
    which is what made the set look flat and scratchy.
    """
    assert icon_grid.OUTLINE >= 2.0 * icon_grid.DETAIL, (
        f"outline {icon_grid.OUTLINE} against detail {icon_grid.DETAIL} is "
        "not enough contrast to separate a shape from what is drawn in it"
    )


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_every_stroke_is_one_of_the_houses_weights(painter, name):
    """
    Two steps, plus deliberate heavy strokes measured in grid units.

    _weight used to scale anything above the outline step by a bare 1.3, so a
    bracelet asking for its heaviest possible ring got 9.1px - thinner than an
    ordinary 12px outline, and barely above a detail line. A stroke that lands
    between the steps is either that bug or a drawing inventing a third
    weight.
    """
    _paint(painter, name)
    unit = SIZE / icon_grid.UNITS
    allowed = {round(tg.DETAIL_WIDTH, 2), round(tg.OUTLINE_WIDTH, 2)}
    stray = set()
    for kind, payload, brush, pen, clip in painter.calls:
        width = round(float(getattr(pen, "width", 0.0)), 2)
        if width in allowed or width <= 1.0:
            continue
        # A heavy stroke is a shape drawn as a line: whole units, and never
        # lighter than the outline it has to hold its own against.
        if width >= tg.OUTLINE_WIDTH and abs(width / unit - round(width / unit)) < 1e-6:
            continue
        stray.add(width)
    assert not stray, (
        f"{name} strokes at {sorted(stray)}px; the house weights are "
        f"{sorted(allowed)}px plus whole-unit heavy strokes"
    )
