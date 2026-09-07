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

import numpy as np
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


# ── legend-size legibility: no two symbols may collapse into each other ──

#: A symbol is only worth drawing if it is not some other symbol. Counting
#: marks used to stand in for this - the cap said "one to three marks, a
#: silhouette and at most the one thing that separates the type" - but a
#: count cannot see whether the result is distinguishable, and it turned out
#: to forbid exactly the detail that makes types tell apart. The artefacts
#: obeyed it and came out empty; the features quietly ignored it and read
#: well. So measure the thing itself: rasterise each symbol at the size a
#: legend actually draws it, and fail when two of them are the same picture.
LEGEND_PX = 64

#: Calibrated against the catalogue rather than guessed. Across all 17,578
#: pairs the median likeness is 59% and the 99.5th percentile is 92% - family
#: resemblance between, say, two jars lives down there. Everything above 95%
#: is a separate, tight cluster of symbols that really are one picture, so
#: that is where the line goes.
MAX_OVERLAP = 0.95

#: 전방후원분 is a Keyhole Tomb is a Kofun (Zenpokouen): the catalogue carries
#: the same monuments under a Korean and a Japanese naming scheme, so those
#: pairs SHOULD draw identically. Covering the same pixels is the correct
#: answer there, not a drawing defect, and redrawing one of them differently
#: would invent a distinction that archaeology does not make. Which name
#: survives is a terminology decision for the catalogue, not for this test.
SAME_MONUMENT_UNDER_TWO_NAMES = (
    ("Keyhole Tomb", "Kofun"),
    ("Kofun (Normal)", "Kofun (Zenpokouen)"),
)


def _is_the_same_monument(a, b):
    for left, right in SAME_MONUMENT_UNDER_TWO_NAMES:
        if {a, b} == {left, right}:
            return True
        if a.startswith(left) and b.startswith(right):
            return True
        if b.startswith(left) and a.startswith(right):
            return True
    return False


#: The drawing code is calibrated for a 256px canvas - _UNIT, DETAIL_WIDTH
#: and OUTLINE_WIDTH are all fixed against it - so painting straight into a
#: 64px grid scales the coordinates but leaves a 12px pen, which paints every
#: symbol into one blob and hides exactly the detail this is here to measure.
#: Paint at the size the code means, then box-filter down to the size a legend
#: shows, which is what QGIS does with the SVG.
PAINT_PX = SIZE
DOWNSAMPLE = PAINT_PX // LEGEND_PX


def _appearance(name, size=LEGEND_PX):
    """
    What a symbol looks like at legend size, as a size x size intensity map.

    Not a silhouette. A binary occupancy map cannot see interior detail at
    all: the mirror's concentric rings sit inside its own filled disc, so
    every pixel they touch is already set, and the mirror measures as
    identical to a plain roof tile when on screen the two are obviously
    different. Intensity carries the tone instead - a fill contributes its
    alpha, a stroke its full weight, darker winning where they meet, which is
    how the house style composes.
    """
    recorder = qr.Painter()
    TemplateGenerator.__new__(TemplateGenerator)._paint_template(
        recorder, name, FakeColor(139, 69, 19), float(PAINT_PX))
    canvas = np.zeros((PAINT_PX, PAINT_PX), dtype=float)
    centres = np.arange(PAINT_PX) + 0.5
    px, py = np.meshgrid(centres, centres)

    for call in recorder.calls:
        brush, pen = call[2], call[3]
        width = float(getattr(pen, "width", 0.0))
        alpha = (brush.alpha / 255.0) if isinstance(brush, FakeColor) else 0.0
        for polygon in _flatten(call[1]):
            points = np.asarray(polygon, dtype=float)
            if len(points) < 2:
                continue
            if alpha > 0.0 and len(points) >= 3:
                inside = np.zeros((PAINT_PX, PAINT_PX), dtype=bool)
                previous = len(points) - 1
                for index in range(len(points)):
                    x1, y1 = points[index]
                    x2, y2 = points[previous]
                    straddles = (y1 > py) != (y2 > py)
                    span = y2 - y1 or 1e-9
                    inside ^= straddles & (px < (x2 - x1) * (py - y1) / span + x1)
                    previous = index
                canvas = np.maximum(canvas, inside * alpha)
            if width > 0.0:
                canvas = np.maximum(canvas, _stroked(points, width, px, py) * 1.0)

    step = DOWNSAMPLE
    return canvas.reshape(size, step, size, step).mean(axis=(1, 3))


def _stroked(points, width, px, py):
    """Pixels within half a pen width of the polyline."""
    half = max(width, 0.6) / 2.0
    covered = np.zeros(px.shape, dtype=bool)
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        dx, dy = x2 - x1, y2 - y1
        length_squared = dx * dx + dy * dy
        if length_squared < 1e-12:
            covered |= (px - x1) ** 2 + (py - y1) ** 2 <= half * half
            continue
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / length_squared, 0.0, 1.0)
        covered |= (px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2 <= half * half
    return covered


def _overlap(a, b):
    """
    How alike two symbols look, 0 to 1.

    Jaccard over intensities rather than over a mask, so it reduces to the
    familiar area IoU for two flat silhouettes but still registers a ring
    drawn across a body.
    """
    union = float(np.maximum(a, b).sum())
    return float(np.minimum(a, b).sum()) / union if union else 0.0


#: Painting 188 symbols is not free, so the maps are built once and kept.
#: They cannot be a module-scoped fixture: painting needs the Qt stand-ins,
#: which "painter" installs per test through monkeypatch.
_APPEARANCE_CACHE = {}


@pytest.fixture
def appearances(painter):
    if not _APPEARANCE_CACHE:
        _APPEARANCE_CACHE.update(
            (name, _appearance(name))
            for name in sorted(TemplateGenerator.TEMPLATE_INFO))
    return _APPEARANCE_CACHE


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_no_symbol_is_another_symbol_at_legend_size(name, appearances):
    """
    Two symbols that cover the same pixels at 64px are one symbol with two
    labels, whatever the drawing code intended. The typology series are where
    this bites: eleven bronze daggers whose types differ only in a hairline
    ridge are eleven identical leaves on a map.
    """
    mine = appearances[name]
    worst, rival = 0.0, None
    for other, theirs in appearances.items():
        if other == name or _is_the_same_monument(name, other):
            continue
        score = _overlap(mine, theirs)
        if score > worst:
            worst, rival = score, other
    assert worst <= MAX_OVERLAP, (
        f"{name} and {rival} cover the same {worst * 100:.0f}% of the tile at "
        f"{LEGEND_PX}px. What separates the two types has to be visible at "
        f"legend size - widen the silhouette difference, or give the type its "
        f"defining feature as an area rather than a hairline"
    )


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
        # A heavy stroke is a shape drawn as a line - a bracelet, a ring
        # ditch - so it is measured on the same grid as everything else: a
        # multiple of the half-unit snap, and never lighter than the outline
        # it has to hold its own against.
        steps = width / (unit * icon_grid.SNAP)
        if width >= tg.OUTLINE_WIDTH and abs(steps - round(steps)) < 1e-6:
            continue
        stray.add(width)
    assert not stray, (
        f"{name} strokes at {sorted(stray)}px; the house weights are "
        f"{sorted(allowed)}px plus heavy strokes on the half-unit grid"
    )


# ── optical weight: every symbol carries a comparable amount of ink ──────

def _flatten(path):
    """Sub-polygons of a recorded path, curves sampled so area is honest."""
    polygons, current, pos = [], [], (0.0, 0.0)
    for command in getattr(path, "commands", ()):
        head = command[0]
        if head == "M":
            if len(current) > 1:
                polygons.append(current)
            pos = (command[1], command[2])
            current = [pos]
        elif head == "L":
            pos = (command[1], command[2])
            current.append(pos)
        elif head == "Q":
            (x0, y0) = pos
            cx, cy, x1, y1 = command[1], command[2], command[3], command[4]
            for step in range(1, 7):
                t = step / 6.0
                u = 1.0 - t
                current.append((u*u*x0 + 2*u*t*cx + t*t*x1,
                                u*u*y0 + 2*u*t*cy + t*t*y1))
            pos = (x1, y1)
        elif head in ("rect", "ellipse"):
            box = command[1]
            x, y, w, h = box.x(), box.y(), box.width(), box.height()
            if head == "rect":
                polygons.append([(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)])
            else:
                cx, cy, rx, ry = x + w/2, y + h/2, w/2, h/2
                polygons.append([
                    (cx + rx*math.cos(2*math.pi*i/20), cy + ry*math.sin(2*math.pi*i/20))
                    for i in range(21)
                ])
        elif head == "close":
            if len(current) > 1:
                polygons.append(current + [current[0]])
            current = [pos]
    if len(current) > 1:
        polygons.append(current)
    return polygons


def _ink(painter):
    """
    How much ink a symbol lays on its tile, in percent.

    Overlapping fills are summed rather than unioned, so a mark drawn over a
    body counts twice in the region they share. That makes this a measure of
    ink laid down rather than ink visible, and it reads a little high for the
    layered symbols - which is the conservative direction for a ceiling.
    """
    total = 0.0
    for kind, payload, brush, pen, clip in painter.calls:
        width = float(getattr(pen, "width", 0.0))
        if kind == "line" and isinstance(payload, tuple) and len(payload) == 4:
            total += math.hypot(payload[2] - payload[0],
                                payload[3] - payload[1]) * width
            continue
        if not hasattr(payload, "commands"):
            continue
        polygons = _flatten(payload)
        if isinstance(brush, FakeColor):
            for polygon in polygons:
                doubled = 0.0
                count = len(polygon)
                for i in range(count):
                    x1, y1 = polygon[i]
                    x2, y2 = polygon[(i + 1) % count]
                    doubled += x1*y2 - x2*y1
                total += abs(doubled) / 2.0 * (brush.alpha / 255.0)
        for polygon in polygons:
            total += sum(
                math.hypot(polygon[i+1][0] - polygon[i][0],
                           polygon[i+1][1] - polygon[i][1])
                for i in range(len(polygon) - 1)
            ) * width * 0.5
    return total / (SIZE * SIZE) * 100.0



# ── detail parity: an artefact is worth as much drawing as a feature ─────

#: The interior detail a symbol carries, as a share of its tile - everything
#: after the silhouette. The features were always drawn properly: a shrine
#: gets a roof and a door, a well gets its rings, a hearth gets its kerb. The
#: artefacts were not, and the mark cap that used to stand here is why - they
#: obeyed it and came out as bare outlines at a third of the features' detail.
#:
#: Held as a ratio rather than an absolute so it measures the imbalance that
#: actually showed on the legend, and so tightening the house style overall
#: does not silently re-open the gap.
MIN_ARTIFACT_DETAIL_RATIO = 0.60


def _interior_detail(name):
    """Ink laid down after the silhouette, as a percentage of the tile."""
    recorder = qr.Painter()
    _paint(recorder, name)
    total = 0.0
    for index, call in enumerate(recorder.calls):
        if index == 0:
            continue
        width = float(getattr(call[3], "width", 0.0))
        for polygon in _flatten(call[1]):
            total += sum(
                math.hypot(polygon[i+1][0] - polygon[i][0],
                           polygon[i+1][1] - polygon[i][1])
                for i in range(len(polygon) - 1)
            ) * max(width, 0.6)
    return total / (SIZE * SIZE) * 100.0


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def test_an_artifact_carries_as_much_detail_as_a_feature(painter):
    by_category = {}
    for name, info in TemplateGenerator.TEMPLATE_INFO.items():
        by_category.setdefault(info.get("category"), []).append(
            _interior_detail(name))
    artifacts = _median(by_category["artifacts"])
    features = _median(by_category["features"])
    assert artifacts >= features * MIN_ARTIFACT_DETAIL_RATIO, (
        f"artefacts carry {artifacts:.1f}% interior detail against the "
        f"features' {features:.1f}% - the artefacts are being drawn as bare "
        f"silhouettes while everything around them gets its distinguishing "
        f"parts"
    )


def test_no_artifact_is_left_as_a_bare_silhouette(painter):
    bare = sorted(
        name for name, info in TemplateGenerator.TEMPLATE_INFO.items()
        if info.get("category") == "artifacts" and _interior_detail(name) == 0.0)
    assert not bare, (
        "these artefacts draw an outline and nothing else, so nothing on them "
        f"says which artefact it is: {bare}"
    )


#: Every symbol was drawn to fill the same safe area, which is not the same as
#: carrying the same visual weight: a solid disc that fills its tile and a
#: needle that spans it are the same size and nowhere near the same ink. Left
#: alone the catalogue ran from 9 percent to 66 - a seven-fold spread, so on a
#: legend the roof tile read as a block and the horse bit went missing.
#:
#: The blade series are corrected up and the vessels down by one factor each
#: (BLADE_SCALE, VESSEL_SCALE), which keeps the differences inside a family.
INK_FLOOR = 12.0
INK_CEILING = 50.0


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_symbols_carry_a_comparable_weight_of_ink(painter, name):
    _paint(painter, name)
    ink = _ink(painter)
    assert INK_FLOOR <= ink <= INK_CEILING, (
        f"{name} covers {ink:.0f}% of its tile; the set is held to "
        f"{INK_FLOOR:.0f}-{INK_CEILING:.0f}% so no symbol shouts and none "
        f"goes missing next to the rest"
    )
