"""
Every template is actually painted here, against a recording stand-in for
QPainter.

QGIS is not available, so the real drawing cannot be rasterised. What can be
checked is what the drawing code does: that it runs at all, that it puts
something on the canvas, and that the geometry stays inside the 256px square.
Off-canvas coordinates are the failure that a static check misses and that
looks, in QGIS, like a clipped or empty symbol.
"""

import math

import pytest

from archeoglyph.generators import template_generator as tg
from archeoglyph.generators.template_generator import TemplateGenerator

SIZE = 256
# Strokes are drawn centred on the path, so a couple of pixels of overhang is
# normal. Anything beyond this is a coordinate mistake, not a stroke width.
TOLERANCE = 12


class FakeColor:
    def __init__(self, *args):
        if len(args) >= 3:
            self._rgb = tuple(int(v) for v in args[:3])
        elif args and isinstance(args[0], FakeColor):
            self._rgb = args[0]._rgb
        else:
            self._rgb = (139, 69, 19)

    def red(self):
        return self._rgb[0]

    def green(self):
        return self._rgb[1]

    def blue(self):
        return self._rgb[2]

    def color(self):
        """QPainter.brush() returns a brush; brushes here are plain colours."""
        return self

    def style(self):
        """Brush style: a colour brush is always a solid one."""
        return "solid"

    def darker(self, _factor=200):
        return FakeColor(*(max(0, c // 2) for c in self._rgb))

    def lighter(self, _factor=150):
        return FakeColor(*(min(255, c * 3 // 2) for c in self._rgb))


class FakePen:
    def __init__(self, color=None, width=1.0, style=None):
        self._color = color if isinstance(color, FakeColor) else FakeColor()
        self.width = width
        self.style = style

    def color(self):
        return self._color

    def setWidth(self, width):
        self.width = width

    def setStyle(self, style):
        self.style = style

    def setColor(self, color):
        self._color = color


class FakePointF:
    """QPointF exposes x()/y() as methods, and the drawing code calls them."""

    def __init__(self, x=0.0, y=0.0):
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y


class FakeRectF:
    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0):
        self.x, self.y, self.w, self.h = (float(v) for v in (x, y, w, h))

    def points(self):
        return [(self.x, self.y), (self.x + self.w, self.y + self.h)]

    def left(self):
        return self.x

    def top(self):
        return self.y

    def right(self):
        return self.x + self.w

    def bottom(self):
        return self.y + self.h

    def width(self):
        return self.w

    def height(self):
        return self.h

    def center(self):
        return FakePointF(self.x + self.w / 2.0, self.y + self.h / 2.0)


class FakePath:
    """Records every point the drawing code puts into a path."""

    def __init__(self, other=None):
        self.points = list(getattr(other, "points", None) or [])
        if isinstance(other, FakeRectF):
            self.points = list(other.points())

    def addPath(self, other):
        self.points.extend(other.points)

    def _add(self, *coords):
        for i in range(0, len(coords), 2):
            self.points.append((float(coords[i]), float(coords[i + 1])))

    def moveTo(self, x, y=None):
        self._add(*_xy(x, y))

    def lineTo(self, x, y=None):
        self._add(*_xy(x, y))

    def quadTo(self, *args):
        self._add(*_flatten(args))

    def cubicTo(self, *args):
        self._add(*_flatten(args))

    def arcTo(self, *_args):
        pass

    def addEllipse(self, rect):
        self.points.extend(rect.points())

    def addRect(self, rect):
        self.points.extend(rect.points())

    def closeSubpath(self):
        pass

    def subtracted(self, other):
        merged = FakePath()
        merged.points = self.points + other.points
        return merged


def _xy(x, y):
    if y is None:          # a QPointF-style argument
        return x.x(), x.y()
    return float(x), float(y)


def _flatten(args):
    out = []
    for value in args:
        if isinstance(value, FakePointF):
            out.extend([value.x(), value.y()])
        else:
            out.append(float(value))
    return out


class RecordingPainter:
    """Collects every coordinate the drawing code touches."""

    def __init__(self):
        self.points = []
        self.brushes = []
        self.fills = []          # the brush in effect at each draw call
        self.operations = 0
        self._pen = FakePen()
        self._brush = None

    # -- state ---------------------------------------------------------
    def pen(self):
        return self._pen

    def brush(self):
        return self._brush

    def setPen(self, pen):
        self._pen = pen if isinstance(pen, FakePen) else FakePen()

    def setBrush(self, brush):
        self._brush = brush
        self.brushes.append(brush)

    def setRenderHint(self, *_args):
        pass

    # -- drawing -------------------------------------------------------
    def _record(self, *coords):
        self.operations += 1
        self.fills.append(self._brush)
        for i in range(0, len(coords), 2):
            self.points.append((float(coords[i]), float(coords[i + 1])))

    def drawLine(self, *args):
        self._record(*_flatten(args))

    def drawPoint(self, *args):
        self._record(*_flatten(args))

    def drawRect(self, *args):
        if len(args) == 1:
            self.operations += 1
            self.fills.append(self._brush)
            self.points.extend(args[0].points())
        else:
            x, y, w, h = (float(v) for v in args)
            self._record(x, y, x + w, y + h)

    def drawEllipse(self, *args):
        self.drawRect(*args)

    def drawArc(self, x, y, w, h, _start, _span):
        self._record(x, y, x + w, y + h)

    def drawPath(self, path):
        self.operations += 1
        self.fills.append(self._brush)
        self.points.extend(path.points)

    def drawPolygon(self, polygon):
        self.operations += 1
        self.fills.append(self._brush)
        self.points.extend(polygon.points)

    def drawText(self, x, y, _text):
        self._record(x, y)


class FakePolygonF:
    def __init__(self, points):
        self.points = [(p.x(), p.y()) for p in points]


@pytest.fixture
def painter(monkeypatch):
    """Swap the Qt drawing types in the module for recording stand-ins."""
    monkeypatch.setattr(tg, "QColor", FakeColor)
    monkeypatch.setattr(tg, "QPen", FakePen)
    monkeypatch.setattr(tg, "QPainterPath", FakePath)
    monkeypatch.setattr(tg, "QPolygonF", FakePolygonF)
    monkeypatch.setattr(tg, "QPointF", FakePointF)
    monkeypatch.setattr(tg, "QRectF", FakeRectF)

    class _Qt:
        NoBrush = object()
        DashLine = object()
        DotLine = object()
        SolidLine = object()
        transparent = object()
        AlignCenter = object()

    monkeypatch.setattr(tg, "Qt", _Qt)
    return RecordingPainter()


def _paint(painter, name):
    generator = TemplateGenerator.__new__(TemplateGenerator)
    generator._paint_template(painter, name, FakeColor(139, 69, 19), SIZE)
    return painter


@pytest.mark.parametrize("name", sorted(TemplateGenerator.TEMPLATE_INFO))
def test_every_template_paints_inside_the_canvas(painter, name):
    _paint(painter, name)

    assert painter.operations > 0, f"{name} drew nothing"
    assert painter.points, f"{name} produced no geometry"

    outside = [
        (x, y) for x, y in painter.points
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
    xs = [x for x, _ in painter.points]
    ys = [y for _, y in painter.points]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    assert max(width, height) >= SIZE * 0.45, (
        f"{name} only spans {width:.0f}x{height:.0f} of {SIZE}px"
    )


def test_unknown_template_falls_back_to_a_shape(painter):
    _paint(painter, "Not A Real Template")
    assert painter.operations > 0


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
        (brush for brush in painter.fills if isinstance(brush, FakeColor)), None
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
