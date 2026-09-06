#!/usr/bin/env python3
"""
Stand-ins for the Qt drawing types, recording what was drawn.

The templates are painted with QPainter, and QGIS is not available here, so
neither the tests nor a preview can use the real thing. These stand-ins accept
the same calls and record them as a command list, which is enough to check the
geometry (tests/test_template_drawing.py) and to replay it as SVG
(scripts/render_templates.py).

One implementation serves both on purpose. If the preview used its own
stand-ins it could drift from the one the tests trust, and then the picture
would be a picture of something QGIS never draws.
"""

import math

# Only the path operations the drawing code actually uses are supported;
# anything else should fail loudly rather than silently draw nothing.
__all__ = [
    "Color", "Pen", "PointF", "RectF", "Path", "PolygonF", "Qt", "Painter",
    "install", "BASE_RGB",
]

BASE_RGB = (139, 69, 19)


class Qt:
    """The Qt namespace constants the drawing code reads."""

    NoBrush = "NoBrush"
    NoPen = "NoPen"
    SolidLine = "SolidLine"
    DashLine = "DashLine"
    DotLine = "DotLine"
    DashDotLine = "DashDotLine"
    RoundCap = "RoundCap"
    FlatCap = "FlatCap"
    SquareCap = "SquareCap"
    RoundJoin = "RoundJoin"
    MiterJoin = "MiterJoin"
    BevelJoin = "BevelJoin"
    transparent = "transparent"
    AlignCenter = "AlignCenter"
    white = "white"
    black = "black"


class Color:
    """QColor: RGB plus an alpha that carries tone."""

    def __init__(self, *args):
        self.alpha = 255
        if not args:
            self._rgb = BASE_RGB
        elif isinstance(args[0], Color):
            self._rgb, self.alpha = args[0]._rgb, args[0].alpha
        elif isinstance(args[0], str):
            self._rgb = _parse_hex(args[0])
        elif len(args) >= 3:
            self._rgb = tuple(max(0, min(255, int(v))) for v in args[:3])
            if len(args) >= 4:
                self.alpha = max(0, min(255, int(args[3])))
        else:
            self._rgb = BASE_RGB

    # -- QColor API ----------------------------------------------------
    def red(self):
        return self._rgb[0]

    def green(self):
        return self._rgb[1]

    def blue(self):
        return self._rgb[2]

    def color(self):
        """QPainter.brush() returns a brush; every brush here is a colour."""
        return self

    def style(self):
        return "solid"

    def darker(self, factor=200):
        scale = 100.0 / max(1.0, float(factor))
        return self._scaled(scale)

    def lighter(self, factor=150):
        scale = float(factor) / 100.0
        return self._scaled(scale)

    def _scaled(self, scale):
        out = Color(*(max(0, min(255, int(c * scale))) for c in self._rgb))
        out.alpha = self.alpha
        return out

    # -- rendering -----------------------------------------------------
    def hex(self):
        return "#%02x%02x%02x" % self._rgb

    def opacity(self):
        return self.alpha / 255.0

    def __repr__(self):
        return f"Color{self._rgb + (self.alpha,)}"


def _parse_hex(text):
    text = str(text).strip().lstrip("#")
    if len(text) == 6:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    return BASE_RGB


class Pen:
    def __init__(self, color=None, width=1.0, style=Qt.SolidLine):
        self._color = color if isinstance(color, Color) else Color()
        self.width = float(width)
        self._style = style
        self.cap = Qt.FlatCap
        self.join = Qt.MiterJoin

    def setCapStyle(self, cap):
        self.cap = cap

    def setJoinStyle(self, join):
        self.join = join

    def color(self):
        return self._color

    def setColor(self, color):
        self._color = color if isinstance(color, Color) else Color()

    def setWidth(self, width):
        self.width = float(width)

    def setStyle(self, style):
        self._style = style

    def style(self):
        return self._style

    def dash_array(self):
        """The SVG stroke-dasharray for this pen's style, or ""."""
        unit = max(1.0, self.width)
        if self._style == Qt.DashLine:
            return f"{unit * 4:.1f} {unit * 2:.1f}"
        if self._style == Qt.DotLine:
            return f"{unit:.1f} {unit * 2:.1f}"
        if self._style == Qt.DashDotLine:
            return f"{unit * 4:.1f} {unit * 2:.1f} {unit:.1f} {unit * 2:.1f}"
        return ""


class PointF:
    """QPointF exposes x()/y() as methods, and the drawing code calls them."""

    def __init__(self, x=0.0, y=0.0):
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y


class RectF:
    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0):
        self.x, self.y, self.w, self.h = (float(v) for v in (x, y, w, h))

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
        return PointF(self.x + self.w / 2.0, self.y + self.h / 2.0)

    def corners(self):
        return [(self.x, self.y), (self.x + self.w, self.y + self.h)]


def _xy(x, y):
    if y is None:                       # a QPointF-style argument
        return x.x(), x.y()
    return float(x), float(y)


def _flatten(args):
    out = []
    for value in args:
        if isinstance(value, PointF):
            out.extend([value.x(), value.y()])
        else:
            out.append(float(value))
    return out


class Path:
    """
    QPainterPath as a command list.

    Commands are ("M"|"L"|"Q", coords...), ("ellipse"|"rect", rect) or
    ("close",). Keeping the structure - not just the points - is what lets the
    same recording be replayed as SVG.
    """

    def __init__(self, other=None):
        self.commands = []
        self.subtracted_from = None
        if isinstance(other, Path):
            self.commands = list(other.commands)
        elif isinstance(other, RectF):
            self.commands = [("rect", other)]

    def addPath(self, other):
        self.commands.extend(other.commands)

    def moveTo(self, x, y=None):
        self.commands.append(("M",) + _xy(x, y))

    def lineTo(self, x, y=None):
        self.commands.append(("L",) + _xy(x, y))

    def quadTo(self, *args):
        self.commands.append(("Q",) + tuple(_flatten(args)))

    def cubicTo(self, *args):
        self.commands.append(("C",) + tuple(_flatten(args)))

    def addEllipse(self, rect):
        self.commands.append(("ellipse", rect))

    def addRect(self, rect):
        self.commands.append(("rect", rect))

    def closeSubpath(self):
        self.commands.append(("close",))

    def subtracted(self, other):
        """
        Qt removes `other`'s area; SVG does the same with an even-odd fill
        over both subpaths, which is how the emitter renders it.
        """
        out = Path()
        out.commands = list(self.commands) + list(other.commands)
        out.subtracted_from = True
        return out

    def points(self):
        """Every coordinate the path touches, for geometry checks."""
        out = []
        for command in self.commands:
            head = command[0]
            if head in ("M", "L", "Q", "C"):
                coords = command[1:]
                out.extend(
                    (float(coords[i]), float(coords[i + 1]))
                    for i in range(0, len(coords) - 1, 2)
                )
            elif head in ("ellipse", "rect"):
                out.extend(command[1].corners())
        return out


class PolygonF:
    def __init__(self, points):
        self.pairs = [(p.x(), p.y()) for p in points]


class Painter:
    """Records draw calls with the pen and brush in effect at each one."""

    def __init__(self):
        self.calls = []      # (kind, payload, brush, pen)
        self.brushes = []    # every setBrush, for the fill-tone contract
        self._pen = Pen()
        self._brush = None
        self._clip = None
        self._stack = []

    # -- state ---------------------------------------------------------
    def pen(self):
        return self._pen

    def brush(self):
        return self._brush

    def setPen(self, pen):
        self._pen = pen if isinstance(pen, Pen) else Pen(color=_as_color(pen))

    def save(self):
        self._stack.append((self._pen, self._brush, self._clip))

    def restore(self):
        if self._stack:
            self._pen, self._brush, self._clip = self._stack.pop()

    def setClipPath(self, path):
        """Internal detail is clipped to the silhouette; record which one."""
        self._clip = path

    def setClipping(self, on):
        if not on:
            self._clip = None

    def setBrush(self, brush):
        self._brush = brush
        self.brushes.append(brush)

    def setRenderHint(self, *_args):
        pass

    def _record(self, kind, payload):
        self.calls.append((kind, payload, self._brush, self._pen, self._clip))

    # -- drawing -------------------------------------------------------
    def drawPath(self, path):
        self._record("path", path)

    def drawPolygon(self, polygon):
        self._record("polygon", polygon)

    def drawLine(self, *args):
        self._record("line", tuple(_flatten(args)))

    def drawPoint(self, *args):
        self._record("point", tuple(_flatten(args)))

    def drawRect(self, *args):
        self._record("rect", _as_rect(args))

    def drawEllipse(self, *args):
        self._record("ellipse", _as_rect(args))

    def drawArc(self, x, y, w, h, start, span):
        # Qt measures in 1/16 degree, anticlockwise from 3 o'clock.
        self._record("arc", (float(x), float(y), float(w), float(h),
                             start / 16.0, span / 16.0))

    def drawText(self, x, y, text):
        self._record("text", (float(x), float(y), str(text)))

    # -- geometry ------------------------------------------------------
    def points(self):
        """Every coordinate touched, for the geometry contracts."""
        out = []
        for kind, payload, _brush, _pen, _clip in self.calls:
            if kind == "path":
                out.extend(payload.points())
            elif kind == "polygon":
                out.extend(payload.pairs)
            elif kind in ("rect", "ellipse"):
                out.extend(payload.corners())
            elif kind == "arc":
                x, y, w, h = payload[:4]
                out.extend([(x, y), (x + w, y + h)])
            elif kind in ("line", "point"):
                coords = payload
                out.extend(
                    (coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)
                )
            elif kind == "text":
                out.append((payload[0], payload[1]))
        return out

    def fills(self):
        """The brush in effect at each draw call, in document order."""
        return [call[2] for call in self.calls]


def _as_color(value):
    return value if isinstance(value, Color) else Color()


def _as_rect(args):
    if len(args) == 1 and isinstance(args[0], RectF):
        return args[0]
    x, y, w, h = (float(v) for v in args[:4])
    return RectF(x, y, w, h)


def install(monkeypatch, module):
    """Point a module's Qt names at these stand-ins."""
    for name, value in (
        ("QColor", Color), ("QPen", Pen), ("QPainterPath", Path),
        ("QPolygonF", PolygonF), ("QPointF", PointF), ("QRectF", RectF),
        ("Qt", Qt),
    ):
        monkeypatch.setattr(module, name, value)


def arc_endpoints(x, y, w, h, start_deg, span_deg):
    """The SVG arc parameters for a Qt drawArc call."""
    rx, ry = w / 2.0, h / 2.0
    cx, cy = x + rx, y + ry
    # Qt's y axis points down, so a positive angle turns anticlockwise on
    # screen, which is a negative rotation in SVG's coordinate system.
    start = math.radians(start_deg)
    end = math.radians(start_deg + span_deg)
    x1, y1 = cx + rx * math.cos(start), cy - ry * math.sin(start)
    x2, y2 = cx + rx * math.cos(end), cy - ry * math.sin(end)
    large = 1 if abs(span_deg) > 180 else 0
    sweep = 0 if span_deg > 0 else 1
    return (x1, y1, rx, ry, large, sweep, x2, y2)
