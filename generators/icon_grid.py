# -*- coding: utf-8 -*-
"""
The 64-unit grid every symbol is drawn on.

These are icons, not illustrations: they sit on a map at 5-10 mm and have to
read as one set. Drawing each one freehand at 256px with whatever coordinates
suited it gave 188 symbols with subtly different weights, margins and centres,
which is why they never looked like a family.

So the canvas is 64 units square, every coordinate snaps to a half unit, and
the shapes come from a small shared vocabulary. A dagger and a spearhead are
the same ``symmetric`` call with different numbers; that is what makes them
look related without anyone tuning them to match.

    unit    = size / 64          (4 px on the 256 px canvas)
    MARGIN  = 4 units            the safe area; artwork lives inside 56x56
    OUTLINE = 3 units            the silhouette stroke
    DETAIL  = 2 units            internal lines

Nothing here knows about archaeology - it is the drawing surface only.
"""

from qgis.PyQt.QtGui import QPainterPath

UNITS = 64
MARGIN = 4
OUTLINE = 3
DETAIL = 2
RADIUS = 3      # clamped to a quarter of the shorter side, so small parts stay crisp

#: Coordinates snap to this fraction of a unit. Half a unit is fine enough for
#: a diagonal to look intentional and coarse enough to keep the set aligned.
SNAP = 0.5

#: The tone ramp.
#:
#: QGIS paints every ``param(fill)`` in a symbol with the one colour the user
#: picked, so a second *hue* would collapse the moment a symbol is recoloured.
#: Depth has to come from opacity, which survives as per-element
#: fill-opacity - so these are three steps of the same hue, and an icon reads
#: as two or three related colours under whatever the user chooses.
#:
#: Three steps, not twenty-one: the catalogue used to carry alphas scattered
#: from 50 to 245, which is the colour equivalent of off-grid coordinates.
SOLID = 255     # the object itself
MID = 165       # a body the detail sits on
SOFT = 95       # ground, spoil, anything behind


#: The material palette.
#:
#: The catalogue used to carry 110 distinct colours for 188 symbols, with
#: saturation anywhere from 0 to 100 percent and lightness from 17 to 83 -
#: neon blues and pure yellows sitting beside muted earths. Nothing made them
#: a set.
#:
#: These thirteen are held to a narrow band of saturation and lightness and
#: vary by hue alone, which is what lets a map full of them read as one
#: family. They are named for the material, because that is how an
#: archaeologist already groups finds - and it means a new template picks its
#: colour by asking what the thing is made of.
PALETTE = {
    "earth":  "#A87C52",   # soil, plain wares, mounds, earthen features
    "clay":   "#A66048",   # fired clay: red wares, kilns, burnt ground
    "stone":  "#7E8C9A",   # stone tools, stone chambers, walling
    "slate":  "#5F6A76",   # dark stone: capstones, roof tiles, ink stones
    "iron":   "#4C535B",   # iron artefacts and smelting
    "bronze": "#B08A46",   # bronze artefacts
    "gold":   "#C2A552",   # gold and gilt ornaments
    "jade":   "#5F9285",   # jade, celadon
    "water":  "#5C88A2",   # glass, water, wells, ditches
    "bone":   "#BFB39D",   # bone, antler, shell, porcelain
    "char":   "#6D645A",   # charcoal, ash, soot
    "field":  "#8B9557",   # paddy, dry field, vegetation
    "mark":   "#B5564F",   # survey and recording marks
}


def tone(color_class, color, level):
    """The symbol colour at one step of the ramp."""
    return color_class(color.red(), color.green(), color.blue(), level)


def ramp(value):
    """Snap an opacity onto the ramp."""
    value = int(value)
    return min((SOFT, MID, SOLID), key=lambda step: abs(step - value))


def snap(value):
    """Round a unit coordinate onto the grid."""
    return round(float(value) / SNAP) * SNAP


class Grid:
    """Converts grid units to canvas pixels and builds the shared shapes."""

    def __init__(self, size=256):
        self.size = float(size)
        self.unit = self.size / UNITS

    # -- coordinates ---------------------------------------------------
    def u(self, value):
        """One coordinate, snapped to the grid and scaled to pixels."""
        return snap(value) * self.unit

    def pt(self, x, y):
        return self.u(x), self.u(y)

    @property
    def centre(self):
        return UNITS / 2.0

    @property
    def inner(self):
        """(left, top, right, bottom) of the safe area, in units."""
        return MARGIN, MARGIN, UNITS - MARGIN, UNITS - MARGIN

    def width(self, units):
        """A stroke width in pixels, for a weight given in units."""
        return snap(units) * self.unit

    # -- primitives ----------------------------------------------------
    def rect(self, x, y, w, h, r=RADIUS):
        """
        A rounded rectangle, built from lines and quads.

        Qt's addRoundedRect would do this, but building it here keeps the
        corners on the grid and keeps the path made of the same commands
        everything else uses, so the preview renderer needs no special case.
        """
        r = min(snap(r), snap(w) / 2.0, snap(h) / 2.0)
        x0, y0 = snap(x), snap(y)
        x1, y1 = snap(x + w), snap(y + h)
        path = QPainterPath()
        path.moveTo(self.u(x0 + r), self.u(y0))
        path.lineTo(self.u(x1 - r), self.u(y0))
        if r:
            path.quadTo(self.u(x1), self.u(y0), self.u(x1), self.u(y0 + r))
        path.lineTo(self.u(x1), self.u(y1 - r))
        if r:
            path.quadTo(self.u(x1), self.u(y1), self.u(x1 - r), self.u(y1))
        path.lineTo(self.u(x0 + r), self.u(y1))
        if r:
            path.quadTo(self.u(x0), self.u(y1), self.u(x0), self.u(y1 - r))
        path.lineTo(self.u(x0), self.u(y0 + r))
        if r:
            path.quadTo(self.u(x0), self.u(y0), self.u(x0 + r), self.u(y0))
        path.closeSubpath()
        return path

    def circle(self, cx, cy, r):
        """A circle as four quads, so it stays on the same command set."""
        cx, cy, r = snap(cx), snap(cy), snap(r)
        k = r * 0.5523
        path = QPainterPath()
        path.moveTo(self.u(cx - r), self.u(cy))
        for (qx, qy, ex, ey) in (
            (cx - r, cy - k * 1.34, cx, cy - r),
            (cx + k * 1.34, cy - r, cx + r, cy),
            (cx + r, cy + k * 1.34, cx, cy + r),
            (cx - k * 1.34, cy + r, cx - r, cy),
        ):
            path.quadTo(self.u(qx), self.u(qy), self.u(ex), self.u(ey))
        path.closeSubpath()
        return path

    def poly(self, points, close=True):
        """A straight-sided shape through grid points."""
        path = QPainterPath()
        for index, (x, y) in enumerate(points):
            if index == 0:
                path.moveTo(*self.pt(x, y))
            else:
                path.lineTo(*self.pt(x, y))
        if close:
            path.closeSubpath()
        return path

    def line(self, x0, y0, x1, y1):
        path = QPainterPath()
        path.moveTo(*self.pt(x0, y0))
        path.lineTo(*self.pt(x1, y1))
        return path

    # -- the shape most artefacts share --------------------------------
    def symmetric(self, profile, curved=False, cx=None):
        """
        A shape mirrored about a vertical axis.

        ``profile`` is a list of ``(half_width, y)`` in units, read top to
        bottom: the right-hand outline. Blades, vessels, mounds and pit
        sections are all this one call, which is what keeps them a family.
        ``curved`` rounds the joins for thrown pottery; blades stay faceted.
        """
        cx = self.centre if cx is None else cx
        right = [(cx + w, y) for w, y in profile]
        left = [(cx - w, y) for w, y in reversed(profile)]

        path = QPainterPath()
        path.moveTo(*self.pt(*right[0]))
        self._run(path, right[1:], curved)
        path.lineTo(*self.pt(*left[0]))
        self._run(path, left[1:], curved)
        path.closeSubpath()
        return path

    def _run(self, path, points, curved):
        if not curved:
            for x, y in points:
                path.lineTo(*self.pt(x, y))
            return
        previous = None
        for index, (x, y) in enumerate(points):
            if previous is None or index == len(points) - 1:
                path.lineTo(*self.pt(x, y))
            else:
                # Bulge the control point outwards from the axis so a vessel
                # wall swells instead of cutting the corner.
                bulge = x + (x - self.centre) * 0.28
                path.quadTo(self.u(bulge), self.u(previous[1] + (y - previous[1]) / 2.0),
                            *self.pt(x, y))
            previous = (x, y)
