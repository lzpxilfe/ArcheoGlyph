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

import math

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
        """
        A circle, as arc segments on the same command set as everything else.

        This used to place its own control points at ``0.74 * r``, which is a
        quarter of the way inside where a quadratic has to reach to touch the
        circle - so every circle in the catalogue flattened at the diagonals
        and read as a lumpy polygon. Deferring to :meth:`arc` means there is
        one piece of curve maths in the file rather than two.
        """
        cx, cy, r = snap(cx), snap(cy), snap(r)
        path = QPainterPath()
        self.arc(path, cx, cy, r, -math.pi / 2.0, 2.0 * math.pi,
                 segments=8, move=True)
        path.closeSubpath()
        return path

    def ellipse(self, cx, cy, rx, ry):
        """
        An oval, for the things that genuinely are one - a river pebble, a
        quern, a saddle. Faking these with a squashed circle or a pair of
        quads is how the catalogue ended up with ovals of six different
        curvatures.
        """
        cx, cy, rx, ry = snap(cx), snap(cy), snap(rx), snap(ry)
        step = math.pi / 4.0
        reach = 1.0 / math.cos(step / 2.0)
        path = QPainterPath()
        path.moveTo(self.u(cx + rx), self.u(cy))
        for index in range(8):
            a0 = step * index
            mid = a0 + step / 2.0
            a1 = a0 + step
            path.quadTo(self.u(cx + rx * reach * math.cos(mid)),
                        self.u(cy + ry * reach * math.sin(mid)),
                        self.u(cx + rx * math.cos(a1)),
                        self.u(cy + ry * math.sin(a1)))
        path.closeSubpath()
        return path

    def arc(self, path, cx, cy, r, start, sweep, segments=6, move=False):
        """
        Append a circular arc, as quads that actually follow the circle.

        The control point sits at r / cos(half step), which is the radius that
        makes a quadratic touch the arc at both ends - sampling the circle and
        joining the samples with straight quads is what leaves a polygon.
        """
        step = sweep / segments
        if move:
            path.moveTo(self.u(cx + r * math.cos(start)),
                        self.u(cy + r * math.sin(start)))
        reach = r / math.cos(step / 2.0)
        for index in range(segments):
            a0 = start + step * index
            a1 = a0 + step
            mid = (a0 + a1) / 2.0
            path.quadTo(self.u(cx + reach * math.cos(mid)),
                        self.u(cy + reach * math.sin(mid)),
                        self.u(cx + r * math.cos(a1)),
                        self.u(cy + r * math.sin(a1)))
        return path

    def keyhole(self, head_cy, head_r, join_y, foot_half, foot_y):
        """
        The 전방후원분 outline: round rear mound and trapezoidal front, as one
        shape.

        Drawn as a circle plus a separate trapezoid the join shows as a seam,
        and the two halves drift apart when either is tuned. Here the tail
        starts exactly on the circle - the waist is derived from the join
        height - so the outline closes on itself by construction.
        """
        cx = self.centre
        offset = min(abs(join_y - head_cy), head_r)
        waist = math.sqrt(max(0.0, head_r ** 2 - offset ** 2))
        start = math.atan2(join_y - head_cy, waist)     # right join, y downward

        path = QPainterPath()
        path.moveTo(*self.pt(cx + waist, join_y))
        # Right join over the top to the left join: the long way round.
        self.arc(path, cx, head_cy, head_r, start, -(math.pi + 2.0 * start))
        path.lineTo(*self.pt(cx - foot_half, foot_y))
        path.lineTo(*self.pt(cx + foot_half, foot_y))
        path.closeSubpath()
        return path

    def spindle(self, r, waist_half, foot_half, top_y, bottom_y):
        """
        쌍방중원분: a round mound with a trapezoidal front at either end.

        The same construction as :meth:`keyhole` - the fronts start exactly on
        the circle, so the outline closes by itself - only mirrored. Drawing
        it as a circle with a bar laid over it, as this used to, left the
        bar's own outline showing through the mound.
        """
        cx = cy = self.centre
        waist = min(abs(waist_half), r)
        offset = math.sqrt(max(0.0, r ** 2 - waist ** 2))
        turn = math.atan2(offset, waist)        # the lower-right join

        path = QPainterPath()
        path.moveTo(*self.pt(cx + waist, cy + offset))
        self.arc(path, cx, cy, r, turn, -2.0 * turn)        # right flank
        path.lineTo(*self.pt(cx + foot_half, top_y))
        path.lineTo(*self.pt(cx - foot_half, top_y))
        path.lineTo(*self.pt(cx - waist, cy - offset))
        self.arc(path, cx, cy, r, math.pi + turn, -2.0 * turn)   # left flank
        path.lineTo(*self.pt(cx - foot_half, bottom_y))
        path.lineTo(*self.pt(cx + foot_half, bottom_y))
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
        self._run(path, right[0], right[1:], curved)
        path.lineTo(*self.pt(*left[0]))
        self._run(path, left[0], left[1:], curved)
        path.closeSubpath()
        return path

    #: How far a curved wall swells past the straight line between two
    #: stations. Enough to read as thrown pottery, not so much that a jar
    #: turns into a balloon.
    BULGE = 0.16

    def _run(self, path, start, points, curved):
        """Walk the outline from ``start`` through ``points``."""
        if not curved:
            for x, y in points:
                path.lineTo(*self.pt(x, y))
            return

        previous = start
        for x, y in points:
            # The control point sits at the midpoint of the segment, pushed
            # away from the axis. Anchoring it to the midpoint rather than to
            # one end is what keeps the wall a curve instead of a corner -
            # which is what made every pot a faceted polygon.
            mid_x = (previous[0] + x) / 2.0
            mid_y = (previous[1] + y) / 2.0
            bulge = mid_x + (mid_x - self.centre) * self.BULGE
            path.quadTo(self.u(bulge), self.u(mid_y), *self.pt(x, y))
            previous = (x, y)
