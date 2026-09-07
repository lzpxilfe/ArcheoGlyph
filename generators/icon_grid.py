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
    OUTLINE = 2.6 units          the silhouette stroke
    DETAIL  = 1.3 units          internal lines

Nothing here knows about archaeology - it is the drawing surface only.
"""

import math

from qgis.PyQt.QtGui import QPainterPath

UNITS = 64
MARGIN = 4

#: The two stroke weights, and the gap between them.
#:
#: These were 3 and 2 units - a ratio of 1.5, which is not enough for the eye
#: to read one as the silhouette and the other as detail. What separates a
#: shape from what is drawn inside it is the *ratio*, not the absolute
#: weight, so the detail line came down rather than the outline going up: at
#: 3 against 1.5 the contrast is still double, and the outline is lighter
#: than it was when the two were 3 and 2.
#:
#: Three units is also still thick enough that the round joins visibly blunt
#: a corner, which is most of what keeps a drawn icon friendly.
OUTLINE = 2.6
DETAIL = 1.3

#: Corner radius, clamped to a quarter of the shorter side so small parts stay
#: crisp. Raised with the outline: a heavier stroke needs a wider corner to
#: turn through, or the join reads as a blob.
RADIUS = 4

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

    def comma(self, cx, cy, outer, head_r, tail_r, start, sweep, segments=10):
        """
        곡옥: a crescent that thins along its length.

        A comma drawn at constant width is a banana, and one drawn as a
        tapering freehand curve closes up at a 3-unit outline and reads as a
        figure nine. Here the back is a true arc and only the belly moves, so
        the head stays fat and the tail stays open however the numbers are
        tuned.

        ``head_r`` and ``tail_r`` are the inner radii at the two ends - the
        larger the tail radius, the thinner the tail.
        """
        path = QPainterPath()
        path.moveTo(self.u(cx + outer * math.cos(start)),
                    self.u(cy + outer * math.sin(start)))
        self.arc(path, cx, cy, outer, start, sweep, segments=segments)

        def belly(fraction):
            angle = start + sweep * fraction
            radius = head_r + (tail_r - head_r) * fraction
            return angle, radius

        for index in range(segments, -1, -1):
            angle, radius = belly(index / float(segments))
            if index == segments:
                path.lineTo(self.u(cx + radius * math.cos(angle)),
                            self.u(cy + radius * math.sin(angle)))
                continue
            mid_angle, mid_radius = belly((index + 0.5) / float(segments))
            reach = mid_radius / math.cos(abs(sweep) / (2.0 * segments))
            path.quadTo(self.u(cx + reach * math.cos(mid_angle)),
                        self.u(cy + reach * math.sin(mid_angle)),
                        self.u(cx + radius * math.cos(angle)),
                        self.u(cy + radius * math.sin(angle)))
        path.closeSubpath()
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

    def _rounded(self, points, close, radius):
        """
        Walk a polygon, turning each corner through a quad instead of a spike.

        ``rect`` has rounded its corners since the grid was built, but every
        other straight-sided shape came to a hard point, which is what made a
        north arrow read as a needle and a gable as a blade. The radius is
        clamped to 40 percent of the shorter adjacent edge, so a long side
        keeps the full corner and a short one is not swallowed by it.
        """
        pts = []
        for x, y in points:
            point = (snap(x), snap(y))
            if not pts or math.hypot(point[0] - pts[-1][0],
                                     point[1] - pts[-1][1]) > 1e-9:
                pts.append(point)
        # A profile that closes to a zero-width tip lands two points on the
        # same spot; left as a pair they give a corner with no edge to cut
        # back along, and the tip stays a needle.
        if close and len(pts) > 1 and math.hypot(pts[0][0] - pts[-1][0],
                                                 pts[0][1] - pts[-1][1]) < 1e-9:
            pts.pop()
        count = len(pts)
        if count < 3:
            path = QPainterPath()
            for index, point in enumerate(pts):
                (path.moveTo if index == 0 else path.lineTo)(
                    self.u(point[0]), self.u(point[1]))
            return path
        path = QPainterPath()

        def lerp(a, b, distance):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                return a
            t = min(distance, length) / length
            return (a[0] + dx * t, a[1] + dy * t)

        started = False
        for index in range(count):
            corner = pts[index]
            before, after = pts[index - 1], pts[(index + 1) % count]
            # The ends of an open polyline are not corners.
            if not close and index in (0, count - 1):
                if not started:
                    path.moveTo(self.u(corner[0]), self.u(corner[1]))
                    started = True
                else:
                    path.lineTo(self.u(corner[0]), self.u(corner[1]))
                continue
            span = min(math.hypot(corner[0] - before[0], corner[1] - before[1]),
                       math.hypot(corner[0] - after[0], corner[1] - after[1]))
            cut = min(radius, span * 0.4)
            entry = lerp(corner, before, cut)
            exit_ = lerp(corner, after, cut)
            if not started:
                path.moveTo(self.u(entry[0]), self.u(entry[1]))
                started = True
            else:
                path.lineTo(self.u(entry[0]), self.u(entry[1]))
            if cut > 0.01:
                path.quadTo(self.u(corner[0]), self.u(corner[1]),
                            self.u(exit_[0]), self.u(exit_[1]))
            else:
                path.lineTo(self.u(corner[0]), self.u(corner[1]))
        if close:
            path.closeSubpath()
        return path

    def poly(self, points, close=True, r=RADIUS):
        """
        A straight-sided shape through grid points, corners turned not spiked.

        Pass ``r=0`` where a shape genuinely needs a crisp point.
        """
        if r <= 0:
            path = QPainterPath()
            for index, (x, y) in enumerate(points):
                if index == 0:
                    path.moveTo(*self.pt(x, y))
                else:
                    path.lineTo(*self.pt(x, y))
            if close:
                path.closeSubpath()
            return path
        return self._rounded(points, close, r)

    def line(self, x0, y0, x1, y1):
        path = QPainterPath()
        path.moveTo(*self.pt(x0, y0))
        path.lineTo(*self.pt(x1, y1))
        return path

    # -- the shape most artefacts share --------------------------------
    #: How far a faceted profile turns its corners. Smaller than RADIUS: a
    #: blade should lose its needle without losing its edge.
    FACET = 2.0

    def symmetric(self, profile, curved=False, cx=None, r=None):
        """
        A shape mirrored about a vertical axis.

        ``profile`` is a list of ``(half_width, y)`` in units, read top to
        bottom: the right-hand outline. Blades, vessels, mounds and pit
        sections are all this one call, which is what keeps them a family.
        ``curved`` rounds the joins for thrown pottery; a faceted profile
        keeps its facets but turns its corners, because a spear point that
        comes to a mathematical point reads as a needle rather than a spear.
        """
        cx = self.centre if cx is None else cx
        right = [(cx + w, y) for w, y in profile]
        left = [(cx - w, y) for w, y in reversed(profile)]

        if not curved:
            radius = self.FACET if r is None else r
            if radius > 0:
                return self._rounded(right + left, True, radius)

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
