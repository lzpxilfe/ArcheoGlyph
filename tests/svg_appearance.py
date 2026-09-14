"""
What a traced SVG looks like at legend size, on the catalogue's own scale.

The drawn catalogue already has an instrument for "are these two symbols the
same picture": test_template_drawing rasterises each painted symbol at the
size a legend actually draws it and compares intensities. The traced symbols
had no such check, which is how a reading that drew one artefact as two
different symbols depending on the lamp went unnoticed.

So this is the same instrument pointed at SVG. It deliberately reuses
``_stroked``, ``_overlap``, ``LEGEND_PX`` and ``MAX_OVERLAP`` from
test_template_drawing rather than defining its own: a second scale would make
the 0.95 bar - which was calibrated over 17,578 catalogue pairs - meaningless
here. Only the front end differs, because ``_appearance`` there takes the name
of a recorded QPainter run and there is no painter in this path.
"""

import re

import numpy as np

from tests.test_template_drawing import (  # noqa: F401  (MAX_OVERLAP re-exported)
    LEGEND_PX,
    MAX_OVERLAP,
    _overlap as overlap,
    _stroked,
)

#: Painted at four times the legend size and box-filtered down, so a hairline
#: that a legend renders as a grey pixel is measured as a grey pixel and not
#: as nothing. The same ratio test_template_drawing uses.
PAINT_PX = LEGEND_PX * 4

_PATH = re.compile(r'<path\b([^>]*?)/>', re.S)
_ATTR = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')
_TOKEN = re.compile(r'[A-Za-z]|-?\d+(?:\.\d+)?')
_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')


#: How many segments a cubic is flattened into. The whole symbol is 64 legend
#: pixels across, so a single curve spans a few of them at most and a dozen
#: segments put the flattening error far below the pixel this is measured in.
CURVE_STEPS = 12


def _cubic(start, c1, c2, end, steps=CURVE_STEPS):
    """A cubic as ``steps`` line segments, the start point excluded."""
    out = []
    for index in range(1, steps + 1):
        t = index / float(steps)
        u = 1.0 - t
        out.append((u * u * u * start[0] + 3 * u * u * t * c1[0]
                    + 3 * u * t * t * c2[0] + t * t * t * end[0],
                    u * u * u * start[1] + 3 * u * u * t * c1[1]
                    + 3 * u * t * t * c2[1] + t * t * t * end[1]))
    return out


def subpaths(d):
    """
    The subpaths of a path's ``d`` as point lists.

    The tracer writes absolute moves, lines, cubics and closes. Moves, lines
    and closes come through exactly; a cubic is flattened, which is the one
    approximation here and is well under the pixel the result is measured in.
    Any other command means the writer has changed and this would silently
    start missing ink, so it is refused loudly rather than skipped.
    """
    out, current = [], []
    tokens = _TOKEN.findall(d)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("M", "L"):
            if token == "M" and len(current) >= 2:
                out.append(current)
                current = []
            index += 1
            while index + 1 < len(tokens) and _NUMBER.fullmatch(tokens[index]):
                current.append((float(tokens[index]), float(tokens[index + 1])))
                index += 2
        elif token == "C":
            index += 1
            while index + 5 < len(tokens) and _NUMBER.fullmatch(tokens[index]):
                values = [float(v) for v in tokens[index:index + 6]]
                start = current[-1] if current else (values[0], values[1])
                current.extend(_cubic(start, values[0:2], values[2:4],
                                      values[4:6]))
                index += 6
        elif token == "Z":
            if current:
                current.append(current[0])
            index += 1
        elif _NUMBER.fullmatch(token):
            index += 1
        else:
            raise AssertionError(
                f"svg_appearance can only measure M/L/C/Z paths and this one "
                f"uses {token!r}. Teach it the new command rather than "
                f"letting the ink go unmeasured.")
    if len(current) >= 2:
        out.append(current)
    return out


def _filled(points, px, py):
    """Even-odd interior of a closed polygon."""
    inside = np.zeros(px.shape, dtype=bool)
    previous = len(points) - 1
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[previous]
        straddles = (y1 > py) != (y2 > py)
        span = (y2 - y1) or 1e-9
        inside ^= straddles & (px < (x2 - x1) * (py - y1) / span + x1)
        previous = index
    return inside


def appearance(svg, size=LEGEND_PX, paint=PAINT_PX):
    """A ``size`` x ``size`` intensity map of a traced symbol, 0 to 1."""
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    side = float(box.group(1)) if box else float(paint)
    scale = paint / side
    canvas = np.zeros((paint, paint), dtype=float)
    centres = np.arange(paint) + 0.5
    px, py = np.meshgrid(centres, centres)

    for body in _PATH.findall(svg):
        attrs = dict(_ATTR.findall(body))
        d = attrs.get("d", "")
        if not d.strip():
            continue
        fill = attrs.get("fill", "none")
        # A gradient still covers its polygon, so it counts as ink at full
        # weight; only an explicit "none" is empty.
        alpha = (float(attrs.get("fill-opacity", 1.0))
                 if fill not in ("none", "") else 0.0)
        width = float(attrs.get("stroke-width", 0.0) or 0.0) * scale
        if attrs.get("stroke", "none") in ("none", ""):
            width = 0.0
        for points in subpaths(d):
            pts = np.asarray(points, dtype=float) * scale
            if len(pts) < 2:
                continue
            if alpha > 0.0 and len(pts) >= 3:
                canvas = np.maximum(canvas, _filled(pts, px, py) * alpha)
            if width > 0.0:
                canvas = np.maximum(canvas, _stroked(pts, width, px, py) * 1.0)

    step = paint // size
    return canvas.reshape(size, step, size, step).mean(axis=(1, 3))
