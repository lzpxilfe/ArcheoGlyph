# -*- coding: utf-8 -*-
"""
SVG construction and post-processing for map symbols.

QGIS-free. Two entry points:

* ``build_svg`` - assemble a symbol from an outline polyline plus internal
  lines, normalised to a square unit viewBox.
* ``finalize_svg`` - post-process any SVG text so it behaves well as a QGIS
  marker: crop the viewBox to the drawn geometry, make it square and centred,
  and expose the body fill / outline stroke as QGIS ``param()`` placeholders so
  the symbol can be recoloured from the Layer Styling panel.

QGIS parametric SVG convention (see QGIS docs, "Parameterizable SVG"):
    fill="param(fill) #hex"  stroke="param(outline) #hex"
    stroke-width="param(outline-width) 1.5"
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

UNIT_BOX = 100.0
DEFAULT_PAD_RATIO = 0.06

_NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER)
_PATH_TOKEN_RE = re.compile(r"[MLHVCSQTAZmlhvcsqtaz]|" + _NUMBER)
_ABSOLUTE_SIMPLE = set("MLHVCSQTZ")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _fmt(value: float) -> str:
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polyline_bbox(points: Iterable[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for pt in points:
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def union_bbox(boxes: Iterable[Optional[Tuple[float, float, float, float]]]):
    result = None
    for box in boxes:
        if box is None:
            continue
        if result is None:
            result = list(box)
        else:
            result[0] = min(result[0], box[0])
            result[1] = min(result[1], box[1])
            result[2] = max(result[2], box[2])
            result[3] = max(result[3], box[3])
    return tuple(result) if result else None


def square_viewbox(bbox, pad_ratio=DEFAULT_PAD_RATIO, min_side=1.0) -> Tuple[float, float, float, float]:
    """Square viewBox (x, y, side, side) centred on bbox with padding."""
    x0, y0, x1, y1 = bbox
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    side = max(min_side, max(w, h) * (1.0 + 2.0 * pad_ratio))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return cx - side / 2.0, cy - side / 2.0, side, side


def _path_bbox(d: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Bounding box of the control points of an absolute-command path.
    Returns None when the path uses relative or arc commands (bbox would be
    unreliable), so callers keep the original viewBox.
    """
    tokens = _PATH_TOKEN_RE.findall(d or "")
    if not tokens:
        return None
    xs: List[float] = []
    ys: List[float] = []
    cmd = None
    nums: List[float] = []

    def flush():
        if cmd is None or not nums:
            return
        if cmd in "MLCSQT":
            for i in range(0, len(nums) - 1, 2):
                xs.append(nums[i])
                ys.append(nums[i + 1])
        elif cmd == "H":
            xs.extend(nums)
        elif cmd == "V":
            ys.extend(nums)

    for tok in tokens:
        if tok.isalpha():
            flush()
            nums = []
            if tok not in _ABSOLUTE_SIMPLE:
                return None
            cmd = tok
        else:
            nums.append(float(tok))
    flush()
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _element_bbox(el: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    tag = _local(el.tag)
    g = el.attrib.get
    try:
        if tag == "path":
            return _path_bbox(g("d", ""))
        if tag == "circle":
            cx, cy, r = float(g("cx", 0)), float(g("cy", 0)), float(g("r", 0))
            return cx - r, cy - r, cx + r, cy + r
        if tag == "ellipse":
            cx, cy = float(g("cx", 0)), float(g("cy", 0))
            rx, ry = float(g("rx", 0)), float(g("ry", 0))
            return cx - rx, cy - ry, cx + rx, cy + ry
        if tag == "rect":
            x, y = float(g("x", 0)), float(g("y", 0))
            return x, y, x + float(g("width", 0)), y + float(g("height", 0))
        if tag == "line":
            x1, y1, x2, y2 = (float(g(k, 0)) for k in ("x1", "y1", "x2", "y2"))
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        if tag in ("polyline", "polygon"):
            nums = [float(v) for v in _NUMBER_RE.findall(g("points", ""))]
            pts = list(zip(nums[0::2], nums[1::2]))
            return polyline_bbox(pts)
    except (TypeError, ValueError):
        return None
    return None


_DRAWABLE = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}


def geometry_bbox(root: ET.Element):
    """Union bbox of all drawable elements outside <defs>. None if unknown."""
    boxes = []
    for el in _iter_drawables(root):
        box = _element_bbox(el)
        if box is None:
            if _local(el.tag) == "path":
                return None  # unsupported path syntax: do not crop
            continue
        boxes.append(box)
    return union_bbox(boxes)


def _iter_drawables(root: ET.Element):
    stack = [root]
    while stack:
        el = stack.pop()
        tag = _local(el.tag)
        if tag in ("defs", "clipPath", "mask", "pattern", "marker", "symbol"):
            continue
        if tag in _DRAWABLE:
            yield el
        stack.extend(list(el))


def smooth_closed_path(points, corner_deg=38.0, tension=1.0, precision=2):
    """
    SVG path data for a closed outline: Catmull-Rom style cubic Beziers
    through the vertices, except at sharp corners (turning angle above
    ``corner_deg``) which stay as hard vertices. Keeps flint edges crisp while
    rendering ovals, discs and vessel profiles as smooth curves.
    """
    import math

    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return ""
    fmt = "{:.%df}" % precision

    def f(v):
        text = fmt.format(v)
        return text.rstrip("0").rstrip(".") if "." in text else text

    def sharp(i):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return True
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        return math.degrees(math.acos(cos)) > corner_deg

    corner = [sharp(i) for i in range(n)]
    parts = [f"M {f(pts[0][0])},{f(pts[0][1])}"]
    for i in range(n):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        if corner[i] and corner[(i + 1) % n]:
            parts.append(f"L {f(p2[0])},{f(p2[1])}")
            continue
        k = tension / 6.0
        c1 = p1 if corner[i] else (p1[0] + (p2[0] - p0[0]) * k, p1[1] + (p2[1] - p0[1]) * k)
        c2 = p2 if corner[(i + 1) % n] else (p2[0] - (p3[0] - p1[0]) * k, p2[1] - (p3[1] - p1[1]) * k)
        parts.append(f"C {f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} {f(p2[0])},{f(p2[1])}")
    parts.append("Z")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------

def _is_solid_paint(value: Optional[str]) -> bool:
    v = (value or "").strip().lower()
    return bool(v) and v != "none" and not v.startswith("url(") and "param(" not in v


def _parse_width(value: Optional[str], default=1.0) -> float:
    m = _NUMBER_RE.search(value or "")
    return float(m.group(0)) if m else default


def parametrize(root: ET.Element) -> Dict[str, object]:
    """
    Expose fill and outline as QGIS param() placeholders.

    * Every drawable with a solid fill gets ``fill="param(fill) <colour>"``.
    * The stroked drawable with the largest bbox is treated as the outline and
      gets ``stroke="param(outline) <colour>"`` plus
      ``stroke-width="param(outline-width) <w>"``.
    Returns the colours/width that were used as fallbacks.
    """
    info: Dict[str, object] = {}
    outline_el = None
    outline_area = -1.0
    for el in _iter_drawables(root):
        fill = el.attrib.get("fill")
        if _is_solid_paint(fill):
            el.set("fill", f"param(fill) {fill.strip()}")
            info.setdefault("fill", fill.strip())
        stroke = el.attrib.get("stroke")
        if _is_solid_paint(stroke):
            box = _element_bbox(el)
            area = 0.0 if box is None else (box[2] - box[0]) * (box[3] - box[1])
            if area > outline_area:
                outline_area = area
                outline_el = el
    if outline_el is not None:
        stroke = outline_el.attrib.get("stroke", "").strip()
        width = _parse_width(outline_el.attrib.get("stroke-width"), 1.0)
        outline_el.set("stroke", f"param(outline) {stroke}")
        outline_el.set("stroke-width", f"param(outline-width) {_fmt(width)}")
        info["outline"] = stroke
        info["outline_width"] = width
    return info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_svg(svg_text: str) -> ET.Element:
    root = ET.fromstring(svg_text.strip())
    if _local(root.tag) != "svg":
        raise ValueError("root element is not <svg>")
    if root.tag == "svg":
        # No namespace declared: re-tag the tree so serialisation emits xmlns.
        for el in root.iter():
            if "}" not in el.tag:
                el.tag = f"{{{SVG_NS}}}{el.tag}"
    return root


def serialize_svg(root: ET.Element) -> str:
    return ET.tostring(root, encoding="unicode")


def describe_provenance(meta) -> str:
    """
    One-line, human-readable record of how a symbol was produced, for the SVG
    <desc>. Keys are taken from a SymbolResult's meta dictionary.
    """
    fields = [
        ("source", "generator"),
        ("style", "style"),
        ("model", "model"),
        ("input", "input image"),
        ("input_kind", "input type"),
        ("plugin_version", "ArchaeoGlyph"),
        ("created", "created"),
    ]
    parts = []
    for key, label in fields:
        value = str((meta or {}).get(key, "")).strip()
        if value:
            parts.append(f"{label}: {value}")
    return "; ".join(parts)


def add_provenance(svg_text: str, meta) -> str:
    """
    Add <title> and <desc> describing how the symbol was made.

    Archaeological symbols end up in published maps, so a reader (or the
    author, months later) should be able to tell whether a symbol was traced
    from a photograph, drawn by a model, or taken from a template. Existing
    title/desc elements are replaced, and the text is XML-escaped by the
    serialiser.
    """
    description = describe_provenance(meta)
    if not description:
        return svg_text
    try:
        root = parse_svg(svg_text)
    except (ET.ParseError, ValueError):
        return svg_text

    for tag in ("title", "desc"):
        for existing in root.findall(f"{{{SVG_NS}}}{tag}"):
            root.remove(existing)

    title = ET.Element(f"{{{SVG_NS}}}title")
    name = str((meta or {}).get("title", "")).strip()
    title.text = name or "ArchaeoGlyph symbol"
    desc = ET.Element(f"{{{SVG_NS}}}desc")
    desc.text = description
    root.insert(0, desc)
    root.insert(0, title)
    return serialize_svg(root)


def finalize_svg(
    svg_text: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    parametrize_colors: bool = True,
) -> Tuple[str, Dict[str, object]]:
    """
    Crop, square and parametrise an SVG for use as a QGIS marker.

    Returns ``(svg_text, info)``. ``info`` may contain ``fill``, ``outline``,
    ``outline_width`` (fallback values used in the param() placeholders),
    ``viewbox`` (x, y, w, h), ``empty`` (True when nothing is drawn) and
    ``parse_error`` (original text returned unchanged).
    """
    try:
        root = parse_svg(svg_text)
    except (ET.ParseError, ValueError) as exc:
        return svg_text, {"parse_error": str(exc)}

    info: Dict[str, object] = {}
    drawables = list(_iter_drawables(root))
    if not drawables:
        info["empty"] = True

    if bbox is None:
        bbox = geometry_bbox(root)
    if bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        vx, vy, vw, vh = square_viewbox(bbox, pad_ratio=pad_ratio)
        root.set("viewBox", f"{_fmt(vx)} {_fmt(vy)} {_fmt(vw)} {_fmt(vh)}")
        for attr in ("width", "height"):
            if attr in root.attrib:
                del root.attrib[attr]
        info["viewbox"] = (vx, vy, vw, vh)
    else:
        vb = [float(v) for v in _NUMBER_RE.findall(root.attrib.get("viewBox", ""))]
        if len(vb) == 4:
            info["viewbox"] = tuple(vb)

    if parametrize_colors:
        info.update(parametrize(root))
    return serialize_svg(root), info


def build_svg(
    outline: Sequence[Sequence[float]],
    internal_lines: Sequence[Sequence[Sequence[float]]] = (),
    *,
    fill_hex: Optional[str] = "#8b6d4a",
    outline_hex: str = "#1e1a16",
    detail_hex: Optional[str] = None,
    outline_width: float = 2.0,
    detail_width: float = 1.0,
    detail_opacity: float = 0.85,
    closed: bool = True,
    pad_ratio: float = DEFAULT_PAD_RATIO,
    unit: float = UNIT_BOX,
    dashed_detail: bool = False,
) -> str:
    """
    Build a parametrised symbol SVG from geometry in image pixel space.

    Coordinates are normalised so the geometry's square bounding box maps to a
    ``unit`` x ``unit`` viewBox; stroke widths are in those units.
    """
    boxes = [polyline_bbox(outline)] + [polyline_bbox(line) for line in internal_lines]
    bbox = union_bbox(boxes)
    if bbox is None:
        return f'<svg xmlns="{SVG_NS}" viewBox="0 0 {_fmt(unit)} {_fmt(unit)}"></svg>'
    vx, vy, side, _ = square_viewbox(bbox, pad_ratio=pad_ratio)
    scale = unit / side

    def tx(pt):
        return (float(pt[0]) - vx) * scale, (float(pt[1]) - vy) * scale

    def path_d(points, close):
        pts = [tx(p) for p in points]
        if len(pts) < 2:
            return ""
        d = "M " + " L ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)
        return d + (" Z" if close else "")

    detail_hex = detail_hex or outline_hex
    parts = [f'<svg xmlns="{SVG_NS}" viewBox="0 0 {_fmt(unit)} {_fmt(unit)}">']
    body = path_d(outline, closed)
    if body:
        if fill_hex:
            parts.append(f'<path d="{body}" fill="param(fill) {fill_hex}" stroke="none"/>')
        parts.append(
            f'<path d="{body}" fill="none" stroke="param(outline) {outline_hex}" '
            f'stroke-width="param(outline-width) {_fmt(outline_width)}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    dash = ' stroke-dasharray="2.2 1.6"' if dashed_detail else ""
    for line in internal_lines:
        d = path_d(line, False)
        if d:
            parts.append(
                f'<path d="{d}" fill="none" stroke="{detail_hex}" stroke-opacity="{_fmt(detail_opacity)}" '
                f'stroke-width="{_fmt(detail_width)}" stroke-linejoin="round" stroke-linecap="round"{dash}/>'
            )
    parts.append("</svg>")
    return "".join(parts)
