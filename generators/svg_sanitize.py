# -*- coding: utf-8 -*-
"""
Parse and sanitise SVG text returned by a language model.

QGIS-free. The previous implementation matched a list of forbidden substrings,
which accepted anything it did not know about (``<script>``, ``<use>``, event
handlers, external references) and never checked that the document was
well-formed. Here the document is parsed, unknown elements and attributes are
dropped, and the result is re-serialised, so only geometry survives.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

ALLOWED_TAGS = {
    "svg", "g", "defs", "path", "circle", "ellipse", "rect", "line",
    "polyline", "polygon", "title", "desc",
}

COMMON_ATTRS = {
    "id", "class", "transform", "fill", "stroke", "stroke-width", "stroke-linecap",
    "stroke-linejoin", "stroke-dasharray", "stroke-dashoffset", "stroke-opacity",
    "stroke-miterlimit", "fill-opacity", "fill-rule", "opacity", "clip-rule",
    "vector-effect", "paint-order",
}

TAG_ATTRS: Dict[str, set] = {
    "svg": COMMON_ATTRS | {"viewBox", "width", "height", "xmlns", "version", "preserveAspectRatio"},
    "g": COMMON_ATTRS,
    "defs": COMMON_ATTRS,
    "path": COMMON_ATTRS | {"d"},
    "circle": COMMON_ATTRS | {"cx", "cy", "r"},
    "ellipse": COMMON_ATTRS | {"cx", "cy", "rx", "ry"},
    "rect": COMMON_ATTRS | {"x", "y", "width", "height", "rx", "ry"},
    "line": COMMON_ATTRS | {"x1", "y1", "x2", "y2"},
    "polyline": COMMON_ATTRS | {"points"},
    "polygon": COMMON_ATTRS | {"points"},
    "title": set(),
    "desc": set(),
}

GEOMETRY_TAGS = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}

_URL_VALUE = re.compile(r"url\s*\(", re.IGNORECASE)
_UNSAFE_VALUE = re.compile(r"(javascript:|data:|expression\s*\()", re.IGNORECASE)
_HEX_COLOR = re.compile(r"^#(?:[0-9a-f]{3}|[0-9a-f]{6})$", re.IGNORECASE)
_RGB_COLOR = re.compile(r"^rgba?\(\s*[\d.%\s,/]+\)$", re.IGNORECASE)

NAMED_COLORS_ALLOWED = {
    "none", "transparent", "black", "white", "gray", "grey", "silver", "dimgray",
    "dimgrey", "darkgray", "darkgrey", "lightgray", "lightgrey", "brown", "sienna",
    "tan", "beige", "olive", "maroon", "navy", "teal",
}


def _local(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def extract_svg(text: str) -> Optional[str]:
    """
    Pull the SVG document out of a model reply (markdown fences, prose, XML
    prolog). Returns None when no ``<svg>...</svg>`` pair is present.
    """
    if not text:
        return None
    start = text.find("<svg")
    if start < 0:
        return None
    end = text.rfind("</svg>")
    if end < start:
        return None
    return text[start:end + len("</svg>")]


def _clean_paint(value: str) -> Optional[str]:
    """Return a safe paint value, or None when it must be dropped."""
    token = (value or "").strip()
    if not token:
        return None
    low = token.lower()
    if _UNSAFE_VALUE.search(low):
        return None
    if _URL_VALUE.search(low):
        return None  # gradients/patterns are not preserved
    if low.startswith("param("):
        return token
    if _HEX_COLOR.match(low) or _RGB_COLOR.match(low):
        return token
    if low in NAMED_COLORS_ALLOWED:
        return token
    return None


def sanitize_svg(svg_text: str) -> Tuple[Optional[str], List[str], Dict[str, int]]:
    """
    Sanitise ``svg_text``.

    :return: ``(clean_svg | None, problems, stats)``. ``problems`` lists the
        reasons content was dropped or the document rejected; ``stats`` counts
        geometry elements and distinct fill/stroke colours.
    """
    problems: List[str] = []
    stats = {"geometry": 0, "colors": 0, "fill_colors": 0}
    if not svg_text or "<svg" not in svg_text:
        return None, ["no SVG document found"], stats

    # Entity declarations are a classic XML attack surface and never needed
    # here; check the raw reply, since extraction would skip a leading DOCTYPE.
    upper = svg_text.upper()
    if "<!ENTITY" in upper or "<!DOCTYPE" in upper:
        return None, ["document type or entity declarations are not allowed"], stats
    body = extract_svg(svg_text) or svg_text

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        return None, [f"malformed SVG ({exc})"], stats

    if _local(root.tag) != "svg":
        return None, ["root element is not <svg>"], stats

    colors, fill_colors = set(), set()

    def clean(element: ET.Element) -> Optional[ET.Element]:
        tag = _local(element.tag)
        if tag not in ALLOWED_TAGS:
            problems.append(f"removed <{tag}>")
            return None
        allowed = TAG_ATTRS.get(tag, COMMON_ATTRS)
        attrib = {}
        for name, value in element.attrib.items():
            key = _local(name)
            if key.lower().startswith("on") or "href" in key.lower():
                problems.append(f"removed attribute {key}")
                continue
            if key == "style":
                problems.append("removed style attribute")
                continue
            if key not in allowed:
                problems.append(f"removed attribute {key}")
                continue
            if key in ("fill", "stroke"):
                cleaned = _clean_paint(value)
                if cleaned is None:
                    problems.append(f"removed unsupported {key} value")
                    continue
                token = cleaned.lower()
                if token not in ("none", "transparent"):
                    colors.add(token)
                    if key == "fill":
                        fill_colors.add(token)
                attrib[key] = cleaned
                continue
            if _UNSAFE_VALUE.search(str(value)):
                problems.append(f"removed unsafe {key} value")
                continue
            attrib[key] = value

        node = ET.Element(f"{{{SVG_NS}}}{tag}", attrib)
        if tag in GEOMETRY_TAGS:
            stats["geometry"] += 1
        for child in list(element):
            kept = clean(child)
            if kept is not None:
                node.append(kept)
        if tag in ("title", "desc"):
            node.text = (element.text or "")[:200]
        return node

    clean_root = clean(root)
    if clean_root is None:
        return None, problems or ["nothing left after sanitisation"], stats
    stats["colors"] = len(colors)
    stats["fill_colors"] = len(fill_colors)
    stats["color_values"] = sorted(colors)
    stats["fill_values"] = sorted(fill_colors)

    if stats["geometry"] == 0:
        return None, problems + ["no geometry elements survived"], stats

    return ET.tostring(clean_root, encoding="unicode"), problems, stats
