import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from archeoglyph.generators import raster_vectorize as rv
from archeoglyph.generators.symbol_result import SymbolResult
from tests import synthetic

NS = "{http://www.w3.org/2000/svg}"


def _paths(svg):
    return ET.fromstring(svg).findall(f".//{NS}path")


def _subpath_points(d):
    """Flatten an absolute M/L/C/Z path into point lists, one per subpath."""
    tokens = re.findall(r"[MLCZ]|-?\d+(?:\.\d+)?", d)
    subpaths, pts, cursor, i = [], [], (0.0, 0.0), 0
    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        if cmd == "Z":
            if pts:
                subpaths.append(pts)
                pts = []
            continue
        if cmd == "M":
            if pts:
                subpaths.append(pts)
            cursor = (float(tokens[i]), float(tokens[i + 1]))
            pts = [cursor]
            i += 2
        elif cmd == "L":
            cursor = (float(tokens[i]), float(tokens[i + 1]))
            pts.append(cursor)
            i += 2
        elif cmd == "C":
            nums = [float(v) for v in tokens[i:i + 6]]
            i += 6
            p0, (c1x, c1y), (c2x, c2y), p3 = cursor, nums[0:2], nums[2:4], (nums[4], nums[5])
            for step in range(1, 9):
                t = step / 8.0
                u = 1 - t
                pts.append((
                    u ** 3 * p0[0] + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t ** 3 * p3[0],
                    u ** 3 * p0[1] + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t ** 3 * p3[1],
                ))
            cursor = p3
    if pts:
        subpaths.append(pts)
    return subpaths


def _fill_iou(svg, truth):
    """Rasterise the filled paths of an SVG and compare with a truth mask."""
    root = ET.fromstring(svg)
    vb = [float(v) for v in root.attrib["viewBox"].split()]
    canvas = np.zeros(truth.shape, dtype=np.uint8)
    scale_x = truth.shape[1] / vb[2]
    scale_y = truth.shape[0] / vb[3]
    for path in root.findall(f".//{NS}path"):
        if path.attrib.get("fill", "none") == "none":
            continue
        for pts in _subpath_points(path.attrib["d"]):
            if len(pts) >= 3:
                scaled = [[(x - vb[0]) * scale_x, (y - vb[1]) * scale_y] for x, y in pts]
                cv2.fillPoly(canvas, [np.array(scaled, dtype=np.int32)], 255)
    a = canvas > 0
    b = truth > 0
    return np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))


def test_two_colour_image_becomes_paths_matching_the_shape():
    img = synthetic.blank(300, color=(255, 255, 255))
    cv2.circle(img, (150, 150), 90, (60, 90, 200), -1)
    cv2.circle(img, (150, 150), 40, (30, 30, 30), -1)
    svg, warnings = rv.vectorize_png(synthetic.encode_png(img), prefer_vtracer=False)
    assert svg and not warnings
    assert len(_paths(svg)) >= 2

    truth = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(truth, (150, 150), 90, 255, -1)
    assert _fill_iou(svg, truth) > 0.9


def test_alpha_defines_the_foreground():
    rgba = synthetic.rgba_cutout(300)
    svg, _ = rv.vectorize_png(synthetic.encode_png(rgba), prefer_vtracer=False)
    assert svg
    assert _fill_iou(svg, synthetic.blade_mask(300)) > 0.85


def test_holes_stay_transparent():
    img = synthetic.blank(300, color=(255, 255, 255))
    cv2.circle(img, (150, 150), 100, (80, 120, 60), -1)
    cv2.circle(img, (150, 150), 45, (255, 255, 255), -1)  # hole punched back to background
    svg, _ = rv.vectorize_png(synthetic.encode_png(img), prefer_vtracer=False)
    assert svg
    filled = [p for p in _paths(svg) if p.attrib.get("fill", "none") != "none"]
    assert filled and all(p.attrib.get("fill-rule") == "evenodd" for p in filled)
    # The ring path must contain two subpaths: outer boundary plus the hole.
    assert any(p.attrib["d"].count("M ") >= 2 for p in filled)


def test_stroke_style_emits_centrelines_not_fills():
    img = synthetic.blank(300, color=(255, 255, 255))
    cv2.rectangle(img, (60, 60), (240, 240), (40, 40, 40), 2)
    cv2.line(img, (60, 150), (240, 150), (40, 40, 40), 2)
    svg, _ = rv.vectorize_png(synthetic.encode_png(img), stroke_style=True, prefer_vtracer=False)
    assert svg
    paths = _paths(svg)
    assert paths and all(p.attrib.get("fill", "none") == "none" for p in paths)


def test_background_only_image_reports_a_warning():
    svg, warnings = rv.vectorize_png(synthetic.encode_png(synthetic.blank(120)), prefer_vtracer=False)
    assert svg is None and warnings


def test_vectorize_result_fills_svg_and_keeps_raster():
    img = synthetic.blank(300, color=(255, 255, 255))
    cv2.circle(img, (150, 150), 80, (70, 110, 180), -1)
    png = synthetic.encode_png(img)
    result = SymbolResult(raster_png=png, source="hf", style="Simple Symbol")
    rv.vectorize_result(result, style="Simple Symbol")
    assert result.is_vector and result.raster_png == png
    assert "param(fill)" in result.svg
    assert result.meta.get("viewbox") is not None


def test_vectorize_result_warns_and_keeps_raster_when_empty():
    result = SymbolResult(raster_png=synthetic.encode_png(synthetic.blank(120)), source="hf")
    rv.vectorize_result(result)
    assert not result.is_vector
    assert result.raster_png and result.warnings


def test_vectorize_result_is_a_no_op_for_vector_results():
    result = SymbolResult(svg="<svg/>", source="gemini")
    before = result.svg
    rv.vectorize_result(result)
    assert result.svg == before and not result.warnings
