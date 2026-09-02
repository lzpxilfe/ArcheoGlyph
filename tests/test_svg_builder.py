import xml.etree.ElementTree as ET

from archeoglyph.generators.autotrace import svg_builder as sb

NS = "{http://www.w3.org/2000/svg}"


def _paths(svg):
    root = ET.fromstring(svg)
    return [el for el in root.iter(f"{NS}path")]


def test_build_svg_normalises_to_unit_box_and_parametrises():
    outline = [(100, 50), (300, 50), (300, 450), (100, 450)]
    lines = [[(150, 250), (250, 250)]]
    svg = sb.build_svg(outline, lines, fill_hex="#aabbcc", outline_hex="#112233")
    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 100 100"
    paths = _paths(svg)
    assert len(paths) == 3
    assert paths[0].attrib["fill"] == "param(fill) #aabbcc"
    assert paths[1].attrib["stroke"] == "param(outline) #112233"
    assert paths[1].attrib["stroke-width"].startswith("param(outline-width)")
    # The tall rectangle (200x400) must be centred horizontally inside the square.
    d = paths[0].attrib["d"]
    xs = [float(tok.split(",")[0]) for tok in d.replace("M ", "").replace(" Z", "").split(" L ")]
    assert abs((min(xs) + max(xs)) / 2 - 50.0) < 0.01
    assert max(xs) - min(xs) < 60


def test_finalize_crops_viewbox_and_injects_params():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800">'
        '<path d="M 400,100 L 600,100 L 600,700 L 400,700 Z" fill="#8b4513" stroke="none"/>'
        '<path d="M 400,100 L 600,100 L 600,700 L 400,700 Z" fill="none" stroke="#222222" stroke-width="2.0"/>'
        '<path d="M 450,300 L 550,300" fill="none" stroke="#555555" stroke-width="1.1"/>'
        "</svg>"
    )
    out, info = sb.finalize_svg(svg)
    root = ET.fromstring(out)
    vb = [float(v) for v in root.attrib["viewBox"].split()]
    assert vb[2] == vb[3]                        # square
    assert vb[2] > 600 and vb[2] < 700          # 600 tall + 6% padding each side
    assert abs((vb[0] + vb[2] / 2) - 500) < 0.01  # centred on the object
    assert info["fill"] == "#8b4513"
    assert info["outline"] == "#222222"
    assert info["outline_width"] == 2.0
    paths = _paths(out)
    assert paths[0].attrib["fill"].startswith("param(fill)")
    assert paths[1].attrib["stroke"].startswith("param(outline)")
    assert paths[2].attrib["stroke"] == "#555555"  # detail line untouched
    assert "ns0:" not in out


def test_finalize_keeps_viewbox_for_relative_paths_and_flags_empty():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50"><path d="m 1,1 l 5,5" stroke="#000"/></svg>'
    out, info = sb.finalize_svg(svg)
    assert ET.fromstring(out).attrib["viewBox"] == "0 0 50 50"
    assert info["viewbox"] == (0.0, 0.0, 50.0, 50.0)
    empty, info2 = sb.finalize_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>')
    assert info2.get("empty") is True


def test_finalize_returns_input_on_parse_error():
    text = "<svg><path d='M 0,0 L 1,1'"
    out, info = sb.finalize_svg(text)
    assert out == text and "parse_error" in info


def test_finalize_handles_svg_without_namespace():
    out, _ = sb.finalize_svg('<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="2" fill="#f00"/></svg>')
    root = ET.fromstring(out)
    assert root.tag == f"{NS}svg"
    assert root.find(f"{NS}circle").attrib["fill"] == "param(fill) #f00"


def test_provenance_is_embedded_and_replaces_previous_entries():
    from archeoglyph.generators.autotrace import svg_builder as sb

    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><title>old</title><path d="M0,0 L1,1"/></svg>'
    meta = {
        "source": "autotrace", "style": "Measured", "input": "sherd_012.jpg",
        "input_kind": "drawing", "plugin_version": "0.2.0", "created": "2026-09-02",
    }
    out = sb.add_provenance(svg, meta)
    root = ET.fromstring(out)
    titles = root.findall(f"{NS}title")
    descs = root.findall(f"{NS}desc")
    assert len(titles) == 1 and len(descs) == 1
    assert titles[0].text == "ArchaeoGlyph symbol"
    text = descs[0].text
    for expected in ("generator: autotrace", "style: Measured", "input image: sherd_012.jpg",
                     "input type: drawing", "ArchaeoGlyph: 0.2.0", "created: 2026-09-02"):
        assert expected in text
    # Geometry survives.
    assert root.findall(f"{NS}path")


def test_provenance_is_skipped_when_there_is_nothing_to_record():
    from archeoglyph.generators.autotrace import svg_builder as sb

    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M0,0"/></svg>'
    assert sb.add_provenance(svg, {}) == svg
    assert sb.add_provenance("not svg", {"source": "x"}) == "not svg"


def test_provenance_keeps_only_the_file_name_of_the_source_image():
    from archeoglyph.generators.symbol_result import SymbolResult

    result = SymbolResult(source="autotrace", style="Line")
    result.record_provenance(image_path="/home/someone/private/dig 2026/sherd.jpg")
    assert result.meta["input"] == "sherd.jpg"
    assert "private" not in str(result.meta)
