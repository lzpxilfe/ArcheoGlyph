import xml.etree.ElementTree as ET

from archeoglyph.generators import svg_sanitize as ss

NS = "{http://www.w3.org/2000/svg}"


def test_extract_svg_handles_fences_and_prose():
    text = "Sure! Here you go:\n```svg\n<svg viewBox='0 0 10 10'><path d='M0,0'/></svg>\n```\nEnjoy."
    assert ss.extract_svg(text).startswith("<svg")
    assert ss.extract_svg(text).endswith("</svg>")
    assert ss.extract_svg("no svg here") is None


def test_script_and_event_handlers_are_removed():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<script>alert(1)</script>'
        '<path d="M0,0 L5,5" fill="#123456" onclick="alert(2)"/>'
        "</svg>"
    )
    clean, problems, stats = ss.sanitize_svg(svg)
    assert clean and "script" not in clean.lower() and "onclick" not in clean.lower()
    assert stats["geometry"] == 1
    assert any("script" in p for p in problems)


def test_external_references_and_styles_are_dropped():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<image href="https://example.invalid/x.png" width="10" height="10"/>'
        '<use xlink:href="#other" xmlns:xlink="http://www.w3.org/1999/xlink"/>'
        '<style>path{fill:red}</style>'
        '<path d="M0,0 L5,5" style="fill:url(#g)" fill="#222"/>'
        "</svg>"
    )
    clean, _problems, stats = ss.sanitize_svg(svg)
    assert clean
    for token in ("image", "use", "style", "href"):
        assert token not in clean.lower()
    assert stats["geometry"] == 1


def test_gradient_fills_are_replaced_not_kept():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<defs><linearGradient id="g"/></defs>'
        '<path d="M0,0" fill="url(#g)"/>'
        "</svg>"
    )
    clean, _p, _s = ss.sanitize_svg(svg)
    assert clean and "url(" not in clean and "linearGradient" not in clean


def test_param_placeholders_survive():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path d="M0,0 L1,1" fill="param(fill) #8b4513" stroke="param(outline) #111"/>'
        "</svg>"
    )
    clean, _p, _s = ss.sanitize_svg(svg)
    assert "param(fill) #8b4513" in clean and "param(outline) #111" in clean


def test_malformed_and_empty_documents_are_rejected():
    assert ss.sanitize_svg("<svg><path d='M0,0'")[0] is None
    assert ss.sanitize_svg("")[0] is None
    assert ss.sanitize_svg('<svg xmlns="http://www.w3.org/2000/svg"></svg>')[0] is None
    assert ss.sanitize_svg("<html><body>hi</body></html>")[0] is None


def test_entity_declarations_are_rejected():
    svg = '<!DOCTYPE svg [<!ENTITY x "y">]><svg xmlns="http://www.w3.org/2000/svg"><path d="M0,0"/></svg>'
    clean, problems, _s = ss.sanitize_svg(svg)
    assert clean is None and problems


def test_colour_statistics_are_reported():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path d="M0,0" fill="#111111" stroke="#222222"/>'
        '<path d="M1,1" fill="#333333" stroke="none"/>'
        "</svg>"
    )
    clean, _p, stats = ss.sanitize_svg(svg)
    assert stats["geometry"] == 2
    assert stats["colors"] == 3 and stats["fill_colors"] == 2
    assert ET.fromstring(clean).findall(f".//{NS}path")
