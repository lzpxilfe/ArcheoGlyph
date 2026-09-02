from archeoglyph.generators.symbol_result import SymbolResult


def test_coerce_wraps_svg_text_and_png_bytes():
    svg = SymbolResult.coerce("<svg xmlns='http://www.w3.org/2000/svg'/>", source="autotrace", style="Line")
    assert svg.is_vector and svg.extension == "svg" and svg.source == "autotrace"
    png = SymbolResult.coerce(b"\x89PNG", source="hf")
    assert not png.is_vector and png.extension == "png" and png.raster_png == b"\x89PNG"
    assert SymbolResult.coerce(None) is None


def test_hash_is_stable_and_distinct():
    a = SymbolResult(svg="<svg/>")
    b = SymbolResult(svg="<svg/>")
    c = SymbolResult(svg="<svg><g/></svg>")
    assert a.content_hash() == b.content_hash() != c.content_hash()
    assert len(a.content_hash()) == 16


def test_warnings_deduplicate_and_empty_detection():
    r = SymbolResult()
    assert r.is_empty
    r.add_warning("x")
    r.add_warning("x")
    r.add_warning("  ")
    assert r.warnings == ["x"]
