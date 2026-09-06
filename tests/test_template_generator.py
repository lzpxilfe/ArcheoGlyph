"""
Template catalog tests.

These exercise the pure resolver only; painting needs a Qt paint device, so
QGIS-dependent parts are covered by the manual check in QGIS.
"""

from archeoglyph.generators.template_generator import TemplateGenerator

COLOR = TemplateGenerator.COLOR


def _resolve(name):
    return TemplateGenerator.__dict__["_resolve_draw"](TemplateGenerator, name)


def test_every_catalog_entry_resolves_to_an_existing_painter():
    unresolved = []
    for name in TemplateGenerator.TEMPLATE_INFO:
        method, _extra = _resolve(name)
        if method is None:
            unresolved.append(name)
            continue
        assert hasattr(TemplateGenerator, method), f"{name} -> missing {method}"
    # Only genuinely generic entries may fall through to the default ellipse.
    assert unresolved == [], unresolved


def test_hearth_is_not_captured_by_the_pit_keyword():
    assert _resolve("Hearth / Fire Pit") == ("_draw_hearth", (COLOR,))
    assert _resolve("Pit") == ("_draw_pit", (COLOR,))
    assert _resolve("Test Pit") == ("_draw_test_pit", (COLOR,))


def test_variant_dispatch_is_preserved():
    assert _resolve("Kofun (Enpun)") == ("_draw_kofun_shape", ("enpun", COLOR))
    assert _resolve("Bronze Dagger (Liaoning-style)") == ("_draw_bronze_dagger_typology", ("liaoning", COLOR))
    assert _resolve("Pottery Rim Sherd (Section)") == ("_draw_pottery_sherd_section", ("rim", COLOR))
    assert _resolve("Projectile Point (Side-notched)") == ("_draw_projectile_point_typology", ("side_notched",))


def test_explicit_draw_key_overrides_keyword_matching():
    assert TemplateGenerator.TEMPLATE_INFO["Hearth / Fire Pit"]["draw"] == ("_draw_hearth", "COLOR")


def test_legacy_aliases_map_into_the_catalog():
    generator = TemplateGenerator.__new__(TemplateGenerator)
    for alias, canonical in TemplateGenerator.LEGACY_TEMPLATE_ALIASES.items():
        assert canonical in TemplateGenerator.TEMPLATE_INFO, f"{alias} -> unknown {canonical}"
        assert generator._normalize_template_type(alias) == canonical


def test_categories_cover_the_catalog():
    covered = set()
    for name, info in TemplateGenerator.TEMPLATE_INFO.items():
        assert info.get("category"), f"{name} has no category"
        assert info.get("default_color", "").startswith("#"), f"{name} has no default colour"
        covered.add(info["category"])
    assert covered <= {"artifacts", "structures", "remains", "features", "survey"}


def test_the_optional_svg_file_key_is_never_assumed(tmp_path):
    """
    Templates drawn in code carry no `file` key, and an entry that has one may
    still have no file on disk. Indexing it directly raised KeyError for every
    code-drawn template, which is most of the catalogue.
    """
    generator = TemplateGenerator(str(tmp_path))
    for name, info in TemplateGenerator.TEMPLATE_INFO.items():
        assert generator._template_file(info) == "", f"{name} resolved a file that is absent"

    (tmp_path / "resources" / "templates").mkdir(parents=True)
    (tmp_path / "resources" / "templates" / "pottery.svg").write_text("<svg/>", encoding="utf-8")
    assert generator._template_file({"file": "pottery.svg"}).endswith("pottery.svg")
    assert generator._template_file({}) == ""
    assert generator._template_file({"file": ""}) == ""
    assert generator._template_file(None) == ""


def test_generate_does_not_index_optional_catalog_keys():
    """The crash was a direct subscript; keep it from coming back."""
    import inspect

    source = inspect.getsource(TemplateGenerator.generate)
    assert "template_info['file']" not in source
    assert 'template_info["file"]' not in source


def test_every_template_takes_its_colour_from_the_palette():
    """
    A catalogue of 188 symbols needs a palette, not 188 opinions.

    These used to carry 110 distinct colours with saturation anywhere from 0
    to 100 percent - neon blue and pure yellow beside muted earths - which is
    why a map full of them never looked like one set. The palette is named by
    material, because that is how the finds are already grouped.
    """
    from archeoglyph.generators.icon_grid import PALETTE

    allowed = set(PALETTE.values())
    stray = sorted({
        info.get("default_color", "")
        for info in TemplateGenerator.TEMPLATE_INFO.values()
        if info.get("default_color", "") not in allowed
    })
    assert not stray, (
        f"colours outside the palette: {stray}. Pick the material the object "
        f"is made of from icon_grid.PALETTE."
    )


def test_the_palette_stays_a_narrow_band():
    """
    The palette varies by hue and holds everything else, which is what lets a
    map full of these read as one family.

    What broke the old set was not grey being grey - stone and iron belong at
    the bottom of the saturation range - but neon sitting beside earth, and
    values running from near-black to near-white. So the ceiling on saturation
    and the band on lightness are what is worth holding.
    """
    import colorsys

    from archeoglyph.generators.icon_grid import PALETTE

    for name, value in PALETTE.items():
        red, green, blue = (int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
        _hue, light, sat = colorsys.rgb_to_hls(red, green, blue)
        assert sat <= 0.55, f"{name} {value} is {sat:.0%} saturated - too loud for the set"
        assert 0.30 <= light <= 0.72, f"{name} {value} sits at {light:.0%} lightness"
