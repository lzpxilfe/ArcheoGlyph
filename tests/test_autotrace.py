import io as std_io
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from archeoglyph.generators.autotrace import colors, io as image_io, model_store, segment
from archeoglyph.generators.autotrace.options import AutoTraceOptions
from archeoglyph.generators.autotrace.pipeline import run_autotrace
from archeoglyph.generators.autotrace.round_motif import polar_unwrap
from tests import synthetic


def _iou(a, b):
    a = a > 0
    b = b > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / max(1, union)


# ---------------------------------------------------------------- io

def test_load_image_keeps_alpha_and_returns_bgr(tmp_path):
    path = synthetic.write_png(tmp_path / "cutout.png", synthetic.rgba_cutout())
    loaded = image_io.load_image(path)
    assert loaded.bgr.shape[2] == 3 and loaded.bgr.dtype == np.uint8
    assert loaded.alpha is not None
    assert _iou(loaded.alpha, synthetic.blade_mask()) > 0.999


def test_load_image_promotes_grayscale(tmp_path):
    gray = np.full((40, 60), 90, dtype=np.uint8)
    path = synthetic.write_png(tmp_path / "gray.png", gray)
    loaded = image_io.load_image(path)
    assert loaded.bgr.shape == (40, 60, 3) and loaded.alpha is None


def test_load_image_applies_exif_orientation(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (300, 200), (200, 200, 200))
    exif = PIL.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise
    path = tmp_path / "rot.jpg"
    img.save(path, exif=exif)
    loaded = image_io.load_image(str(path))
    assert loaded.bgr.shape[:2] == (300, 200)


# ---------------------------------------------------------------- segment

def test_alpha_channel_wins_over_heuristics():
    rgba = synthetic.rgba_cutout()
    mask = segment.select_mask(rgba[:, :, :3], backend="auto", alpha=rgba[:, :, 3])
    assert _iou(mask, synthetic.blade_mask()) > 0.98


def test_opencv_mask_recovers_blade_silhouette():
    mask = segment.get_mask_opencv(synthetic.ellipse_blade())
    # Blur + closing add a 1-2 px rim, so accept a slightly generous mask.
    assert _iou(mask, synthetic.blade_mask()) > 0.92


def test_opencv_mask_keeps_dark_object_on_white():
    img = synthetic.dark_flint_on_white()
    truth = (img[:, :, 0] < 100).astype(np.uint8) * 255
    mask = segment.get_mask_opencv(img)
    assert _iou(mask, truth) > 0.85


def test_smooth_mask_edges_fills_holes_even_when_object_touches_corner():
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[0:60, 0:60] = 255
    mask[20:30, 20:30] = 0  # hole
    out = segment.smooth_mask_edges(mask)
    assert out[25, 25] == 255
    assert out[100, 100] == 0


def test_select_mask_falls_back_when_model_fails():
    img = synthetic.ellipse_blade()

    def broken(_bgr):
        raise RuntimeError("model exploded")

    mask = segment.select_mask(img, backend="onnx", onnx_fn=broken)
    assert _iou(mask, synthetic.blade_mask()) > 0.92


# ---------------------------------------------------------------- colours / geometry

def test_dominant_color_is_exact_for_flat_object_and_deterministic():
    img = synthetic.ellipse_blade(color=(60, 90, 140))  # BGR
    mask = synthetic.blade_mask()
    assert colors.extract_dominant_color(img, mask) == "#8c5a3c"
    assert colors.extract_material_palette(img, mask) == colors.extract_material_palette(img, mask)


def test_polar_unwrap_is_radius_major():
    img = np.zeros((200, 200), dtype=np.uint8)
    import cv2
    cv2.circle(img, (100, 100), 60, 255, 2)
    polar = polar_unwrap(img, 100, 100, 100, n_theta=180, n_rad=100)
    assert polar.shape == (100, 180)
    row = int(np.argmax(polar.mean(axis=1)))
    assert abs(row - 59) <= 3


# ---------------------------------------------------------------- pipeline

def _run(img, **kw):
    opts = AutoTraceOptions(**kw)
    return run_autotrace(img, opts, lambda bgr: segment.get_mask_opencv(bgr))


def _path_count(svg):
    return len(ET.fromstring(svg).findall(".//{http://www.w3.org/2000/svg}path"))


@pytest.mark.parametrize("style", ["Simple Symbol", "Line", "Measured"])
def test_pipeline_is_deterministic_and_valid(style):
    img = synthetic.ellipse_blade()
    a = _run(img, style=style)
    b = _run(img, style=style)
    assert a == b
    root = ET.fromstring(a)
    assert root.attrib["viewBox"].startswith("0 0 ")
    assert _path_count(a) >= 1


def test_synthetic_structure_lines_are_opt_in():
    img = synthetic.ellipse_blade()
    plain = _run(img, style="Simple Symbol", synthetic_structure=False)
    schematic = _run(img, style="Simple Symbol", synthetic_structure=True)
    assert _path_count(schematic) > _path_count(plain)


def test_oval_keeps_its_outline_instead_of_a_circle():
    img = synthetic.blank(400)
    import cv2
    cv2.ellipse(img, (200, 200), (150, 110), 0, 0, 360, (60, 90, 140), -1)  # aspect 0.73 -> roundish
    svg = _run(img, style="Line")
    from archeoglyph.generators.autotrace import svg_builder
    body = ET.fromstring(svg).find(".//{http://www.w3.org/2000/svg}path").attrib["d"]
    x0, y0, x1, y1 = svg_builder._path_bbox(body)
    width = x1 - x0
    height = y1 - y0
    assert 0.65 < height / width < 0.82


def test_options_normalize_bad_values():
    opts = AutoTraceOptions(detail_mode="turbo", round_strategy="?", factuality=250, color="  ").normalized()
    assert opts.detail_mode == "fast" and opts.round_strategy == "image_first"
    assert opts.factuality == 100 and opts.color is None


# ---------------------------------------------------------------- model store

def _fake_urlopen(payload):
    class _Resp(std_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return lambda request, timeout=0: _Resp(payload)


def test_model_download_verifies_hash_and_size(tmp_path, monkeypatch):
    import hashlib

    payload = b"onnx-bytes" * 100
    good = model_store.ModelSpec(
        key="t", label="t", filename="t.onnx", url="https://example.invalid/t.onnx",
        sha256=hashlib.sha256(payload).hexdigest(), size=len(payload), input_size=8,
        mean=(0, 0, 0), std=(1, 1, 1),
    )
    monkeypatch.setattr(model_store.urllib.request, "urlopen", _fake_urlopen(payload))
    path = model_store.download_model(good, str(tmp_path))
    assert path.endswith("t.onnx") and model_store.is_installed(good, str(tmp_path))
    assert model_store.verify_model(good, str(tmp_path))

    bad = model_store.ModelSpec(**{**good.__dict__, "sha256": "0" * 64, "filename": "bad.onnx"})
    with pytest.raises(model_store.ModelStoreError):
        model_store.download_model(bad, str(tmp_path))
    assert not model_store.is_installed(bad, str(tmp_path))
    assert not [n for n in (tmp_path / "archeoglyph" / "models").iterdir() if n.name.endswith(".part")]


# ---------------------------------------------------------------- drawings

def test_line_drawing_input_keeps_inner_strokes():
    img = synthetic.line_drawing_sherd()
    explicit = _run(img, style="Line", input_kind="drawing")
    auto = _run(img, style="Line")
    assert explicit == auto, "auto detection should route the drawing the same way"
    assert _path_count(explicit) >= 2
    photo_mode = _run(img, style="Line", input_kind="photo")
    assert isinstance(photo_mode, str) and "<svg" in photo_mode


def test_grabcut_refinement_skips_instead_of_failing_on_a_full_mask():
    """
    A mask covering the whole frame leaves GrabCut no background to sample.
    That used to raise inside a silent except; it must return None instead.
    """
    img = synthetic.ellipse_blade()
    full = np.full(img.shape[:2], 255, dtype=np.uint8)
    assert segment.refine_with_grabcut(img, full) is None

    empty = np.zeros(img.shape[:2], dtype=np.uint8)
    assert segment.refine_with_grabcut(img, empty) is None


def test_grabcut_refinement_still_runs_when_both_classes_exist():
    img = synthetic.dark_flint_on_white()
    truth = (img[:, :, 0] < 100).astype(np.uint8) * 255
    refined = segment.refine_with_grabcut(img, truth)
    assert refined is not None
    assert _iou(refined, truth) > 0.85


def test_an_elongated_find_is_stood_upright():
    """
    Excavated material is photographed lying down - a blade is laid on the
    bench and shot from above - and traced as-is it comes out as a bar. The
    slender bronze dagger did exactly that: a horizontal lens that read as
    no artefact at all next to the upright drawn template.
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from archeoglyph.generators.autotrace.segment import stand_upright

    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(mask, (200, 200), (150, 24), 0, 0, 360, 255, -1)   # lying down
    bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    _rot_bgr, rot_mask = stand_upright(bgr, mask)
    contours, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _x, _y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    assert height > width * 2.0, (
        f"a {150 * 2}x{24 * 2} find came out {width}x{height}; an elongated "
        f"artefact has to stand up whatever way it was photographed"
    )


def test_standing_upright_leaves_an_already_upright_find_alone():
    """Rotating a find that is already upright only costs it interpolation."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from archeoglyph.generators.autotrace.segment import stand_upright

    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(mask, (200, 200), (24, 150), 0, 0, 360, 255, -1)
    bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    _rot_bgr, rot_mask = stand_upright(bgr, mask)
    assert rot_mask.shape == mask.shape
    assert int(np.abs(rot_mask.astype(int) - mask.astype(int)).sum()) == 0


def test_a_round_find_is_never_spun():
    """A mirror or a tile end has no long axis; turning one only blurs it."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from archeoglyph.generators.autotrace.segment import stand_upright

    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(mask, (200, 200), 150, 255, -1)
    bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    _rot_bgr, rot_mask = stand_upright(bgr, mask)
    assert rot_mask.shape == mask.shape
    assert int(np.abs(rot_mask.astype(int) - mask.astype(int)).sum()) == 0


def test_a_traced_outline_takes_the_artefact_colour():
    """
    A traced symbol has to look like it belongs to the drawn catalogue.

    In QGIS the outline colour is the user's, so a flat "#111111" cost
    nothing there. Everywhere else - preview sheets, documentation, any plain
    SVG viewer - it put a near-black ring around a traced artefact standing
    beside a catalogue drawn in its own muted colour, and the same stroke
    width read as far heavier than it is. The catalogue darkens the fill by
    QColor.darker(140); this does the same.
    """
    from archeoglyph.generators.autotrace import pipeline as pl

    img = synthetic.ellipse_blade()
    for style in ("Line", "Measured"):
        svg = _run(img, style=style)
        heaviest, colour = 0.0, None
        for el in ET.fromstring(svg).iter():
            stroke = str(el.attrib.get("stroke", "")).strip()
            if not stroke.startswith("#"):
                continue
            width = float(el.attrib.get("stroke-width", 0.0) or 0.0)
            if width > heaviest:
                heaviest, colour = width, stroke.split()[-1]
        assert colour is not None, f"{style} drew no stroked outline"
        assert colour != "#111111", (
            f"{style} still draws its outline in a flat near-black")
        channels = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
        assert max(channels) - min(channels) > 4, (
            f"{style} drew its outline in the grey {colour}, which carries "
            f"none of the artefact's own colour")

    assert pl.HOUSE_OUTLINE_DARKEN == pytest.approx(1.0 / 1.4), (
        "the catalogue darkens an outline with QColor.darker(140); these two "
        "have to stay in step or a traced symbol stops matching a drawn one")


def _photograph_like(size=420):  # noqa: E302
    """
    A find on a lit table, with a soft cast shadow.

    test_pipeline_is_deterministic_and_valid uses a flat ellipse on a flat
    ground, which the chroma split alone resolves - so GrabCut never runs and
    the determinism bug below went unseen through the whole suite. Soft
    shading and a shadow are what send the mask down the refinement path.
    """
    cv2 = pytest.importorskip("cv2")
    img = np.full((size, size, 3), 232, dtype=np.uint8)
    ramp = np.linspace(-16, 16, size, dtype=np.float32)[None, :, None]
    img = np.clip(img.astype(np.float32) + ramp, 0, 255).astype(np.uint8)
    centre = (size // 2, int(size * 0.46))
    cv2.ellipse(img, (centre[0], int(size * 0.78)), (int(size * 0.28), int(size * 0.07)),
                0, 0, 360, (198, 200, 203), -1)          # the cast shadow
    cv2.GaussianBlur(img, (31, 31), 0, dst=img)
    cv2.circle(img, centre, int(size * 0.27), (96, 104, 122), -1)
    cv2.circle(img, centre, int(size * 0.13), (74, 82, 100), -1)
    return img


def test_the_same_photograph_gives_the_same_mask():
    """
    GrabCut seeds its colour models with k-means, and OpenCV's k-means draws
    from one global RNG whose state advances with every call. Unpinned, the
    same photograph gave a different silhouette every time it was traced: on
    eight of nine Korean finds three calls in a row returned three different
    masks, and the same three in the same order in a fresh process, so a user
    pressing the button twice got two different symbols.

    Three calls, not two: the second and third differed from each other as
    well as from the first.
    """
    img = _photograph_like()
    masks = [segment.get_mask_opencv(img.copy()) for _ in range(3)]
    first = masks[0]
    assert np.count_nonzero(first) > 0, "the fixture gave no silhouette to compare"
    for index, other in enumerate(masks[1:], start=2):
        differing = int(np.count_nonzero(first != other))
        assert differing == 0, (
            f"call {index} returned a mask differing from the first in "
            f"{differing} pixels; the silhouette is not a function of the image")


def test_the_grabcut_vote_is_not_a_single_draw():
    """
    Pinning one seed is deterministic but only freezes one draw out of the
    spread, and a single draw can be a bad one - on the lotus tile the first
    fixed seed swallowed the whole white support block, twice the area of any
    unseeded run. An odd number of seeds keeps the majority from tying.
    """
    assert len(segment.GRABCUT_SEEDS) >= 3, (
        "one or two seeds is a draw, not a vote")
    assert len(segment.GRABCUT_SEEDS) % 2 == 1, (
        "an even number of seeds can tie on a boundary pixel")
    assert len(set(segment.GRABCUT_SEEDS)) == len(segment.GRABCUT_SEEDS)


def test_a_mark_too_small_to_see_at_legend_size_is_not_drawn():
    """
    A symbol is 64 grid units and a legend shows it at 64 pixels, so a unit is
    a legend pixel and icon_grid.DETAIL - the internal line weight - is
    exactly one. A mark spanning two or three pixels is not a line at that
    size, it is a speck, and the styles were emitting twenty-four of them
    inside a round artefact.
    """
    from archeoglyph.generators.autotrace.geometry import (
        LEGEND_MARK_MIN_SPAN, keep_marks_that_read)

    extent = 640.0
    floor = LEGEND_MARK_MIN_SPAN * extent
    speck = [[100, 100], [100 + floor * 0.4, 100 + floor * 0.4]]
    stroke = [[100, 100], [100 + floor * 3.0, 100 + floor * 3.0]]

    kept = keep_marks_that_read([speck, stroke], extent)
    assert len(kept) == 1, f"expected the speck dropped and the stroke kept, got {kept}"
    assert kept[0] == stroke

    assert keep_marks_that_read([speck] * 20, extent) == [], (
        "twenty specks are still twenty specks")


def test_the_ink_budget_is_paid_in_marks_not_in_invisible_lines():
    """
    The budget used to be paid entirely in stroke weight, and on the two roof
    tile ends that put the interior line at 0.53 and 0.71 of a grid unit. A
    unit is a legend pixel, so the ornament was drawn thinner than the legend
    can show: four hundred curves, none of them visible.

    Past the floor the bill is settled by dropping marks instead, longest
    first, because a mark that cannot be seen carries nothing.
    """
    from archeoglyph.generators.autotrace.geometry import (
        keep_marks_within_ink_budget, polyline_length)

    long_line = [[0, 0], [100, 0]]
    short_line = [[0, 10], [10, 10]]
    lines = [short_line, long_line, short_line]

    kept, drawn = keep_marks_within_ink_budget(lines, 120.0)
    assert kept == lines, "it all fits; nothing should be dropped"
    assert drawn == pytest.approx(120.0)

    kept, drawn = keep_marks_within_ink_budget(lines, 105.0)
    assert kept == [long_line], "the long mark carries the most; it stays"
    assert drawn == pytest.approx(100.0)

    # Never nothing: one mark is kept even when the budget cannot afford it,
    # because a symbol with a blank interior is a worse answer than a heavy one.
    kept, drawn = keep_marks_within_ink_budget(lines, 1.0)
    assert kept == [long_line]

    assert polyline_length(long_line) == pytest.approx(100.0)
    assert polyline_length([[0, 0]]) == 0.0


def test_a_round_artefact_keeps_its_interior_lines_at_legend_weight():
    """
    The contract the budget may not break: whatever it does to fit the ink
    inside the catalogue's band, what is left on the page is drawn at a weight
    the legend can render. A symbol is 64 units shown at 64 pixels, so a grid
    unit is a legend pixel and icon_grid.DETAIL is the floor.
    """
    from archeoglyph.generators import icon_grid
    from archeoglyph.generators.autotrace import svg_builder as sb

    img = synthetic.mirror_with_rings()
    for style in ("Line", "Measured"):
        out, info = sb.finalize_svg(_run(img, style=style))
        side = float(info["viewbox"][2])
        widths = []
        for node in ET.fromstring(out).iter():
            raw = node.attrib.get("stroke-width")
            if raw is None:
                continue
            widths.append(float(re.search(r"[\d.]+$", raw.strip()).group(0)))
        assert widths, f"{style} drew no strokes"
        floor = side * icon_grid.DETAIL / icon_grid.UNITS
        assert min(widths) >= floor * 0.999, (
            f"{style} drew a {min(widths):.2f} stroke on a {side:.0f} symbol; "
            f"the legend floor is {floor:.2f}. The budget has to be paid in "
            f"marks once the weight reaches the floor, not in more thinning")


def test_the_two_relief_readings_are_different_pictures():
    """
    Decoration is either cut into the surface or raised out of it, and the two
    want different ink. INCISED inks the dark side - the shadow in a groove -
    which is the drawing on a lotus roof tile end. MODELLED inks where the
    relief changes fastest, which is the drawing on a dragon tile, whose body
    is raised: the groove reading finds only the shadowed flank of each coil
    and returns squiggles where this returns the coil.

    Which one an artefact wants cannot be told from the photograph - the mean
    mark width is 1.79 percent of the artefact on the lotus tile and 1.70 on
    the dragon - so both are traced and merged. This is what makes that worth
    doing: they are not the same picture.
    """
    cv2 = pytest.importorskip("cv2")
    from archeoglyph.generators.autotrace.enhance import (
        INCISED, MODELLED, relief_ink_sheet)

    size, radius = 400, 150
    centre = (size // 2, size // 2)
    face = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(face, centre, radius, 255, -1)

    plate = np.full((size, size), 150, dtype=np.uint8)
    cv2.circle(plate, (centre[0] - 60, centre[1]), 34, 96, -1)   # a cut hollow
    cv2.circle(plate, (centre[0] + 60, centre[1]), 34, 205, -1)  # a raised boss
    bgr = cv2.cvtColor(cv2.GaussianBlur(plate, (0, 0), 3.0), cv2.COLOR_GRAY2BGR)

    def _ink(reading):
        sheet = relief_ink_sheet(bgr, face, radius, reading=reading)
        return cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY) < 128

    groove, relief = _ink(INCISED), _ink(MODELLED)
    assert groove.any() and relief.any(), "a reading returned no ink at all"

    overlap = float((groove & relief).sum()) / float((groove | relief).sum())
    assert overlap < 0.75, (
        f"the two readings agree on {overlap:.0%} of their ink; if they were "
        f"the same picture there would be nothing to gain by merging them")

    # The groove reading is the default, so a caller that does not name one
    # still gets what it got before the split.
    default = cv2.cvtColor(relief_ink_sheet(bgr, face, radius),
                           cv2.COLOR_BGR2GRAY) < 128
    assert (default == groove).all()


def test_a_traced_symbol_is_no_busier_than_the_busiest_drawn_one():
    """
    The cap comes from the catalogue this has to sit beside: over its 188
    symbols the median artefact carries 2 interior marks, the ninetieth
    percentile 5, and the busiest 11. A traced symbol may be as busy as the
    busiest drawn one and no busier - it was carrying 24.
    """
    from archeoglyph.generators.autotrace.geometry import (
        MAX_INTERIOR_MARKS, keep_marks_that_read)

    extent = 640.0
    long_enough = extent * 0.3
    lines = [[[0, i * 4], [long_enough * (1.0 - i * 0.01), i * 4]]
             for i in range(40)]
    kept = keep_marks_that_read(lines, extent)
    assert len(kept) == MAX_INTERIOR_MARKS

    spans = [max(p[0] for p in line) - min(p[0] for p in line) for line in kept]
    assert spans == sorted(spans, reverse=True), (
        "the cap kept an arbitrary eleven; it has to keep the largest eleven")


def test_bold_interior_structure_survives_the_filter():
    """
    The filter must not be a way of drawing nothing. A mirror with two bold
    concentric rings has interior structure that belongs in the symbol.
    """
    img = synthetic.mirror_with_rings()
    for style in ("Line", "Measured"):
        svg = _run(img, style=style)
        assert _path_count(svg) >= 2, (
            f"{style} kept only the silhouette of a disc with two bold rings")


def test_relief_becomes_ink_and_the_lighting_does_not():
    """
    The decoration on a roof tile end is height, and a photograph carries
    height only as shading - which is why reading marks straight off the
    photograph produced lighting artefacts, including a band across three
    quarters of a lotus tile's face where its lit and shadowed halves met.

    Subtracting a wide blur removes the lamp, which is broad, and keeps the
    grooves, which are not. What is left is a rubbing of the object.
    """
    cv2 = pytest.importorskip("cv2")
    from archeoglyph.generators.autotrace.enhance import relief_ink_sheet

    size, radius = 400, 150
    centre = (size // 2, size // 2)
    face = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(face, centre, radius, 255, -1)

    plate = np.full((size, size), 150, dtype=np.uint8)
    for step in range(8):                       # eight grooves, the decoration
        angle = 2.0 * np.pi * step / 8.0
        cv2.line(plate, centre,
                 (int(centre[0] + radius * 0.85 * np.cos(angle)),
                  int(centre[1] + radius * 0.85 * np.sin(angle))), 96, 5)
    lamp = np.linspace(-46, 46, size, dtype=np.float32)[None, :]
    lit = np.clip(plate.astype(np.float32) + lamp, 0, 255).astype(np.uint8)

    sheet = relief_ink_sheet(cv2.cvtColor(lit, cv2.COLOR_GRAY2BGR), face, radius)
    ink = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY) < 128

    left = int(ink[:, :size // 2].sum())
    right = int(ink[:, size // 2:].sum())
    assert left > 0 and right > 0, "the grooves did not survive at all"
    assert min(left, right) > 0.45 * max(left, right), (
        f"the ink is lopsided - {left} on the lit side against {right} on the "
        f"shadowed one - so the lamp came through as decoration")


def test_a_flat_lit_disc_yields_almost_no_ink():
    """A plain disc under the same lamp has nothing to draw."""
    cv2 = pytest.importorskip("cv2")
    from archeoglyph.generators.autotrace.enhance import relief_ink_sheet

    size, radius = 400, 150
    face = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(face, (size // 2, size // 2), radius, 255, -1)
    lamp = np.linspace(-46, 46, size, dtype=np.float32)[None, :]
    flat = np.clip(np.full((size, size), 150, dtype=np.float32) + lamp,
                   0, 255).astype(np.uint8)

    sheet = relief_ink_sheet(cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR), face, radius)
    ink = int((cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY) < 128).sum())
    assert ink < 0.02 * int((face > 0).sum()), (
        f"a plain disc under a lamp produced {ink} pixels of ink; the lamp is "
        f"being drawn as decoration")


def _interior_ink_share(svg):
    """Arc length times stroke width, over the symbol's own box."""
    import re as _re

    box = float(_re.search(r'viewBox="([^"]+)"', svg).group(1).split()[2])
    total = 0.0
    for path in _re.finditer(
            r'<path[^>]*d="([^"]+)"[^>]*stroke-width="(?:param\(outline-width\) )?([\d.]+)"',
            svg):
        numbers = [float(n) for n in _re.findall(r"-?\d+\.?\d*", path.group(1))]
        xs, ys = numbers[0::2], numbers[1::2]
        if len(xs) < 2:
            continue
        arc = sum(float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
                  for i in range(len(xs) - 1))
        total += arc * float(path.group(2))
    return total / max(box * box, 1.0)


def test_a_dense_drawing_is_drawn_lighter_not_thicker():
    """
    A stroke weight chosen for a symbol with five marks buries a plate with
    four hundred traced curves: measured before this, the two roof tile ends
    laid down 58 and 61 percent of their own tile in ink, where the drawn
    catalogue's busiest symbol covers 48. The weight is scaled until the
    drawing lands back at the catalogue's median.
    """
    from archeoglyph.generators.autotrace import pipeline as pl

    cv2 = pytest.importorskip("cv2")
    size = 460
    img = np.full((size, size, 3), 236, dtype=np.uint8)
    centre = (size // 2, size // 2)
    cv2.circle(img, centre, 170, (120, 126, 140), -1)
    rng = np.random.default_rng(11)
    for _ in range(260):                        # a densely decorated face
        angle = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.15, 0.9) * 170
        start = (int(centre[0] + rad * np.cos(angle)),
                 int(centre[1] + rad * np.sin(angle)))
        end = (int(start[0] + rng.uniform(-30, 30)),
               int(start[1] + rng.uniform(-30, 30)))
        cv2.line(img, start, end, (74, 78, 92), 3)

    for style in ("Line", "Measured"):
        share = _interior_ink_share(_run(img, style=style))
        assert share <= pl.INTERIOR_INK_CEILING * 1.05, (
            f"{style} covered {share * 100:.0f}% of the symbol in ink against "
            f"the catalogue's {pl.INTERIOR_INK_CEILING * 100:.0f}%")


def test_the_ink_budget_comes_from_the_drawn_catalogue():
    """
    Both numbers are measurements of the 188 drawn symbols, not choices: the
    median covers 27 percent of its tile and the busiest 1.76 times that.
    """
    from archeoglyph.generators.autotrace import pipeline as pl

    assert pl.INTERIOR_INK_MEDIAN == pytest.approx(0.27)
    assert pl.INTERIOR_INK_CEILING == pytest.approx(1.76 * pl.INTERIOR_INK_MEDIAN)
    assert pl.INTERIOR_INK_MEDIAN < pl.INTERIOR_INK_CEILING, (
        "the target has to sit below the trigger or the scaling would fight "
        "itself")
