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
def test_a_decorated_disc_does_not_come_out_as_a_plain_one():
    """
    The marker style draws two interior marks, which is right for a silhouette
    that carries its own meaning and useless for a decorated disc: a lotus
    roof tile end and a plain disc were coming out as the same grey circle,
    which was the largest thing this set got wrong.

    So a flat decorated face is traced whatever the style asked for, and when
    the trace finds more readable marks than the busiest drawn symbol carries,
    the marker draws that ornament instead of its two structural cues.
    """
    plain = _path_count(_run(synthetic.plain_disc(), style="Simple Symbol"))
    rosette = _path_count(_run(synthetic.rosette_disc(), style="Simple Symbol"))

    # Six, because six petals is the drawn register: this project's own
    # 수막새 symbol is a circle, six petals and a boss, and a rosette read
    # from a photograph comes out as one shape per fold - eight here. The
    # margin used to be eight, set when a bare disc drew fourteen paths of
    # read noise and a rosette twenty-seven; with the noise gone a bare disc
    # draws three and the rosette its petals, and eight would fail a
    # correctly read six-petal tile.
    assert rosette >= plain + 6, (
        f"a rosette drew {rosette} paths and a bare disc of the same size "
        f"{plain}; at marker size the two artefacts read as the same object")

    # And the control holds in the other direction: a disc with two rings and
    # a boss is not "decorated" in this sense - it is inside the count a drawn
    # symbol carries, so it keeps the marker's own structural reading, which
    # is a smaller thing than the rosette's ornament.
    #
    # The margin here used to be measured against the bare disc, and it is
    # not any more, because the bare disc got much cleaner when the relief
    # reading stopped thresholding a slope: it drew fourteen paths of read
    # noise and now draws three. A rings-and-boss mirror is compared against
    # the rosette instead - the two readings this is meant to tell apart -
    # and against the bare disc only for the direction of the difference.
    rings = _path_count(_run(synthetic.mirror_with_rings(),
                             style="Simple Symbol"))
    assert rosette >= rings + 6, (
        f"a rosette drew {rosette} paths and two rings and a boss {rings}; "
        "the ornament route is firing on the structural artefact")
    assert rings > plain, (
        f"two rings and a boss drew {rings} paths against a bare disc's "
        f"{plain}; the structural reading has stopped saying anything")


def test_only_a_vessel_gets_the_rim_and_shoulder_bands():
    """
    Structural bands are the *shape* of a pot - where its outline changes
    curvature - and they fire on anything at all unless they are gated: a
    bronze mirror and a slender dagger get them as readily as a jar.

    The gate reads the silhouette and never the photograph, deliberately: six
    attempts to find a vessel's decorated zone in its photograph all failed,
    and this is not a seventh. An open pot is wide at the rim, widest near the
    top, and narrower at the base.
    """
    from archeoglyph.generators.autotrace.segment import get_mask_opencv
    from archeoglyph.generators.autotrace.structure import looks_like_a_vessel

    def _is_vessel(img):
        return looks_like_a_vessel(get_mask_opencv(img))

    assert _is_vessel(synthetic.open_vessel()), "a pot is a vessel"
    assert not _is_vessel(synthetic.plain_disc()), "a disc is not a vessel"
    assert not _is_vessel(synthetic.ellipse_blade()), "a blade is not a vessel"
    assert not _is_vessel(synthetic.mirror_with_rings()), (
        "a deep bowl passes the pipeline's is_roundish test and so does a "
        "mirror; the profile has to do the separating, not the roundness")


def test_a_vessel_is_drawn_with_its_rim_and_a_blade_is_not():
    vessel = _path_count(_run(synthetic.open_vessel(), style="Measured"))
    blade = _path_count(_run(synthetic.ellipse_blade(), style="Measured"))
    assert vessel > blade, (
        f"the pot drew {vessel} paths and the blade {blade}; the pot should "
        f"carry a rim and a shoulder that the blade has no business having")


def test_a_symbol_without_a_typology_code_is_unchanged():
    """
    The code is opt-in. An empty one has to leave the symbol exactly as it
    was, byte for byte, or every user who does not classify their finds pays
    for a feature they did not ask for.
    """
    img = synthetic.ellipse_blade()
    assert _run(img, style="Measured") == _run(img, style="Measured", type_code="")
    assert _run(img, style="Measured") == _run(img, style="Measured", type_code="   ")


def test_a_typology_code_is_drawn_where_it_can_be_read():
    """
    Inside the artefact when it fits there and underneath it when it does not:
    a slender blade is narrow enough that fitting five characters across it
    shrank them to a squint, so past a floor the code goes under the drawing,
    which is what an archaeological plate does anyway.

    Whichever it is, the strokes are drawn at the legend floor - the code is
    of no use in a symbol too small to read it.
    """
    import xml.etree.ElementTree as ET

    from archeoglyph.generators import icon_grid
    from archeoglyph.generators.autotrace import svg_builder as sb

    for name in ("ellipse_blade", "open_vessel", "plain_disc"):
        img = getattr(synthetic, name)()
        bare = _run(img, style="Measured")
        coded = _run(img, style="Measured", type_code="IIa2b")
        assert _path_count(coded) > _path_count(bare), f"{name} drew no code"

        out, info = sb.finalize_svg(coded)
        side = float(info["viewbox"][2])
        widths = []
        for node in ET.fromstring(out).iter():
            raw = node.attrib.get("stroke-width")
            if raw is not None:
                widths.append(float(re.search(r"[\d.]+$", raw.strip()).group(0)))
        floor = side * icon_grid.DETAIL / icon_grid.UNITS
        assert min(widths) >= floor * 0.999, (
            f"{name} drew a {min(widths):.2f} stroke on a {side:.0f} symbol; "
            f"the legend floor is {floor:.2f}")


def test_a_typology_code_is_set_tall_enough_to_read():
    """
    Measured where the code is actually drawn, not where it was requested.

    The stroke width had a floor and the height had none, so a code could be
    shrunk to fit inside a narrow artefact and still pass: at the old floor of
    0.09 the cap height came to 5.8 legend pixels over a four-row glyph, which
    is 1.44 pixels a row against a stroke a whole pixel wide - an E with its
    bars touching. The floor is now the grid's own, two detail units a row.
    """
    from archeoglyph.generators import icon_grid
    from archeoglyph.generators.autotrace import pipeline as pl
    from archeoglyph.generators.autotrace import svg_builder as sb
    from archeoglyph.generators.autotrace.stroke_font import CELL_H
    from tests.svg_appearance import LEGEND_PX

    for name in ("ellipse_blade", "open_vessel", "plain_disc"):
        img = getattr(synthetic, name)()
        bare = set(re.findall(r'<path[^>]*/>', _run(img, style="Measured")))
        for code in ("II", "IIa2b", "IIIabcd"):
            coded = _run(img, style="Measured", type_code=code)
            added = [p for p in re.findall(r'<path[^>]*/>', coded)
                     if p not in bare]
            assert added, f"{name} drew no code for {code}"

            numbers = [float(v) for path in added
                       for d in re.findall(r'\sd="([^"]*)"', path)
                       for v in re.findall(r'-?\d+(?:\.\d+)?', d)]
            assert numbers, f"{name}'s {code} paths carried no coordinates"
            ys = numbers[1::2]
            _out, info = sb.finalize_svg(coded)
            side = float(info["viewbox"][2])
            cap = (max(ys) - min(ys)) / side * LEGEND_PX
            rows = cap / CELL_H
            assert rows >= 2.0 * icon_grid.DETAIL * 0.999, (
                f"{name} set {code} {cap:.1f} legend pixels tall, which is "
                f"{rows:.2f} pixels a glyph row against a "
                f"{icon_grid.DETAIL:.0f} pixel stroke - the rows close up")
            assert cap >= pl.TYPE_CODE_MIN_HEIGHT * LEGEND_PX * 0.999


def test_a_typology_code_is_trimmed_and_bounded():
    from archeoglyph.generators.autotrace.options import (
        MAX_TYPE_CODE, AutoTraceOptions)

    assert AutoTraceOptions(type_code="  IIa2b  ").normalized().type_code == "IIa2b"
    assert AutoTraceOptions(type_code="II  a").normalized().type_code == "II a"
    long_code = AutoTraceOptions(type_code="X" * 40).normalized().type_code
    assert len(long_code) == MAX_TYPE_CODE
    assert AutoTraceOptions().normalized().type_code == ""


def test_a_vessel_band_is_drawn_once():
    """
    The rim and shoulder come from estimate_profile_bands, and so do the
    schematic structure lines the user can switch on. The vessel reading was
    added outside the block that clears the schematic ones, so with that switch
    on its first band was drawn twice - the identical polyline, laid down four
    times once each is haloed. Measured on a photographed comb pot: 15 paths
    became 19, one of them repeated four times.

    Asserted on the drawing rather than on how many times the estimator was
    asked. The count was a stand-in for this, and only a good one while both
    readings came off a single call: what actually keeps them apart is that
    the vessel bands are read at all only when the schematic list came out
    empty, so the two are never both drawn however many calls it takes.

    Neither form of this test reproduces the original fault on this control -
    deleting the guard leaves the open-vessel output byte for byte the same,
    because the ink budget trims back to the same marks. It is kept as a
    statement of the property, not as a demonstration of the bug, and that is
    worth saying rather than leaving the next reader to assume it has teeth
    it does not have.
    """
    from collections import Counter

    img = synthetic.open_vessel()
    for synthetic_structure in (False, True):
        svg = _run(img, style="Measured",
                   synthetic_structure=synthetic_structure)
        repeats = Counter(re.findall(r'\sd="([^"]*)"', svg))
        worst, count = repeats.most_common(1)[0]
        # Twice is this style's halo - every interior line is laid down at two
        # stroke widths. Four times is one band drawn from both readings.
        assert count <= 2, (
            f"synthetic_structure={synthetic_structure}: one geometry was "
            f"drawn {count} times, so the same band is in internal_lines "
            f"twice ({worst[:60]}...)")


def test_a_shorter_band_list_is_not_a_prefix_of_a_longer_one():
    """
    Which is why the pipeline asks for the count it means to spend.

    estimate_profile_bands picks its candidates by curvature strength and then
    sorts what it kept by position, so the first of two is not the one it
    would have chosen if asked for one. Reading two and slicing to one - which
    the pipeline did, to save a call - therefore drew a band the reading never
    selected: on this control, one band is the shoulder at y=234 and two are
    [191, 234], so the slice handed back y=191.
    """
    from archeoglyph.generators.autotrace.structure import estimate_profile_bands

    mask = segment.get_mask_opencv(synthetic.open_vessel(400))
    one, two = estimate_profile_bands(mask, 1), estimate_profile_bands(mask, 2)
    assert one and len(two) == 2
    assert two[:1] != one, (
        "the first of two bands now equals the one band, so the prefix trick "
        "would be safe again - but the ordering that made it unsafe is the "
        "estimator's own, and this test is what would notice it changing")


def test_a_straight_sided_silhouette_is_not_a_vessel_by_default():
    """
    The vessel test asks where the silhouette is widest. np.argmax reports the
    *first* row of a tie, so a straight-sided shape - every row the same width -
    answered "at 0.0 of my height", i.e. always a vessel. The plateau's centre
    is the honest answer to the same question, and on the nine photographed
    finds it widens the margin rather than narrowing it: the two pots read 0.08
    and 0.18 against 0.40 for the nearest thing that is not a pot.
    """
    import numpy as np

    from archeoglyph.generators.autotrace.structure import looks_like_a_vessel

    # A straight-sided box: every row ties at the maximum width.
    box = np.zeros((200, 200), np.uint8)
    box[20:190, 40:160] = 255
    assert not looks_like_a_vessel(box), (
        "a box with parallel sides is not an open vessel")

    # A box that narrows only at the very foot is still not one.
    footed = box.copy()
    footed[150:190, 40:160] = 0
    footed[150:190, 70:130] = 255
    assert not looks_like_a_vessel(footed)

    # A real pot still reads as one.
    from archeoglyph.generators.autotrace.segment import get_mask_opencv
    assert looks_like_a_vessel(get_mask_opencv(synthetic.open_vessel()))


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

# ------------------------------------------------- telling symbols apart

#: The subjects the discrimination test compares, all synthetic: the real
#: photographs are not committed, and a control can be made twice under two
#: lamps, which is the comparison that matters most here.
def _discrimination_subjects():
    return {
        "plain disc": synthetic.lit_plain_disc(600),
        "mirror rings": synthetic.mirror_with_rings(600),
        "knobbed disc": synthetic.knobbed_disc(600, knobs=2),
        "vessel": synthetic.open_vessel(600),
        "blade": synthetic.ellipse_blade(600),
        "rosette 6": synthetic.lit_relief_disc(600, folds=6),
        "rosette 8": synthetic.lit_relief_disc(600, folds=8),
        "rosette 9": synthetic.lit_relief_disc(600, folds=9),
        "rosette 8 softbox": synthetic.diffuse_relief_disc(600, folds=8),
        "rosette 9 softbox": synthetic.diffuse_relief_disc(600, folds=9),
    }


#: Tracing ten subjects is not free, so the maps are built once and kept -
#: the same bargain test_template_drawing strikes for its 188 painted ones.
_TRACED_CACHE = {}


def _traced_appearances(style="Measured"):
    from archeoglyph.generators.autotrace.segment import get_mask_opencv
    from tests.svg_appearance import appearance

    if style not in _TRACED_CACHE:
        maps = {}
        for name, image in _discrimination_subjects().items():
            result = run_autotrace(image, AutoTraceOptions(style=style),
                                   get_mask_opencv)
            maps[name] = appearance(result if isinstance(result, str)
                                    else result.svg)
        _TRACED_CACHE[style] = maps
    return _TRACED_CACHE[style]


def test_no_traced_symbol_is_another_at_legend_size():
    """
    The catalogue's own bar, applied to the tracer.

    test_template_drawing holds the 188 drawn symbols to this: rasterise at
    the size a legend draws them and fail when two are the same picture. The
    traced symbols were never held to anything, and a symbol nobody can tell
    from the next one is not worth the photograph it came from.
    """
    from tests.svg_appearance import LEGEND_PX, MAX_OVERLAP, overlap

    maps = _traced_appearances()
    names = sorted(maps)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            likeness = overlap(maps[first], maps[second])
            assert likeness < MAX_OVERLAP, (
                f"{first} and {second} are the same picture at "
                f"{LEGEND_PX}px ({likeness:.3f}); a legend cannot tell them "
                f"apart, so the reading is drawing nothing that separates "
                f"them")


def test_one_artefact_under_two_lamps_is_one_symbol():
    """
    The other half of the bar, and the half that caught the real defect.

    Two symbols being different is only worth something if two photographs of
    ONE artefact come out the same; otherwise the reading is drawing the
    lighting. It was: a disc lit by a lamp and the same disc under a softbox
    overlapped 0.256, which is less than a rosette overlaps a plain disc, and
    less than two different fold counts overlap each other. The phase, the
    element's length and the boss radius were all being read off a signal
    that changes with the lamp. Now they are conventions keyed to the
    measured count, and the pair scores 0.612.

    So the floor is not a tuned number - it is the claim that a change of
    lamp matters LESS than a change of artefact, which is what "the symbol is
    the artefact's" means.
    """
    from tests.svg_appearance import overlap

    maps = _traced_appearances()
    types = max(overlap(maps[f"rosette {a}"], maps[f"rosette {b}"])
                for a, b in ((6, 8), (8, 9), (6, 9)))
    for folds in (8, 9):
        lamps = overlap(maps[f"rosette {folds}"],
                        maps[f"rosette {folds} softbox"])
        assert lamps > types, (
            f"a {folds}-fold rosette photographed under two lamps agrees "
            f"{lamps:.3f} with itself, but two different fold counts agree "
            f"{types:.3f} - the symbol is recording the lighting rather than "
            f"the artefact")
        assert lamps >= 0.50, (
            f"a {folds}-fold rosette under two lamps agrees only "
            f"{lamps:.3f} with itself")
