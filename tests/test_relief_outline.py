# -*- coding: utf-8 -*-
"""
A raised element's boundary, drawn as a closed curve.

The controls here are synthetic *photographs* of relief - a height field this
project writes, shaded by a lamp - rather than drawn grooves, because the
reading under test puts the shading back together into a height before it
draws anything. The ground truth is therefore known by construction: eight
petals, a boss and a raised rim on one disc, and nothing at all on the other.

Two properties matter more than any count. A curve must be closed, because a
boundary is; and the same artefact photographed at two resolutions must be
read the same way, because the earlier readings were not - a quantile of the
slope gave 5 curves at one working size and 12 at another, which is what sent
this module back to the drawing board.
"""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from archeoglyph.generators.autotrace import relief_outline as ro  # noqa: E402
from archeoglyph.generators.autotrace.segment import get_mask_opencv  # noqa: E402

from . import synthetic  # noqa: E402


def _read(image):
    """``(curves, mask, radius)`` for one synthetic photograph."""
    mask = get_mask_opencv(image)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    assert contours, "the control image has no silhouette"
    _x, _y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    radius = max(width, height) / 2.0
    return ro.raised_outlines(image, mask, radius), mask, radius


def test_a_decorated_face_comes_back_as_several_closed_curves():
    curves, _mask, _radius = _read(synthetic.lit_relief_disc())
    assert len(curves) >= 6, (
        f"a disc carrying eight petals, a boss and a rim gave {len(curves)} "
        "curves")
    for curve in curves:
        assert curve[0] == curve[-1], "a boundary that does not close"
        assert len(curve) >= 4


def test_a_bare_face_is_left_bare():
    curves, _mask, _radius = _read(synthetic.lit_plain_disc())
    assert curves == [], (
        f"the bare control was given {len(curves)} curves; a reading that "
        "draws on a plain face is drawing its own noise")


def test_the_reading_does_not_depend_on_the_working_resolution():
    small, _m, _r = _read(synthetic.lit_relief_disc(size=400))
    large, _m2, _r2 = _read(synthetic.lit_relief_disc(size=800))
    assert abs(len(small) - len(large)) <= 2, (
        f"{len(small)} curves at 400 and {len(large)} at 800: the reading "
        "changes with the working size")


def test_the_reading_does_not_depend_on_where_the_lamp_stands():
    counts = [len(_read(synthetic.lit_relief_disc(azimuth=angle))[0])
              for angle in (20.0, 65.0, 110.0, 200.0)]
    assert min(counts) >= 5, f"a lamp move emptied the reading: {counts}"
    assert max(counts) - min(counts) <= 4, (
        f"the reading swings with the lamp: {counts}")


def test_the_height_map_puts_the_relief_the_right_way_up():
    """The raised elements must be the high ground, not the low."""
    image = synthetic.lit_relief_disc()
    mask = get_mask_opencv(image)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    _x, _y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    surface, _azimuth = ro.relief_height(image, mask, max(width, height) / 2.0)
    assert surface is not None
    truth, face = synthetic._disc_height(image.shape[0], True)
    raised = (truth > 0.5) & (face > 0)
    flat = (truth <= 0.0) & (face > 0)
    assert float(surface[raised].mean()) > float(surface[flat].mean()), (
        "the height map is upside down - the ornament reads as grooves")


def test_a_curve_over_flat_ground_is_refused():
    """The step gate, on its own: no step, no curve."""
    image = synthetic.lit_relief_disc()
    mask = get_mask_opencv(image)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    _x, _y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    radius = max(width, height) / 2.0
    surface, _azimuth = ro.relief_height(image, mask, radius)
    flat = np.zeros_like(surface)
    circle = [[int(image.shape[1] / 2 + radius * 0.5 * np.cos(t)),
               int(image.shape[0] / 2 + radius * 0.5 * np.sin(t))]
              for t in np.linspace(0, 2 * np.pi, 64)]
    circle.append(list(circle[0]))
    assert ro.step_across(circle, flat, radius) == 0.0
    assert ro.step_across(circle, surface, radius) > 0.0


def test_nothing_to_read_is_not_an_error():
    blank = synthetic.blank(200, color=(180, 180, 180))
    mask = np.zeros(blank.shape[:2], dtype=np.uint8)
    assert ro.raised_outlines(blank, mask, 60.0) == []
    assert ro.raised_outlines(blank, mask, 0.0) == []
    assert ro.relief_height(blank, mask, 60.0) == (None, 0.0)


def test_a_pair_of_knobs_on_a_bare_face_is_drawn():
    """The multi-knobbed mirror's knobs, on a face that has nothing else."""
    for count in (2, 3):
        image = synthetic.knobbed_disc(size=600, knobs=count)
        curves, mask, radius = _read(image)
        knobs = ro.paired_knobs(image, mask, radius)
        assert len(knobs) == count, (
            f"{count} knobs at one radius came back as {len(knobs)}")
        for knob in knobs:
            assert knob[0] == knob[-1]


def test_a_lone_spot_on_a_bare_face_is_refused():
    """One raised spot is a corrosion blister as often as a knob."""
    image = synthetic.knobbed_disc(size=600, knobs=1)
    _curves, mask, radius = _read(image)
    assert ro.paired_knobs(image, mask, radius) == []


def test_the_knob_reading_draws_nothing_on_the_controls():
    for image in (synthetic.lit_plain_disc(size=600),
                  synthetic.diffuse_plain_disc(size=600),
                  synthetic.plain_disc(size=600)):
        _curves, mask, radius = _read(image)
        assert ro.paired_knobs(image, mask, radius) == []


def test_a_ring_of_petals_is_not_knobs():
    """Eight compact bumps at one radius are a rosette; the count refuses them."""
    image = synthetic.lit_relief_disc(size=600)
    _curves, mask, radius = _read(image)
    assert ro.paired_knobs(image, mask, radius) == []


def test_the_folded_cell_is_the_petal():
    """One closed cell per fold, each sitting on a planted petal."""
    from archeoglyph.generators.autotrace import round_motif as rm
    for maker in (synthetic.lit_relief_disc, synthetic.diffuse_relief_disc):
        image = maker(size=600, folds=8)
        _curves, mask, radius = _read(image)
        lines = ro.line_map(image, mask, radius)
        # The frame is centred the way the pipeline centres it: by the
        # feature vote, with the one-cycle recentring only as the fallback.
        from archeoglyph.generators.autotrace.feature_symmetry import vote_for_centre
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _sx, _sy, face = rm.trimmed_face_circle(max(contours, key=cv2.contourArea))
        # ... and the fold is counted on the height map, as the pipeline does.
        surface, _azimuth = ro.relief_height(image, mask, radius)
        frame = rm.find_rotational_frame(surface, mask,
                                         centre=vote_for_centre(gray, mask, face))
        assert frame is not None and frame.folds == 8, (
            f"the eight-fold control read as {frame and frame.folds}")
        cells = rm.fold_line_cells(lines, frame)
        petals = synthetic.petal_centres(600, 8)
        centre = (300.0, 300.0)
        # The boss's ring comes back too, as a circle round the centre; the
        # rest must be exactly the eight petals.
        rings = [c for c in cells
                 if np.hypot(*(np.asarray(c, dtype=np.float32).mean(axis=0) - centre)) < radius * 0.05]
        cells = [c for c in cells if c not in rings]
        assert len(rings) <= 1, f"{len(rings)} rings round the centre for one boss"
        assert len(cells) == 8, f"{len(cells)} cells for eight petals"
        for cell in cells:
            assert cell[0] == cell[-1]
            pts = np.asarray(cell, dtype=np.float32)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            assert any(np.hypot(cx - px, cy - py) < radius * 0.08 for px, py in petals), (
                f"a cell centred at ({cx:.0f},{cy:.0f}) sits on no petal")
            # ... and contains the petal's centre. The planted petal covers
            # 0.039 of the disc and the cell comes back at 0.009 to 0.015,
            # because the synthetic dome is slope all over and its line ring
            # is thick, so the cut sits well inside the rim; on the tile the
            # rims are thin and the cell is the petal (0.033 of the face). What
            # a cell must never be is a pocket beside the centre.
            share = abs(cv2.contourArea(pts.astype(np.int32))) / float(np.count_nonzero(mask))
            assert 0.005 <= share <= 0.08, f"a cell of {share:.3f} of the face for a petal of 0.039"
            assert any(cv2.pointPolygonTest(pts.astype(np.int32), (float(px), float(py)), False) >= 0
                       for px, py in petals), "a cell that does not contain its petal's centre"
