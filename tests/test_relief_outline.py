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
