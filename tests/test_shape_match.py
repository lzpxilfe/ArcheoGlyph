import cv2
import numpy as np

from archeoglyph.generators import shape_match as sm


def _disc(size=200, radius=70):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    return mask


def test_overlap_scores_on_identical_and_disjoint_masks():
    disc = _disc()
    same = sm.overlap_scores(disc, disc)
    assert same["iou"] == 1.0 and same["recall"] == 1.0 and same["precision"] == 1.0

    other = np.zeros_like(disc)
    cv2.circle(other, (20, 20), 10, 255, -1)
    apart = sm.overlap_scores(disc, other)
    assert apart["iou"] == 0.0


def test_boundary_band_is_a_ring_around_the_edge():
    disc = _disc()
    band = sm.boundary_band(disc, width=4)
    assert band[100, 100] == 0            # centre is not in the band
    assert band[100, 30] > 0              # left edge (radius 70 from centre 100)
    assert band[0, 0] == 0                # far background


def test_stroke_output_passes_for_an_outline_and_fails_for_a_scribble():
    disc = _disc()
    outline = np.zeros_like(disc)
    cv2.circle(outline, (100, 100), 70, 255, 2)
    ok, reason = sm.matches_reference(disc, outline, stroke_style=True)
    assert ok, reason

    scribble = np.zeros_like(disc)
    cv2.line(scribble, (5, 5), (40, 40), 255, 2)
    bad, reason = sm.matches_reference(disc, scribble, stroke_style=True)
    assert not bad and "outline" in reason


def test_stroke_style_would_fail_the_filled_comparison():
    """The old check compared thin strokes against a filled mask; it could not pass."""
    disc = _disc()
    outline = np.zeros_like(disc)
    cv2.circle(outline, (100, 100), 70, 255, 2)
    filled_ok, _ = sm.matches_reference(disc, outline, stroke_style=False)
    stroke_ok, _ = sm.matches_reference(disc, outline, stroke_style=True)
    assert not filled_ok and stroke_ok


def test_filled_output_passes_when_shapes_agree():
    disc = _disc()
    slightly_smaller = _disc(radius=67)
    ok, reason = sm.matches_reference(disc, slightly_smaller, stroke_style=False)
    assert ok, reason

    shifted = np.zeros_like(disc)
    cv2.circle(shifted, (150, 150), 70, 255, -1)
    bad, reason = sm.matches_reference(disc, shifted, stroke_style=False)
    assert not bad and "mismatch" in reason


def test_empty_prediction_and_empty_reference():
    disc = _disc()
    empty = np.zeros_like(disc)
    ok, reason = sm.matches_reference(disc, empty, stroke_style=False)
    assert not ok and reason
    # Nothing meaningful to compare against -> accept.
    assert sm.matches_reference(empty, disc)[0]


def test_masks_from_png_round_trip():
    canvas = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.circle(canvas, (60, 60), 40, (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    mask = sm.mask_from_png(bytes(buf))
    assert mask is not None and mask[60, 60] and not mask[5, 5]

    painted = sm.painted_mask_from_png(bytes(buf))
    assert painted is not None and painted[60, 60] and not painted[5, 5]
