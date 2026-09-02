import numpy as np
import cv2

from archeoglyph.generators import ink_centerline as ink
from tests import synthetic


def _length(pline):
    return sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(pline, pline[1:]))


def test_single_stroke_yields_one_long_polyline():
    image, drawn_length = synthetic.single_dark_stroke()
    polylines = ink.extract_ink_polylines(image)
    assert polylines, "expected at least one centreline"
    assert 0.85 * drawn_length <= _length(polylines[0]) <= 1.15 * drawn_length
    assert len(polylines) == 1


def test_ink_score_is_normalised_and_quiet_on_background():
    image, _ = synthetic.single_dark_stroke()
    score = ink.compute_ink_score(image)
    assert score.dtype == np.float32
    assert score.max() <= 1.0 + 1e-6
    assert score[:20, -20:].max() < 0.05


def test_isoluminant_red_stroke_is_detected():
    polylines = ink.extract_ink_polylines(synthetic.red_stroke_on_gray())
    assert polylines and _length(polylines[0]) > 150


def test_bright_stroke_produces_no_halo_lines():
    img = synthetic.blank(300, color=(200, 200, 200))
    cv2.line(img, (30, 150), (270, 150), (255, 255, 255), 3, cv2.LINE_AA)
    assert ink.extract_ink_polylines(img) == []


def test_faint_stroke_survives_next_to_a_strong_one():
    img = synthetic.blank(400, color=(250, 250, 250))
    cv2.line(img, (20, 40), (180, 40), (20, 20, 20), 3, cv2.LINE_AA)        # strong, top-left tile
    cv2.line(img, (220, 360), (380, 360), (205, 205, 205), 3, cv2.LINE_AA)  # faint, far tile
    polylines = ink.extract_ink_polylines(img)
    ys = sorted(int(np.mean([p[1] for p in pl])) for pl in polylines)
    assert len(polylines) == 2 and abs(ys[0] - 40) <= 2 and abs(ys[1] - 360) <= 2


def test_y_junction_gives_three_branches_sharing_the_node():
    img = synthetic.y_junction()
    polylines = ink.extract_ink_polylines(img)
    assert len(polylines) == 3
    center = (150, 150)
    for pl in polylines:
        d0 = np.hypot(pl[0][0] - center[0], pl[0][1] - center[1])
        d1 = np.hypot(pl[-1][0] - center[0], pl[-1][1] - center[1])
        assert min(d0, d1) <= 3.0


def test_straight_line_through_a_crossing_is_joined():
    img = synthetic.blank(300, color=(250, 250, 250))
    cv2.line(img, (20, 150), (280, 150), (20, 20, 20), 3, cv2.LINE_AA)
    cv2.line(img, (150, 20), (150, 280), (20, 20, 20), 3, cv2.LINE_AA)
    polylines = ink.extract_ink_polylines(img)
    assert len(polylines) == 2
    assert all(_length(pl) > 240 for pl in polylines)


def test_spur_pruning_removes_short_branch_only():
    skel = np.zeros((40, 60), dtype=bool)
    skel[20, 5:55] = True          # main line
    skel[17:20, 30] = True         # 3-px spur off the middle
    pruned = ink.prune_spurs(skel, max_len=6)
    assert pruned[20, 5:55].all()
    assert not pruned[17:20, 30].any()


def test_looks_like_drawing_separates_line_art_from_photos():
    assert ink.looks_like_drawing(synthetic.line_drawing_sherd())[0]
    assert not ink.looks_like_drawing(synthetic.ellipse_blade())[0]
    assert not ink.looks_like_drawing(synthetic.mirror_with_rings())[0]


def test_polylines_to_text_and_guide_image():
    pl = [(0, 0), (1, 1), (2, 2), (3, 3), (10, 10), (10, 20)]
    text = ink.polylines_to_text([pl], epsilon=1.0)
    assert text == "0,0 10,10 10,20"
    image, _ = synthetic.single_dark_stroke()
    canvas = ink.compose_guide_image(image, synthetic.blade_mask(300), [pl])
    assert canvas.shape == image.shape
    assert canvas[5, 250].tolist() == [255, 255, 255]
    assert (canvas[:, :, 2] > 200).sum() > (canvas[:, :, 0] > 200).sum()  # red silhouette present


def test_constraint_image_has_dark_lines_on_light_canvas():
    image, _ = synthetic.single_dark_stroke()
    canvas = ink.render_ink_constraint_image(image)
    assert canvas is not None and canvas.shape == image.shape
    gray = canvas.mean(axis=2)
    assert (gray < 100).sum() > 50
    assert gray.mean() > 180
