import numpy as np

from archeoglyph.generators import ink_centerline as ink
from tests import synthetic


def test_single_stroke_yields_one_long_polyline():
    image, drawn_length = synthetic.single_dark_stroke()
    polylines = ink.extract_ink_polylines(image)
    assert polylines, "expected at least one centreline"
    longest = max(polylines, key=len)
    length = sum(
        float(np.hypot(b[0] - a[0], b[1] - a[1]))
        for a, b in zip(longest, longest[1:])
    )
    assert 0.7 * drawn_length <= length <= 1.3 * drawn_length


def test_ink_score_is_normalised_and_quiet_on_background():
    image, _ = synthetic.single_dark_stroke()
    score = ink.compute_ink_score(image)
    assert score.dtype == np.float32
    assert score.max() <= 1.0 + 1e-6
    # Far corner of the canvas holds no ink.
    assert score[:20, -20:].max() < 0.05


def test_constraint_image_has_dark_lines_on_light_canvas():
    image, _ = synthetic.single_dark_stroke()
    canvas = ink.render_ink_constraint_image(image)
    assert canvas is not None and canvas.shape == image.shape
    gray = canvas.mean(axis=2)
    assert (gray < 100).sum() > 50
    assert gray.mean() > 180
