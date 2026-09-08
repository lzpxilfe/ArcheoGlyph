# -*- coding: utf-8 -*-
"""
Rotational motif recovery, checked against drawings this file makes itself.

A bronze mirror and a roof tile end are decorated in n-fold rotational
symmetry, and that repeat is the whole of what makes them readable at legend
size. Reading it off a photograph took six failed attempts, all of which
shared a cause: the geometry was fitted to the object mask, which includes
the cast shadow, so the sampling ring swept the shadow rather than the
decorated face.

The controls here are synthetic on purpose. The threshold has to be set
against inputs whose answer is known by construction, not tuned until three
photographs come out well - that would only prove the tuning.
"""

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from archeoglyph.generators.autotrace import round_motif as rm  # noqa: E402

SIZE = 480
CENTRE = (SIZE // 2, SIZE // 2)
FACE_R = 170


def _disc(centre=CENTRE, radius=FACE_R):
    img = np.full((SIZE, SIZE), 235, dtype=np.uint8)
    cv2.circle(img, centre, radius, 190, -1)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, centre, radius, 255, -1)
    return img, mask


def _petals(img, folds, centre=CENTRE, radius=FACE_R, tone=110):
    """A ring of identical petals - the thing a roof tile end actually has."""
    for index in range(folds):
        angle = 2.0 * math.pi * index / folds
        cx = int(centre[0] + 0.58 * radius * math.cos(angle))
        cy = int(centre[1] + 0.58 * radius * math.sin(angle))
        cv2.ellipse(img, (cx, cy), (int(radius * 0.20), int(radius * 0.13)),
                    math.degrees(angle), 0, 360, tone, -1)
    return img


def test_a_six_fold_motif_is_recovered_as_six():
    img, mask = _disc()
    _petals(img, 6)
    frame = rm.find_rotational_frame(img, mask)
    assert frame is not None, "a six-petal disc has a repeat to find"
    assert frame.folds == 6, f"read {frame.folds} petals, not 6"
    assert frame.score >= rm.FRAME_MIN_SCORE


def test_an_eight_fold_motif_is_recovered_as_eight():
    img, mask = _disc()
    _petals(img, 8)
    frame = rm.find_rotational_frame(img, mask)
    assert frame is not None
    assert frame.folds == 8, f"read {frame.folds} petals, not 8"


def test_a_disc_with_no_repeat_scores_below_the_threshold():
    """
    The contract that matters most: inventing decoration is worse than
    admitting there is none. A dragon-motif tile must not be given petals.
    """
    img, mask = _disc()
    rng = np.random.default_rng(20260907)
    for _ in range(7):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        rad = rng.uniform(0.25, 0.8) * FACE_R
        cx = int(CENTRE[0] + rad * math.cos(angle))
        cy = int(CENTRE[1] + rad * math.sin(angle))
        axes = (int(rng.uniform(14, 34)), int(rng.uniform(10, 26)))
        cv2.ellipse(img, (cx, cy), axes, rng.uniform(0, 180), 0, 360, 120, -1)
    frame = rm.find_rotational_frame(img, mask)
    score = 0.0 if frame is None else frame.score
    assert score < rm.FRAME_MIN_SCORE, (
        f"scattered blobs scored {score:.3f}, at or above the "
        f"{rm.FRAME_MIN_SCORE} needed to draw a motif - the tracer would "
        f"stamp a repeat onto an artefact that has none")


def test_the_search_finds_the_face_when_the_mask_is_pulled_off_centre():
    """
    The mask includes the cast shadow, which drags its centroid away from the
    decorated face. Every earlier attempt failed on exactly this.
    """
    img, mask = _disc()
    _petals(img, 6)
    shadow = mask.copy()
    cv2.circle(shadow, (CENTRE[0], CENTRE[1] + 80), FACE_R, 255, -1)

    frame = rm.find_rotational_frame(img, shadow)
    assert frame is not None
    assert frame.folds == 6
    drift = math.hypot(frame.cx - CENTRE[0], frame.cy - CENTRE[1])
    off_centre = math.hypot(0, 40)          # the shadowed mask's own centroid
    assert drift < off_centre, (
        f"the frame sat {drift:.0f}px from the face centre, no better than "
        f"the shadowed mask centroid at {off_centre:.0f}px")


def test_folding_drops_a_streak_that_lives_in_one_sector():
    """Median across sectors, not mean: a lamp streak is not decoration."""
    img, mask = _disc()
    _petals(img, 6)
    cv2.line(img, CENTRE,
             (CENTRE[0] + FACE_R, CENTRE[1]), 40, 21)     # one sector only
    frame = rm.find_rotational_frame(img, mask)
    assert frame is not None and frame.folds == 6
    wedge = rm.fold_rotational_motif(img, frame)
    assert wedge, "the petals should survive folding"
    lines = rm.replay_rotational_motif(wedge, frame)
    assert lines, "a folded motif should replay as polylines"
    # Every stamp is the same shape, so their sizes must agree closely.
    spans = [max(max(p[0] for p in line) - min(p[0] for p in line),
                 max(p[1] for p in line) - min(p[1] for p in line))
             for line in lines]
    assert max(spans) <= 2.2 * min(spans), (
        "replayed stamps differ in size, so the streak was folded in as if "
        "it were part of the motif")


def test_replayed_motifs_stay_inside_the_face():
    img, mask = _disc()
    _petals(img, 8)
    frame = rm.find_rotational_frame(img, mask)
    assert frame is not None
    lines = rm.replay_rotational_motif(rm.fold_rotational_motif(img, frame), frame)
    assert lines
    for line in lines:
        for x, y in line:
            assert math.hypot(x - CENTRE[0], y - CENTRE[1]) <= FACE_R * 1.05, (
                "a motif was stamped outside the decorated face")


def test_a_clean_repeat_gives_the_same_answer_when_the_input_moves():
    """
    Stability is what separates a motif from a lucky optimiser run.

    Shifting the image by a pixel cannot change how many petals it has, so a
    score that moves under that shift is not measuring petals. The drawn
    control holds still; the photographs of relief decoration do not, which
    is why FRAME_MIN_SCORE sits above the range they wander over.
    """
    seen = set()
    for shift in range(5):
        img, mask = _disc()
        _petals(img, 6)
        frame = rm.find_rotational_frame(np.roll(img, shift, axis=1),
                                         np.roll(mask, shift, axis=1))
        assert frame is not None
        seen.add((frame.folds, round(frame.score, 3)))
    assert len(seen) == 1, (
        f"a one-pixel shift changed the reading: {sorted(seen)}")


def test_the_threshold_clears_the_noise_a_flat_disc_produces():
    """A blank disc has no repeat, so whatever it scores is the noise floor."""
    worst = 0.0
    for shift in range(4):
        img, mask = _disc()
        frame = rm.find_rotational_frame(np.roll(img, shift, axis=1),
                                         np.roll(mask, shift, axis=1))
        if frame is not None:
            worst = max(worst, frame.score)
    assert worst < rm.FRAME_MIN_SCORE, (
        f"a plain disc reached {worst:.3f} against a {rm.FRAME_MIN_SCORE} "
        f"gate; the gate has to sit above what nothing at all produces")


def test_every_fold_is_stamped_or_none_is():
    """
    A motif that stops three sectors round reads as damage, not decoration.

    Truncating the replayed lines by count cuts at a sector boundary, which
    is what left an eight-petal tile with petals over one quadrant and a
    bare rest. The wedge is trimmed by shape instead, and then every fold is
    emitted.
    """
    img, mask = _disc()
    _petals(img, 8)
    frame = rm.find_rotational_frame(img, mask)
    assert frame is not None and frame.folds == 8
    wedge = rm.fold_rotational_motif(img, frame)
    assert 0 < len(wedge) <= rm.MAX_WEDGE_SHAPES
    lines = rm.replay_rotational_motif(wedge, frame)
    assert len(lines) == len(wedge) * frame.folds, (
        "the replay dropped sectors instead of shapes")

    # The stamps have to go all the way round, not bunch in one quadrant.
    centres = []
    for line in lines:
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        centres.append(math.degrees(math.atan2(
            sum(ys) / len(ys) - frame.cy, sum(xs) / len(xs) - frame.cx)) % 360.0)
    for quadrant in range(4):
        low, high = quadrant * 90.0, (quadrant + 1) * 90.0
        assert any(low <= c < high for c in centres), (
            f"no motif was stamped between {low:.0f} and {high:.0f} degrees; "
            f"the replay covers {sorted(round(c) for c in centres)}")


def test_trimming_the_contour_beats_fitting_all_of_it():
    """
    A photograph on a table carries a shadow skirt fused to the silhouette,
    and a fit to the whole contour sits between the disc and the skirt.

    The gain from trimming is consistent but small, and this test says so
    rather than claiming the skirt is removed: across four skirt sizes the
    trimmed circle must beat a plain ellipse fit every time, and must land on
    the face for the skirts a lit photograph actually produces.
    """
    for width, height, offset, tolerance in ((0.6, 0.15, 0.85, 0.04),
                                             (0.7, 0.18, 0.90, 0.08),
                                             (0.8, 0.22, 0.95, 0.12),
                                             (0.9, 0.35, 1.00, 0.22)):
        _img, mask = _disc()
        cv2.ellipse(mask, (CENTRE[0], int(CENTRE[1] + offset * FACE_R)),
                    (int(FACE_R * width), int(FACE_R * height)), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        main = max(contours, key=cv2.contourArea)

        plain = cv2.fitEllipse(main)
        cx, cy, _radius = rm.trimmed_face_circle(main)
        drift = math.hypot(cx - CENTRE[0], cy - CENTRE[1])
        plain_drift = math.hypot(plain[0][0] - CENTRE[0], plain[0][1] - CENTRE[1])

        assert drift < plain_drift, (
            f"a {width}x{height} skirt: trimming gained nothing, {drift:.0f}px "
            f"against {plain_drift:.0f}px for a plain fit")
        assert drift < tolerance * FACE_R, (
            f"a {width}x{height} skirt left the circle {drift:.0f}px off")


def test_the_centre_walks_back_onto_the_decorated_face():
    """
    Off-centre, the same feature returns at a different radius on the far side
    of the face, so the radius where the contrast sits swings once per turn.
    Nulling that swing needs no fold count, which is the point - the fold
    score was never a safe objective to move the centre with.
    """
    img, _mask = _disc()
    _petals(img, 8)
    gray = img.astype(np.float32)
    off_x, off_y = CENTRE[0] + 0.20 * FACE_R, CENTRE[1] - 0.16 * FACE_R
    before = math.hypot(off_x - CENTRE[0], off_y - CENTRE[1])

    cx, cy = rm.recentre_on_decoration(gray, off_x, off_y, FACE_R)
    after = math.hypot(cx - CENTRE[0], cy - CENTRE[1])
    assert after < before / 2.0, (
        f"the centre moved from {before:.0f}px off to {after:.0f}px off")


def test_a_frame_that_sweeps_across_the_motif_does_not_decide_it():
    """
    Ballots are weighed by their score, not counted.

    A frame whose ellipse sweeps across a circular motif reads it distorted
    and scores low. Counting its ballot equally is what turned a clean
    six-fold disc into a seven, so the weighing is the contract here: adding
    frames that see nothing must not change the answer.
    """
    img, _mask = _disc()
    _petals(img, 6)
    gray = img.astype(np.float32)
    folds, _agreement, score, _scale, _ratio, _angle = rm.survey_folds(
        gray, CENTRE[0], CENTRE[1], FACE_R)
    assert folds == 6 and score >= rm.FRAME_MIN_SCORE

    original = rm.SURVEY_RATIOS
    try:
        # Ratios that squash the sampling ring flat see nothing but noise.
        rm.SURVEY_RATIOS = original + (0.30, 0.24, 0.18, 0.12)
        again = rm.survey_folds(gray, CENTRE[0], CENTRE[1], FACE_R)
    finally:
        rm.SURVEY_RATIOS = original
    assert again[0] == 6, (
        f"adding {len(again)} blind frames changed the reading to {again[0]}")


def test_the_gate_sits_above_every_control_and_below_every_repeat():
    """
    The threshold is set from what has no motif, never from what does.

    Drawn positives and drawn controls both, in one place, so that a change to
    the scoring that moves them together is caught here rather than in a
    photograph nobody can commit.
    """
    def read(build):
        img, mask = build()
        frame = rm.find_rotational_frame(img, mask)
        return 0.0 if frame is None else frame.score

    def blobs(seed):
        def build():
            img, mask = _disc()
            rng = np.random.default_rng(seed)
            for _ in range(7):
                angle = rng.uniform(0.0, 2.0 * math.pi)
                rad = rng.uniform(0.25, 0.8) * FACE_R
                cv2.ellipse(img,
                            (int(CENTRE[0] + rad * math.cos(angle)),
                             int(CENTRE[1] + rad * math.sin(angle))),
                            (int(rng.uniform(14, 34)), int(rng.uniform(10, 26))),
                            rng.uniform(0, 180), 0, 360, 120, -1)
            return img, mask
        return build

    def grain(seed):
        def build():
            img, mask = _disc()
            rng = np.random.default_rng(seed)
            noisy = np.clip(img.astype(np.float32) + rng.normal(0, 9.0, img.shape),
                            0, 255).astype(np.uint8)
            return noisy, mask
        return build

    controls = [read(_disc)]
    controls += [read(blobs(seed)) for seed in (20260907, 11, 202, 5150)]
    controls += [read(grain(seed)) for seed in (3, 41)]
    repeats = [read(lambda folds=folds: (_petals(_disc()[0], folds), _disc()[1]))
               for folds in (6, 8, 12)]

    assert max(controls) < rm.FRAME_MIN_SCORE, (
        f"a control reached {max(controls):.3f} against a "
        f"{rm.FRAME_MIN_SCORE} gate")
    assert min(repeats) > rm.FRAME_MIN_SCORE, (
        f"a drawn repeat only reached {min(repeats):.3f}, at or below the "
        f"{rm.FRAME_MIN_SCORE} gate that is supposed to let it through")
    assert min(repeats) > 2.0 * max(controls), (
        f"only {min(repeats) / max(max(controls), 1e-9):.1f}x separates the "
        f"weakest repeat from the loudest control; the gate has no room")
