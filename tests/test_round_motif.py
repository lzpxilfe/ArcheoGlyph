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
