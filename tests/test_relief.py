# -*- coding: utf-8 -*-
"""
Relief recovered from a stack of differently lit frames.

Reading shallow relief off one photograph failed eight times, always to the
same thing: stains and surface grain look like decoration when you only have
one frame to look at. They stop looking like it the moment the light moves,
because a stain does not change with the lamp and a sloped surface does.

The stacks here are synthetic, and deliberately so. Every one is built from
a height field this file writes, lit by Lambertian shading from known
directions, so the answer is known by construction rather than by whether
the output happens to please. No real multi-light photographs of Korean
artefacts were available to this work; that is recorded in
docs/auto_trace_limits.md rather than papered over.
"""

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from archeoglyph.generators.autotrace import relief as rl  # noqa: E402
from archeoglyph.generators.autotrace import round_motif as rm  # noqa: E402

SIZE = 420
CENTRE = (SIZE // 2, SIZE // 2)
FACE_R = 150
#: Five, not four. Four lamps evenly around a disc put their own
#: four-fold arrangement into the relief map, and an eight-petal tile
#: then reads as four - see test_four_lamps_cannot_answer_a_four_fold.
LIGHTS = [(math.cos(a), math.sin(a)) for a in
          np.linspace(0.0, 2.0 * math.pi, rl.ADVISED_LIGHTS, endpoint=False)]


def _height_field(folds, radius=FACE_R):
    """A disc carrying `folds` raised petals - the height, not a picture."""
    height = np.zeros((SIZE, SIZE), dtype=np.float32)
    if folds:
        for index in range(folds):
            angle = 2.0 * math.pi * index / folds
            cx = int(CENTRE[0] + 0.58 * radius * math.cos(angle))
            cy = int(CENTRE[1] + 0.58 * radius * math.sin(angle))
            cv2.ellipse(height, (cx, cy),
                        (int(radius * 0.20), int(radius * 0.13)),
                        math.degrees(angle), 0, 360, 1.0, -1)
    height = cv2.GaussianBlur(height, (0, 0), sigmaX=5.0)
    disc = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(disc, CENTRE, radius, 255, -1)
    return height * (disc > 0), disc


def _light(height, albedo, direction):
    """Lambertian shading: brightness follows the surface angle to the lamp."""
    gy, gx = np.gradient(height.astype(np.float32))
    lx, ly = direction
    shade = 0.62 + 0.9 * (-gx * lx - gy * ly)
    frame = np.clip(albedo * shade, 0.0, 1.0) * 255.0
    return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _stack(folds, stains=False, radius=FACE_R):
    height, disc = _height_field(folds, radius)
    albedo = np.full((SIZE, SIZE), 0.80, dtype=np.float32)
    albedo[disc == 0] = 0.95
    if stains:
        rng = np.random.default_rng(20260907)
        for _ in range(9):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            r = rng.uniform(0.2, 0.85) * radius
            cv2.ellipse(albedo,
                        (int(CENTRE[0] + r * math.cos(angle)),
                         int(CENTRE[1] + r * math.sin(angle))),
                        (int(rng.uniform(12, 30)), int(rng.uniform(9, 22))),
                        rng.uniform(0, 180), 0, 360, float(rng.uniform(0.4, 0.6)), -1)
    return [_light(height, albedo, d) for d in LIGHTS], disc


def test_a_relief_motif_is_recovered_from_a_light_stack():
    frames, mask = _stack(8)
    relief = rl.relief_from_light_stack(frames)
    assert relief is not None
    frame = rm.find_rotational_frame(relief, mask)
    assert frame is not None, "the relief map should carry the repeat"
    assert frame.folds == 8, f"read {frame.folds} petals, not 8"
    assert frame.score >= rm.FRAME_MIN_SCORE, (
        f"scored {frame.score:.3f}, under the {rm.FRAME_MIN_SCORE} gate")


def test_stains_survive_one_frame_and_vanish_from_the_stack():
    """
    This is the whole reason for the multi-light path.

    Discolouration is the thing that beat every single-image attempt. It does
    not move with the lamp, so it cancels in the stack while the relief does
    not.
    """
    frames, mask = _stack(8, stains=True)
    inside = mask > 0

    single = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    single = single[inside]
    single = (single - single.mean()) / (single.std() + 1e-6)

    relief = rl.relief_from_light_stack(frames)
    assert relief is not None
    stacked = relief.astype(np.float32)[inside]
    stacked = (stacked - stacked.mean()) / (stacked.std() + 1e-6)

    clean, _mask2 = _stack(8, stains=False)
    truth = rl.relief_from_light_stack(clean).astype(np.float32)[inside]
    truth = (truth - truth.mean()) / (truth.std() + 1e-6)

    single_err = float(np.abs(single - truth).mean())
    stacked_err = float(np.abs(stacked - truth).mean())
    assert stacked_err < single_err * 0.75, (
        f"the stack is {stacked_err:.3f} from the unstained relief and one "
        f"frame is {single_err:.3f}; the stains are not being divided out")


def test_a_relief_with_no_repeat_is_still_refused():
    """Multi-light does not license inventing decoration."""
    frames, mask = _stack(0)
    rng = np.random.default_rng(11)
    height, disc = _height_field(0)
    for _ in range(6):
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.25, 0.8) * FACE_R
        cv2.ellipse(height,
                    (int(CENTRE[0] + r * math.cos(angle)),
                     int(CENTRE[1] + r * math.sin(angle))),
                    (int(rng.uniform(14, 32)), int(rng.uniform(10, 24))),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
    height = cv2.GaussianBlur(height, (0, 0), sigmaX=5.0) * (disc > 0)
    albedo = np.full((SIZE, SIZE), 0.8, dtype=np.float32)
    frames = [_light(height, albedo, d) for d in LIGHTS]

    relief = rl.relief_from_light_stack(frames)
    assert relief is not None
    frame = rm.find_rotational_frame(relief, mask)
    score = 0.0 if frame is None else frame.score
    assert score < rm.FRAME_MIN_SCORE, (
        f"scattered relief scored {score:.3f} against a "
        f"{rm.FRAME_MIN_SCORE} gate; the tracer would invent a repeat")


def test_one_frame_is_not_a_stack():
    """With a single photograph there is nothing to compare against."""
    frames, _mask = _stack(8)
    assert rl.relief_from_light_stack(frames[:1]) is None
    assert rl.relief_from_light_stack([]) is None
    assert rl.relief_from_light_stack(None) is None


def test_frames_that_drifted_are_brought_back_together():
    frames, _mask = _stack(8)
    shifted = [frames[0]]
    for index, frame in enumerate(frames[1:], start=1):
        matrix = np.float32([[1, 0, 3 * index], [0, 1, -2 * index]])
        shifted.append(cv2.warpAffine(frame, matrix, (SIZE, SIZE),
                                      borderMode=cv2.BORDER_REPLICATE))
    aligned, moved = rl.align_light_stack(shifted)
    assert moved >= 2, "the drifted frames were not moved back"
    assert len(aligned) == len(shifted), "a frame was dropped, not aligned"


def test_frames_of_different_sizes_are_refused():
    frames, _mask = _stack(8)
    odd = list(frames)
    odd[1] = cv2.resize(odd[1], (SIZE // 2, SIZE // 2))
    assert rl.relief_from_light_stack(odd) is None


def test_the_lamp_count_does_not_leak_into_the_reading():
    """
    The lamps are an arrangement in a circle too, and they used to be read
    as one: with four of them an eight-petal tile came out as four, and
    nothing in the stack could say whether that four was the artefact or the
    lighting. relief.py carried a guard that refused any reading equal to the
    frame count for exactly that reason.

    The confound came from the frame, not from the lamps. While the face was
    located by searching centres for the best fold score, a four-lobed
    lighting pattern was one of the things that search could lock onto. With
    the face fixed by geometry first the confound does not reproduce, and the
    guard is gone with it - it would now only refuse correct answers, since an
    eight-fold artefact lit by eight lamps reads as eight.

    Six fold counts against five lamp counts, all read correctly.
    """
    for true_folds in (5, 6, 7, 8, 9, 11):
        height, disc = _height_field(true_folds)
        albedo = np.full((SIZE, SIZE), 0.80, dtype=np.float32)
        albedo[disc == 0] = 0.95
        for lamps in (3, 4, 5, 6, 8):
            directions = [(math.cos(a), math.sin(a)) for a in
                          np.linspace(0.0, 2.0 * math.pi, lamps, endpoint=False)]
            relief = rl.relief_from_light_stack(
                [_light(height, albedo, d) for d in directions])
            frame = rm.find_rotational_frame(relief, disc)
            assert frame is not None and frame.folds == true_folds, (
                f"{true_folds} petals under {lamps} lamps read as "
                f"{frame.folds if frame else 0}")
