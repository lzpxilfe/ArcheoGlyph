# -*- coding: utf-8 -*-
"""
The second opinion on where a decorated face is.

round_motif finds the face by geometry, which is this project's own
invention. feature_symmetry checks it against Loy and Eklundh's voting
method, which arrives at the centre a completely different way - matched
feature pairs each vote, and the centre falls out as the peak rather than
being handed in as an input.

The contract has two halves and the second matters more: it has to agree
where the face really is, and it has to stay silent rather than object when
there is nothing to match. A guard that fires on silence would refuse every
plain artefact.
"""

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from archeoglyph.generators.autotrace import feature_symmetry as fs  # noqa: E402
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
    for index in range(folds):
        angle = 2.0 * math.pi * index / folds
        cx = int(centre[0] + 0.58 * radius * math.cos(angle))
        cy = int(centre[1] + 0.58 * radius * math.sin(angle))
        cv2.ellipse(img, (cx, cy), (int(radius * 0.20), int(radius * 0.13)),
                    math.degrees(angle), 0, 360, tone, -1)
    return img


def test_the_vote_finds_the_centre_of_a_repeat():
    img, mask = _disc()
    _petals(img, 8)
    voted = fs.vote_for_centre(img, mask, FACE_R)
    assert voted is not None, "eight identical petals give plenty to match"
    drift = math.hypot(voted[0] - CENTRE[0], voted[1] - CENTRE[1])
    assert drift < fs.CENTRE_AGREEMENT * FACE_R, (
        f"the vote put the centre {drift:.0f}px from the face")


def test_the_vote_ignores_a_shadow_skirt_fused_to_the_silhouette():
    """
    This is the case the whole module exists for.

    A photograph on a table carries a shadow skirt, and every method that
    starts from the silhouette's centroid is dragged off the face by it. The
    vote never looks at the silhouette - it only looks at what matches what -
    so the skirt cannot move it.
    """
    img, mask = _disc()
    _petals(img, 8)
    skirt = mask.copy()
    cv2.ellipse(skirt, (CENTRE[0], CENTRE[1] + FACE_R),
                (int(FACE_R * 0.9), int(FACE_R * 0.35)), 0, 0, 360, 255, -1)

    contours, _ = cv2.findContours(skirt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    plain = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    plain_drift = math.hypot(plain[0][0] - CENTRE[0], plain[0][1] - CENTRE[1])

    voted = fs.vote_for_centre(img, skirt, FACE_R)
    assert voted is not None
    drift = math.hypot(voted[0] - CENTRE[0], voted[1] - CENTRE[1])
    assert drift < plain_drift, (
        f"the vote sat {drift:.0f}px off, no better than the {plain_drift:.0f}px "
        f"an ellipse fit to the shadowed silhouette manages")
    assert drift < fs.CENTRE_AGREEMENT * FACE_R


def test_silence_is_not_disagreement():
    """
    A plain disc has nothing to match, so the vote has no opinion. Reading
    that as a conflict would refuse decoration on every artefact whose
    surface is too worn to give features - exactly the ones a tracer is most
    likely to be handed.
    """
    img, mask = _disc()
    frame = rm.find_rotational_frame(img, mask)
    assert frame is not None
    assert fs.centre_disagrees(frame, None, FACE_R) is False
    assert fs.centre_disagrees(None, (10.0, 10.0), FACE_R) is False


def test_a_frame_on_the_wrong_part_of_the_face_is_caught():
    img, mask = _disc()
    _petals(img, 8)
    frame = rm.find_rotational_frame(img, mask)
    assert frame is not None
    voted = fs.vote_for_centre(img, mask, FACE_R)
    assert voted is not None
    assert not fs.centre_disagrees(frame, voted, FACE_R), (
        "the two methods disagree on a clean drawn repeat, so the guard would "
        "refuse readings that are right")

    frame.cx += 0.30 * FACE_R
    assert fs.centre_disagrees(frame, voted, FACE_R), (
        "a frame pushed a third of a radius off the face was not caught")


def test_the_tolerance_sits_between_what_was_measured():
    """
    Nine photographed finds, measured on deterministic masks: the two methods
    agree to 0.011, 0.017 and 0.072 of the face radius on the three decorated
    discs, and disagree by 0.235 and up on the six finds that are not one.
    The tolerance has to sit in that gap or it is measuring noise.

    The numbers here are the reproducible ones. An earlier version of this
    test used 0.02 and 0.35, taken before get_mask_opencv was deterministic,
    when the same photograph gave a different mask on every call.
    """
    worst_agreement, best_disagreement = 0.072, 0.235
    assert fs.CENTRE_AGREEMENT > worst_agreement, (
        f"{fs.CENTRE_AGREEMENT} would refuse the lotus tile, which the two "
        f"methods place within {worst_agreement} of a radius of each other")
    assert fs.CENTRE_AGREEMENT < best_disagreement, (
        f"{fs.CENTRE_AGREEMENT} would accept a comb-pattern jar, where they "
        f"are {best_disagreement} of a radius apart")
