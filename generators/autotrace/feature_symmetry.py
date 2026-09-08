# -*- coding: utf-8 -*-
"""
A second opinion on where a round artefact's decorated face is.

round_motif finds the face by geometry: a circle fitted to the silhouette,
then the centre walked downhill on the one-cycle wave the decoration makes.
That works, but it is this project's own invention, and the cost of it being
wrong is the worst failure this tracer has - decoration stamped onto an
artefact that has none.

So the answer is checked against a published method that arrives at it a
completely different way. Loy and Eklundh (ECCV 2006), the baseline the CVPR
symmetry competitions use, match features to each other and let every similar
pair vote: two patches that look alike are related by some rotation, and a
rotation through a known angle about an unknown centre pins that centre down
exactly. The centre is an *output* of the vote rather than an input to it,
which is precisely the part this project kept getting wrong.

On nine photographed finds the two agree to 0.01-0.02 of the face radius on
every round one - the lotus roof tile end, the dragon tile, the bronze mirror
- where a plain ellipse fit to the silhouette is 0.21 of a radius away on the
lotus. On every find that is not a decorated disc they disagree by more than
0.35 of a radius, which is the honest answer for a dagger.

**Only the centre is taken from this method.** Its fold-count step was
measured on the same photographs and does not separate an eight-petal tile
from a dragon: both came out at a matched-pair concentration of 0.28, the
tile reading 8 and the dragon 4. Four statistics on those pair angles were
tried - Rayleigh on the descriptor orientation difference, Rayleigh on the
positional rotation, bootstrap stability of the winning count, and agreement
between radial bands - and each either named the wrong count for the tile or
scored a control above it. The count therefore still comes from
round_motif.survey_folds, and this module answers only what it answers well.
"""

import math

import numpy as np

from ...log import log_exception

try:
    import cv2
except ImportError:  # pragma: no cover - mirrors the rest of the package
    cv2 = None

TWO_PI = 2.0 * math.pi

#: Features to look for. More than this is grain on a photographed surface,
#: and the pair matching below is quadratic in the count.
MAX_FEATURES = 1500

#: SIFT's own contrast floor. The default of 0.04 finds almost nothing on
#: shallow relief lit from one side, which is the whole case of interest.
CONTRAST_FLOOR = 0.02

#: How many near neighbours each feature is matched against. A petal has
#: seven siblings, so a handful is enough and more only adds noise.
NEIGHBOURS = 8

#: A pure rotation preserves scale, so a pair seen at different scales is not
#: one; and a pair rotated by almost nothing says nothing about where the
#: centre is, because the bisector construction below goes to infinity.
MIN_SCALE_RATIO = 0.72
MIN_HALF_ANGLE_SINE = 0.12

#: Descriptor distance past which a "match" is a coincidence.
MAX_DESCRIPTOR_DISTANCE = 280.0

#: Vote accumulator resolution, in pixels, and the blur applied to it as a
#: fraction of the face radius.
VOTE_STEP = 4
VOTE_SIGMA = 0.05

#: How far the two methods may sit apart before the reading is refused, as a
#: fraction of the face radius. Measured: they agree to 0.02 on every
#: decorated disc and disagree by 0.35 or more on everything that is not one,
#: so this sits six times above the agreement and three times below the
#: disagreement, with nothing measured in between.
CENTRE_AGREEMENT = 0.12


def _pairs(gray, mask):
    """Similar-looking feature pairs, with the rotation each implies."""
    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES,
                           contrastThreshold=CONTRAST_FLOOR)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < 8:
        return None
    points = np.array([k.pt for k in keypoints], dtype=np.float64)
    angles = np.radians(np.array([k.angle for k in keypoints], dtype=np.float64))
    scales = np.array([k.size for k in keypoints], dtype=np.float64)

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, descriptors,
                                                  k=min(NEIGHBOURS, len(keypoints)))
    left, right, distance = [], [], []
    for group in matches:
        for match in group:
            if match.queryIdx < match.trainIdx:
                left.append(match.queryIdx)
                right.append(match.trainIdx)
                distance.append(match.distance)
    if len(left) < 8:
        return None
    left = np.asarray(left)
    right = np.asarray(right)
    distance = np.asarray(distance)

    ratio = (np.minimum(scales[left], scales[right])
             / np.maximum(scales[left], scales[right]))
    turn = (angles[right] - angles[left]) % TWO_PI
    usable = (
        (ratio > MIN_SCALE_RATIO)
        & (np.sin(turn / 2.0) > MIN_HALF_ANGLE_SINE)
        & (distance < MAX_DESCRIPTOR_DISTANCE)
    )
    if int(usable.sum()) < 8:
        return None
    weight = np.exp(-distance[usable] / 120.0) * ratio[usable]
    return points, left[usable], right[usable], turn[usable], weight


def vote_for_centre(gray_img, mask, radius):
    """
    Where matched feature pairs say the centre of rotation is.

    Returns (cx, cy), or None when the image has too little structure to say.
    None means "no opinion" and must not be read as disagreement - a plain
    disc genuinely has nothing to match.
    """
    if cv2 is None or gray_img is None or mask is None or not (radius > 0):
        return None
    try:
        gray = gray_img if gray_img.dtype == np.uint8 else np.clip(
            gray_img, 0, 255).astype(np.uint8)
        found = _pairs(gray, mask)
        if found is None:
            return None
        points, left, right, turn, weight = found

        # Rotating the left point about c by turn lands on the right one, so c
        # sits on their bisector at cot(turn/2) times half their separation.
        offset = points[right] - points[left]
        middle = (points[left] + points[right]) / 2.0
        normal = np.stack([-offset[:, 1], offset[:, 0]], axis=1)
        centres = middle + normal * (0.5 / np.tan(turn / 2.0))[:, None]

        height, width = mask.shape[:2]
        inside = (
            (centres[:, 0] >= 0) & (centres[:, 0] < width)
            & (centres[:, 1] >= 0) & (centres[:, 1] < height)
        )
        if int(inside.sum()) < 8:
            return None

        rows = height // VOTE_STEP + 1
        columns = width // VOTE_STEP + 1
        accumulator = np.zeros((rows, columns), dtype=np.float32)
        xs = (centres[inside, 0] / VOTE_STEP).astype(int)
        ys = (centres[inside, 1] / VOTE_STEP).astype(int)
        np.add.at(accumulator, (ys, xs), weight[inside].astype(np.float32))
        accumulator = cv2.GaussianBlur(
            accumulator, (0, 0),
            sigmaX=max(1.0, radius * VOTE_SIGMA / VOTE_STEP))
        if float(accumulator.max()) <= 0.0:
            return None
        row, column = np.unravel_index(int(np.argmax(accumulator)),
                                       accumulator.shape)
        return float(column * VOTE_STEP), float(row * VOTE_STEP)
    except Exception as exc:
        log_exception("vote_for_centre", exc)
        return None


def centre_disagrees(frame, voted, radius, tolerance=CENTRE_AGREEMENT):
    """
    Whether the two methods place the face in different places.

    False when there is no second opinion: silence is not disagreement.
    """
    if frame is None or voted is None or not (radius > 0):
        return False
    apart = math.hypot(float(voted[0]) - frame.cx, float(voted[1]) - frame.cy)
    return bool(apart > tolerance * float(radius))
