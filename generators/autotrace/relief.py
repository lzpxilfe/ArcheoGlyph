# -*- coding: utf-8 -*-
"""
Relief from a stack of photographs lit from different directions.

Shallow relief - the decoration on a bronze mirror or a roof tile end - is
height, not colour, and a single photograph does not carry height. Eight
attempts at reading it from one image all lost to the same thing: surface
stains, discolouration and grain, which look exactly like decoration to any
measurement made on one frame.

Those two separate the moment the light moves. A stain is the same however
the object is lit; a raised petal edge is bright on one side under one lamp
position and bright on the other under the next. So the per-pixel *variation
across the stack* is relief with the albedo divided out, which is the
simplified form of the Reflectance Transformation Imaging archaeology
already uses to record this kind of decoration.

Three or four frames from a phone with a desk lamp moved between them is
enough. This module does not solve for surface normals: it needs no light
calibration, and the fold-and-replay reading downstream only wants a clean
single-channel image.
"""

import numpy as np

from ...log import log, log_exception

try:
    import cv2
except ImportError:  # pragma: no cover - mirrors the rest of the package
    cv2 = None

#: Below this a "stack" cannot say anything a single photograph could not:
#: with one frame there is nothing to compare against.
MIN_STACK = 2

#: Lamp positions are themselves an arrangement in a circle, and with few of
#: them that arrangement is stamped into the relief map. Four lamps around a
#: disc make an eight-petal tile read as four - measured, not supposed. Five
#: or more, and an odd count for preference, keeps the lamps out of the
#: answer; below this the reading is not offered at all.
ADVISED_LIGHTS = 5

#: How far ECC is allowed to move a frame onto the first, as a fraction of
#: the image's smaller side. Hand-held frames drift a little; anything past
#: this is a different photograph, not a shifted one.
MAX_ALIGN_SHIFT = 0.08


def _gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return image.astype(np.float32)


def align_light_stack(images):
    """
    Bring every frame onto the first by translation.

    Returns (aligned, moved) where ``moved`` counts the frames that actually
    needed shifting. A frame that cannot be aligned is kept as it is and
    said so in the log: silently dropping it would leave a relief map built
    from fewer lights than the user thinks they gave.
    """
    if cv2 is None or not images:
        return list(images or []), 0
    reference = _gray(images[0])
    reference = reference / (reference.max() or 1.0)
    limit = MAX_ALIGN_SHIFT * float(min(reference.shape[:2]))
    aligned, moved = [images[0]], 0
    for index, frame in enumerate(images[1:], start=1):
        try:
            probe = _gray(frame)
            probe = probe / (probe.max() or 1.0)
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
            cv2.findTransformECC(reference, probe, warp, cv2.MOTION_TRANSLATION,
                                 criteria, None, 5)
            shift = float(np.hypot(warp[0, 2], warp[1, 2]))
            if shift > limit:
                log(f"Light-stack frame {index + 1} sits {shift:.0f}px from the "
                    f"first, past the {limit:.0f}px this expects; using it "
                    f"unaligned. Relief needs the camera to stay put.",
                    level="warning")
                aligned.append(frame)
                continue
            if shift > 0.5:
                moved += 1
            aligned.append(cv2.warpAffine(
                frame, warp, (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE))
        except Exception as exc:
            log(f"Light-stack frame {index + 1} could not be aligned "
                f"({type(exc).__name__}); using it as it is.", level="warning")
            aligned.append(frame)
    return aligned, moved


def relief_from_light_stack(images, align=True):
    """
    A single-channel relief map, or None when the stack cannot give one.

    Each frame is normalised for overall exposure first - the lamp does not
    only move, it also changes how much light reaches the object - and the
    map is the spread of each pixel across the stack. Flat, evenly coloured
    ground varies not at all and lands near zero; a stain does the same;
    only surfaces whose angle to the lamp changes light up.
    """
    if cv2 is None:
        return None
    frames = [f for f in (images or []) if f is not None]
    if len(frames) < MIN_STACK:
        return None
    try:
        shape = frames[0].shape[:2]
        if any(f.shape[:2] != shape for f in frames):
            log("Light-stack frames are different sizes; relief needs the "
                "same framing in every shot.", level="warning")
            return None
        if align:
            frames, _moved = align_light_stack(frames)

        stack = []
        for frame in frames:
            gray = _gray(frame)
            # Divide out exposure, not brightness structure: a lamp moved
            # closer makes everything brighter without changing the relief.
            mean = float(gray.mean())
            stack.append(gray / (mean if mean > 1e-6 else 1.0))
        stack = np.stack(stack, axis=0)

        relief = stack.max(axis=0) - stack.min(axis=0)
        spread = float(relief.max() - relief.min())
        if spread < 1e-9:
            log("Every frame in the light stack is identical; relief needs "
                "the light moved between shots.", level="warning")
            return None
        relief = (relief - relief.min()) / spread
        return (relief * 255.0).astype(np.uint8)
    except Exception as exc:
        log_exception("relief_from_light_stack", exc)
        return None


def fold_is_confounded_by_the_lights(folds, frame_count):
    """
    Whether a fold count is indistinguishable from the lamp arrangement.

    With four lamps evenly around a disc, a four-fold reading may be the
    artefact or may be the lighting, and nothing in the stack can separate
    them - so it is refused rather than reported. Take another frame from a
    new position and the ambiguity goes away.
    """
    return bool(folds) and bool(frame_count) and int(folds) == int(frame_count)
