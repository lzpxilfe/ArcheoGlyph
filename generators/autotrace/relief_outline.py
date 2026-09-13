# -*- coding: utf-8 -*-
"""
Shallow relief read as closed curves, not as ink strokes.

``ink_centerline`` is a *centreline* tracer: it skeletonises dark strokes and
follows their middle. That is exactly right for a rubbing or a drawing, where
the ink is the record. It is the wrong instrument for a photograph of relief,
where what an illustrator draws is the *boundary of a raised element* - the
outline of a lotus petal, the outline of a dragon's body.

Two instruments were tried here before this one, and both are recorded in
``docs/auto_trace_limits.md`` because both were refuted by measurement:

* Skeletonising the ribbon of steep shading that a boundary makes. A ribbon
  that loops back on itself has a junction and a spur at every turn, so this
  fragments by construction - 310 and 290 curves on the two roof tile ends,
  of which 7 and 3 were closed. Closing the ribbon's gaps first made it worse
  (the lotus tile's longest curve fell from 0.82 of its width to 0.24).
* Taking the ribbon's own outer boundary, plus a polar trace of the disc's
  concentric rings. Closed, but not *stable*: a quantile of the slope is a
  knife-edge, and the same artefact read at two resolutions gave 5 curves and
  12. Normalising the reading scale did not fix it.

What this module does instead is put the shading back together into a height
before drawing anything. Under a low lamp the shading of shallow relief is, to
first order, the height's derivative along the lamp; integrating along that
direction turns edges back into *regions*, and the level set of a region is a
closed curve by construction. Measured on the three discs, the reading gives
the same curves at both working resolutions - 16, 4 and 1 - which the earlier
readings never did.

Nothing here invents evidence. A curve is kept only where the surface really
steps across it, and a plain face - a worn bronze mirror - yields its rim and
nothing else, which is the right answer for it.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from ...log import log_exception


#: How much of the picture is lamp rather than relief, as a share of the
#: artefact's radius. A lamp falls off broadly and relief does not.
LAMP_SCALE = 0.22

#: How far the integral remembers, as a share of the radius. This is the
#: largest raised element the reading can see whole: at 0.02 a lotus petal
#: comes apart into its own facets, at 0.16 the petals merge into the
#: rosette. The reading is a band-limited integral, so this also sets where
#: the integral stops drifting.
INTEGRAL_REACH = 0.06

#: Cross-wise smoothing, as a share of the radius. A one-dimensional integral
#: streaks along its own direction; this is what closes the streaks up without
#: blurring the elements themselves.
INTEGRAL_SMOOTH = 0.015

#: Where the outline is cut, as a quantile of the height inside the face. At
#: 0.75 a lotus petal splits into an inner and an outer lobe; at 0.60 it comes
#: out as one closed curve, which is what an illustrator draws.
HEIGHT_LEVEL = 0.60

#: How far in from the silhouette the reading stops, as a share of the radius.
#: The silhouette's own shading is the steepest thing in the frame and it is
#: already drawn as the outline.
FACE_INSET = 0.03

#: How much the surface must actually step across a curve for it to be drawn,
#: in units of the height map's own spread. Measured over the three discs at
#: both working resolutions: a lotus tile's petals step 0.52 to 1.06 and a
#: dragon tile's body 0.41 to 0.87, while the flat face of a worn bronze
#: mirror offers curves that step 0.17 to 0.28 - and its one real feature, the
#: raised rim, steps 0.64. At 0.40 the mirror keeps its rim and loses the rest.
MIN_STEP = 0.40

#: The smallest closed curve worth drawing, as a share of the artefact's area,
#: and how hard the curves are simplified, as a share of their own perimeter.
#: A symbol is read at 64 pixels, where a curve enclosing a hundredth of the
#: artefact is six pixels across - already the smallest mark that says
#: anything. Below that the lotus tile's chips of relief came out as a rash of
#: dots around its petals.
MIN_REGION_AREA = 0.010
OUTLINE_EPSILON = 0.005

#: How much of a curve may be the inset frame rather than the surface, before
#: it has to justify itself by encircling the face. A raised element cut off
#: by the inset is traced along the inset, and that part of the outline is not
#: evidence of anything. A real rim ring runs along it too - and goes all the
#: way round, which is what tells the two apart.
MAX_FRAME_SHARE = 0.15

#: How much a curve is smoothed before it is drawn, as a share of its own
#: perimeter. A level set of a photographed surface is ragged at the pixel
#: scale and an illustrator's outline is not; simplifying instead of smoothing
#: only trades the rag for a polygon.
OUTLINE_SMOOTH = 0.018


def lamp_azimuth(shading, mask):
    """
    Which way the lamp falls, from the structure tensor of the shading.

    A raised element lit from one side shades as a bright-then-dark doublet
    *along* the lamp and hardly at all across it, so the shading's gradients
    line up with the lamp. The tensor gives an axis rather than a direction -
    the sign is settled by ``relief_height``, which knows relief protrudes.
    """
    gx = cv2.Sobel(shading, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(shading, cv2.CV_32F, 0, 1, ksize=5)
    inside = mask > 0
    if not inside.any():
        return 0.0
    jxx = float((gx[inside] ** 2).mean())
    jyy = float((gy[inside] ** 2).mean())
    jxy = float((gx[inside] * gy[inside]).mean())
    return 0.5 * float(np.arctan2(2.0 * jxy, jxx - jyy))


def _integral_kernel(reach):
    """
    A band-limited integral, as a correlation kernel.

    Integrating a derivative outright drifts without bound, so the memory is
    let decay: the forward integral is a sum over past samples weighted
    ``a**s`` and the backward integral the same over future ones, and their
    difference - which is what this kernel is - recovers the height without
    the drift and without the half-element shift a one-sided integral has.
    """
    length = max(1.0, float(reach))
    decay = float(np.exp(-1.0 / length))
    span = max(3, int(round(length * 5.0)))
    steps = np.arange(-span, span + 1, dtype=np.float32)
    kernel = -0.5 * np.sign(steps) * (decay ** np.abs(steps))
    return kernel.reshape(1, -1)


def relief_height(bgr_img, mask, radius, azimuth=None):
    """
    The surface's height, in units of its own spread, and the lamp's azimuth.

    Returns ``(None, 0.0)`` when there is nothing to read.
    """
    if cv2 is None or bgr_img is None or mask is None or not (radius > 0):
        return None, 0.0
    try:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        inside = mask > 0
        if not inside.any():
            return None, 0.0
        # Flatten the ground to the artefact's own tone first. The step from
        # object to background is far larger than any relief on it, and left
        # in place the integral turns it into a cliff just inside the
        # silhouette - which came back as a crescent drawn on the bare face of
        # the plain control, where the right answer is nothing at all.
        gray[~inside] = float(np.median(gray[inside]))
        lamp = cv2.GaussianBlur(gray, (0, 0),
                                sigmaX=max(9.0, float(radius) * LAMP_SCALE))
        shading = gray - lamp
        shading[~inside] = 0.0
        if azimuth is None:
            azimuth = lamp_azimuth(shading, mask)

        # Turn the lamp onto the x axis, so the integral is one row at a time.
        height, width = shading.shape[:2]
        turn = cv2.getRotationMatrix2D((width / 2.0, height / 2.0),
                                       float(np.degrees(azimuth)), 1.0)
        side = int(round(float(np.hypot(width, height))))
        turn[0, 2] += (side - width) / 2.0
        turn[1, 2] += (side - height) / 2.0
        turned = cv2.warpAffine(shading, turn, (side, side))
        turned[cv2.warpAffine(mask, turn, (side, side),
                              flags=cv2.INTER_NEAREST) == 0] = 0.0

        surface = cv2.filter2D(turned, cv2.CV_32F,
                               _integral_kernel(radius * INTEGRAL_REACH))
        surface = cv2.GaussianBlur(
            surface, (0, 0),
            sigmaX=max(0.8, radius * 0.004),
            sigmaY=max(0.8, radius * INTEGRAL_SMOOTH))
        surface = cv2.warpAffine(surface, cv2.invertAffineTransform(turn),
                                 (width, height))
        surface[mask == 0] = 0.0

        spread = float(np.percentile(np.abs(surface[inside]), 98))
        if not (spread > 0):
            return None, float(azimuth)
        surface = np.clip(surface / spread, -2.0, 2.0)
        # The tensor gives the lamp's axis, not which end of it the lamp is
        # on, and the wrong end reads every groove as a ridge. Relief
        # protrudes from its ground: a few high elements over a broad low
        # field, which is a positive skew. A negative one means the map is
        # upside down.
        skew = float(np.mean(surface[inside] ** 3))
        if skew < 0:
            surface = -surface
        return surface, float(azimuth)
    except Exception as exc:
        log_exception("relief_height", exc)
        return None, 0.0


def _resample(points, step):
    """Points along the polyline, roughly ``step`` apart."""
    line = np.asarray(points, dtype=np.float32)
    run = np.concatenate([[0.0], np.cumsum(np.hypot(*(line[1:] - line[:-1]).T))])
    total = float(run[-1])
    if not (total > 0):
        return line
    at = np.linspace(0.0, total, max(16, int(total / max(1.0, float(step)))))
    return np.stack([np.interp(at, run, line[:, 0]),
                     np.interp(at, run, line[:, 1])], axis=1)


def _is_the_frame(points, frame_distance, radius, centre):
    """
    Whether this curve is mostly the inset frame and not a ring round the face.

    The worn bronze mirror's one offering was a crescent that ran along the
    inset for a third of its length and cut across the face for the rest,
    drawn just inside the silhouette the symbol already has. A roof tile's rim
    ring runs along the inset too, and encircles the face, and is worth
    drawing.
    """
    line = _resample(points, max(1.0, float(radius) * 0.01))
    rows, cols = frame_distance.shape[:2]
    on_frame = frame_distance[np.clip(line[:, 1].astype(int), 0, rows - 1),
                              np.clip(line[:, 0].astype(int), 0, cols - 1)]
    if float((on_frame < float(radius) * 0.02).mean()) <= MAX_FRAME_SHARE:
        return False
    return cv2.pointPolygonTest(np.asarray(points, np.int32),
                                (float(centre[0]), float(centre[1])), False) < 0


def _smooth_closed(points, radius, smoothing=OUTLINE_SMOOTH):
    """``points`` resampled evenly and smoothed round the loop."""
    line = _resample(points, max(1.0, float(radius) * 0.01))[:-1]
    count = len(line)
    if count < 8:
        return points
    width = int(round(count * float(smoothing)))
    if width < 1:
        return points
    window = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1)
                            / float(width)) ** 2)
    window /= window.sum()
    pad = len(window) // 2
    wrapped = np.concatenate([line[-pad:], line, line[:pad]], axis=0)
    smooth = np.stack([np.convolve(wrapped[:, axis], window, mode="valid")
                       for axis in (0, 1)], axis=1)
    out = [[int(round(x)), int(round(y))] for x, y in smooth]
    out.append(list(out[0]))
    return out


def step_across(points, surface, radius, reach=0.02):
    """
    How far the surface steps across ``points``, in units of its own spread.

    This is what separates a raised element's boundary from an iso-line
    wandering over a flat face: both are closed curves at the same height, and
    only one of them has a surface that changes across it.
    """
    line = _resample(points, max(1.0, float(radius) * 0.01))
    if len(line) < 3:
        return 0.0
    along = np.gradient(line, axis=0)
    normal = np.stack([-along[:, 1], along[:, 0]], axis=1)
    length = np.hypot(normal[:, 0], normal[:, 1])
    length[length == 0] = 1.0
    normal = normal / length[:, None] * (float(radius) * float(reach))
    rows, cols = surface.shape[:2]

    def sample(at):
        return surface[np.clip(at[:, 1].astype(int), 0, rows - 1),
                       np.clip(at[:, 0].astype(int), 0, cols - 1)]

    return float(np.median(np.abs(sample(line + normal) - sample(line - normal))))


def raised_outlines(bgr_img, mask, radius, level=HEIGHT_LEVEL,
                    min_step=MIN_STEP, min_area=MIN_REGION_AREA,
                    surface=None):
    """
    The raised elements' boundaries, as closed polylines.

    Each is a level set of the height, so it comes back closed - first point
    equal to last - which is what an illustrator draws and what skeletonising
    could never produce.
    """
    if cv2 is None:
        return []
    try:
        if surface is None:
            surface, _azimuth = relief_height(bgr_img, mask, radius)
        if surface is None:
            return []
        inset = max(3, int(round(float(radius) * FACE_INSET))) | 1
        face = cv2.erode(mask, np.ones((inset, inset), np.uint8))
        if not (face > 0).any():
            return []
        frame = cv2.distanceTransform(255 - cv2.Canny(face, 50, 150),
                                      cv2.DIST_L2, 5)
        moments = cv2.moments(face, binaryImage=True)
        centre = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        cut = float(np.quantile(surface[face > 0], float(level)))
        band = cv2.bitwise_and(((surface >= cut) * 255).astype(np.uint8), face)
        # A level set of a photographed surface is ragged at the pixel scale;
        # an outline drawn at legend size is not.
        smooth = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, int(radius * 0.02)) | 1,) * 2)
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, smooth)
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, smooth)

        contours, _hierarchy = cv2.findContours(band, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
        floor = float(min_area) * float(np.count_nonzero(mask))
        outlines = []
        for contour in contours:
            if cv2.contourArea(contour) < floor:
                continue
            simple = cv2.approxPolyDP(
                contour, OUTLINE_EPSILON * cv2.arcLength(contour, True), True)
            points = [[int(p[0][0]), int(p[0][1])] for p in simple]
            if len(points) < 3:
                continue
            points.append(list(points[0]))
            if _is_the_frame(points, frame, radius, centre):
                continue
            # Smoothed first, then tested. A curve is drawn smoothed, and a
            # smoothed curve sits a little off the steepest line, so testing
            # the ragged one and drawing the smooth one meant the same curve
            # could pass here and fail the identical test downstream.
            points = _smooth_closed(points, radius)
            if step_across(points, surface, radius) < float(min_step):
                continue
            outlines.append(points)
        return outlines
    except Exception as exc:
        log_exception("raised_outlines", exc)
        return []
