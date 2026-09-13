# -*- coding: utf-8 -*-
"""
Synthetic image fixtures drawn with OpenCV so tests need no real photos.

All functions return uint8 numpy arrays in OpenCV's BGR (or BGRA) order.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def blank(size=400, color=(240, 240, 240)):
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = np.array(color, dtype=np.uint8)
    return canvas


def ellipse_blade(size=400, color=(60, 90, 140)):
    """Filled vertical ellipse resembling a blade silhouette on a light background."""
    img = blank(size)
    cv2.ellipse(img, (size // 2, size // 2), (size // 8, size // 3), 0, 0, 360, color, -1)
    return img


def blade_mask(size=400):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 8, size // 3), 0, 0, 360, 255, -1)
    return mask


def mirror_with_rings(size=400):
    """Bronze-ish disc with two dark concentric rings and a central boss."""
    img = blank(size, color=(235, 235, 235))
    c = (size // 2, size // 2)
    r = int(size * 0.4)
    cv2.circle(img, c, r, (70, 120, 160), -1)
    cv2.circle(img, c, int(r * 0.75), (40, 60, 90), 3)
    cv2.circle(img, c, int(r * 0.45), (40, 60, 90), 3)
    cv2.circle(img, c, int(r * 0.12), (40, 60, 90), -1)
    return img


def plain_disc(size=400):
    """The same disc as mirror_with_rings, with nothing on its face.

    The control for every question of the form "does this artefact carry
    decoration": whatever is drawn on the mirror and not here is the reading,
    and whatever is drawn on both is the shape.
    """
    img = blank(size, color=(235, 235, 235))
    cv2.circle(img, (size // 2, size // 2), int(size * 0.4), (70, 120, 160), -1)
    return img


def open_vessel(size=400):
    """A pot: wide at the rim, widest near the top, narrowing to the base."""
    img = blank(size, color=(246, 246, 246))
    pts = np.array([
        [size * 0.18, size * 0.14],
        [size * 0.82, size * 0.14],
        [size * 0.74, size * 0.58],
        [size * 0.60, size * 0.88],
        [size * 0.40, size * 0.88],
        [size * 0.26, size * 0.58],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (150, 118, 92))
    return img


def rosette_disc(size=400):
    """A disc carrying more ornament than a drawn symbol would ever hold.

    Rim, bead ring, eight petals and a boss - the layout of a lotus roof tile
    end, cut as grooves so the relief reading has something to find. The
    control for it is plain_disc, which is the same disc with a bare face.
    """
    img = plain_disc(size)
    c = (size // 2, size // 2)
    r = int(size * 0.4)
    groove = (44, 74, 100)
    cv2.circle(img, c, int(r * 0.90), groove, 3)
    cv2.circle(img, c, int(r * 0.74), groove, 3)
    for step in range(12):                       # the bead ring
        angle = 2.0 * np.pi * step / 12.0
        cv2.circle(img, (int(c[0] + r * 0.82 * np.cos(angle)),
                         int(c[1] + r * 0.82 * np.sin(angle))),
                   max(2, int(r * 0.05)), groove, 2)
    for step in range(8):                        # the petals
        angle = 2.0 * np.pi * step / 8.0
        cv2.ellipse(img, (int(c[0] + r * 0.44 * np.cos(angle)),
                          int(c[1] + r * 0.44 * np.sin(angle))),
                    (int(r * 0.26), int(r * 0.15)),
                    float(np.degrees(angle)), 0, 360, groove, 3)
    cv2.circle(img, c, int(r * 0.14), groove, 3)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def _lit(height_field, mask, azimuth=35.0, face=(150, 150, 155),
         ground=(238, 238, 240)):
    """A height field photographed under a low lamp, as a BGR image.

    The control for the relief reading has to be a *photograph* of relief -
    shading, not ink - because that reading puts the shading back together
    into a height before it draws anything. Drawn grooves would test a
    different instrument entirely.
    """
    smooth = cv2.GaussianBlur(height_field, (0, 0), sigmaX=height_field.shape[0] * 0.008)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    lamp = np.radians(azimuth)
    shade = gx * np.cos(lamp) + gy * np.sin(lamp)
    img = np.empty(height_field.shape + (3,), dtype=np.uint8)
    img[:] = np.array(ground, dtype=np.uint8)
    lit = np.clip(np.array(face, dtype=np.float32)[None, None, :]
                  + (shade * 900.0)[:, :, None], 0, 255).astype(np.uint8)
    img[mask > 0] = lit[mask > 0]
    return img


def _diffuse(height_field, mask, face=(150, 150, 155), ground=(238, 238, 240)):
    """A height field photographed under a softbox, as a BGR image.

    Museum photographs are mostly lit this way, and it is a different
    picture from the lamp in ``_lit``: with light arriving from every side a
    groove is dark from every side and a dome is bright on top, so brightness
    follows *concavity* rather than the slope along one direction. A reading
    built for a lamp integrates this along a direction that does not exist.
    """
    smooth = cv2.GaussianBlur(height_field, (0, 0), sigmaX=height_field.shape[0] * 0.008)
    wide = cv2.GaussianBlur(smooth, (0, 0), sigmaX=height_field.shape[0] * 0.03)
    openness = smooth - wide
    img = np.empty(height_field.shape + (3,), dtype=np.uint8)
    img[:] = np.array(ground, dtype=np.uint8)
    lit = np.clip(np.array(face, dtype=np.float32)[None, None, :]
                  + (openness * 140.0)[:, :, None], 0, 255).astype(np.uint8)
    img[mask > 0] = lit[mask > 0]
    return img


def _disc_height(size, motif, folds=8):
    """``(height field, silhouette)`` for a disc, with or without ornament."""
    field = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    c = (size // 2, size // 2)
    r = int(size * 0.4)
    cv2.circle(mask, c, r, 255, -1)
    if not motif:
        return field, mask
    cv2.circle(field, c, r, 1.0, -1)                       # the rim, raised
    cv2.circle(field, c, int(r * 0.88), 0.0, -1)           # the face, sunk
    for step in range(int(folds)):                         # the petals
        angle = 2.0 * np.pi * step / float(folds)
        cv2.ellipse(field, (int(c[0] + r * 0.46 * np.cos(angle)),
                            int(c[1] + r * 0.46 * np.sin(angle))),
                    (int(r * 0.26), int(r * 0.15)),
                    float(np.degrees(angle)), 0, 360, 1.0, -1)
    cv2.circle(field, c, int(r * 0.14), 1.3, -1)           # the boss
    return field, mask


def petal_centres(size=400, folds=8):
    """Where ``lit_relief_disc``'s petals are, for a test to check against."""
    c = size // 2
    r = size * 0.4
    return [(c + r * 0.46 * np.cos(2.0 * np.pi * step / float(folds)),
             c + r * 0.46 * np.sin(2.0 * np.pi * step / float(folds)))
            for step in range(int(folds))]


def lit_relief_disc(size=400, motif=True, azimuth=35.0, folds=8):
    """A disc carrying a ring of petals, a boss and a raised rim, under a lamp.

    Ground truth for the relief reading: ``folds`` petals plus a boss and a
    rim, every one of which should come back as its own closed curve, and the
    repeat is known by construction - which is what lets a test ask whether
    the fold count was read correctly rather than whether the picture looks
    right. ``motif=False`` is the same disc with a bare face and is the
    control: a bare face has nothing to read, and a reading that draws
    something on it is drawing noise.
    """
    field, mask = _disc_height(size, motif, folds=folds)
    return _lit(field, mask, azimuth=azimuth)


def lit_plain_disc(size=400, azimuth=35.0):
    """``lit_relief_disc`` with nothing on its face."""
    return lit_relief_disc(size=size, motif=False, azimuth=azimuth)


def diffuse_relief_disc(size=400, motif=True, folds=8):
    """``lit_relief_disc``'s disc under a softbox instead of a lamp."""
    field, mask = _disc_height(size, motif, folds=folds)
    return _diffuse(field, mask)


def diffuse_plain_disc(size=400):
    """``diffuse_relief_disc`` with nothing on its face."""
    return diffuse_relief_disc(size=size, motif=False)


def knobbed_disc(size=400, knobs=2, azimuth=35.0):
    """A bare disc carrying ``knobs`` small raised loops at one radius.

    The multi-knobbed bronze mirror (다뉴세문경) is named for these: its face
    is fine hatching no symbol can carry, and its knobs come as a pair, or
    three, set at the same distance from the centre. One knob alone is the
    control - a lone raised spot on a bare face is a corrosion blister as
    often as a knob, and the reading must refuse it.
    """
    field, mask = _disc_height(size, False)
    c = size / 2.0
    r = size * 0.4
    for step in range(int(knobs)):
        angle = -np.pi / 2.0 + (step - (knobs - 1) / 2.0) * 0.85
        cv2.ellipse(field, (int(c + r * 0.36 * np.cos(angle)),
                            int(c + r * 0.36 * np.sin(angle))),
                    (int(r * 0.07), int(r * 0.05)), 0, 0, 360, 1.2, -1)
    return _lit(field, mask, azimuth=azimuth)


def dark_flint_on_white(size=400):
    """Dark grey flint shape on white paper with a soft cast shadow to the lower right."""
    img = blank(size, color=(252, 252, 252))
    pts = np.array([
        [size * 0.50, size * 0.12],
        [size * 0.66, size * 0.40],
        [size * 0.60, size * 0.85],
        [size * 0.40, size * 0.85],
        [size * 0.34, size * 0.40],
    ], dtype=np.int32)
    shadow = np.zeros_like(img)
    cv2.fillPoly(shadow, [pts + np.array([12, 14])], (200, 200, 200))
    shadow = cv2.GaussianBlur(shadow, (0, 0), 6)
    img = np.where(shadow > 0, np.minimum(img, 255 - (255 - shadow) // 4), img).astype(np.uint8)
    cv2.fillPoly(img, [pts], (55, 55, 60))
    return img


def rgba_cutout(size=400):
    """Object with a fully transparent background; alpha is the exact mask."""
    bgr = ellipse_blade(size, color=(120, 80, 40))
    alpha = blade_mask(size)
    return np.dstack([bgr, alpha])


def red_stroke_on_gray(size=300):
    """Isoluminant red stroke on a mid-gray ground (invisible in luminance alone)."""
    img = blank(size, color=(120, 120, 120))
    cv2.line(img, (30, size // 2), (size - 30, size // 2), (60, 60, 200), 3, cv2.LINE_AA)
    return img


def single_dark_stroke(size=300, thickness=3):
    """One dark diagonal stroke on white. Returns (image, drawn_length_px)."""
    img = blank(size, color=(250, 250, 250))
    p0 = (40, 60)
    p1 = (size - 40, size - 60)
    cv2.line(img, p0, p1, (20, 20, 20), thickness, cv2.LINE_AA)
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return img, length


def y_junction(size=300, thickness=3):
    """Three strokes meeting at one junction (Y shape)."""
    img = blank(size, color=(250, 250, 250))
    c = (size // 2, size // 2)
    for end in ((size // 2, 30), (40, size - 40), (size - 40, size - 40)):
        cv2.line(img, c, end, (20, 20, 20), thickness, cv2.LINE_AA)
    return img


def line_drawing_sherd(size=400):
    """Thin black outline drawing of a rim sherd profile on white, like a report figure."""
    img = blank(size, color=(255, 255, 255))
    pts = np.array([
        [size * 0.30, size * 0.20], [size * 0.70, size * 0.20],
        [size * 0.72, size * 0.30], [size * 0.62, size * 0.80],
        [size * 0.38, size * 0.80], [size * 0.28, size * 0.30],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (int(size * 0.33), int(size * 0.32)), (int(size * 0.67), int(size * 0.32)), (0, 0, 0), 1, cv2.LINE_AA)
    return img


def encode_png(image):
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return bytes(buf)


def write_png(path, image):
    path = str(path)
    assert cv2.imwrite(path, image)
    return path
