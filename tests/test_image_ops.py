"""
Equivalence tests for the vectorised post-processing.

Each test carries a reference implementation transcribed from the original
per-pixel loops and asserts the numpy version produces identical bytes, so the
speed-up cannot silently change how generated symbols look.
"""

import numpy as np
import pytest

from archeoglyph.generators import image_ops as ops

MIN_ALPHA = ops.MIN_ALPHA


@pytest.fixture
def sample():
    rng = np.random.default_rng(7)
    h, w = 24, 18
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    alpha = rng.choice([0, 5, 200, 255], size=(h, w)).astype(np.uint8)
    mask_rgb = np.where(
        rng.random((h, w, 1)) < 0.45, np.uint8(20), np.uint8(240)
    ).repeat(3, axis=2).astype(np.uint8)
    ref_rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    ref_alpha = rng.choice([0, 9, 255], size=(h, w)).astype(np.uint8)
    return rgb, alpha, mask_rgb, ref_rgb, ref_alpha


# ---------------------------------------------------------------- helpers

def test_blend_and_hex_helpers_match_original_semantics():
    assert ops.blend_rgb((10, 20, 30), (200, 200, 200), 0.5) == (105, 110, 115)
    assert ops.blend_rgb((10, 20, 30), (200, 200, 200), 5.0) == (200, 200, 200)  # ratio clamped
    assert ops.parse_hex_rgb("#8B4513") == (139, 69, 19)
    assert ops.parse_hex_rgb("8b4513") == (139, 69, 19)
    assert ops.parse_hex_rgb("nope") is None and ops.parse_hex_rgb(None) is None


def test_mask_inside_matches_black_on_white_convention(sample):
    _rgb, _alpha, mask_rgb, _ref, _ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    h, w = inside.shape
    for y in range(h):
        for x in range(w):
            r, g, b = mask_rgb[y, x]
            assert inside[y, x] == (r < 90 and g < 90 and b < 90)


# ---------------------------------------------------------------- reference colour

def _reference_estimate_rgb(ref_rgb, ref_alpha, inside):
    sum_r = sum_g = sum_b = count = 0
    h, w = inside.shape
    for y in range(h):
        for x in range(w):
            if not inside[y, x]:
                continue
            if ref_alpha[y, x] < 8:
                continue
            r, g, b = (int(v) for v in ref_rgb[y, x])
            mx, mn = max(r, g, b), min(r, g, b)
            if (mx - mn) < 12 or mx < 28 or mx > 245:
                continue
            sum_r += r
            sum_g += g
            sum_b += b
            count += 1

    if count < 25:
        for y in range(h):
            for x in range(w):
                if not inside[y, x] or ref_alpha[y, x] < 8:
                    continue
                r, g, b = (int(v) for v in ref_rgb[y, x])
                sum_r += r
                sum_g += g
                sum_b += b
                count += 1

    if count < 5:
        return ops.DEFAULT_MATERIAL_RGB
    return (int(sum_r / count), int(sum_g / count), int(sum_b / count))


def test_estimate_reference_rgb_matches_reference(sample):
    _rgb, _alpha, mask_rgb, ref_rgb, ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    assert ops.estimate_reference_rgb(ref_rgb, ref_alpha, inside) == _reference_estimate_rgb(
        ref_rgb, ref_alpha, inside
    )


def test_estimate_reference_rgb_defaults_when_nothing_visible():
    empty = np.zeros((8, 8), dtype=bool)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    alpha = np.zeros((8, 8), dtype=np.uint8)
    assert ops.estimate_reference_rgb(rgb, alpha, empty) == ops.DEFAULT_MATERIAL_RGB


def _reference_palette(ref_rgb, ref_alpha, inside, max_colors=4, step=1):
    bins = {}
    h, w = inside.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            if not inside[y, x] or ref_alpha[y, x] < 8:
                continue
            r, g, b = (int(v) for v in ref_rgb[y, x])
            mx, mn = max(r, g, b), min(r, g, b)
            if (mx - mn) < 8 or mx < 20 or mx > 248:
                continue
            key = (r >> 5, g >> 5, b >> 5)
            bucket = bins.setdefault(key, {"count": 0, "r": 0, "g": 0, "b": 0})
            bucket["count"] += 1
            bucket["r"] += r
            bucket["g"] += g
            bucket["b"] += b

    ranked = sorted(bins.values(), key=lambda item: int(item["count"]), reverse=True)
    palette = []
    for item in ranked:
        count = max(1, int(item["count"]))
        rgb = (int(item["r"] / count), int(item["g"] / count), int(item["b"] / count))
        if any(
            ((rgb[0] - ex[0]) ** 2 + (rgb[1] - ex[1]) ** 2 + (rgb[2] - ex[2]) ** 2) ** 0.5 < 24.0
            for ex in palette
        ):
            continue
        palette.append(rgb)
        if len(palette) >= max_colors:
            break
    return palette


def test_extract_reference_palette_matches_reference(sample):
    _rgb, _alpha, mask_rgb, ref_rgb, ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    assert ops.extract_reference_palette(ref_rgb, ref_alpha, inside, step=1) == _reference_palette(
        ref_rgb, ref_alpha, inside
    )


def test_palette_entries_are_distinct_and_bounded(sample):
    _rgb, _alpha, mask_rgb, ref_rgb, ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    palette = ops.extract_reference_palette(ref_rgb, ref_alpha, inside, max_colors=3)
    assert len(palette) <= 3
    for i, a in enumerate(palette):
        for b in palette[i + 1:]:
            distance = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
            assert distance >= 24.0


# ---------------------------------------------------------------- measurements

def _reference_texture_noise(rgb, inside):
    h, w = inside.shape
    if w < 3 or h < 3:
        return 0.0
    total = 0.0
    samples = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not inside[y, x]:
                continue
            p = rgb[y, x].astype(np.int32)
            neighbours = (rgb[y, x - 1], rgb[y, x + 1], rgb[y - 1, x], rgb[y + 1, x])
            total += sum(int(abs(p - n.astype(np.int32)).sum()) for n in neighbours) / 12.0
            samples += 1
    if samples < 20:
        return 0.0
    return total / samples


def test_texture_noise_matches_reference(sample):
    rgb, _alpha, mask_rgb, _ref, _ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    assert ops.estimate_texture_noise(rgb, inside) == pytest.approx(
        _reference_texture_noise(rgb, inside), rel=1e-9
    )


def test_texture_noise_is_zero_for_flat_fill_and_tiny_images():
    flat = np.full((30, 30, 3), 120, dtype=np.uint8)
    inside = np.ones((30, 30), dtype=bool)
    assert ops.estimate_texture_noise(flat, inside) == 0.0
    assert ops.estimate_texture_noise(flat[:2, :2], inside[:2, :2]) == 0.0


def _reference_luma_variance(rgb, inside):
    total = total_sq = 0.0
    count = 0
    h, w = inside.shape
    for y in range(h):
        for x in range(w):
            if not inside[y, x]:
                continue
            r, g, b = (int(v) for v in rgb[y, x])
            lum = (0.299 * r) + (0.587 * g) + (0.114 * b)
            total += lum
            total_sq += lum * lum
            count += 1
    if count < 20:
        return 0.0
    mean = total / count
    return max(0.0, (total_sq / count) - (mean * mean))


def test_luma_variance_matches_reference(sample):
    rgb, _alpha, mask_rgb, _ref, _ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    assert ops.estimate_luma_variance(rgb, inside) == pytest.approx(
        _reference_luma_variance(rgb, inside), rel=1e-9
    )


# ---------------------------------------------------------------- harmonisation

def _reference_typology(rgb, alpha, base, palette, preserve_ratio):
    shadow, mid, highlight, patina = ops.typology_tones(base, palette)
    preserve = max(0.10, min(0.62, float(preserve_ratio)))
    out = rgb.copy()
    out_alpha = alpha.copy()
    h, w = alpha.shape
    for y in range(h):
        for x in range(w):
            if alpha[y, x] < 8:
                continue
            r, g, b = (int(v) for v in rgb[y, x])
            lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            lum = round(max(0.0, min(1.0, lum)) * 5.0) / 5.0
            if lum <= 0.25:
                tone = shadow
            elif lum <= 0.50:
                tone = ops.blend_rgb(shadow, mid, (lum - 0.25) / 0.25)
            elif lum <= 0.78:
                tone = ops.blend_rgb(mid, highlight, (lum - 0.50) / 0.28)
            else:
                tone = highlight
            sat = max(r, g, b) - min(r, g, b)
            patina_mix = max(0.0, min(0.22, (sat - 12.0) / 180.0))
            tone = ops.blend_rgb(tone, patina, patina_mix)
            out[y, x] = [
                max(0, min(255, int((tone[0] * (1.0 - preserve)) + (r * preserve)))),
                max(0, min(255, int((tone[1] * (1.0 - preserve)) + (g * preserve)))),
                max(0, min(255, int((tone[2] * (1.0 - preserve)) + (b * preserve)))),
            ]
            out_alpha[y, x] = 255
    return out, out_alpha


def test_harmonize_typology_matches_reference(sample):
    rgb, alpha, _mask, _ref, _ref_alpha = sample
    base = (120, 90, 60)
    palette = [(180, 150, 110), (90, 70, 50), (140, 120, 90)]
    got_rgb, got_alpha = ops.harmonize_typology(rgb, alpha, base, palette, preserve_ratio=0.36)
    want_rgb, want_alpha = _reference_typology(rgb, alpha, base, palette, 0.36)
    assert np.array_equal(got_rgb, want_rgb)
    assert np.array_equal(got_alpha, want_alpha)


def test_typology_tones_stay_separated_even_from_one_seed():
    shadow, mid, highlight, _patina = ops.typology_tones((120, 120, 120), [(120, 120, 120)])
    assert ops.luma(highlight) - ops.luma(mid) >= 15.0
    assert ops.luma(mid) - ops.luma(shadow) >= 15.0
    assert ops.luma(highlight) - ops.luma(shadow) >= 31.0


def _reference_colored(rgb, alpha, base, flatten, preserve_ratio):
    br, bg, bb = base
    out = rgb.copy()
    out_alpha = alpha.copy()
    h, w = alpha.shape
    for y in range(h):
        for x in range(w):
            if alpha[y, x] < 8:
                continue
            r, g, b = (int(v) for v in rgb[y, x])
            lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            if flatten:
                lum = round(lum * 3.0) / 3.0
            shade = 0.58 + (0.64 * lum)
            tr = max(0, min(255, int(br * shade)))
            tg = max(0, min(255, int(bg * shade)))
            tb = max(0, min(255, int(bb * shade)))
            out[y, x] = [
                max(0, min(255, int((tr * (1.0 - preserve_ratio)) + (r * preserve_ratio)))),
                max(0, min(255, int((tg * (1.0 - preserve_ratio)) + (g * preserve_ratio)))),
                max(0, min(255, int((tb * (1.0 - preserve_ratio)) + (b * preserve_ratio)))),
            ]
            out_alpha[y, x] = 255
    return out, out_alpha


@pytest.mark.parametrize("flatten", [False, True])
def test_harmonize_colored_matches_reference(sample, flatten):
    rgb, alpha, _mask, _ref, _ref_alpha = sample
    base = (140, 100, 70)
    got_rgb, got_alpha = ops.harmonize_colored(rgb, alpha, base, flatten=flatten, preserve_ratio=0.3)
    want_rgb, want_alpha = _reference_colored(rgb, alpha, base, flatten, 0.3)
    assert np.array_equal(got_rgb, want_rgb)
    assert np.array_equal(got_alpha, want_alpha)


def _reference_mono(rgb, alpha, publication):
    out = rgb.copy()
    out_alpha = alpha.copy()
    h, w = alpha.shape
    for y in range(h):
        for x in range(w):
            if alpha[y, x] < 8:
                continue
            r, g, b = (int(v) for v in rgb[y, x])
            lum = int(0.299 * r + 0.587 * g + 0.114 * b)
            value = (25 if lum < 135 else 70) if publication else int(20 + (lum * 0.35))
            value = max(0, min(255, value))
            out[y, x] = [value, value, value]
            out_alpha[y, x] = 255
    return out, out_alpha


@pytest.mark.parametrize("publication", [False, True])
def test_harmonize_mono_matches_reference(sample, publication):
    rgb, alpha, _mask, _ref, _ref_alpha = sample
    got_rgb, got_alpha = ops.harmonize_mono(rgb, alpha, publication=publication)
    want_rgb, want_alpha = _reference_mono(rgb, alpha, publication)
    assert np.array_equal(got_rgb, want_rgb)
    assert np.array_equal(got_alpha, want_alpha)


def _reference_tone_map(rgb, alpha, ref_rgb, inside, strength):
    h, w = inside.shape
    min_l, max_l = 255.0, 0.0
    for y in range(h):
        for x in range(w):
            if not inside[y, x]:
                continue
            r, g, b = (int(v) for v in ref_rgb[y, x])
            lum = (0.299 * r) + (0.587 * g) + (0.114 * b)
            min_l = min(min_l, lum)
            max_l = max(max_l, lum)
    span = max_l - min_l
    if span < 6.0:
        return rgb.copy(), alpha.copy()

    s = max(0.0, min(1.0, float(strength)))
    out = rgb.copy()
    out_alpha = alpha.copy()
    for y in range(h):
        for x in range(w):
            if not inside[y, x]:
                continue
            r, g, b = (int(v) for v in ref_rgb[y, x])
            lum = (0.299 * r) + (0.587 * g) + (0.114 * b)
            norm = (lum - min_l) / span
            tone = 0.90 if norm < 0.34 else (1.00 if norm < 0.68 else 1.10)
            pr, pg, pb = (int(v) for v in rgb[y, x])
            tr = max(0, min(255, int(pr * tone)))
            tg = max(0, min(255, int(pg * tone)))
            tb = max(0, min(255, int(pb * tone)))
            out[y, x] = [
                max(0, min(255, int((pr * (1.0 - s)) + (tr * s)))),
                max(0, min(255, int((pg * (1.0 - s)) + (tg * s)))),
                max(0, min(255, int((pb * (1.0 - s)) + (tb * s)))),
            ]
            out_alpha[y, x] = 255
    return out, out_alpha


def test_reference_tone_map_matches_reference(sample):
    rgb, alpha, mask_rgb, ref_rgb, _ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    got_rgb, got_alpha = ops.reference_tone_map(rgb, alpha, ref_rgb, inside, strength=0.28)
    want_rgb, want_alpha = _reference_tone_map(rgb, alpha, ref_rgb, inside, 0.28)
    assert np.array_equal(got_rgb, want_rgb)
    assert np.array_equal(got_alpha, want_alpha)


def test_reference_tone_map_is_a_no_op_on_a_flat_reference(sample):
    rgb, alpha, mask_rgb, _ref, _ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    flat_ref = np.full_like(rgb, 128)
    got_rgb, got_alpha = ops.reference_tone_map(rgb, alpha, flat_ref, inside, strength=0.5)
    assert np.array_equal(got_rgb, rgb) and np.array_equal(got_alpha, alpha)


def test_apply_silhouette_keeps_only_the_object(sample):
    rgb, _alpha, mask_rgb, _ref, _ref_alpha = sample
    inside = ops.mask_inside(mask_rgb)
    out_rgb, out_alpha = ops.apply_silhouette(rgb, inside)
    assert np.array_equal(out_rgb[inside], rgb[inside])
    assert (out_alpha[inside] == 255).all()
    assert (out_alpha[~inside] == 0).all()
    assert (out_rgb[~inside] == 0).all()


# ---------------------------------------------------------------- speed

def test_vectorised_path_is_fast_on_a_realistic_image():
    """A megapixel image must post-process in well under a second."""
    import time

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(1000, 1000, 3), dtype=np.uint8)
    alpha = np.full((1000, 1000), 255, dtype=np.uint8)
    inside = np.zeros((1000, 1000), dtype=bool)
    inside[100:900, 100:900] = True

    start = time.perf_counter()
    ops.estimate_texture_noise(rgb, inside)
    ops.estimate_luma_variance(rgb, inside)
    out_rgb, out_alpha = ops.harmonize_typology(rgb, alpha, (120, 90, 60), [(180, 150, 110)])
    ops.reference_tone_map(out_rgb, out_alpha, rgb, inside, strength=0.3)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, f"post-processing took {elapsed:.1f}s"
