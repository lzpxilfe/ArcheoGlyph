import io as std_io
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from archeoglyph.generators.autotrace import colors, io as image_io, model_store, segment
from archeoglyph.generators.autotrace.options import AutoTraceOptions
from archeoglyph.generators.autotrace.pipeline import run_autotrace
from archeoglyph.generators.autotrace.round_motif import polar_unwrap
from tests import synthetic


def _iou(a, b):
    a = a > 0
    b = b > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / max(1, union)


# ---------------------------------------------------------------- io

def test_load_image_keeps_alpha_and_returns_bgr(tmp_path):
    path = synthetic.write_png(tmp_path / "cutout.png", synthetic.rgba_cutout())
    loaded = image_io.load_image(path)
    assert loaded.bgr.shape[2] == 3 and loaded.bgr.dtype == np.uint8
    assert loaded.alpha is not None
    assert _iou(loaded.alpha, synthetic.blade_mask()) > 0.999


def test_load_image_promotes_grayscale(tmp_path):
    gray = np.full((40, 60), 90, dtype=np.uint8)
    path = synthetic.write_png(tmp_path / "gray.png", gray)
    loaded = image_io.load_image(path)
    assert loaded.bgr.shape == (40, 60, 3) and loaded.alpha is None


def test_load_image_applies_exif_orientation(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (300, 200), (200, 200, 200))
    exif = PIL.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise
    path = tmp_path / "rot.jpg"
    img.save(path, exif=exif)
    loaded = image_io.load_image(str(path))
    assert loaded.bgr.shape[:2] == (300, 200)


# ---------------------------------------------------------------- segment

def test_alpha_channel_wins_over_heuristics():
    rgba = synthetic.rgba_cutout()
    mask = segment.select_mask(rgba[:, :, :3], backend="auto", alpha=rgba[:, :, 3])
    assert _iou(mask, synthetic.blade_mask()) > 0.98


def test_opencv_mask_recovers_blade_silhouette():
    mask = segment.get_mask_opencv(synthetic.ellipse_blade())
    # Blur + closing add a 1-2 px rim, so accept a slightly generous mask.
    assert _iou(mask, synthetic.blade_mask()) > 0.92


def test_opencv_mask_keeps_dark_object_on_white():
    img = synthetic.dark_flint_on_white()
    truth = (img[:, :, 0] < 100).astype(np.uint8) * 255
    mask = segment.get_mask_opencv(img)
    assert _iou(mask, truth) > 0.85


def test_smooth_mask_edges_fills_holes_even_when_object_touches_corner():
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[0:60, 0:60] = 255
    mask[20:30, 20:30] = 0  # hole
    out = segment.smooth_mask_edges(mask)
    assert out[25, 25] == 255
    assert out[100, 100] == 0


def test_select_mask_falls_back_when_model_fails():
    img = synthetic.ellipse_blade()

    def broken(_bgr):
        raise RuntimeError("model exploded")

    mask = segment.select_mask(img, backend="onnx", onnx_fn=broken)
    assert _iou(mask, synthetic.blade_mask()) > 0.92


# ---------------------------------------------------------------- colours / geometry

def test_dominant_color_is_exact_for_flat_object_and_deterministic():
    img = synthetic.ellipse_blade(color=(60, 90, 140))  # BGR
    mask = synthetic.blade_mask()
    assert colors.extract_dominant_color(img, mask) == "#8c5a3c"
    assert colors.extract_material_palette(img, mask) == colors.extract_material_palette(img, mask)


def test_polar_unwrap_is_radius_major():
    img = np.zeros((200, 200), dtype=np.uint8)
    import cv2
    cv2.circle(img, (100, 100), 60, 255, 2)
    polar = polar_unwrap(img, 100, 100, 100, n_theta=180, n_rad=100)
    assert polar.shape == (100, 180)
    row = int(np.argmax(polar.mean(axis=1)))
    assert abs(row - 59) <= 3


# ---------------------------------------------------------------- pipeline

def _run(img, **kw):
    opts = AutoTraceOptions(**kw)
    return run_autotrace(img, opts, lambda bgr: segment.get_mask_opencv(bgr))


def _path_count(svg):
    return len(ET.fromstring(svg).findall(".//{http://www.w3.org/2000/svg}path"))


@pytest.mark.parametrize("style", ["Simple Symbol", "Line", "Measured"])
def test_pipeline_is_deterministic_and_valid(style):
    img = synthetic.ellipse_blade()
    a = _run(img, style=style)
    b = _run(img, style=style)
    assert a == b
    root = ET.fromstring(a)
    assert root.attrib["viewBox"].startswith("0 0 ")
    assert _path_count(a) >= 1


def test_synthetic_structure_lines_are_opt_in():
    img = synthetic.ellipse_blade()
    plain = _run(img, style="Simple Symbol", synthetic_structure=False)
    schematic = _run(img, style="Simple Symbol", synthetic_structure=True)
    assert _path_count(schematic) > _path_count(plain)


def test_oval_keeps_its_outline_instead_of_a_circle():
    img = synthetic.blank(400)
    import cv2
    cv2.ellipse(img, (200, 200), (150, 110), 0, 0, 360, (60, 90, 140), -1)  # aspect 0.73 -> roundish
    svg = _run(img, style="Line")
    body = ET.fromstring(svg).find(".//{http://www.w3.org/2000/svg}path").attrib["d"]
    xs, ys = [], []
    for tok in body.replace("M", "").replace("Z", "").split("L"):
        tok = tok.strip()
        if tok:
            x, y = tok.split(",")
            xs.append(float(x))
            ys.append(float(y))
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    assert 0.65 < height / width < 0.82


def test_options_normalize_bad_values():
    opts = AutoTraceOptions(detail_mode="turbo", round_strategy="?", factuality=250, color="  ").normalized()
    assert opts.detail_mode == "fast" and opts.round_strategy == "image_first"
    assert opts.factuality == 100 and opts.color is None


# ---------------------------------------------------------------- model store

def _fake_urlopen(payload):
    class _Resp(std_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return lambda request, timeout=0: _Resp(payload)


def test_model_download_verifies_hash_and_size(tmp_path, monkeypatch):
    import hashlib

    payload = b"onnx-bytes" * 100
    good = model_store.ModelSpec(
        key="t", label="t", filename="t.onnx", url="https://example.invalid/t.onnx",
        sha256=hashlib.sha256(payload).hexdigest(), size=len(payload), input_size=8,
        mean=(0, 0, 0), std=(1, 1, 1),
    )
    monkeypatch.setattr(model_store.urllib.request, "urlopen", _fake_urlopen(payload))
    path = model_store.download_model(good, str(tmp_path))
    assert path.endswith("t.onnx") and model_store.is_installed(good, str(tmp_path))
    assert model_store.verify_model(good, str(tmp_path))

    bad = model_store.ModelSpec(**{**good.__dict__, "sha256": "0" * 64, "filename": "bad.onnx"})
    with pytest.raises(model_store.ModelStoreError):
        model_store.download_model(bad, str(tmp_path))
    assert not model_store.is_installed(bad, str(tmp_path))
    assert not [n for n in (tmp_path / "archeoglyph" / "models").iterdir() if n.name.endswith(".part")]
