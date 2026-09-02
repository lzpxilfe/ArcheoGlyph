"""
The QImage <-> numpy bridge, exercised against a stand-in that mimics Qt's
buffer API (padded rows, sip-style constBits with setsize, PNG saving).

QGIS is not available here, so this checks the conversion logic itself: row
padding is honoured, channel order is RGB, and the PNG fallback produces the
same arrays as the direct buffer path.
"""

import numpy as np
import pytest

from archeoglyph.generators.huggingface_generator import HuggingFaceGenerator


class _Buffer:
    """sip.voidptr stand-in: bytes(...) only works after setsize()."""

    def __init__(self, payload):
        self._payload = payload
        self._size = None

    def setsize(self, size):
        self._size = int(size)

    def __bytes__(self):
        if self._size is None:
            raise RuntimeError("setsize() was not called")
        return self._payload[: self._size]


class FakeQImage:
    """
    Minimal QImage stand-in over an RGBA array, with padded scanlines so the
    bridge's stride handling is actually tested.
    """

    padding_pixels = 3
    readable_buffer = True
    encoder = None  # set by the test to emulate save()

    def __init__(self, rgba):
        self._rgba = np.ascontiguousarray(rgba.astype(np.uint8))

    def width(self):
        return int(self._rgba.shape[1])

    def height(self):
        return int(self._rgba.shape[0])

    def bytesPerLine(self):
        return 4 * (self.width() + self.padding_pixels)

    def sizeInBytes(self):
        return self.bytesPerLine() * self.height()

    def convertToFormat(self, _fmt):
        return self

    def constBits(self):
        if not self.readable_buffer:
            raise RuntimeError("buffer not accessible")
        padded = np.zeros((self.height(), self.width() + self.padding_pixels, 4), dtype=np.uint8)
        padded[:, : self.width(), :] = self._rgba
        return _Buffer(padded.tobytes())

    def save(self, _buffer, _fmt):
        return bool(self.encoder and self.encoder(self._rgba))


@pytest.fixture
def rgba():
    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, size=(9, 7, 4), dtype=np.uint8)
    data[0, 0] = [255, 0, 0, 255]      # red, to pin channel order
    data[0, 1] = [0, 0, 255, 128]      # blue, semi transparent
    return data


def test_direct_buffer_path_handles_padding_and_channel_order(rgba):
    image = FakeQImage(rgba)
    rgb, alpha = HuggingFaceGenerator._qimage_to_arrays(image)

    assert rgb.shape == (9, 7, 3) and alpha.shape == (9, 7)
    assert rgb[0, 0].tolist() == [255, 0, 0]
    assert rgb[0, 1].tolist() == [0, 0, 255]
    assert alpha[0, 1] == 128
    assert np.array_equal(rgb, rgba[:, :, :3])
    assert np.array_equal(alpha, rgba[:, :, 3])


def test_png_fallback_produces_the_same_arrays(rgba, monkeypatch):
    cv2 = pytest.importorskip("cv2")

    encoded = {}

    def _encode(data):
        bgra = data[:, :, [2, 1, 0, 3]]
        ok, buf = cv2.imencode(".png", bgra)
        encoded["payload"] = bytes(buf) if ok else None
        return ok

    image = FakeQImage(rgba)
    image.readable_buffer = False       # force the fallback
    image.encoder = _encode
    monkeypatch.setattr(
        HuggingFaceGenerator, "_image_to_png_bytes",
        staticmethod(lambda img: _encode(img._rgba) and encoded["payload"]),
    )

    rgb, alpha = HuggingFaceGenerator._qimage_to_arrays(image)
    assert np.array_equal(rgb, rgba[:, :, :3])
    assert np.array_equal(alpha, rgba[:, :, 3])


def test_byte_count_falls_back_across_qt_versions():
    class _Old(FakeQImage):
        def __init__(self, rgba):
            super().__init__(rgba)

        sizeInBytes = None  # Qt < 5.10

        def byteCount(self):
            return self.bytesPerLine() * self.height()

    image = _Old(np.zeros((4, 5, 4), dtype=np.uint8))
    assert HuggingFaceGenerator._image_byte_count(image) == image.byteCount()

    class _Ancient(FakeQImage):
        sizeInBytes = None
        byteCount = None

    ancient = _Ancient(np.zeros((4, 5, 4), dtype=np.uint8))
    assert HuggingFaceGenerator._image_byte_count(ancient) == ancient.bytesPerLine() * 4


def test_round_trip_through_arrays_is_lossless(rgba, monkeypatch):
    """rgb/alpha -> QImage -> rgb/alpha must return the same pixels."""
    created = {}

    def _fake_qimage(buffer, width, height, stride, _fmt):
        data = np.frombuffer(bytes(buffer), dtype=np.uint8)
        data = data[: height * stride].reshape(height, stride // 4, 4)[:, :width, :]
        image = FakeQImage(data.copy())
        created["image"] = image
        return image

    _fake_qimage.Format_RGBA8888 = 17

    monkeypatch.setattr(FakeQImage, "copy", lambda self: self, raising=False)
    monkeypatch.setattr(
        "archeoglyph.generators.huggingface_generator.QImage", _fake_qimage, raising=False
    )

    image = HuggingFaceGenerator._arrays_to_qimage(rgba[:, :, :3], rgba[:, :, 3])
    rgb, alpha = HuggingFaceGenerator._qimage_to_arrays(image)
    assert np.array_equal(rgb, rgba[:, :, :3])
    assert np.array_equal(alpha, rgba[:, :, 3])
