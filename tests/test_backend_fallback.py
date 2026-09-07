# -*- coding: utf-8 -*-
"""
Picking a mask backend that cannot run must not look like it ran.

Testing Auto Trace against real museum photographs turned up a trap: with
``mask_backend`` set to "onnx" but no model installed, the generator quietly
handed the work to OpenCV and returned a perfectly plausible symbol. The
results were indistinguishable from a real ONNX run unless you went looking,
so a comparison between the two backends silently compared OpenCV with
itself. Under "auto" that fallback is the intended behaviour; under an
explicit choice it has to be said out loud.
"""

from archeoglyph.generators import contour_generator as cg


class _Settings:
    def __init__(self, **values):
        self._values = values

    def value(self, key, default=None, **_):
        return self._values.get(key, default)


def _generator(backend, monkeypatch, *, runtime=True, model=None):
    monkeypatch.setattr(cg, "onnx_available", lambda: runtime)
    monkeypatch.setattr(cg, "installed_model", lambda base, key: model)
    return cg.ContourGenerator(_Settings(**{"ArcheoGlyph/mask_backend": backend}))


def test_choosing_onnx_without_the_runtime_says_so(monkeypatch):
    said = []
    monkeypatch.setattr(cg, "log", lambda msg, level="info": said.append((level, msg)))
    gen = _generator("onnx", monkeypatch, runtime=False)
    assert gen._onnx_backend() is None
    assert len(said) == 1
    level, message = said[0]
    assert level == "warning"
    assert "onnxruntime" in message and "OpenCV" in message


def test_choosing_onnx_without_a_model_names_where_it_looked(monkeypatch):
    said = []
    monkeypatch.setattr(cg, "log", lambda msg, level="info": said.append((level, msg)))
    gen = _generator("onnx", monkeypatch, model=None)
    assert gen._onnx_backend() is None
    assert len(said) == 1
    level, message = said[0]
    assert level == "warning"
    # Without the directory the message is unactionable - the user cannot
    # tell where to put the file.
    assert cg.models_dir(cg.profile_base_dir()) in message


def test_the_warning_is_said_once_not_once_per_image(monkeypatch):
    said = []
    monkeypatch.setattr(cg, "log", lambda msg, level="info": said.append(msg))
    gen = _generator("onnx", monkeypatch, runtime=False)
    for _ in range(5):
        gen._onnx_backend()
    assert len(said) == 1


def test_auto_falls_back_without_complaining(monkeypatch):
    """Under "auto" the fallback is the design, so it must stay quiet."""
    said = []
    monkeypatch.setattr(cg, "log", lambda msg, level="info": said.append(msg))
    gen = _generator("auto", monkeypatch, runtime=False)
    assert gen._onnx_backend() is None
    gen2 = _generator("auto", monkeypatch, model=None)
    assert gen2._onnx_backend() is None
    assert said == []


def test_opencv_never_reaches_the_onnx_path(monkeypatch):
    said = []
    monkeypatch.setattr(cg, "log", lambda msg, level="info": said.append(msg))
    gen = _generator("opencv", monkeypatch, runtime=False)
    assert gen._onnx_backend() is None
    assert said == []
