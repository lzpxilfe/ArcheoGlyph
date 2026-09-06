# -*- coding: utf-8 -*-
"""
ArchaeoGlyph - Contour Generator (Auto Trace) facade.

Thin QGIS-facing wrapper around the QGIS-free ``generators.autotrace``
package: reads settings, owns model backends and caches, and exposes the
public API used by the dialog and the AI generators.
"""

import os

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None
from qgis.PyQt.QtCore import QSettings

from .style_control_utils import (
    STYLE_CONTROL_EXAGGERATION,
    STYLE_CONTROL_FACTUALITY,
    STYLE_CONTROL_SYMBOLIC_LOOSENESS,
    resolve_style_controls,
)
from .autotrace.options import AutoTraceOptions
from .autotrace.io import adaptive_prescale, load_image, resize_alpha
from .autotrace.model_store import DEFAULT_MODEL_KEY, installed_model
from .autotrace.pipeline import run_autotrace
from .autotrace.sam_backend import SamBackend
from .autotrace.segment import OnnxSalientBackend, normalize_backend, onnx_available, select_mask
from ..log import log, log_exception

EMPTY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'


def profile_base_dir():
    """QGIS profile directory (model store root), with a plain-Python fallback."""
    try:
        from qgis.core import QgsApplication

        base = QgsApplication.qgisSettingsDirPath()
        if base:
            return base
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), ".archeoglyph")


def _settings_bool(settings, key, default=False):
    value = settings.value(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class ContourGenerator:
    """Generates SVG symbols from images using OpenCV (plus optional models)."""

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else QSettings()
        self._sam = SamBackend(self.settings)
        self._onnx = None
        self._onnx_path = None
        self._mask_cache = {}
        self._image_cache = None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _require_cv(self):
        if cv2 is None or np is None:
            raise ImportError(
                "OpenCV and NumPy are required for Auto Trace. "
                "Please install via 'pip install opencv-python-headless numpy'."
            )

    def _load(self, image_path):
        mtime = os.path.getmtime(image_path)
        if self._image_cache is not None and self._image_cache[0] == (image_path, mtime):
            return self._image_cache[1]
        loaded = load_image(image_path)
        if loaded is None:
            raise ValueError("Failed to load image.")
        self._image_cache = ((image_path, mtime), loaded)
        return loaded

    def _backend_key(self):
        return normalize_backend(self.settings.value('ArcheoGlyph/mask_backend', 'auto'))

    def _onnx_backend(self):
        backend = self._backend_key()
        if backend not in ("auto", "onnx") or not onnx_available():
            return None
        key = str(self.settings.value('ArcheoGlyph/onnx_bg_model', DEFAULT_MODEL_KEY) or DEFAULT_MODEL_KEY)
        found = installed_model(profile_base_dir(), key)
        if not found:
            return None
        spec, path = found
        if self._onnx is None or self._onnx_path != path:
            self._onnx = OnnxSalientBackend(path, spec)
            self._onnx_path = path
        return self._onnx

    def _mask_provider(self, loaded):
        """Callable ``processing_bgr -> mask`` with per-image caching."""
        backend = self._backend_key()
        onnx = self._onnx_backend()
        onnx_fn = onnx.get_mask if onnx is not None else None
        sam_fn = self._sam.get_mask if (backend in ("sam", "auto") and self._sam.configured()) else None
        try:
            mtime = os.path.getmtime(loaded.path) if loaded.path else 0
        except OSError:
            mtime = 0

        def provider(processing_bgr):
            key = (loaded.path, mtime, processing_bgr.shape[:2], backend, onnx_fn is not None, sam_fn is not None)
            mask = self._mask_cache.get(key)
            if mask is None:
                alpha = resize_alpha(loaded.alpha, processing_bgr.shape)
                mask = select_mask(processing_bgr, backend=backend, alpha=alpha, onnx_fn=onnx_fn, sam_fn=sam_fn)
                if len(self._mask_cache) >= 8:
                    self._mask_cache.clear()
                self._mask_cache[key] = mask
            return None if mask is None else mask.copy()

        return provider

    def _options(self, **kwargs):
        controls = resolve_style_controls(
            settings=self.settings,
            factuality=kwargs.get("factuality"),
            symbolic_looseness=kwargs.get("symbolic_looseness"),
            exaggeration=kwargs.get("exaggeration"),
        )
        synthetic = kwargs.get("synthetic_structure")
        if synthetic is None:
            synthetic = _settings_bool(self.settings, 'ArcheoGlyph/autotrace_synthetic_structure', False)
        return AutoTraceOptions(
            style=str(kwargs.get("style") or ""),
            color=kwargs.get("color"),
            symmetry=bool(kwargs.get("symmetry", False)),
            force_lowres_upscale=bool(kwargs.get("force_lowres_upscale", False)),
            detail_mode=kwargs.get("detail_mode") or self.settings.value("ArcheoGlyph/autotrace_detail_mode", "fast"),
            round_strategy=kwargs.get("round_strategy") or self.settings.value("ArcheoGlyph/round_strategy", "image_first"),
            factuality=controls[STYLE_CONTROL_FACTUALITY],
            symbolic_looseness=controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS],
            exaggeration=controls[STYLE_CONTROL_EXAGGERATION],
            synthetic_structure=bool(synthetic),
            input_kind=kwargs.get("input_kind") or self.settings.value("ArcheoGlyph/autotrace_input_kind", "auto"),
        ).normalized()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_path,
        style=None,
        color=None,
        symmetry=False,
        force_lowres_upscale=False,
        detail_mode=None,
        round_strategy=None,
        factuality=None,
        symbolic_looseness=None,
        exaggeration=None,
        synthetic_structure=None,
        input_kind=None,
    ):
        """
        Generate contour SVG from an image file.

        :return: SVG string (analysis-pixel coordinates; see generate_result
            for the cropped, parametrised version used by the UI)
        """
        self._require_cv()
        loaded = self._load(image_path)
        options = self._options(
            style=style, color=color, symmetry=symmetry, force_lowres_upscale=force_lowres_upscale,
            detail_mode=detail_mode, round_strategy=round_strategy, factuality=factuality,
            symbolic_looseness=symbolic_looseness, exaggeration=exaggeration,
            synthetic_structure=synthetic_structure, input_kind=input_kind,
        )
        return run_autotrace(loaded.bgr, options, self._mask_provider(loaded))

    def generate_result(self, image_path, **kwargs):
        """
        Run ``generate`` and return a SymbolResult whose SVG is cropped to the
        object, squared, and parametrised for QGIS (param(fill)/param(outline)).
        """
        from .symbol_result import SymbolResult
        from .autotrace.svg_builder import finalize_svg

        from .autotrace.svg_builder import add_provenance
        from ..defaults import PLUGIN_VERSION

        svg = self.generate(image_path, **kwargs)
        svg, info = finalize_svg(svg)
        result = SymbolResult(svg=svg, source="autotrace", style=str(kwargs.get("style") or ""), meta=info)
        result.record_provenance(
            image_path=image_path,
            input_kind=kwargs.get("input_kind"),
            plugin_version=PLUGIN_VERSION,
        )
        result.svg = add_provenance(result.svg, result.meta)
        if info.get("empty"):
            result.add_warning("No object silhouette was found in the image.")
        if info.get("parse_error"):
            result.add_warning(f"SVG could not be post-processed: {info['parse_error']}")
        return result

    def analyze(self, image_path):
        """
        Load, prescale and segment an image once (cached).
        :return: (processing_bgr, mask) or (None, None) when unusable
        """
        self._require_cv()
        loaded = self._load(image_path)
        processing_bgr, _scale = adaptive_prescale(loaded.bgr)
        mask = self._mask_provider(loaded)(processing_bgr)
        if mask is None or int(np.count_nonzero(mask)) < 40:
            return processing_bgr, None
        return processing_bgr, mask

    def get_silhouette_bytes(self, image_path):
        """Black-on-white silhouette PNG bytes for AI guidance, or None."""
        try:
            bgr, mask = self.analyze(image_path)
        except Exception as e:
            # Without this the AI backends simply get no guidance image and
            # produce a looser symbol, with nothing to say why.
            log_exception("Could not build the silhouette guide for AI generation", e)
            return None
        if bgr is None or mask is None:
            log("No silhouette guide: the image produced no usable mask.")
            return None
        out_img = np.full((bgr.shape[0], bgr.shape[1], 3), 255, dtype=np.uint8)
        out_img[mask == 255] = [0, 0, 0]
        success, encoded = cv2.imencode('.png', out_img)
        return encoded.tobytes() if success else None

    def get_ink_constraint_bytes(self, image_path, fmt="png"):
        """Ink centreline constraint image bytes for AI guidance, or None."""
        try:
            from .ink_centerline import render_ink_constraint_bytes
        except ImportError as e:
            log_exception("Ink centreline is unavailable, so AI gets no line guide", e)
            return None
        try:
            bgr, mask = self.analyze(image_path)
            if bgr is None or mask is None:
                log("No ink guide: the image produced no usable mask.")
                return None
            return render_ink_constraint_bytes(bgr, mask=mask, fmt=fmt)
        except Exception as e:
            log_exception("Could not build the ink guide for AI generation", e)
            return None
