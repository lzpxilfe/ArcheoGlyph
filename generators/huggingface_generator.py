# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Hugging Face Generator
Generates symbols using Hugging Face Inference API with evidence-first fallback.
"""

import base64
import importlib.util
import io
import os
import time
from urllib.parse import urlparse

from qgis.PyQt.QtCore import QByteArray, QBuffer, QIODevice, QSettings, Qt
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.PyQt.QtSvg import QSvgRenderer

from ..auth import get_api_key, set_api_key
from ..defaults import (
    HF_DEFAULT_MODEL_ID,
    HF_GUIDANCE_MODELS,
    HF_IMG2IMG_MODELS,
    HF_LEGACY_MODEL_ALIASES,
    HF_MAX_MODELS_PER_ROUTE,
    HF_TXT2IMG_MODELS,
)
from .contour_generator import ContourGenerator
from .style_control_utils import resolve_style_controls, style_controls_prompt_hint
import numpy as np

from ..log import log, log_exception
from . import image_ops
from .symbol_result import SymbolResult
from .subject_terms import find_subjects
from .style_utils import (
    STYLE_COLORED,
    STYLE_LINE,
    STYLE_MEASURED,
    STYLE_TYPOLOGY,
    normalize_style,
)


class HuggingFaceGenerator:
    """
    Generator using Hugging Face Inference API for symbol creation.
    """

    DEFAULT_MODEL_ID = HF_DEFAULT_MODEL_ID
    LOAD_RETRIES = 2

    def __init__(self):
        """Initialize the Hugging Face generator."""
        self.settings = QSettings()
        self.api_key = get_api_key("huggingface", self.settings)
        self.model_id = self.settings.value('ArcheoGlyph/hf_model_id', self.DEFAULT_MODEL_ID)
        self.contour_gen = ContourGenerator()

    def set_api_key(self, api_key):
        """Save the API key (QGIS authentication database when available)."""
        self.api_key = api_key
        set_api_key("huggingface", api_key, self.settings)

    def get_api_key(self):
        """Get API key from storage."""
        return self.api_key

    @staticmethod
    def hub_available():
        """True when huggingface_hub is importable."""
        return importlib.util.find_spec("huggingface_hub") is not None

    def _client(self):
        """
        Inference client with automatic provider routing.

        The old code called the ``hf-inference`` provider directly, but the
        default and fallback models are served by third-party providers
        through the router, which is why every request returned 404/403.
        """
        if not self.hub_available():
            raise ImportError(
                "The 'huggingface_hub' package is required for Hugging Face generation. "
                "Install it with: pip install huggingface_hub"
            )
        from huggingface_hub import InferenceClient

        return InferenceClient(api_key=(self.api_key or "").strip(), provider="auto", timeout=90)

    @staticmethod
    def _pil_to_qimage(image):
        """Convert a PIL image returned by the hub client into a QImage."""
        if image is None:
            return None
        if isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
        result = QImage()
        return result if result.loadFromData(payload) else None

    def _call_model(self, client, model, prompt, image_bytes=None, steps=None, guidance=None):
        """
        One inference call. Instruction-edit models get only the prompt and the
        image; guidance and negative prompts go to models that support them.
        """
        parameters = {}
        if steps:
            parameters["num_inference_steps"] = int(steps)
        if guidance and model in HF_GUIDANCE_MODELS:
            parameters["guidance_scale"] = float(guidance)
            parameters["negative_prompt"] = self._negative_prompt()

        if image_bytes is not None:
            from PIL import Image

            source = Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
            return client.image_to_image(source, prompt=prompt, model=model, **parameters)
        return client.text_to_image(prompt, model=model, **parameters)

    def _normalize_model_id(self, model_id):
        """
        Normalize user input into 'organization/model-name' format.
        Accepts raw model ids, 'models/...' prefixes, or full huggingface.co URLs.
        """
        value = (model_id or "").strip().replace("\\", "/")
        if not value:
            return self.DEFAULT_MODEL_ID

        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc and "huggingface.co" in parsed.netloc:
            value = parsed.path.strip("/")

        for prefix in ("hf-inference/models/", "models/"):
            if value.startswith(prefix):
                value = value[len(prefix):]

        value = "/".join([part.strip() for part in value.strip("/").split("/") if part.strip()])

        value = HF_LEGACY_MODEL_ALIASES.get(value, value)

        if "/" not in value:
            return self.DEFAULT_MODEL_ID
        return value

    def _get_error_detail(self, response):
        """Extract compact error detail from HF JSON/text responses."""
        try:
            data = response.json()
            if isinstance(data, dict):
                detail = str(data.get("error", data))
                if data.get("estimated_time") is not None:
                    detail += f" (estimated_time={data['estimated_time']}s)"
                return detail
            return str(data)
        except Exception:
            text = response.text.strip()
            return text if text else ""

    def _build_prompt(
        self,
        prompt,
        style=None,
        color=None,
        evidence_mode=False,
        factuality=None,
        symbolic_looseness=None,
        exaggeration=None,
    ):
        """Build an evidence-focused generation prompt."""
        style_key = self._normalize_style(style)
        parts = [
            "single isolated archaeological artifact",
            "documentary illustration",
            "preserve measured proportions",
            "preserve observed material characteristics",
            "subtle material shading only",
            "flat symbol-friendly rendering",
            "centered object",
            "plain neutral background",
            "no extra objects",
            "no decorative motif invention",
            "no engraved ornament invention",
            "no texture collage",
        ]
        if evidence_mode:
            parts.extend([
                "preserve silhouette and edge geometry from the reference image",
                "preserve observed chips wear cracks and asymmetry",
                "retain observed engraved motifs and relief lines from the reference image",
                "do not invent new internal patterns",
            ])
        if style_key == STYLE_LINE:
            parts.append("style hint: monochrome line drawing, clean contour and key internal lines")
        elif style_key == STYLE_MEASURED:
            parts.append(
                "style hint: black and white measured drawing, technical publication style, "
                "preserve observed internal motifs as simplified factual linework"
            )
        elif style_key == STYLE_TYPOLOGY:
            parts.append(
                "style hint: archaeological typology icon, standardized silhouette, "
                "bold outline, central axis cue, 1-3 structural bands, "
                "2-3 analogous muted tones from observed material palette"
            )
        else:
            parts.append(
                "style hint: archaeological catalog icon, bold contour, "
                "2-3 structural lines (rim/shoulder/base), flat 2-3 tone shading, no texture"
            )
        if color:
            parts.append(f"material color constrained to {color}")
        parts.append(
            self._style_control_hint(
                factuality=factuality,
                symbolic_looseness=symbolic_looseness,
                exaggeration=exaggeration,
            )
        )
        if prompt:
            parts.append(prompt)
            # The prompt is a phrase list here, so name the type rather than
            # appending the sentence Gemini gets.
            subjects = find_subjects(prompt)
            if subjects:
                named = " and ".join(english for _korean, english in subjects[:3])
                parts.append(f"the artifact is a {named}")
        return ", ".join(parts)

    def _negative_prompt(self):
        return (
            "landscape, scenery, architecture, village, people, animals, trees, sky, clouds, "
            "multiple objects, dramatic scene, fantasy scene, text, watermark, logo, map, diagram, "
            "invented decorative background pattern, mandala-style radial fantasy pattern, "
            "mosaic, tattoo pattern, collage texture, "
            "brush strokes, painterly texture, concept art, surreal art"
        )

    def _style_control_hint(self, factuality=None, symbolic_looseness=None, exaggeration=None):
        """Read style sliders and return compact prompt guidance."""
        controls = resolve_style_controls(
            settings=self.settings,
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        return style_controls_prompt_hint(controls, prefix="control hint")

    def _normalize_style(self, style):
        """Map style labels to canonical style keys."""
        return normalize_style(style)

    def _prompt_influence_score(self, prompt):
        """
        Estimate how strongly user text prompt should influence stylization.
        Returns a score in [0.0, 1.0].
        """
        text = str(prompt or "").strip().lower()
        if not text:
            return 0.0

        if text == "archaeological artifact from reference photo":
            return 0.0

        score = 0.35
        if len(text) >= 18:
            score += 0.15

        style_terms = (
            "game", "icon", "stylized", "symbol", "emblem", "catalog",
            "flat", "vector", "clean", "minimal", "badge", "ui",
            "typology", "typological", "dagger", "artifact class", "classification"
        )
        if any(term in text for term in style_terms):
            score += 0.30

        strong_terms = ("fantasy", "rpg", "cel", "toon", "illustration")
        if any(term in text for term in strong_terms):
            score += 0.20

        return max(0.0, min(1.0, score))







    # ------------------------------------------------------------------
    # Post-processing (numpy; see generators/image_ops.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_byte_count(image):
        """Buffer size across Qt versions (sizeInBytes is Qt 5.10+)."""
        for name in ("sizeInBytes", "byteCount"):
            getter = getattr(image, name, None)
            if getter is not None:
                try:
                    return int(getter())
                except Exception:
                    continue
        return int(image.bytesPerLine()) * int(image.height())

    @classmethod
    def _qimage_to_arrays(cls, image):
        """
        QImage -> (rgb uint8 (h, w, 3), alpha uint8 (h, w)).

        Uses the raw buffer when it is readable, and falls back to a PNG
        round-trip, which is still orders of magnitude faster than reading
        pixel by pixel.
        """
        converted = image.convertToFormat(QImage.Format_RGBA8888)
        width, height = int(converted.width()), int(converted.height())
        try:
            buffer = converted.constBits()
            setsize = getattr(buffer, "setsize", None)
            if setsize is not None:
                setsize(cls._image_byte_count(converted))
            raw = np.frombuffer(bytes(buffer), dtype=np.uint8)
            # Rows may be padded, so stride by bytesPerLine rather than width.
            stride = int(converted.bytesPerLine()) // 4
            raw = raw[: height * stride * 4].reshape(height, stride, 4)[:, :width, :]
            return np.ascontiguousarray(raw[:, :, :3]), np.ascontiguousarray(raw[:, :, 3])
        except Exception as e:
            log_exception("Falling back to PNG decoding for image conversion", e)

        payload = cls._image_to_png_bytes(converted)
        if payload is None:
            raise RuntimeError("Could not read the image buffer.")
        import cv2

        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError("Could not decode the image buffer.")
        if decoded.ndim == 2:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGRA)
        if decoded.shape[2] == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2BGRA)
        rgb = np.ascontiguousarray(decoded[:, :, :3][:, :, ::-1])
        return rgb, np.ascontiguousarray(decoded[:, :, 3])

    @staticmethod
    def _image_to_png_bytes(image):
        """Encode a QImage as PNG bytes, or None on failure."""
        data = QByteArray()
        buffer = QBuffer(data)
        if not buffer.open(QIODevice.WriteOnly):
            return None
        ok = image.save(buffer, "PNG")
        buffer.close()
        return bytes(data) if ok else None

    @staticmethod
    def _arrays_to_qimage(rgb, alpha):
        """(rgb, alpha) -> QImage. The buffer is copied, so it outlives the arrays."""
        height, width = alpha.shape[:2]
        rgba = np.ascontiguousarray(np.dstack([rgb, alpha]).astype(np.uint8, copy=False))
        image = QImage(rgba.data, width, height, 4 * width, QImage.Format_RGBA8888)
        # copy() detaches from the numpy buffer, which is about to go out of scope.
        return image.copy()

    def _mask_arrays(self, mask_img, image_path=None, size=None):
        """
        Silhouette mask as a boolean array, plus the reference photo resampled
        to the same size when ``image_path`` is given.
        """
        mask_rgb, _mask_alpha = self._qimage_to_arrays(mask_img)
        inside = image_ops.mask_inside(mask_rgb)
        if image_path is None:
            return inside, None, None
        reference = QImage(image_path)
        if reference.isNull():
            return inside, None, None
        target = size or (mask_img.width(), mask_img.height())
        reference = reference.scaled(target[0], target[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        ref_rgb, ref_alpha = self._qimage_to_arrays(reference)
        return inside, ref_rgb, ref_alpha

    def _parse_hex_rgb(self, hex_color):
        """Parse #RRGGBB to (r,g,b), return None if invalid."""
        return image_ops.parse_hex_rgb(hex_color)

    def _rgb_to_hex(self, rgb):
        """Convert an (r, g, b) tuple to #RRGGBB."""
        r, g, b = image_ops.clamp_rgb(rgb)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def _blend_rgb(self, base_rgb, mix_rgb, mix_ratio=0.35):
        """Blend two RGB tuples while preserving base tone identity."""
        return image_ops.blend_rgb(base_rgb, mix_rgb, mix_ratio)

    def _estimate_reference_rgb(self, image_path, mask_img, forced_hex=None):
        """Estimate artifact color from the reference photo inside the silhouette."""
        forced = image_ops.parse_hex_rgb(forced_hex)
        if forced:
            return forced
        inside, ref_rgb, ref_alpha = self._mask_arrays(mask_img, image_path)
        if ref_rgb is None:
            return image_ops.DEFAULT_MATERIAL_RGB
        return image_ops.estimate_reference_rgb(ref_rgb, ref_alpha, inside)

    def _extract_reference_palette(self, image_path, mask_img, forced_hex=None, max_colors=4):
        """Dominant material tones from the reference photo inside the silhouette."""
        forced = image_ops.parse_hex_rgb(forced_hex)
        if forced:
            return [forced]
        if mask_img is None or mask_img.isNull():
            return []
        inside, ref_rgb, ref_alpha = self._mask_arrays(mask_img, image_path)
        if ref_rgb is None:
            return []
        return image_ops.extract_reference_palette(ref_rgb, ref_alpha, inside, max_colors=max_colors)

    def _harmonize_typology_output(self, image, base_rgb, palette_rgb=None, preserve_ratio=0.34):
        """Map output onto analogous tone blocks instead of one flat colour."""
        rgb, alpha = self._qimage_to_arrays(image)
        out_rgb, out_alpha = image_ops.harmonize_typology(
            rgb, alpha, base_rgb, palette_rgb, preserve_ratio=preserve_ratio
        )
        return self._arrays_to_qimage(out_rgb, out_alpha)

    def _harmonize_colored_output(self, image, base_rgb, flatten=False, preserve_ratio=0.18):
        """Reduce painterly drift by harmonizing output to reference material color."""
        rgb, alpha = self._qimage_to_arrays(image)
        out_rgb, out_alpha = image_ops.harmonize_colored(
            rgb, alpha, base_rgb, flatten=flatten, preserve_ratio=preserve_ratio
        )
        return self._arrays_to_qimage(out_rgb, out_alpha)

    def _harmonize_mono_output(self, image, publication=False):
        """Convert output to stable monochrome for line/publication styles."""
        rgb, alpha = self._qimage_to_arrays(image)
        out_rgb, out_alpha = image_ops.harmonize_mono(rgb, alpha, publication=publication)
        return self._arrays_to_qimage(out_rgb, out_alpha)

    def _estimate_texture_noise(self, image, mask_img):
        """Estimate high-frequency texture noise inside the masked artifact area."""
        if image is None or mask_img is None:
            return 0.0
        rgb, _alpha = self._qimage_to_arrays(image)
        mask_rgb, _mask_alpha = self._qimage_to_arrays(mask_img)
        inside = image_ops.mask_inside(mask_rgb)
        height = min(rgb.shape[0], inside.shape[0])
        width = min(rgb.shape[1], inside.shape[1])
        return image_ops.estimate_texture_noise(rgb[:height, :width], inside[:height, :width])

    def _estimate_luma_variance(self, image, mask_img):
        """Estimate luminance variance inside the silhouette area."""
        if image is None or mask_img is None:
            return 0.0
        rgb, _alpha = self._qimage_to_arrays(image)
        mask_rgb, _mask_alpha = self._qimage_to_arrays(mask_img)
        inside = image_ops.mask_inside(mask_rgb)
        height = min(rgb.shape[0], inside.shape[0])
        width = min(rgb.shape[1], inside.shape[1])
        return image_ops.estimate_luma_variance(rgb[:height, :width], inside[:height, :width])

    def _apply_reference_tone_map(self, image, image_path, mask_img, strength=0.5):
        """Apply a coarse three-level tone map taken from the reference photo."""
        rgb, alpha = self._qimage_to_arrays(image)
        inside, ref_rgb, _ref_alpha = self._mask_arrays(
            mask_img, image_path, size=(image.width(), image.height())
        )
        if ref_rgb is None:
            return image
        height = min(rgb.shape[0], inside.shape[0], ref_rgb.shape[0])
        width = min(rgb.shape[1], inside.shape[1], ref_rgb.shape[1])
        if height < 2 or width < 2:
            return image
        out_rgb, out_alpha = image_ops.reference_tone_map(
            rgb[:height, :width], alpha[:height, :width],
            ref_rgb[:height, :width], inside[:height, :width],
            strength=strength,
        )
        return self._arrays_to_qimage(out_rgb, out_alpha)

    def _qimage_to_base64_png(self, image):
        """Encode QImage to base64 PNG string."""
        if image is None or image.isNull():
            return None

        ba = QByteArray()
        buffer = QBuffer(ba)
        if not buffer.open(QIODevice.WriteOnly):
            return None
        ok = image.save(buffer, "PNG")
        buffer.close()
        if not ok:
            return None
        return base64.b64encode(bytes(ba)).decode("utf-8")






    def _render_svg_to_image(self, svg_code):
        """Render SVG string to QImage."""
        if not svg_code:
            return None

        renderer = QSvgRenderer(QByteArray(svg_code.encode('utf-8')))
        if not renderer.isValid():
            return None

        view_box = renderer.viewBoxF()
        width = int(view_box.width()) if view_box.width() > 0 else 512
        height = int(view_box.height()) if view_box.height() > 0 else 512

        width = max(64, min(width, 1024))
        height = max(64, min(height, 1024))

        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)

        painter = QPainter(image)
        renderer.render(painter)
        painter.end()

        return image

    def _generate_evidence_fallback(self, image_path, style=None, color=None, symmetry=False):
        """
        Deterministic fallback based on extracted contour from input image.
        This avoids imaginative drift when remote model output is off-target.
        """
        try:
            svg_code = self.contour_gen.generate(
                image_path=image_path,
                style=style,
                color=color,
                symmetry=symmetry
            )
            image = self._render_svg_to_image(svg_code)
            if image is None:
                return None

            style_key = self._normalize_style(style)
            if style_key != STYLE_COLORED:
                return image

            silhouette_bytes = self.contour_gen.get_silhouette_bytes(image_path)
            if not silhouette_bytes:
                return image
            mask_img = QImage()
            if not mask_img.loadFromData(silhouette_bytes):
                return image

            image = image.scaled(mask_img.width(), mask_img.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            image = self._harmonize_colored_output(
                image,
                self._estimate_reference_rgb(image_path, mask_img, forced_hex=color),
                flatten=False,
                preserve_ratio=0.24,
            )
            image = self._apply_reference_tone_map(image, image_path, mask_img, strength=0.58)
            return image
        except Exception as e:
            log_exception("Rendering the contour SVG failed", e)
            return None

    def _apply_reference_mask(
        self,
        generated_image,
        image_path,
        symmetry=False,
        style=None,
        color=None,
        prompt_influence=0.0,
        used_contour_seed=False,
    ):
        """
        Force generated result to follow reference silhouette and linework.
        """
        try:
            style_key = self._normalize_style(style)
            silhouette_bytes = self.contour_gen.get_silhouette_bytes(image_path)
            if not silhouette_bytes:
                return generated_image

            mask_img = QImage()
            if not mask_img.loadFromData(silhouette_bytes):
                return generated_image

            target_w, target_h = mask_img.width(), mask_img.height()
            generated = generated_image.scaled(
                target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ).convertToFormat(QImage.Format_ARGB32)

            # Keep only what falls inside the measured silhouette.
            mask_rgb, _mask_alpha = self._qimage_to_arrays(mask_img)
            generated_rgb, _generated_alpha = self._qimage_to_arrays(generated)
            inside = image_ops.mask_inside(mask_rgb)
            out_rgb, out_alpha = image_ops.apply_silhouette(generated_rgb, inside)
            out = self._arrays_to_qimage(out_rgb, out_alpha)

            texture_noise = self._estimate_texture_noise(generated, mask_img)

            if style_key == STYLE_TYPOLOGY:
                typology_base = self._estimate_reference_rgb(image_path, mask_img, forced_hex=color)
                typology_palette = self._extract_reference_palette(
                    image_path,
                    mask_img,
                    forced_hex=color,
                    max_colors=4,
                )
                out = self._harmonize_typology_output(
                    out,
                    typology_base,
                    palette_rgb=typology_palette,
                    preserve_ratio=0.36,
                )
                out = self._apply_reference_tone_map(out, image_path, mask_img, strength=0.28)
            elif style_key == STYLE_COLORED:
                flatten_threshold = 24.0 + (8.0 * float(prompt_influence))
                flatten = texture_noise >= flatten_threshold
                base_ratio = 0.30 + (0.22 * float(prompt_influence))
                preserve_ratio = max(0.16, min(0.56, base_ratio - (0.08 if flatten else 0.0)))
                out = self._harmonize_colored_output(
                    out,
                    self._estimate_reference_rgb(image_path, mask_img, forced_hex=color),
                    flatten=flatten,
                    preserve_ratio=preserve_ratio,
                )
            else:
                out = self._harmonize_mono_output(out, publication=(style_key == STYLE_MEASURED))

            # If colored output is too flat, inject measured tone structure from reference image.
            if style_key == STYLE_COLORED:
                luma_var = self._estimate_luma_variance(out, mask_img)
                luma_threshold = 110.0 - (32.0 * float(prompt_influence))
                if luma_var < luma_threshold:
                    tone_strength = max(0.24, 0.52 - (0.24 * float(prompt_influence)))
                    out = self._apply_reference_tone_map(out, image_path, mask_img, strength=tone_strength)

            overlay_linework = str(
                self.settings.value('ArcheoGlyph/hf_overlay_linework', 'false')
            ).strip().lower() in ("1", "true", "yes", "on")
            overlay_opacity = 1.0
            if style_key == STYLE_LINE:
                overlay_linework = True
                overlay_opacity = 0.94
            elif style_key == STYLE_MEASURED:
                # Respect user setting for measured mode to avoid collapsing into Auto Trace look.
                if used_contour_seed:
                    overlay_linework = True
                    overlay_opacity = 0.44
                else:
                    overlay_opacity = 0.40 if overlay_linework else 0.0
            if style_key == STYLE_TYPOLOGY:
                overlay_linework = True
                overlay_opacity = max(0.55, 0.72 - (0.15 * float(prompt_influence)))
            if style_key == STYLE_COLORED:
                if overlay_linework:
                    overlay_opacity = max(0.18, 0.52 - (0.30 * float(prompt_influence)))
                elif used_contour_seed and float(prompt_influence) < 0.42:
                    overlay_linework = True
                    overlay_opacity = max(0.18, 0.34 - (0.16 * float(prompt_influence)))
                elif texture_noise >= (46.0 + (8.0 * float(prompt_influence))):
                    overlay_linework = True
                    overlay_opacity = 0.24
            if style_key == STYLE_COLORED and overlay_linework and texture_noise >= 28.0:
                overlay_opacity = min(0.50, overlay_opacity + 0.08)

            if overlay_linework:
                # Optional: overlay factual linework if user explicitly enables it.
                if style_key == STYLE_TYPOLOGY:
                    overlay_style = STYLE_LINE
                elif style_key == STYLE_MEASURED:
                    overlay_style = STYLE_MEASURED
                else:
                    overlay_style = STYLE_LINE
                line_svg = self.contour_gen.generate(
                    image_path=image_path,
                    style=overlay_style,
                    color=None,
                    symmetry=symmetry
                )
                line_img = self._render_svg_to_image(line_svg)
                if line_img:
                    line_img = line_img.scaled(
                        target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                    )
                    painter = QPainter(out)
                    painter.setRenderHint(QPainter.Antialiasing, True)
                    painter.setOpacity(overlay_opacity)
                    painter.drawImage(0, 0, line_img)
                    painter.end()

            # If output remains highly noisy, fall back to deterministic factual contour.
            if style_key in (STYLE_COLORED, STYLE_TYPOLOGY):
                final_noise = self._estimate_texture_noise(out, mask_img)
                base_noise_threshold = 48.0 if style_key == STYLE_COLORED else 44.0
                fallback_noise_threshold = base_noise_threshold + (14.0 * float(prompt_influence))
                # With strong prompt input, avoid collapsing back to contour too early.
                if final_noise >= fallback_noise_threshold and float(prompt_influence) < 0.78:
                    fallback = self._generate_evidence_fallback(
                        image_path=image_path,
                        style=style,
                        color=color,
                        symmetry=symmetry,
                    )
                    if fallback:
                        return fallback

            return out
        except Exception as e:
            log_exception("Applying the reference silhouette failed", e)
            return generated_image

    def _try_models(
        self,
        client,
        models_to_try,
        prompt,
        error_logs,
        image_bytes=None,
        steps=None,
        guidance=None,
        image_path=None,
        symmetry=False,
        style=None,
        color=None,
        prompt_influence=0.0,
        used_contour_seed=False,
        cancel_check=None,
    ):
        """Try each model in turn; return the first usable QImage."""
        for model in models_to_try[:HF_MAX_MODELS_PER_ROUTE]:
            if cancel_check and cancel_check():
                return None
            if not model or len(model) < 3:
                continue

            for attempt in range(self.LOAD_RETRIES + 1):
                try:
                    raw = self._call_model(
                        client, model, prompt, image_bytes=image_bytes, steps=steps, guidance=guidance
                    )
                except Exception as exc:
                    message = str(exc)
                    lowered = message.lower()
                    loading = "503" in message or "loading" in lowered or "currently loading" in lowered
                    if loading and attempt < self.LOAD_RETRIES:
                        delay = 5.0 * (attempt + 1)
                        log(f"Model {model} is loading; retrying in {delay:.0f}s")
                        time.sleep(delay)
                        continue
                    error_logs.append(f"Model {model}: {message[:220]}")
                    break

                image = self._pil_to_qimage(raw)
                if image is None or image.isNull():
                    error_logs.append(f"Model {model}: response was not a usable image")
                    break
                if image_path and os.path.exists(image_path):
                    return self._apply_reference_mask(
                        generated_image=image,
                        image_path=image_path,
                        symmetry=symmetry,
                        style=style,
                        color=color,
                        prompt_influence=prompt_influence,
                        used_contour_seed=used_contour_seed,
                    )
                return image

        return None

    def generate(
        self,
        prompt,
        style=None,
        color=None,
        image_path=None,
        symmetry=False,
        factuality=None,
        symbolic_looseness=None,
        exaggeration=None,
        cancel_check=None,
    ):
        """
        Generate a symbol with the Hugging Face inference providers.

        :param prompt: Text prompt
        :param style: style text
        :param color: optional override color
        :param image_path: optional reference image path
        :param symmetry: optional symmetry hint
        :param cancel_check: optional callable returning True to abort
        :return: SymbolResult
        """
        api_key = (self.api_key or "").strip()
        if not api_key:
            raise ValueError("Hugging Face API token is missing. Please set it in Settings.")

        self.model_id = self._normalize_model_id(self.model_id)
        # Settings are owned by the GUI thread; the worker only reads them.
        client = self._client()

        def _route(preferred, pool):
            models = [preferred] if preferred in pool else []
            for candidate in pool:
                normalized = self._normalize_model_id(candidate)
                if normalized not in models:
                    models.append(normalized)
            return models

        txt2img_models = _route(self.model_id, HF_TXT2IMG_MODELS)
        img2img_models = _route(self.model_id, HF_IMG2IMG_MODELS)

        error_logs = []
        prompt_influence = self._prompt_influence_score(prompt)
        try:
            p_text = str(prompt or "").strip()
            p_short = p_text if len(p_text) <= 120 else (p_text[:117] + "...")
            log(f"HF prompt influence={prompt_influence:.2f} prompt='{p_short}'")
        except Exception:
            pass
        base_prompt = self._build_prompt(
            prompt=prompt,
            style=style,
            color=color,
            evidence_mode=bool(image_path and os.path.exists(image_path)),
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        style_key = self._normalize_style(style)
        control_values = resolve_style_controls(
            settings=self.settings,
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        factuality_v = int(control_values.get("factuality", 0))
        symbolic_v = int(control_values.get("symbolic_looseness", 0))
        exaggeration_v = int(control_values.get("exaggeration", 0))

        # 1) If reference image exists, try img2img/edit path first.
        has_reference = bool(image_path and os.path.exists(image_path))
        use_contour_seed = False
        contour_seed = None
        contour_seed_b64 = None
        reference_hex = color
        if has_reference:
            if style_key == STYLE_LINE:
                use_contour_seed = True
            elif style_key == STYLE_MEASURED:
                # Measured HF should usually learn motifs from the real photo, not only contour seed.
                use_contour_seed = (
                    factuality_v >= 95 and
                    symbolic_v <= 8 and
                    exaggeration_v <= 8 and
                    float(prompt_influence) < 0.05
                )
            elif style_key == STYLE_TYPOLOGY:
                use_contour_seed = (
                    factuality_v >= 92 and
                    symbolic_v <= 18 and
                    exaggeration_v <= 14 and
                    float(prompt_influence) < 0.08
                )
            else:
                use_contour_seed = (
                    factuality_v >= 86 and
                    symbolic_v <= 24 and
                    exaggeration_v <= 20 and
                    float(prompt_influence) < 0.12
                )

            # Build deterministic Auto Trace seed only when requested by heuristic.
            if use_contour_seed:
                contour_seed = self._generate_evidence_fallback(
                    image_path=image_path,
                    style=style,
                    color=color,
                    symmetry=symmetry,
                )
                if contour_seed is not None:
                    contour_seed_b64 = self._qimage_to_base64_png(contour_seed)

            # Line guide: black strokes measured from the artifact on white.
            # For stroke styles this *is* the structure we want redrawn, so it
            # is sent as the input image rather than described in the prompt.
            guide_bytes = None
            try:
                guide_bytes = self.contour_gen.get_ink_constraint_bytes(image_path)
            except Exception as e:
                log_exception("Line guide extraction failed", e)

            if not reference_hex:
                silhouette_bytes = self.contour_gen.get_silhouette_bytes(image_path)
                if silhouette_bytes:
                    mask_img = QImage()
                    if mask_img.loadFromData(silhouette_bytes):
                        reference_hex = self._rgb_to_hex(self._estimate_reference_rgb(image_path, mask_img))

        if has_reference:
            try:
                if style_key in (STYLE_LINE, STYLE_MEASURED) and guide_bytes:
                    input_bytes = guide_bytes
                    img2img_prompt = (
                        f"{base_prompt}, the input image already shows the artifact's measured strokes "
                        "in black on white; keep every stroke in place and redraw them as a clean "
                        "archaeological line symbol, add nothing new"
                    )
                    img_strength = 0.2
                    source_tag = "line_guide"
                elif contour_seed_b64:
                    input_bytes = base64.b64decode(contour_seed_b64)
                    img2img_prompt = (
                        f"{base_prompt}, "
                        "input image is an archaeological contour seed, preserve its exact silhouette and proportions, "
                        "only refine internal tone transitions, keep flat vector-like shading, no new ornaments"
                    )
                    if reference_hex:
                        img2img_prompt += f", keep material hue near {reference_hex}"
                    img_strength = 0.18 + (0.14 * float(prompt_influence))
                    source_tag = "contour_seed"
                else:
                    with open(image_path, 'rb') as f:
                        input_bytes = f.read()
                    img2img_prompt = (
                        f"{base_prompt}, preserve measured silhouette and proportions from the reference photo, "
                        "retain observed engraved motifs and relief zones as simplified factual linework, "
                        "do not invent motifs, allow stylistic simplification into a readable archaeological symbol icon"
                    )
                    img_strength = 0.28 + (0.18 * float(prompt_influence))
                    source_tag = "reference_photo"

                img_steps = int(24 + (4 * float(prompt_influence)))
                img_guidance = 4.0 + (1.2 * float(prompt_influence))
                img_strength = max(0.12, min(0.62, float(img_strength)))
                try:
                    log(
                        f"HF image source={source_tag} "
                        f"strength={img_strength:.2f} guidance={img_guidance:.2f} steps={img_steps}"
                    )
                except Exception:
                    pass

                result = self._try_models(
                    client=client,
                    models_to_try=img2img_models,
                    prompt=img2img_prompt,
                    error_logs=error_logs,
                    image_bytes=input_bytes,
                    steps=img_steps,
                    guidance=img_guidance,
                    image_path=image_path,
                    symmetry=symmetry,
                    style=style,
                    color=color,
                    prompt_influence=prompt_influence,
                    used_contour_seed=(source_tag == "contour_seed"),
                    cancel_check=cancel_check,
                )
                if result:
                    return SymbolResult.coerce(result, source="huggingface", style=str(style or ""))
            except Exception as exc:
                error_logs.append(f"Reference img2img setup failed: {exc}")

        # 2) In reference mode we skip txt2img to avoid imaginative drift.
        # txt2img is used only when there is no photo reference.
        if not has_reference:
            result = self._try_models(
                client=client,
                models_to_try=txt2img_models,
                prompt=base_prompt,
                error_logs=error_logs,
                steps=30,
                guidance=5.0,
                image_path=image_path,
                symmetry=symmetry,
                style=style,
                color=color,
                prompt_influence=prompt_influence,
                cancel_check=cancel_check,
            )
            if result:
                return SymbolResult.coerce(result, source="huggingface", style=str(style or ""))

        # 3) Final deterministic evidence fallback if all remote calls fail.
        if has_reference:
            contour_result = contour_seed
            if contour_result is None:
                contour_result = self._generate_evidence_fallback(
                    image_path=image_path,
                    style=style,
                    color=color,
                    symmetry=symmetry
                )
            if contour_result:
                result = SymbolResult.coerce(
                    contour_result, source="autotrace-fallback", style=str(style or "")
                )
                report = "; ".join(error_logs[:2]) if error_logs else "no model returned an image"
                result.add_warning(f"Hugging Face output was not used ({report}); showing an Auto Trace symbol.")
                return result

        report = "\n".join(error_logs[:10]) if error_logs else "No response received from any model."
        hint_lines = []
        full_log = "\n".join(error_logs)

        if "401" in full_log or "unauthorized" in full_log.lower():
            hint_lines.append("Check token validity and ensure it has read/inference permissions.")
        if "403" in full_log or "forbidden" in full_log.lower():
            hint_lines.append("Model may be gated. Accept model terms on Hugging Face or choose a public model.")
        if "404" in full_log or "not found" in full_log.lower():
            hint_lines.append("Model id may be invalid or not deployed on this provider.")
            hint_lines.append(f"Try '{HF_DEFAULT_MODEL_ID}' or 'Qwen/Qwen-Image'.")
        if "gated" in full_log.lower():
            hint_lines.append("Accept the model's licence on huggingface.co, or pick an ungated model.")
        if image_path:
            hint_lines.append("Reference photo mode was enabled, but no valid remote image was returned.")

        hint = ""
        if hint_lines:
            hint = "\n\nTIP:\n- " + "\n- ".join(hint_lines)

        raise Exception(f"All models failed.\n\nERROR LOG:\n{report}{hint}")
