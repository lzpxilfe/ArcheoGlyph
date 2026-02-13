# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Hugging Face Generator
Generates symbols using Hugging Face Inference API with evidence-first fallback.
"""

import base64
import os
from urllib.parse import urlparse

import requests
from qgis.PyQt.QtCore import QByteArray, QSettings, Qt
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.PyQt.QtSvg import QSvgRenderer

from .contour_generator import ContourGenerator
from .style_utils import (
    STYLE_COLORED,
    STYLE_LINE,
    STYLE_MEASURED,
    normalize_style,
)


class HuggingFaceGenerator:
    """
    Generator using Hugging Face Inference API for symbol creation.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
    INFERENCE_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model_id}"

    def __init__(self):
        """Initialize the Hugging Face generator."""
        self.settings = QSettings()
        self.api_key = self.settings.value('ArcheoGlyph/huggingface_api_key', '')
        self.model_id = self.settings.value('ArcheoGlyph/hf_model_id', self.DEFAULT_MODEL_ID)
        self.contour_gen = ContourGenerator()

    def set_api_key(self, api_key):
        """Save API key to settings."""
        self.api_key = api_key
        self.settings.setValue('ArcheoGlyph/huggingface_api_key', api_key)

    def get_api_key(self):
        """Get API key from settings."""
        return self.api_key

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

        aliases = {
            "stabilityai/stable-diffusion-2-1": self.DEFAULT_MODEL_ID,
            "runwayml/stable-diffusion-v1-5": self.DEFAULT_MODEL_ID,
            "stable-diffusion-v1-5/stable-diffusion-v1-5": self.DEFAULT_MODEL_ID,
            "stabilityai/stable-diffusion-xl-base-1.0": self.DEFAULT_MODEL_ID,
        }
        value = aliases.get(value, value)

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

    def _build_prompt(self, prompt, style=None, color=None, evidence_mode=False):
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
            "no decorative motif",
            "no engraved ornament invention",
            "no texture collage",
        ]
        if evidence_mode:
            parts.extend([
                "preserve silhouette and edge geometry from the reference image",
                "preserve observed chips wear cracks and asymmetry",
                "do not invent new internal patterns",
            ])
        if style_key == STYLE_LINE:
            parts.append("style hint: monochrome line drawing, clean contour and key internal lines")
        elif style_key == STYLE_MEASURED:
            parts.append("style hint: black and white measured drawing, technical publication style")
        else:
            parts.append("style hint: restrained flat symbol color, non-painterly")
        if color:
            parts.append(f"material color constrained to {color}")
        if prompt:
            parts.append(prompt)
        return ", ".join(parts)

    def _negative_prompt(self):
        return (
            "landscape, scenery, architecture, village, people, animals, trees, sky, clouds, "
            "multiple objects, dramatic scene, fantasy scene, text, watermark, logo, map, diagram, "
            "ornament, decorative pattern, mosaic, tattoo pattern, mandala, collage texture, "
            "brush strokes, painterly texture, concept art, surreal art"
        )

    def _normalize_style(self, style):
        """Map style labels to canonical style keys."""
        return normalize_style(style)

    def _parse_hex_rgb(self, hex_color):
        """Parse #RRGGBB to (r,g,b), return None if invalid."""
        value = str(hex_color or "").strip().lstrip("#")
        if len(value) != 6:
            return None
        try:
            return (
                int(value[0:2], 16),
                int(value[2:4], 16),
                int(value[4:6], 16),
            )
        except Exception:
            return None

    def _estimate_reference_rgb(self, image_path, mask_img, forced_hex=None):
        """Estimate artifact color from reference image constrained by silhouette mask."""
        forced = self._parse_hex_rgb(forced_hex)
        if forced:
            return forced

        ref = QImage(image_path)
        if ref.isNull():
            return (88, 112, 92)

        ref = ref.scaled(mask_img.width(), mask_img.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        ref = ref.convertToFormat(QImage.Format_ARGB32)

        sum_r = 0
        sum_g = 0
        sum_b = 0
        count = 0

        for y in range(mask_img.height()):
            for x in range(mask_img.width()):
                mp = mask_img.pixelColor(x, y)
                inside = (mp.red() < 90 and mp.green() < 90 and mp.blue() < 90)
                if not inside:
                    continue

                px = ref.pixelColor(x, y)
                if px.alpha() < 8:
                    continue
                mx = max(px.red(), px.green(), px.blue())
                mn = min(px.red(), px.green(), px.blue())
                sat = mx - mn
                if sat < 12 or mx < 28 or mx > 245:
                    continue

                sum_r += px.red()
                sum_g += px.green()
                sum_b += px.blue()
                count += 1

        if count < 25:
            for y in range(mask_img.height()):
                for x in range(mask_img.width()):
                    mp = mask_img.pixelColor(x, y)
                    inside = (mp.red() < 90 and mp.green() < 90 and mp.blue() < 90)
                    if not inside:
                        continue
                    px = ref.pixelColor(x, y)
                    if px.alpha() < 8:
                        continue
                    sum_r += px.red()
                    sum_g += px.green()
                    sum_b += px.blue()
                    count += 1

        if count < 5:
            return (88, 112, 92)
        return (int(sum_r / count), int(sum_g / count), int(sum_b / count))

    def _estimate_texture_noise(self, image, mask_img):
        """Estimate high-frequency texture noise inside masked artifact area."""
        if image is None or mask_img is None:
            return 0.0

        img = image.convertToFormat(QImage.Format_ARGB32)
        w = min(img.width(), mask_img.width())
        h = min(img.height(), mask_img.height())
        if w < 3 or h < 3:
            return 0.0

        diff_sum = 0.0
        samples = 0
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                mp = mask_img.pixelColor(x, y)
                inside = (mp.red() < 90 and mp.green() < 90 and mp.blue() < 90)
                if not inside:
                    continue

                p = img.pixelColor(x, y)
                l = img.pixelColor(x - 1, y)
                r = img.pixelColor(x + 1, y)
                u = img.pixelColor(x, y - 1)
                d = img.pixelColor(x, y + 1)

                dl = (
                    abs(p.red() - l.red()) + abs(p.green() - l.green()) + abs(p.blue() - l.blue()) +
                    abs(p.red() - r.red()) + abs(p.green() - r.green()) + abs(p.blue() - r.blue()) +
                    abs(p.red() - u.red()) + abs(p.green() - u.green()) + abs(p.blue() - u.blue()) +
                    abs(p.red() - d.red()) + abs(p.green() - d.green()) + abs(p.blue() - d.blue())
                ) / 12.0
                diff_sum += dl
                samples += 1

        if samples < 20:
            return 0.0
        return diff_sum / samples

    def _estimate_luma_variance(self, image, mask_img):
        """Estimate luminance variance inside silhouette area."""
        if image is None or mask_img is None:
            return 0.0

        img = image.convertToFormat(QImage.Format_ARGB32)
        w = min(img.width(), mask_img.width())
        h = min(img.height(), mask_img.height())
        if w < 2 or h < 2:
            return 0.0

        total = 0.0
        total_sq = 0.0
        count = 0
        for y in range(h):
            for x in range(w):
                mp = mask_img.pixelColor(x, y)
                inside = (mp.red() < 90 and mp.green() < 90 and mp.blue() < 90)
                if not inside:
                    continue

                px = img.pixelColor(x, y)
                lum = (0.299 * px.red()) + (0.587 * px.green()) + (0.114 * px.blue())
                total += lum
                total_sq += (lum * lum)
                count += 1

        if count < 20:
            return 0.0
        mean = total / count
        var = (total_sq / count) - (mean * mean)
        return max(0.0, var)

    def _apply_reference_tone_map(self, image, image_path, mask_img, strength=0.5):
        """
        Apply a coarse (3-level) tone map from the reference photo.
        Keeps factual highlights/shadows without introducing painterly noise.
        """
        ref = QImage(image_path)
        if ref.isNull():
            return image

        out = image.convertToFormat(QImage.Format_ARGB32)
        ref = ref.scaled(out.width(), out.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        ref = ref.convertToFormat(QImage.Format_ARGB32)

        w = min(out.width(), mask_img.width())
        h = min(out.height(), mask_img.height())
        if w < 2 or h < 2:
            return out

        min_l = 255.0
        max_l = 0.0
        for y in range(h):
            for x in range(w):
                mp = mask_img.pixelColor(x, y)
                inside = (mp.red() < 90 and mp.green() < 90 and mp.blue() < 90)
                if not inside:
                    continue
                rp = ref.pixelColor(x, y)
                lum = (0.299 * rp.red()) + (0.587 * rp.green()) + (0.114 * rp.blue())
                if lum < min_l:
                    min_l = lum
                if lum > max_l:
                    max_l = lum

        span = max_l - min_l
        if span < 6.0:
            return out

        s = max(0.0, min(1.0, float(strength)))
        for y in range(h):
            for x in range(w):
                mp = mask_img.pixelColor(x, y)
                inside = (mp.red() < 90 and mp.green() < 90 and mp.blue() < 90)
                if not inside:
                    continue

                rp = ref.pixelColor(x, y)
                lum = (0.299 * rp.red()) + (0.587 * rp.green()) + (0.114 * rp.blue())
                norm = (lum - min_l) / span
                if norm < 0.34:
                    tone = 0.90
                elif norm < 0.68:
                    tone = 1.00
                else:
                    tone = 1.10

                px = out.pixelColor(x, y)
                tr = max(0, min(255, int(px.red() * tone)))
                tg = max(0, min(255, int(px.green() * tone)))
                tb = max(0, min(255, int(px.blue() * tone)))
                nr = int((px.red() * (1.0 - s)) + (tr * s))
                ng = int((px.green() * (1.0 - s)) + (tg * s))
                nb = int((px.blue() * (1.0 - s)) + (tb * s))
                px.setRed(max(0, min(255, nr)))
                px.setGreen(max(0, min(255, ng)))
                px.setBlue(max(0, min(255, nb)))
                px.setAlpha(255)
                out.setPixelColor(x, y, px)
        return out

    def _harmonize_colored_output(self, image, base_rgb, flatten=False, preserve_ratio=0.18):
        """Reduce painterly drift by harmonizing output to reference material color."""
        out = image.convertToFormat(QImage.Format_ARGB32)
        br, bg, bb = base_rgb
        for y in range(out.height()):
            for x in range(out.width()):
                px = out.pixelColor(x, y)
                if px.alpha() < 8:
                    continue
                lum = (0.299 * px.red() + 0.587 * px.green() + 0.114 * px.blue()) / 255.0
                if flatten:
                    lum = round(lum * 3.0) / 3.0
                shade = 0.58 + (0.64 * lum)
                tr = max(0, min(255, int(br * shade)))
                tg = max(0, min(255, int(bg * shade)))
                tb = max(0, min(255, int(bb * shade)))
                nr = int((tr * (1.0 - preserve_ratio)) + (px.red() * preserve_ratio))
                ng = int((tg * (1.0 - preserve_ratio)) + (px.green() * preserve_ratio))
                nb = int((tb * (1.0 - preserve_ratio)) + (px.blue() * preserve_ratio))
                px.setRed(max(0, min(255, nr)))
                px.setGreen(max(0, min(255, ng)))
                px.setBlue(max(0, min(255, nb)))
                px.setAlpha(255)
                out.setPixelColor(x, y, px)
        return out

    def _harmonize_mono_output(self, image, publication=False):
        """Convert output to stable monochrome for line/publication styles."""
        out = image.convertToFormat(QImage.Format_ARGB32)
        for y in range(out.height()):
            for x in range(out.width()):
                px = out.pixelColor(x, y)
                if px.alpha() < 8:
                    continue
                lum = int(0.299 * px.red() + 0.587 * px.green() + 0.114 * px.blue())
                if publication:
                    v = 25 if lum < 135 else 70
                else:
                    v = int(20 + (lum * 0.35))
                v = max(0, min(255, v))
                px.setRed(v)
                px.setGreen(v)
                px.setBlue(v)
                px.setAlpha(255)
                out.setPixelColor(x, y, px)
        return out

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
        except Exception:
            return None

    def _apply_reference_mask(self, generated_image, image_path, symmetry=False, style=None, color=None):
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

            out = QImage(target_w, target_h, QImage.Format_ARGB32_Premultiplied)
            out.fill(Qt.transparent)

            for y in range(target_h):
                for x in range(target_w):
                    mask_px = mask_img.pixelColor(x, y)
                    # In silhouette mask: object is black, background is white.
                    inside = (
                        mask_px.red() < 90 and
                        mask_px.green() < 90 and
                        mask_px.blue() < 90
                    )
                    if inside:
                        px = generated.pixelColor(x, y)
                        px.setAlpha(255)
                        out.setPixelColor(x, y, px)

            texture_noise = self._estimate_texture_noise(generated, mask_img)

            if style_key == STYLE_COLORED:
                flatten = texture_noise >= 24.0
                preserve_ratio = 0.16 if flatten else 0.30
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
                if luma_var < 110.0:
                    out = self._apply_reference_tone_map(out, image_path, mask_img, strength=0.52)

            overlay_linework = str(
                self.settings.value('ArcheoGlyph/hf_overlay_linework', 'false')
            ).strip().lower() in ("1", "true", "yes", "on")
            if style_key in (STYLE_LINE, STYLE_MEASURED):
                overlay_linework = True
            if style_key == STYLE_COLORED and texture_noise >= 28.0:
                overlay_linework = True

            if overlay_linework:
                # Optional: overlay factual linework if user explicitly enables it.
                line_svg = self.contour_gen.generate(
                    image_path=image_path,
                    style=STYLE_LINE,
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
                    painter.drawImage(0, 0, line_img)
                    painter.end()

            # If output remains highly noisy, fall back to deterministic factual contour.
            if style_key == STYLE_COLORED:
                final_noise = self._estimate_texture_noise(out, mask_img)
                if final_noise >= 34.0:
                    fallback = self._generate_evidence_fallback(
                        image_path=image_path,
                        style=style,
                        color=color,
                        symmetry=symmetry,
                    )
                    if fallback:
                        return fallback

            return out
        except Exception:
            return generated_image

    def _try_models(
        self,
        models_to_try,
        headers,
        payload,
        error_logs,
        timeout=60,
        image_path=None,
        symmetry=False,
        style=None,
        color=None,
    ):
        """Try a payload across model list and return first valid QImage."""
        for model in models_to_try:
            if not model or len(model) < 3:
                continue

            api_url = self.INFERENCE_URL_TEMPLATE.format(model_id=model)

            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as exc:
                error_logs.append(f"Model {model}: request failed - {exc}")
                continue

            if response.status_code == 200:
                image = QImage()
                if image.loadFromData(response.content):
                    if image_path and os.path.exists(image_path):
                        return self._apply_reference_mask(
                            generated_image=image,
                            image_path=image_path,
                            symmetry=symmetry,
                            style=style,
                            color=color,
                        )
                    return image

                detail = self._get_error_detail(response)
                if detail:
                    error_logs.append(f"Model {model}: invalid image response - {detail[:220]}")
                else:
                    error_logs.append(f"Model {model}: invalid image response")
                continue

            detail = self._get_error_detail(response)
            suffix = f" - {detail[:220]}" if detail else ""

            if response.status_code == 401:
                error_logs.append(f"Model {model}: AUTH ERROR (401){suffix}")
            elif response.status_code == 403:
                error_logs.append(f"Model {model}: FORBIDDEN (403){suffix}")
            elif response.status_code == 404:
                error_logs.append(f"Model {model}: NOT FOUND (404){suffix}")
            elif response.status_code == 410:
                error_logs.append(f"Model {model}: GONE (410){suffix}")
            elif response.status_code == 503:
                error_logs.append(f"Model {model}: LOADING (503){suffix}")
            else:
                error_logs.append(f"Model {model}: HTTP {response.status_code}{suffix}")

        return None

    def generate(self, prompt, style=None, color=None, image_path=None, symmetry=False):
        """
        Generate symbol image using HF API.

        :param prompt: Text prompt
        :param style: style text
        :param color: optional override color
        :param image_path: optional reference image path
        :param symmetry: optional symmetry hint
        :return: QImage
        """
        api_key = (self.api_key or "").strip()
        if not api_key:
            raise ValueError("Hugging Face API token is missing. Please set it in Settings.")

        self.model_id = self._normalize_model_id(self.model_id)
        if "flat-design-icons" in self.model_id:
            self.model_id = self.DEFAULT_MODEL_ID
        self.settings.setValue('ArcheoGlyph/hf_model_id', self.model_id)

        models_to_try = [self.model_id]
        for fallback in [
            self.DEFAULT_MODEL_ID,
            "Qwen/Qwen-Image-Edit-2509",
            "Qwen/Qwen-Image-Edit",
            "black-forest-labs/FLUX.1-Kontext-dev",
            "black-forest-labs/FLUX.2-dev",
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-3.5-large",
        ]:
            normalized = self._normalize_model_id(fallback)
            if normalized not in models_to_try:
                models_to_try.append(normalized)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }

        error_logs = []
        base_prompt = self._build_prompt(
            prompt=prompt,
            style=style,
            color=color,
            evidence_mode=bool(image_path and os.path.exists(image_path))
        )

        # 1) If reference image exists, try img2img/edit path first.
        has_reference = bool(image_path and os.path.exists(image_path))
        if has_reference:
            try:
                with open(image_path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode('utf-8')

                img2img_models = []
                for mid in [
                    "Qwen/Qwen-Image-Edit-2509",
                    "black-forest-labs/FLUX.2-dev",
                    "black-forest-labs/FLUX.1-Kontext-dev",
                    "Qwen/Qwen-Image-Edit",
                ] + models_to_try:
                    normalized = self._normalize_model_id(mid)
                    if normalized not in img2img_models:
                        img2img_models.append(normalized)

                img2img_payload = {
                    "inputs": image_b64,
                    "parameters": {
                        "prompt": base_prompt,
                        "negative_prompt": self._negative_prompt(),
                        "num_inference_steps": 24,
                        "guidance_scale": 4.0,
                        "strength": 0.22,
                    }
                }

                result = self._try_models(
                    models_to_try=img2img_models,
                    headers=headers,
                    payload=img2img_payload,
                    error_logs=error_logs,
                    timeout=75,
                    image_path=image_path,
                    symmetry=symmetry,
                    style=style,
                    color=color,
                )
                if result:
                    return result
            except Exception as exc:
                error_logs.append(f"Reference img2img setup failed: {exc}")

        # 2) In reference mode we skip txt2img to avoid imaginative drift.
        # txt2img is used only when there is no photo reference.
        if not has_reference:
            txt2img_payload = {
                "inputs": base_prompt,
                "parameters": {
                    "negative_prompt": self._negative_prompt(),
                    "num_inference_steps": 30,
                    "guidance_scale": 5.0,
                }
            }

            result = self._try_models(
                models_to_try=models_to_try,
                headers=headers,
                payload=txt2img_payload,
                error_logs=error_logs,
                timeout=60,
                image_path=image_path,
                symmetry=symmetry,
                style=style,
                color=color,
            )
            if result:
                return result

        # 3) Final deterministic evidence fallback if all remote calls fail.
        if has_reference:
            contour_result = self._generate_evidence_fallback(
                image_path=image_path,
                style=style,
                color=color,
                symmetry=symmetry
            )
            if contour_result:
                return contour_result

        report = "\n".join(error_logs[:10]) if error_logs else "No response received from any model."
        hint_lines = []
        full_log = "\n".join(error_logs)

        if "AUTH ERROR (401)" in full_log:
            hint_lines.append("Check token validity and ensure it has read/inference permissions.")
        if "FORBIDDEN (403)" in full_log:
            hint_lines.append("Model may be gated. Accept model terms on Hugging Face or choose a public model.")
        if "NOT FOUND (404)" in full_log:
            hint_lines.append("Model id may be invalid or not deployed on this provider.")
            hint_lines.append("Try 'Qwen/Qwen-Image-Edit-2509' or 'Qwen/Qwen-Image'.")
        if "GONE (410)" in full_log:
            hint_lines.append("Legacy endpoint is retired. Use router.huggingface.co/hf-inference/models/... endpoint.")
        if image_path:
            hint_lines.append("Reference photo mode was enabled, but no valid remote image was returned.")

        hint = ""
        if hint_lines:
            hint = "\n\nTIP:\n- " + "\n- ".join(hint_lines)

        raise Exception(f"All models failed.\n\nERROR LOG:\n{report}{hint}")
