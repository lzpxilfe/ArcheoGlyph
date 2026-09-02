# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Google Gemini Generator
Generates stylized archaeological symbols using Google Gemini API.
"""

import base64
import os
import re
from qgis.PyQt.QtCore import QSettings, Qt, QByteArray, QRectF
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.PyQt.QtSvg import QSvgRenderer

from ..defaults import (
    PLUGIN_VERSION,
    GEMINI_EXCLUDED_KEYWORDS,
    GEMINI_IMAGE_MODEL_CANDIDATES,
    GEMINI_INSTALL_PACKAGE,
    GEMINI_TEXT_MODEL_CANDIDATES,
)



# Import ContourGenerator for hybrid workflow
from .contour_generator import ContourGenerator
from .style_control_utils import (
    STYLE_CONTROL_EXAGGERATION,
    STYLE_CONTROL_FACTUALITY,
    STYLE_CONTROL_SYMBOLIC_LOOSENESS,
    resolve_style_controls,
)
from ..auth import get_api_key, set_api_key
from ..log import log, log_exception
from .autotrace.svg_builder import add_provenance, finalize_svg
from .shape_match import mask_from_png, matches_reference, painted_mask_from_png
from .svg_sanitize import sanitize_svg
from .symbol_result import SymbolResult
from .style_utils import (
    STYLE_COLORED,
    STYLE_LINE,
    STYLE_MEASURED,
    STYLE_TYPOLOGY,
    normalize_style,
)

class GeminiGenerator:
    """Generator using Google Gemini API for symbol creation."""
    
    # Shape analysis preamble (prepended to all style prompts).
    # Forces the AI to carefully study the artifact's contour before drawing.
    _SHAPE_PREAMBLE = (
        "You are an expert archaeological illustrator. "
        "STEP 1 - SHAPE ANALYSIS: Analyze this artifact image carefully. "
        "Identify measured outline, asymmetry, and diagnostic form transitions. "
        "Keep silhouette factual, suppressing only tiny visual noise that hurts legibility. "
        "STEP 2 - SCALE ANALYSIS: Determine the exact aspect ratio of the object. "
        "STEP 3 - SVG GENERATION: Create a factual archaeological symbol SVG. "
        "\n\n"
        "ABSOLUTE RULES:\n"
        "- Preserve measured proportions and major shape cues from the reference.\n"
        "- Maintain the exact aspect ratio of the original image.\n"
        "- Use smooth vector geometry with sufficient control points for curved sections.\n"
        "- Output must read as an archaeological symbol icon, not a painting.\n\n"
    )

    _IMAGE_SHAPE_PREAMBLE = (
        "You are an expert archaeological illustrator. "
        "STEP 1 - SHAPE ANALYSIS: Analyze this artifact image carefully. "
        "Identify measured outline, asymmetry, and diagnostic form transitions. "
        "Keep silhouette factual, suppressing only tiny visual noise that hurts legibility. "
        "STEP 2 - SCALE ANALYSIS: Preserve the artifact's exact aspect ratio and major proportions. "
        "STEP 3 - IMAGE GENERATION: Create one factual archaeological symbol image. "
        "\n\n"
        "ABSOLUTE RULES:\n"
        "- Preserve measured proportions and major shape cues from the reference.\n"
        "- Maintain the exact aspect ratio of the original image.\n"
        "- Output exactly one isolated artifact symbol.\n"
        "- Render clean, legible, symbol-like edges rather than painterly texture.\n\n"
    )

    # Style prompts: only control rendering style, never the shape.
    STYLE_PROMPTS = {
        STYLE_COLORED: (
            "RENDERING STYLE: Archaeological catalog symbol icon. "
            "1. SHAPE RULES: Strictly trace provided silhouette constraints. "
            "2. OUTLINE: Clean black outline (about 1-2px equivalent). "
            "3. INTERNAL STRUCTURE: Add 1-3 factual feature lines only (rim/shoulder/base or blade midline). "
            "4. SHADING: Optional 2-3 flat tone regions only, no painterly texture. "
            "5. FORBIDDEN: no scenery, no architecture, no decorative motifs."
        ),
        STYLE_TYPOLOGY: (
            "RENDERING STYLE: Archaeological typology catalog icon. "
            "1. SHAPE RULES: Preserve measured proportions and diagnostic silhouette. "
            "2. OUTLINE: Bold and clean outer contour. "
            "3. INTERNAL STRUCTURE: Add 1-3 structural lines (e.g., rim/shoulder/base or blade midline). "
            "4. SHADING: Use flat muted tones only (2-3 analogous tone blocks from observed material), never painterly texture. "
            "5. FORBIDDEN COLOR: do not use a single flat fill across the whole object. "
            "5. FORBIDDEN: no scenery, no decorative motifs, no invented ornaments."
        ),
        STYLE_LINE: (
            "RENDERING STYLE: Archaeological Line Drawing. "
            "Draw ONLY the precise outline of the artifact and major internal lines. "
            "Use clean, consistent black strokes (1-2px). "
            "NO shading, NO stippling, NO hatching, NO fill. "
            "Pure abstraction of the form. Transparent background."
        ),
        STYLE_MEASURED: (
            "RENDERING STYLE: Traditional Archaeological Ink Illustration (Pen & Ink). "
            "Strictly MONOCHROME. "
            "1. OUTLINE: Precise fine line. "
            "2. SHADING: Use STIPPLING (dots) to show volume, curvature, and texture. "
            "3. TECHNIQUE: Traditional hand-drawn academic style (Pax Sapientica). "
            "NO solid color fills. NO greyscale gradients. Only Black Ink dots and lines."
        )
    }

    _SVG_FORMAT = (
        "\n\nOUTPUT: Provide ONLY valid SVG code. No markdown. No explanation. "
        "Start with <svg> and end with </svg>. "
        "Set viewBox to match the artifact's aspect ratio (e.g., '0 0 1000 1500'). "
        "Fit the artifact tightly within the viewBox. "
        "Use <path d='...'> with C (cubic bezier) commands. "
        "Use ABSOLUTE coordinates. "
        "Ensure the path is closed (ends with Z)."
    )

    _IMAGE_OUTPUT_RULES = (
        "\n\nOUTPUT: Return one generated image only. "
        "Prefer transparent background. Plain white background is acceptable if transparency is unavailable. "
        "No border, no frame, no caption, no watermark, no scene."
    )

    _NO_EXAGGERATION_RULES = (
        "\n\nREALISM RULES:\n"
        "- Do NOT exaggerate proportions, edges, thickness, or decorative details.\n"
        "- Do NOT cartoonize, beautify, or idealize the artifact.\n"
        "- Keep the rendering neutral and documentary.\n"
        "- Preserve observed damage, asymmetry, and surface wear from the reference image.\n"
        "- Output exactly one isolated artifact object.\n"
        "- Do not add any scene/background elements (ground, sky, plants, architecture, people).\n"
    )

    
    MAX_MODELS_PER_ROUTE = 3
    MAX_RETRIES_PER_MODEL = 2

    def __init__(self):
        """Initialize the Gemini generator."""
        self.settings = QSettings()
        self.api_key = get_api_key("gemini", self.settings)
        self.contour_gen = ContourGenerator()
        self._model_list_cache = None
        
    def set_api_key(self, api_key):
        """Save the API key (QGIS authentication database when available)."""
        self.api_key = api_key
        set_api_key("gemini", api_key, self.settings)

    def get_api_key(self):
        """Get API key from storage."""
        return self.api_key
        
    def _normalize_style(self, style):
        """Map various style labels to canonical styles."""
        return normalize_style(style)

    def _style_control_hint(self, factuality=None, symbolic_looseness=None, exaggeration=None):
        """Read style sliders and return prompt guidance text."""
        controls = resolve_style_controls(
            settings=self.settings,
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        return (
            f"\nSTYLE CONTROL: factuality={controls[STYLE_CONTROL_FACTUALITY]}/100, "
            f"symbol_looseness={controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS]}/100, "
            f"exaggeration={controls[STYLE_CONTROL_EXAGGERATION]}/100."
        )

    def _normalize_model_name(self, model_name):
        """Normalize model names by removing whitespace and optional SDK prefix."""
        normalized = str(model_name or "").strip()
        if normalized.startswith("models/"):
            normalized = normalized.replace("models/", "", 1)
        return normalized

    def _is_excluded_model(self, model_name):
        """Filter out unstable or unsupported Gemini utility models."""
        low = str(model_name or "").strip().lower()
        return any(keyword in low for keyword in GEMINI_EXCLUDED_KEYWORDS)

    def _is_image_model(self, model_name):
        """Detect Gemini image-generation/edit models."""
        return "image" in str(model_name or "").strip().lower()

    def _model_rank(self, model_name):
        """Rank Gemini models by family recency and practical utility."""
        low = str(model_name or "").strip().lower()
        major = 0
        minor = 0
        match = re.search(r"gemini-(\d+)(?:\.(\d+))?", low)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2) or 0)

        score = (major * 1000) + (minor * 100)
        if self._is_image_model(low):
            score += 160
        if "pro" in low:
            score += 60
        if "flash" in low:
            score += 45
        if "preview" in low:
            score += 8
        if "lite" in low:
            score -= 12
        if "exp" in low:
            score -= 20
        return score

    def _style_prompt_for_output(self, style_key, output_kind):
        """Return style prompt tuned for SVG or raster output."""
        if output_kind == "image":
            if style_key == STYLE_COLORED:
                return (
                    "RENDERING STYLE: Archaeological catalog symbol icon (NOT painting). "
                    "1. SHAPE RULES: Strictly trace the provided silhouette constraints. "
                    "2. OUTLINE: Use a clean black or very dark outline. "
                    "3. INTERNAL STRUCTURE: Add 1-3 structural feature lines that follow form. "
                    "4. SHADING: Optional 2-3 flat muted tone regions only. No painterly texture. "
                    "5. BACKGROUND: transparent or pure white only. "
                    "6. FORBIDDEN: No scenery, no landscape, no architecture, no decorative background."
                )
            if style_key == STYLE_TYPOLOGY:
                return (
                    "RENDERING STYLE: Typological archaeological symbol icon. "
                    "1. Preserve the measured silhouette and diagnostic form transitions. "
                    "2. Use a bold outer contour and clean axis-centered composition. "
                    "3. Add 1-3 structural lines only (e.g., shoulder/band/midline). "
                    "4. Use muted flat color blocks (2-3 analogous tones), no texture noise. "
                    "5. Keep visible tone separation; do not collapse to a single flat fill. "
                    "6. Avoid decorative elements and scenic context."
                )
            return self.STYLE_PROMPTS.get(style_key, self.STYLE_PROMPTS[STYLE_COLORED])

        if style_key == STYLE_COLORED:
            return (
                "RENDERING STYLE: Archaeological catalog symbol icon (NOT painting). "
                "1. SHAPE RULES: Strictly trace the provided silhouette constraints. "
                "2. OUTLINE: Use a clean black outline (about 1-2px equivalent). "
                "3. INTERNAL STRUCTURE: Add 1-3 structural feature lines that follow form "
                "(for example rim/shoulder/base or blade midline), and do not invent ornament. "
                "4. SHADING: Optional 2-3 flat tone regions only. No painterly texture. "
                "5. FORBIDDEN: No scenery, no landscape, no architecture, no decorative background. "
                "6. SVG PURITY: Use simple vector paths only; do not use gradients, filters, images, or masks."
            )
        if style_key == STYLE_TYPOLOGY:
            return (
                "RENDERING STYLE: Typological archaeological symbol icon. "
                "1. Preserve the measured silhouette and diagnostic form transitions. "
                "2. Use a bold outer contour and clean axis-centered composition. "
                "3. Add 1-3 structural lines only (e.g., shoulder/band/midline). "
                "4. Use muted flat color blocks (2-3 analogous tones), no texture noise. "
                "5. Do not render as one single flat fill color; keep visible tone separation. "
                "6. Avoid decorative elements and avoid scenic context."
            )
        return self.STYLE_PROMPTS.get(style_key, self.STYLE_PROMPTS[STYLE_COLORED])

    def _build_prompt(
        self,
        style_key,
        user_prompt_text,
        color,
        symmetry,
        silhouette_bytes,
        factuality,
        symbolic_looseness,
        exaggeration,
        output_kind="svg",
        guide_bytes=None,
        ink_polyline_text="",
    ):
        """Build prompt for SVG or raster Gemini generation."""
        full_prompt = (
            self._SHAPE_PREAMBLE if output_kind == "svg" else self._IMAGE_SHAPE_PREAMBLE
        )
        full_prompt += self._style_prompt_for_output(style_key, output_kind)
        full_prompt += self._NO_EXAGGERATION_RULES
        full_prompt += self._style_control_hint(
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        if user_prompt_text:
            full_prompt += (
                "\nUSER NOTE (respect if consistent with factual evidence): "
                + user_prompt_text
            )

        if symmetry:
            full_prompt += (
                "\n\nSYMMETRY RULE: Apply bilateral symmetry only when the artifact appears "
                "naturally symmetrical in the photo. Do not force perfect symmetry for damaged "
                "or asymmetrical objects."
            )

        if color and style_key in (STYLE_COLORED, STYLE_TYPOLOGY):
            full_prompt += (
                f"\n\nCOLOR INSTRUCTIONS:"
                f"\n1. Detect and use the artifact's observed material color from the photo."
                f"\n2. If user color {color} conflicts with the photo, prioritize the photo."
                f"\n3. Use 2-3 analogous muted tones from the observed material."
                f"\n4. Keep color variations subtle and realistic; avoid saturated fantasy tones."
                f"\n5. Keep the outline clean and documentary."
            )

        # Describe exactly the images that are attached, in the same order.
        image_notes = ["Image 1: the original photograph (material and colour reference)."]
        if silhouette_bytes:
            image_notes.append(
                f"Image {len(image_notes) + 1}: a black-and-white silhouette. "
                "Reproduce this outline; do not redraw the object's shape from imagination."
            )
        if guide_bytes:
            image_notes.append(
                f"Image {len(image_notes) + 1}: a line guide on a white ground - black strokes are the "
                "real incisions, ridges and painted lines measured from the artifact, the red contour is "
                "its outline. Draw only these strokes as internal detail; invent nothing else."
            )
        if len(image_notes) > 1:
            full_prompt += "\n\nREFERENCE IMAGES (in order):\n" + "\n".join(image_notes)

        if ink_polyline_text:
            full_prompt += (
                "\n\nMEASURED STROKE PATHS (image pixel coordinates, one polyline per line). "
                "Emit these as <path> elements verbatim, then add the outline and fills around them:\n"
                + ink_polyline_text
            )

        full_prompt += self._SVG_FORMAT if output_kind == "svg" else self._IMAGE_OUTPUT_RULES
        return full_prompt

    def _list_genai_models(self, client):
        """List available Gemini model names (cached for this generator)."""
        if self._model_list_cache is not None:
            return list(self._model_list_cache)
        names = []
        try:
            for model in client.models.list():
                name = self._normalize_model_name(getattr(model, "name", ""))
                if not name:
                    continue
                low = name.lower()
                if "gemini" not in low or self._is_excluded_model(name):
                    continue
                names.append(name)
        except Exception:
            self._model_list_cache = []
            return []

        deduped = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        self._model_list_cache = list(deduped)
        return deduped

    def _expand_model_candidates(self, available_names, aliases, image_mode=None):
        """
        Resolve exact or prefix matches against live model names.

        ``image_mode`` restricts prefix matches to the right route, so an alias
        like "gemini-2.5-flash" cannot pull in "gemini-2.5-flash-image".
        """
        normalized_available = {}
        for name in list(available_names or []):
            normalized = self._normalize_model_name(name)
            if normalized and normalized not in normalized_available:
                normalized_available[normalized] = normalized

        chosen = []
        for alias in list(aliases or []):
            normalized_alias = self._normalize_model_name(alias)
            if not normalized_alias:
                continue
            if normalized_alias in normalized_available:
                exact = normalized_available[normalized_alias]
                if exact not in chosen:
                    chosen.append(exact)
                continue

            matches = [
                name for name in normalized_available
                if name.startswith(normalized_alias) and not self._is_excluded_model(name)
            ]
            if image_mode is not None:
                matches = [name for name in matches if self._is_image_model(name) == bool(image_mode)]
            if matches:
                matches.sort(key=self._model_rank, reverse=True)
                best = matches[0]
                if best not in chosen:
                    chosen.append(best)

        return chosen

    def _resolve_models_to_try(self, available_names, preferred_model, image_mode=False):
        """Build text or image model fallback list."""
        route_candidates = (
            list(GEMINI_IMAGE_MODEL_CANDIDATES)
            if image_mode else
            list(GEMINI_TEXT_MODEL_CANDIDATES)
        )
        aliases = []
        preferred = self._normalize_model_name(preferred_model)
        if preferred and image_mode == self._is_image_model(preferred):
            aliases.append(preferred)
        aliases.extend(route_candidates)

        models_to_try = self._expand_model_candidates(available_names, aliases, image_mode=image_mode)

        route_pool = []
        for name in list(available_names or []):
            normalized = self._normalize_model_name(name)
            if not normalized or self._is_excluded_model(normalized):
                continue
            if image_mode != self._is_image_model(normalized):
                continue
            route_pool.append(normalized)
        route_pool.sort(key=self._model_rank, reverse=True)

        for name in route_pool:
            if name not in models_to_try:
                models_to_try.append(name)

        if not models_to_try:
            models_to_try = [self._normalize_model_name(name) for name in route_candidates if name]
        # A click must not turn into dozens of billed calls.
        return models_to_try[:self.MAX_MODELS_PER_ROUTE]

    def _build_genai_contents(self, sdk_types, full_prompt, image_path, image_data,
                              silhouette_bytes=None, guide_bytes=None):
        """
        Assemble the request parts. The order must match the image list
        described in the prompt: photo, silhouette, line guide.
        """
        contents = [full_prompt]
        contents.append(
            sdk_types.Part.from_bytes(
                data=image_data,
                mime_type=self._get_mime_type(image_path),
            )
        )
        for payload in (silhouette_bytes, guide_bytes):
            if payload:
                contents.append(sdk_types.Part.from_bytes(data=payload, mime_type="image/png"))
        return contents

    def _extract_response_parts(self, response):
        """Collect response parts from modern Google GenAI SDK responses."""
        parts = list(getattr(response, "parts", []) or [])
        if parts:
            return parts

        for candidate in list(getattr(response, "candidates", []) or []):
            content = getattr(candidate, "content", None)
            candidate_parts = list(getattr(content, "parts", []) or [])
            if candidate_parts:
                parts.extend(candidate_parts)
        return parts

    def _extract_image_from_response(self, response):
        """Extract the first image payload from a Gemini response."""
        for part in self._extract_response_parts(response):
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue

            mime_type = str(getattr(inline_data, "mime_type", "") or "").strip().lower()
            if not mime_type.startswith("image/"):
                continue

            payload = getattr(inline_data, "data", None)
            if payload is None:
                continue

            if isinstance(payload, str):
                try:
                    payload = base64.b64decode(payload)
                except Exception:
                    continue
            elif isinstance(payload, bytearray):
                payload = bytes(payload)

            if not isinstance(payload, (bytes, bytearray)):
                continue

            image = QImage()
            if image.loadFromData(bytes(payload)):
                return image

        return None

    def _postprocess_image_result(self, generated_image, image_path, style, color, symmetry, prompt):
        """Reuse factual masking logic for Gemini image outputs."""
        if generated_image is None or generated_image.isNull():
            return generated_image
        if not image_path or not os.path.exists(image_path):
            return generated_image

        try:
            from .huggingface_generator import HuggingFaceGenerator

            helper = HuggingFaceGenerator()
            prompt_influence = helper._prompt_influence_score(prompt)
            return helper._apply_reference_mask(
                generated_image=generated_image,
                image_path=image_path,
                symmetry=symmetry,
                style=style,
                color=color,
                prompt_influence=prompt_influence,
                used_contour_seed=False,
            )
        except Exception:
            return generated_image

    SYSTEM_INSTRUCTION = (
        "You convert archaeological artifact photographs into flat, factual point "
        "symbols for GIS maps. A symbol must stay legible at 5-10 mm on a printed map: "
        "one object, centred, tight bounds, no scene, no text, no shadow, no perspective. "
        "You never invent decoration that is not visible in the reference material."
    )

    def _guide_material(self, image_path, style_key):
        """
        Build the reference material sent alongside the photo.

        :return: (silhouette PNG or None, guide PNG or None, polyline text)
        """
        silhouette_bytes = None
        guide_bytes = None
        polyline_text = ""
        try:
            silhouette_bytes = self.contour_gen.get_silhouette_bytes(image_path)
        except Exception as e:
            log_exception("Silhouette extraction failed", e)

        try:
            from .ink_centerline import compose_guide_image, extract_ink_polylines, polylines_to_text
            import cv2

            bgr, mask = self.contour_gen.analyze(image_path)
            if bgr is not None and mask is not None:
                polylines = extract_ink_polylines(bgr, mask=mask)
                if polylines:
                    guide = compose_guide_image(bgr, mask, polylines)
                    ok, buf = cv2.imencode(".png", guide)
                    if ok:
                        guide_bytes = bytes(buf)
                    polyline_text = polylines_to_text(polylines, max_lines=30)
        except Exception as e:
            log_exception("Line guide extraction failed", e)

        return silhouette_bytes, guide_bytes, polyline_text

    def _validate_svg(self, svg_text, style_key, silhouette_bytes):
        """
        Sanitise and check a model-produced SVG.

        :return: (clean svg | None, reason). ``reason`` explains a rejection.
        """
        clean, problems, stats = sanitize_svg(svg_text)
        if not clean:
            return None, "; ".join(problems[:3]) or "unusable SVG"

        limit = 26 if style_key in (STYLE_COLORED, STYLE_TYPOLOGY) else 60
        if stats["geometry"] > limit:
            return None, f"too many shapes for a map symbol ({stats['geometry']})"
        max_colors = 6 if style_key in (STYLE_COLORED, STYLE_TYPOLOGY) else 3
        if stats["colors"] > max_colors:
            return None, f"too many distinct colours ({stats['colors']})"

        if silhouette_bytes:
            reference = mask_from_png(silhouette_bytes)
            if reference is not None:
                height, width = reference.shape[:2]
                rendered = self._render_svg_to_image(clean, width, height)
                if rendered is None:
                    return None, "the generated SVG could not be rendered"
                png = SymbolResult.coerce(rendered).raster_png
                painted = painted_mask_from_png(png)
                if painted is not None:
                    stroke_style = style_key in (STYLE_LINE, STYLE_MEASURED)
                    ok, reason = matches_reference(
                        reference, painted,
                        stroke_style=stroke_style,
                        strict=(style_key == STYLE_COLORED),
                    )
                    if not ok:
                        return None, reason
        return clean, ""

    def _autotrace_fallback(self, image_path, style, color, symmetry, reason):
        """Deterministic Auto Trace result, clearly labelled as a substitution."""
        result = self.contour_gen.generate_result(
            image_path=image_path, style=style, color=color, symmetry=symmetry
        )
        if result is None:
            return None
        result.source = "autotrace-fallback"
        result.add_warning(f"Gemini output was not used ({reason}); showing an Auto Trace symbol.")
        return result

    def generate(
        self,
        image_path,
        prompt="",
        style=STYLE_COLORED,
        color="#000000",
        symmetry=False,
        factuality=None,
        symbolic_looseness=None,
        exaggeration=None,
        cancel_check=None,
    ):
        """
        Generate a symbol from the input image using Gemini.

        :param cancel_check: optional callable returning True to abort
        :return: SymbolResult (vector SVG, or a raster image from an image model)
        """
        if not self.api_key:
            raise ValueError(
                "Gemini API key not configured. Please set your API key in the settings."
            )

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                f"{GEMINI_INSTALL_PACKAGE} package not installed. "
                f"Please run: pip install {GEMINI_INSTALL_PACKAGE}"
            )

        def cancelled():
            return bool(cancel_check and cancel_check())

        client = genai.Client(api_key=self.api_key)

        with open(image_path, 'rb') as f:
            image_data = f.read()

        user_prompt_text = str(prompt or "").strip()
        style_key = self._normalize_style(style)
        silhouette_bytes, guide_bytes, polyline_text = self._guide_material(image_path, style_key)

        preferred_model = self._normalize_model_name(
            self.settings.value('ArcheoGlyph/gemini_model_id', '')
        )
        available_models = self._list_genai_models(client)

        route_order = [("svg", False), ("image", True)]
        if preferred_model and self._is_image_model(preferred_model):
            route_order = [("image", True), ("svg", False)]

        last_error = None
        last_svg_issue = None
        quota_blocked = False
        for output_kind, image_mode in route_order:
            if cancelled():
                break
            models_to_try = self._resolve_models_to_try(
                available_names=available_models,
                preferred_model=preferred_model,
                image_mode=image_mode,
            )
            full_prompt = self._build_prompt(
                style_key=style_key,
                user_prompt_text=user_prompt_text,
                color=color,
                symmetry=symmetry,
                silhouette_bytes=silhouette_bytes,
                factuality=factuality,
                symbolic_looseness=symbolic_looseness,
                exaggeration=exaggeration,
                output_kind=output_kind,
                guide_bytes=guide_bytes,
                ink_polyline_text=(polyline_text if output_kind == "svg" else ""),
            )
            contents = self._build_genai_contents(
                sdk_types=genai_types,
                full_prompt=full_prompt,
                image_path=image_path,
                image_data=image_data,
                silhouette_bytes=silhouette_bytes,
                guide_bytes=guide_bytes,
            )
            config = self._generation_config(genai_types, image_mode)

            for model_name in models_to_try:
                if cancelled():
                    break
                for attempt in range(self.MAX_RETRIES_PER_MODEL + 1):
                    if cancelled():
                        break
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=config,
                        )

                        if image_mode:
                            image = self._extract_image_from_response(response)
                            if image is not None and not image.isNull():
                                image = self._postprocess_image_result(
                                    generated_image=image,
                                    image_path=image_path,
                                    style=style,
                                    color=color,
                                    symmetry=symmetry,
                                    prompt=user_prompt_text,
                                )
                                result = SymbolResult.coerce(image, source="gemini", style=str(style))
                                result.record_provenance(
                                    image_path=image_path,
                                    model=model_name,
                                    plugin_version=PLUGIN_VERSION,
                                )
                                return result
                            break

                        response_text = str(getattr(response, "text", "") or "")
                        if response_text:
                            svg_code, issue = self._validate_svg(
                                response_text, style_key, silhouette_bytes
                            )
                            if svg_code:
                                svg_code, info = finalize_svg(svg_code)
                                result = SymbolResult(
                                    svg=svg_code, source="gemini", style=str(style), meta=info
                                )
                                result.record_provenance(
                                    image_path=image_path,
                                    model=model_name,
                                    plugin_version=PLUGIN_VERSION,
                                )
                                result.svg = add_provenance(result.svg, result.meta)
                                return result
                            last_svg_issue = issue
                        break

                    except Exception as e:
                        error_str = str(e)
                        lowered = error_str.lower()
                        is_quota_exhausted = (
                            "quota exceeded" in lowered or
                            "limit: 0" in lowered or
                            "generate_content_free_tier" in lowered
                        )
                        status = getattr(e, "code", None) or getattr(e, "status_code", None)
                        is_rate_limit = (
                            status == 429 or
                            "resourceexhausted" in lowered or
                            "rate limit" in lowered or
                            "too many requests" in lowered
                        )

                        if is_quota_exhausted:
                            last_error = e
                            quota_blocked = True
                            break

                        if is_rate_limit and attempt < self.MAX_RETRIES_PER_MODEL:
                            import random
                            import time

                            delay = (2 * (2 ** attempt)) + random.uniform(0, 1)
                            log(f"Gemini rate limit ({model_name}); retrying in {delay:.1f}s")
                            time.sleep(delay)
                            continue

                        last_error = e
                        break

        if cancelled():
            raise RuntimeError("Generation cancelled.")

        if quota_blocked and last_error:
            raise Exception(
                "Gemini quota exhausted (HTTP 429). "
                "Wait for the quota to reset, or use Auto Trace in the meantime. "
                f"Original error: {last_error}"
            )

        reason = last_svg_issue or (str(last_error) if last_error else "no usable model response")
        fallback = self._autotrace_fallback(image_path, style, color, symmetry, reason)
        if fallback is not None:
            return fallback

        if last_error:
            raise last_error
        raise Exception(f"Failed to generate symbol: {reason}")

    def _generation_config(self, sdk_types, image_mode):
        """Build the generation config for a route, tolerating older SDKs."""
        if image_mode:
            return {"response_modalities": ["IMAGE", "TEXT"]}
        options = {
            "temperature": 0.2,
            "max_output_tokens": 8192,
            "system_instruction": self.SYSTEM_INSTRUCTION,
        }
        try:
            return sdk_types.GenerateContentConfig(**options)
        except Exception:
            return options

    def _get_mime_type(self, file_path):
        """Get MIME type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        return mime_types.get(ext, 'image/png')
        


    def _render_svg_to_image(self, svg_code, width, height):
        """Render SVG into a fixed-size transparent image."""
        if not svg_code or width < 2 or height < 2:
            return None

        renderer = QSvgRenderer(QByteArray(svg_code.encode('utf-8')))
        if not renderer.isValid():
            return None

        renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        image = QImage(int(width), int(height), QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)

        view_box = renderer.viewBoxF()
        if not view_box.isValid() or view_box.width() <= 0 or view_box.height() <= 0:
            default_size = renderer.defaultSize()
            if default_size.isValid() and default_size.width() > 0 and default_size.height() > 0:
                view_box = QRectF(0.0, 0.0, float(default_size.width()), float(default_size.height()))
            else:
                view_box = QRectF(0.0, 0.0, float(width), float(height))

        scale = min(float(width) / view_box.width(), float(height) / view_box.height())
        target_w = view_box.width() * scale
        target_h = view_box.height() * scale
        target_rect = QRectF((width - target_w) * 0.5, (height - target_h) * 0.5, target_w, target_h)

        painter = QPainter(image)
        renderer.render(painter, target_rect)
        painter.end()
        return image

