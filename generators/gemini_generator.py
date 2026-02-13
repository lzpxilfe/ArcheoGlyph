# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Google Gemini Generator
Generates stylized archaeological symbols using Google Gemini API.
"""

import os
import re
from qgis.PyQt.QtCore import QSettings, Qt, QByteArray, QRectF
from qgis.PyQt.QtGui import QImage, QPainter
from qgis.PyQt.QtSvg import QSvgRenderer




# Import ContourGenerator for hybrid workflow
from .contour_generator import ContourGenerator
from .style_utils import (
    STYLE_COLORED,
    STYLE_LINE,
    STYLE_MEASURED,
    normalize_style,
)

class GeminiGenerator:
    """Generator using Google Gemini API for symbol creation."""
    
    # ── Shape analysis preamble (prepended to ALL style prompts) ──
    # Forces the AI to carefully study the artifact's contour before drawing.
    _SHAPE_PREAMBLE = (
        "You are an expert archaeological illustrator. "
        "STEP 1 — SHAPE ANALYSIS: Analyze this artifact image pixel-by-pixel. "
        "Identify the EXACT outline including all asymmetric curves, damage, notches, "
        "and subtle irregularities. Do not idealize the shape. "
        "STEP 2 — SCALE ANALISYS: Determine the aspect ratio of the object. "
        "STEP 3 — SVG GENERATION: Create a high-fidelity SVG tracing. "
        "\n\n"
        "ABSOLUTE RULES:\n"
        "- The SVG outline MUST match the artifact silhouette EXACTLY.\n"
        "- Do NOT simplify. Capture every small bump and curve.\n"
        "- Maintain the exact aspect ratio of the original image.\n"
        "- Use at minimum 100+ control points with cubic bezier curves (C command) "
        "to ensure high precision. Do NOT use straight lines for curved sections.\n"
        "- The viewer must be able to identify the specific individual artifact "
        "from your outline, not just the type.\n\n"
    )

    # Style prompts — only control RENDERING style, never the shape.
    STYLE_PROMPTS = {
        STYLE_COLORED: (
            "RENDERING STYLE: Premium Vector Game Asset / RPG Item Icon. "
            "1. SHAPE RULES: STICTLY TRACE the provided SILHOUETTE MASK (Image 2). The mask defines the EXACT geometry. "
            "   Do not deviate from the mask's outline. "
            "2. OUTLINE: Draw a bold, clean, consistent BLACK outline (2-3px) around the entire object defined by the mask. "
            "3. SHADING: Use 'Cel Shading' or '2-Tone Shading' (Base Color + Shadow Color) based on the REFERENCE PHOTO (Image 1). "
            "4. AESTHETIC: Flat Design but with depth. Like a high-quality strategy game unit or resource icon. "
            "NO gradient meshes. NO realistic texture noise. Clean vector shapes."
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

    _NO_EXAGGERATION_RULES = (
        "\n\nREALISM RULES:\n"
        "- Do NOT exaggerate proportions, edges, thickness, or decorative details.\n"
        "- Do NOT cartoonize, beautify, or idealize the artifact.\n"
        "- Keep the rendering neutral and documentary.\n"
        "- Preserve observed damage, asymmetry, and surface wear from the reference image.\n"
        "- Output exactly one isolated artifact object.\n"
        "- Do not add any scene/background elements (ground, sky, plants, architecture, people).\n"
    )

    _DISALLOWED_SVG_TOKENS = (
        "<image",
        "<foreignobject",
        "<filter",
        "<lineargradient",
        "<radialgradient",
        "<pattern",
        "<mask",
        "<text",
        "<clippath",
    )
    
    def __init__(self):
        """Initialize the Gemini generator."""
        self.settings = QSettings()
        self.api_key = self.settings.value('ArcheoGlyph/gemini_api_key', '')
        self.contour_gen = ContourGenerator()
        
    def set_api_key(self, api_key):
        """Save API key to settings."""
        self.api_key = api_key
        self.settings.setValue('ArcheoGlyph/gemini_api_key', api_key)
        
    def get_api_key(self):
        """Get API key from settings."""
        return self.api_key
        
    def _normalize_style(self, style):
        """Map various style labels to canonical styles."""
        return normalize_style(style)

    def generate(self, image_path, style=STYLE_COLORED, color="#000000", symmetry=False):
        """
        Generate a symbol from the input image using Gemini.
        Returns the SVG code as a string.
        """
        if not self.api_key:
            raise ValueError(
                "Gemini API key not configured. Please set your API key in the settings."
            )
            
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Please run: pip install google-generativeai"
            )
            
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Read and encode the input image
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        # Build the full prompt
        # Build the full prompt
        style_key = self._normalize_style(style)
        style_prompt = self.STYLE_PROMPTS.get(style_key, self.STYLE_PROMPTS[STYLE_COLORED])
        # Keep colored output non-exaggerated and documentary.
        if style_key == STYLE_COLORED:
            style_prompt = (
                "RENDERING STYLE: Neutral archaeological plate symbol (NOT painting). "
                "1. SHAPE RULES: Strictly trace the provided silhouette constraints. "
                "2. OUTLINE: Use a clean black outline (about 1-2px equivalent). "
                "3. SHADING: Optional 1-2 flat tone regions only. No painterly texture. "
                "4. FORBIDDEN: No scenery, no landscape, no architecture, no decorative background. "
                "5. SVG PURITY: Use simple vector paths only; do not use gradients, filters, images, or masks."
            )

        prompt = self._SHAPE_PREAMBLE + style_prompt
        prompt += self._NO_EXAGGERATION_RULES
        
        # Hybrid Workflow: Get Silhouette
        silhouette_bytes = None
        try:
             silhouette_bytes = self.contour_gen.get_silhouette_bytes(image_path)
        except Exception as e:
             print(f"Silhouette extraction failed: {e}")
             
        if symmetry:
            prompt += (
                "\n\nSYMMETRY RULE: Apply bilateral symmetry only when the artifact appears "
                "naturally symmetrical in the photo. Do not force perfect symmetry for damaged "
                "or asymmetrical objects."
            )
        
        if color and style_key == STYLE_COLORED:
             prompt += (
                 f"\n\nCOLOR INSTRUCTIONS:"
                 f"\n1. Detect and use the artifact's observed material color from the photo."
                 f"\n2. If user color {color} conflicts with the photo, prioritize the photo."
                 f"\n3. Keep color variations subtle and realistic; avoid saturated fantasy tones."
                 f"\n4. Keep the outline black and clean."
             )
        elif color:
             # For Line Drawing / Publication, color serves as a hint/tint but dominant style rules apply
             pass
             
        # Add Hybrid Logic Instructions to Prompt if applicable
        if silhouette_bytes and style_key == STYLE_COLORED:
             prompt += "\n\nCRITICAL INSTRUCTION: I have provided TWO images. \nImage 1: Original Photo (Textural/Color Reference). \nImage 2: Black & White Silhouette (SHAPE CONSTRAINT). \n\nYOU MUST DRAW THE SYMBOL TO MATCH THE EXACT SHAPE OF IMAGE 2 (THE SILHOUETTE). IGNORE THE SHAPE OF IMAGE 1 IF IT DIFFERS. APPLY THE COLORS/TEXTURES OF IMAGE 1 ONTO THE SHAPE OF IMAGE 2."

        prompt += self._SVG_FORMAT
        
        # Construct content parts
        parts = []
        parts.append(prompt)
        parts.append({"mime_type": self._get_mime_type(image_path), "data": image_data}) # Image 1: Reference
        
        if silhouette_bytes and style_key == STYLE_COLORED:
             parts.append({"mime_type": "image/png", "data": silhouette_bytes}) # Image 2: Mask
        
        # Priorities: Pro models (better vision) > Flash models (faster)
        # We want QUALITY for this task as requested by user.
        # Priorities: v3.0 (Latest Preview) > v2.0 (Stable until Mar 2026) > v1.5 (Legacy)
        # We want STABILITY and SPEED. 'gemini-2.0' is ending soon, v3 is next.
        start_models = ['gemini-3-flash-preview', 'gemini-3-pro-preview', 'gemini-2.0-flash', 'gemini-1.5-flash']
        
        # Explicitly exclude models known to be unstable or quota-restricted for general use
        # "deep-research" caused 429 errors.
        excluded_keywords = ['deep-research', 'experimental']

        models_to_try = []
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # normalize names
            available_map = {m.replace('models/', ''): m for m in available_models}
            
            # Helper to check if a model name should be excluded
            def is_excluded(name):
                return any(keyword in name.lower() for keyword in excluded_keywords)

            # 1. Try preferred start models if they exist (exact or prefix match)
            for m_pref in start_models:
                # Direct match
                if m_pref in available_map and not is_excluded(m_pref):
                    models_to_try.append(available_map[m_pref])
                    continue
                    
                # Prefix match (e.g. 'gemini-1.5-pro' matches 'gemini-1.5-pro-001')
                # We find the *latest* version if multiple exist (usually lexicographically last is best estimate if versioned)
                matches = [name for name in available_map.keys() if name.startswith(m_pref) and not is_excluded(name)]
                if matches:
                    # distinct versions, pick the shortest name (usually the alias) or the latest version
                    # simplistic: sort and pick last (often highest version number)
                    matches.sort()
                    best_match = matches[-1]
                    models_to_try.append(available_map[best_match])
            
            # 2. If none of the specific pro models found, fall back to any 'pro' model
            if not models_to_try:
                for name, full_name in available_map.items():
                    if 'pro' in name.lower() and not is_excluded(name) and full_name not in models_to_try:
                        models_to_try.append(full_name)
            
            # 3. If still nothing, try any 'flash' model
            if not models_to_try:
                for name, full_name in available_map.items():
                    if 'flash' in name.lower() and not is_excluded(name) and full_name not in models_to_try:
                        models_to_try.append(full_name)
                        
            # 4. Last resort: whatever is available (filtered)
            if not models_to_try:
                models_to_try = [m for m in available_models if not is_excluded(m)]
                
        except Exception:
             # Offline fallback or API error list
            models_to_try = ['models/gemini-1.5-pro', 'models/gemini-1.5-flash']
        
        last_error = None
        last_svg_issue = None
        for model_name in models_to_try:
            # Retry logic with exponential backoff
            max_retries = 3
            base_delay = 2  # seconds
            
            for attempt in range(max_retries + 1):
                try:
                    model = genai.GenerativeModel(model_name)
                    
                    # Generate text (SVG code)
                    response = model.generate_content(parts)
                    
                    # If successful, extract SVG code and return as string
                    if response.text:
                        svg_code = self._extract_svg(response.text)
                        if svg_code:
                            is_safe, issue = self._is_svg_documentary_safe(svg_code, style_key=style_key)
                            if is_safe:
                                if silhouette_bytes:
                                    is_match, shape_issue = self._matches_reference_silhouette(
                                        svg_code=svg_code,
                                        silhouette_bytes=silhouette_bytes,
                                        style_key=style_key,
                                    )
                                    if is_match:
                                        return svg_code
                                    last_svg_issue = shape_issue
                                else:
                                    return svg_code
                            else:
                                last_svg_issue = issue
                        
                    # If we got a response but no SVG, maybe try next model
                    break 
 
                         
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "Quota" in error_str or "ResourceExhausted" in error_str
                    
                    if is_rate_limit and attempt < max_retries:
                        import time
                        import random
                        # Exponential backoff with jitter
                        delay = (base_delay * (2 ** attempt)) + (random.uniform(0, 1))
                        print(f"Gemini API rate limit hit ({model_name}). Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        last_error = e
                        # If it's not a rate limit, or we ran out of retries, try the next model
                        break
        
        if last_error:
            raise last_error

        # Final factual fallback: deterministic contour extraction.
        try:
            fallback_svg = self.contour_gen.generate(
                image_path=image_path,
                style=style,
                color=color,
                symmetry=symmetry
            )
            if fallback_svg:
                return fallback_svg
        except Exception as e:
            if not last_error:
                last_error = e

        if last_error:
            raise last_error
        if last_svg_issue:
            raise Exception(f"Gemini output rejected as non-documentary: {last_svg_issue}")

        raise Exception("Failed to generate symbol: No suitable AI model found.")
        
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
        
    def _extract_svg(self, text):
        """Extract SVG code from response text."""
        start = text.find('<svg')
        end = text.find('</svg>')
        
        if start != -1 and end != -1:
            return text[start:end+6]
        return None

    def _is_svg_documentary_safe(self, svg_code, style_key=None):
        """Reject SVG outputs that look painterly or non-symbolic."""
        if not svg_code:
            return False, "empty SVG"

        lower = svg_code.lower()
        if "<svg" not in lower or "</svg>" not in lower:
            return False, "invalid SVG envelope"
        if "<path" not in lower:
            return False, "no path geometry found"

        for token in self._DISALLOWED_SVG_TOKENS:
            if token in lower:
                return False, f"contains disallowed element: {token}"

        path_count = lower.count("<path")
        if path_count <= 0:
            return False, "no path elements found"
        if style_key == STYLE_COLORED and path_count > 18:
            return False, f"too many path elements for factual colored style ({path_count})"
        if style_key in (STYLE_LINE, STYLE_MEASURED) and path_count > 42:
            return False, f"too many path elements for line/measured style ({path_count})"

        # Reject overly decorative color palettes in documentary mode.
        fills = re.findall(r'fill\s*=\s*["\']([^"\']+)["\']', svg_code, flags=re.IGNORECASE)
        strokes = re.findall(r'stroke\s*=\s*["\']([^"\']+)["\']', svg_code, flags=re.IGNORECASE)
        colors = set()
        for val in fills + strokes:
            token = val.strip().lower()
            if token in ("none", "transparent", "currentcolor", ""):
                continue
            colors.add(token)

        if style_key == STYLE_COLORED and len(colors) > 6:
            return False, f"too many distinct colors ({len(colors)})"
        if style_key in (STYLE_LINE, STYLE_MEASURED):
            for c in colors:
                if c in ("#000", "#000000", "black", "#111", "#111111", "#222", "#222222"):
                    continue
                if re.fullmatch(r'#[0-9a-f]{6}', c):
                    try:
                        r = int(c[1:3], 16)
                        g = int(c[3:5], 16)
                        b = int(c[5:7], 16)
                        if abs(r - g) <= 10 and abs(g - b) <= 10:
                            continue
                    except Exception:
                        pass
                return False, f"non-monochrome color detected in line/measured mode: {c}"

        return True, ""

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

    def _matches_reference_silhouette(self, svg_code, silhouette_bytes, style_key=None):
        """Validate generated SVG silhouette against contour-derived reference mask."""
        if not silhouette_bytes:
            return True, ""

        ref_mask = QImage()
        if not ref_mask.loadFromData(silhouette_bytes):
            return True, ""

        rendered = self._render_svg_to_image(svg_code, ref_mask.width(), ref_mask.height())
        if rendered is None:
            return False, "failed to rasterize SVG for silhouette check"

        inter = 0
        union = 0
        ref_count = 0
        pred_count = 0

        h = min(ref_mask.height(), rendered.height())
        w = min(ref_mask.width(), rendered.width())
        for y in range(h):
            for x in range(w):
                rp = ref_mask.pixelColor(x, y)
                ref_inside = (rp.red() < 90 and rp.green() < 90 and rp.blue() < 90)

                gp = rendered.pixelColor(x, y)
                pred_inside = (
                    gp.alpha() > 16 and
                    not (gp.red() > 248 and gp.green() > 248 and gp.blue() > 248 and gp.alpha() > 220)
                )

                if ref_inside:
                    ref_count += 1
                if pred_inside:
                    pred_count += 1
                if ref_inside and pred_inside:
                    inter += 1
                if ref_inside or pred_inside:
                    union += 1

        if ref_count < 40:
            return True, ""
        if union <= 0 or pred_count <= 0:
            return False, "empty rendered geometry against reference silhouette"

        iou = float(inter) / float(union)
        recall = float(inter) / float(ref_count)
        precision = float(inter) / float(pred_count)

        if style_key == STYLE_COLORED:
            ok = (iou >= 0.72 and recall >= 0.84 and precision >= 0.72)
        elif style_key == STYLE_LINE:
            ok = (iou >= 0.42 and recall >= 0.66)
        else:
            ok = (iou >= 0.50 and recall >= 0.72)

        if ok:
            return True, ""
        return False, f"silhouette mismatch (IoU={iou:.2f}, recall={recall:.2f}, precision={precision:.2f})"
