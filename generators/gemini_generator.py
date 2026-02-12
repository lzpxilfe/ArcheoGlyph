# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Google Gemini Generator
Generates stylized archaeological symbols using Google Gemini API.
"""

import os
from qgis.PyQt.QtCore import QSettings



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
        "🎯 Colored Silhouette (채색 실루엣)": (
            "RENDERING STYLE: Premium Vector Game Asset / RPG Item Icon. "
            "1. SHAPE RULES: STICTLY TRACE the original artifact. DO NOT exaggerate features. DO NOT make it thinner or thicker. "
            "   Capture the EXACT silhouette constraints. "
            "2. OUTLINE: Draw a bold, clean, consistent BLACK outline (2-3px) around the entire object. "
            "3. SHADING: Use 'Cel Shading' or '2-Tone Shading' (Base Color + Shadow Color) to show volume. "
            "4. AESTHETIC: Flat Design but with depth. Like a high-quality strategy game unit or resource icon. "
            "NO gradient meshes. NO realistic texture noise. Clean vector shapes."
        ),
        "📐 Line Drawing (선화)": (
            "RENDERING STYLE: Archaeological Line Drawing. "
            "Draw ONLY the precise outline of the artifact and major internal lines. "
            "Use clean, consistent black strokes (1-2px). "
            "NO shading, NO stippling, NO hatching, NO fill. "
            "Pure abstraction of the form. Transparent background."
        ),
        "🏛️ Publication (실측 도면)": (
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
    
    def __init__(self):
        """Initialize the Gemini generator."""
        self.settings = QSettings()
        self.api_key = self.settings.value('ArcheoGlyph/gemini_api_key', '')
        
    def set_api_key(self, api_key):
        """Save API key to settings."""
        self.api_key = api_key
        self.settings.setValue('ArcheoGlyph/gemini_api_key', api_key)
        
    def get_api_key(self):
        """Get API key from settings."""
        return self.api_key
        
    def generate(self, image_path, style="🎯 Colored Silhouette (채색 실루엣)", color="#000000", symmetry=False):
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
        style_prompt = self.STYLE_PROMPTS.get(style, self.STYLE_PROMPTS["🎯 Colored Silhouette (채색 실루엣)"])
        prompt = self._SHAPE_PREAMBLE + style_prompt
        
        if symmetry:
            prompt += "\n\nCRITICAL: The object must be PERFECTLY SYMMETRICAL (bilateral symmetry). Mirror the left side to the right if needed to create a perfect canonical view."
        
        if color and "Colored Silhouette" in style:
             prompt += (
                 f"\n\nCOLOR INSTRUCTIONS:"
                 f"\n1. PRIMARY FILL COLOR: ANALYZE the input image and DETECT the dominant color of the artifact. Use that detected color."
                 f"\n   (NOTE: The user suggested {color}, but you should ignore it if it doesn't match the image's actual material color)."
                 f"\n2. HARMONY: You MAY mix in subtle variations, analogous colors, or gradients to make it look natural and artistic."
                 f"\n3. OUTLINE: Keep the outline BLACK."
                 f"\n4. AVOID flat, cartoonish single-color fills. Make it look like a high-quality hand-painted archaeological illustration."
             )
        elif color:
             # For Line Drawing / Publication, color serves as a hint/tint but dominant style rules apply
             pass
            
        prompt += self._SVG_FORMAT
        
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
        for model_name in models_to_try:
            # Retry logic with exponential backoff
            max_retries = 3
            base_delay = 2  # seconds
            
            for attempt in range(max_retries + 1):
                try:
                    model = genai.GenerativeModel(model_name)
                    
                    image_part = {
                        'mime_type': self._get_mime_type(image_path),
                        'data': image_data
                    }
                    
                    # Generate text (SVG code)
                    response = model.generate_content(
                        [prompt, image_part]
                    )
                    
                    # If successful, extract SVG code and return as string
                    if response.text:
                        svg_code = self._extract_svg(response.text)
                        if svg_code:
                            return svg_code
                        
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
