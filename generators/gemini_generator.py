# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Google Gemini Generator
Generates stylized archaeological symbols using Google Gemini API.
"""

import os
import json
import base64
import tempfile
from qgis.PyQt.QtGui import QPixmap, QImage
from qgis.PyQt.QtCore import QSettings


class GeminiGenerator:
    """Generator using Google Gemini API for symbol creation."""
    
    # Style prompts for different archaeological symbol styles
    STYLE_PROMPTS = {
        "🎨 Cute / Kawaii": (
            "Create a cute, kawaii-style icon of this archaeological artifact. "
            "Make it adorable with rounded shapes, soft colors, and friendly appearance. "
            "Simple, clean design suitable for use as a map symbol. "
            "High contrast."
        ),
        "📐 Minimal": (
            "Create a minimalist line art icon of this archaeological artifact. "
            "Use simple geometric shapes and clean lines. "
            "Black and white or single color, suitable for technical drawings. "
            "High contrast."
        ),
        "🏛️ Classic Archaeological": (
            "Create a classic archaeological illustration style icon of this artifact. "
            "Traditional stippling and cross-hatching techniques. "
            "Academic and professional appearance, suitable for publications. "
            "High contrast."
        )
    }
    
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
        
    def generate(self, image_path, style, color=None):
        """
        Generate a symbol from the input image using Gemini.
        Returns a QPixmap of the generated SVG.
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
            
        # Get the appropriate prompt
        prompt = self.STYLE_PROMPTS.get(style, self.STYLE_PROMPTS["🎨 Cute / Kawaii"])
        
        if color:
            prompt += f" Use {color} as the primary color."
            
        prompt += (
            " OUTPUT FORMAT: Provide ONLY valid SVG (Scalable Vector Graphics) code. "
            "Do not use markdown code blocks. Start directly with <svg> and end with </svg>. "
            "Ensure the SVG is square (e.g. viewBox='0 0 512 512')."
        )
        
        # List available models from the API
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    name = m.name
                    if name.startswith('models/'):
                        name = name.replace('models/', '')
                    available_models.append(name)
        except Exception:
            pass 
            
        # Prioritize models
        models_to_try = []
        
        # 1. Flash (Fastest)
        for m in available_models:
            if 'flash' in m.lower():
                models_to_try.append(m)
        
        # 2. Pro (Stable)
        for m in available_models:
            if 'pro' in m.lower() and m not in models_to_try:
                models_to_try.append(m)
                
        # 3. Others (Fallback)
        for m in available_models:
            if m not in models_to_try:
                models_to_try.append(m)
        
        if not models_to_try:
             models_to_try = ['gemini-1.5-flash', 'gemini-pro']

        last_error = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                
                image_part = {
                    'mime_type': self._get_mime_type(image_path),
                    'data': image_data
                }
                
                # Generate text (SVG code)
                response = model.generate_content(
                    [prompt, image_part]
                    # No generation_config needed for text output
                )
                
                # If successful, extract text and convert to pixmap
                if response.text:
                    svg_code = self._extract_svg(response.text)
                    if svg_code:
                        return self._svg_to_pixmap(svg_code)
                     
            except Exception as e:
                last_error = e
                continue 
        
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

    def _svg_to_pixmap(self, svg_code):
        """Render SVG code to QPixmap."""
        from qgis.PyQt.QtCore import QByteArray
        from qgis.PyQt.QtSvg import QSvgRenderer
        from qgis.PyQt.QtGui import QPainter
        
        renderer = QSvgRenderer(QByteArray(svg_code.encode('utf-8')))
        
        if not renderer.isValid():
            raise ValueError("Invalid SVG code generated")
            
        pixmap = QPixmap(256, 256)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        
        return pixmap
