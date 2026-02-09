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
            "Transparent background, centered, high contrast."
        ),
        "📐 Minimal": (
            "Create a minimalist line art icon of this archaeological artifact. "
            "Use simple geometric shapes and clean lines. "
            "Black and white or single color, suitable for technical drawings. "
            "Transparent background, centered, high contrast."
        ),
        "🏛️ Classic Archaeological": (
            "Create a classic archaeological illustration style icon of this artifact. "
            "Traditional stippling and cross-hatching techniques. "
            "Academic and professional appearance, suitable for publications. "
            "Transparent background, centered, high contrast."
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
        
        :param image_path: Path to the input artifact image
        :param style: Style preset name
        :param color: Optional hex color for the symbol
        :return: QPixmap of generated symbol or None on failure
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
            
        prompt += " Output a 256x256 pixel PNG image with transparent background."
        
        # Create the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Prepare the image part
        image_part = {
            'mime_type': self._get_mime_type(image_path),
            'data': image_data
        }
        
        # Generate
        response = model.generate_content(
            [prompt, image_part],
            generation_config={
                'response_mime_type': 'image/png'
            }
        )
        
        # Extract generated image
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data'):
                        image_bytes = part.inline_data.data
                        return self._bytes_to_pixmap(image_bytes)
                        
        return None
        
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
        
    def _bytes_to_pixmap(self, image_bytes):
        """Convert raw bytes to QPixmap."""
        image = QImage()
        image.loadFromData(image_bytes)
        return QPixmap.fromImage(image)
