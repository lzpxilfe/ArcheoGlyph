# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Hugging Face Generator
Generates icon-style symbols using Hugging Face Inference API and vectorizes them.
"""

import os
import json
import requests
import tempfile
import time
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtGui import QImage, QColor

# Import the contour generator for vectorization
from .contour_generator import ContourGenerator

class HuggingFaceGenerator:
    """
    Generator using Hugging Face Inference API for symbol creation.
    Specifically targets 'icon' generation models.
    """
    
    def __init__(self):
        """Initialize the Hugging Face generator."""
        self.settings = QSettings()
        self.api_key = self.settings.value('ArcheoGlyph/huggingface_api_key', '')
        # Load model ID, default to reliable SD 1.5
        self.model_id = self.settings.value(
            'ArcheoGlyph/hf_model_id', 
            'stabilityai/stable-diffusion-2-1'
        )
        self.contour_gen = ContourGenerator()
        
    def set_api_key(self, api_key):
        """Save API key to settings."""
        self.api_key = api_key
        self.settings.setValue('ArcheoGlyph/huggingface_api_key', api_key)
        
    def get_api_key(self):
        """Get API key from settings."""
        return self.api_key
        
    def generate(self, prompt, style=None, color=None):
        """
        Generate an icon using Hugging Face Inference API.
        
        :param prompt: Text prompt for the icon
        :param style: Style name (optional)
        :param color: Color hex code (optional)
        :return: QImage of the generated symbol
        """
        if not self.api_key:
             raise ValueError("Hugging Face API Token is missing. Please set it in Settings.")

        # Construct API URL from model ID
        # SANITIZE: Remove leading/trailing spaces which cause 404/410
        self.model_id = self.model_id.strip()

        # Construct API URL from model ID
        if not self.model_id:
             self.model_id = "stabilityai/stable-diffusion-2-1" # Standard reliable model
        
        # Check against common 410 URLs and divert
        if "flat-design-icons" in self.model_id:
            self.model_id = "stabilityai/stable-diffusion-2-1"
            
        # MIGRATION: Auto-switch from old v1.5 default to v2.1
        if self.model_id == "runwayml/stable-diffusion-v1-5":
             self.model_id = "stabilityai/stable-diffusion-2-1"
             self.settings.setValue('ArcheoGlyph/hf_model_id', self.model_id)
        
        # List of models to try (Original choice -> Fallbacks)
        models_to_try = [self.model_id]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Add reliable fallbacks if not already in list
        # PRIORITY: Use valid, non-gated, fast models first.
        # SD 1.5 is the most reliable "open" model.
        # SDXL is great but often GATED (requires agreeing to terms), so it fails with 403.
        fallbacks = [
            "stabilityai/stable-diffusion-2-1", # Newer, more reliable
            "prompthero/openjourney",           # Great for artistic styles
            "CompVis/stable-diffusion-v1-4"     # Solar powered old reliable
        ]

        for fb in fallbacks:
            if fb not in models_to_try:
                models_to_try.append(fb)

        # Enhance prompt for icon generation
        # User wants "Pax Sapientica" style: Scientific, volumetric, not flat.
        style_keywords = "scientific archaeological illustration, watercolor style, volumetric lighting, 3d, highly detailed, texture, white background"
        if style:
            style_keywords += f", {style}"
        if color:
            style_keywords += f", {color} tones"

        enhanced_prompt = f"{style_keywords}, {prompt}"

        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "negative_prompt": "blur, grainy, low resolution, complexity, background, shadow, text, watermark, flat, simple, minimalist, icon",
                "num_inference_steps": 35,
                "guidance_scale": 8.5,
            }
        }

        error_logs = []
        for model in models_to_try:
            try:
                # API Endpoint
                # Check for empty model string
                if not model or len(model) < 3: continue
                
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                
                # Check specific error codes
                if response.status_code == 410:
                    error_logs.append(f"Model {model}: GONE (410) - Check URL")
                    continue
                elif response.status_code == 404:
                    error_logs.append(f"Model {model}: NOT FOUND (404) - Check typo")
                    continue
                elif response.status_code == 503:
                    error_logs.append(f"Model {model}: LOADING (503)")
                    continue
                elif response.status_code == 401:
                    error_logs.append(f"Model {model}: AUTH ERROR (401) - Check Token")
                    continue
                elif response.status_code == 403:
                    error_logs.append(f"Model {model}: FORBIDDEN (403) - Gated Model? (Accept Terms on HF)")
                    continue
                    
                response.raise_for_status()
                
                # If we get here, it worked!
                image = QImage()
                image.loadFromData(response.content)
                return image
                
            except Exception as e:
                error_logs.append(f"Model {model} Error: {str(e)}")
                continue
                
        # If all failed, construct detailed report
        report = "\n".join(error_logs[:6]) 
        
        hint = ""
        full_log = str(error_logs)
        if "403" in full_log:
             hint = "\n\n💡 TIP: 'Forbidden' usually means a Gated Model (e.g. SDXL). Use 'runwayml/stable-diffusion-v1-5' instead."
        if "410" in full_log or "401" in full_log:
             hint = "\n\n💡 TIP: Check TOKEN PERMISSIONS. Ensure you used a 'READ' (Classic) token."
             
        raise Exception(f"All models failed.\n\nERROR LOG:\n{report}\n{hint}")
                

