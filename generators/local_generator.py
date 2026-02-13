# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Local Stable Diffusion Generator
Generates stylized archaeological symbols using local Stable Diffusion.
Supports both ComfyUI and Automatic1111 WebUI backends.
"""

import os
import json
import base64
import tempfile
from qgis.PyQt.QtGui import QImage
from qgis.PyQt.QtCore import QSettings

from .style_utils import (
    STYLE_COLORED,
    STYLE_LINE,
    STYLE_MEASURED,
    normalize_style,
)


class LocalGenerator:
    """Generator using local Stable Diffusion for symbol creation."""
    
    # Default endpoints for different backends
    BACKENDS = {
        'automatic1111': {
            'txt2img': '/sdapi/v1/txt2img',
            'img2img': '/sdapi/v1/img2img',
            'default_port': 7860
        },
        'comfyui': {
            'prompt': '/prompt',
            'history': '/history',
            'default_port': 8188
        }
    }
    
    # Style prompts for different archaeological symbol styles
    STYLE_PROMPTS = {
        STYLE_COLORED: (
            "accurate archaeological artifact silhouette, flat color fill, "
            "clean shape, precise outline, map symbol, "
            "transparent background, centered, high contrast, "
            "digital art, vector style"
        ),
        STYLE_LINE: (
            "minimalist line art icon, archaeological artifact, "
            "simple geometric shapes, clean lines, monochrome, "
            "technical drawing style, transparent background, centered, "
            "vector illustration, blueprint style"
        ),
        STYLE_MEASURED: (
            "classic archaeological illustration, artifact drawing, "
            "stippling cross-hatching, academic professional, publication quality, "
            "transparent background, centered, scientific illustration, "
            "museum catalog style"
        )
    }
    
    NEGATIVE_PROMPT = (
        "blurry, low quality, text, watermark, signature, "
        "complex background, cluttered, realistic photo, "
        "multiple objects, frame, border"
    )
    
    def __init__(self):
        """Initialize the local generator."""
        self.settings = QSettings()
        self.backend = self.settings.value('ArcheoGlyph/sd_backend', 'automatic1111')
        self.server_url = self.settings.value('ArcheoGlyph/sd_server', 'http://127.0.0.1:7860')
        
    def set_server(self, url, backend='automatic1111'):
        """Save server settings."""
        self.server_url = url
        self.backend = backend
        self.settings.setValue('ArcheoGlyph/sd_server', url)
        self.settings.setValue('ArcheoGlyph/sd_backend', backend)
        
    def test_connection(self):
        """Test connection to the local SD server."""
        try:
            import requests
            if self.backend == 'automatic1111':
                response = requests.get(f"{self.server_url}/sdapi/v1/sd-models", timeout=5)
            else:
                response = requests.get(f"{self.server_url}/system_stats", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
            
    def generate(self, image_path, style, color=None):
        """
        Generate a symbol from the input image using local Stable Diffusion.
        
        :param image_path: Path to the input artifact image
        :param style: Style preset name
        :param color: Optional hex color for the symbol
        :return: QImage of generated symbol or None on failure
        """
        if not self.test_connection():
            raise ConnectionError(
                f"Cannot connect to Stable Diffusion server at {self.server_url}. "
                "Please ensure the server is running."
            )
            
        prompt = self.STYLE_PROMPTS.get(self._normalize_style(style), self.STYLE_PROMPTS[STYLE_COLORED])
        
        if color:
            prompt += f", {color} color scheme"
            
        if self.backend == 'automatic1111':
            return self._generate_a1111(image_path, prompt)
        else:
            return self._generate_comfyui(image_path, prompt)
            
    def _generate_a1111(self, image_path, prompt):
        """Generate using Automatic1111 WebUI API."""
        import requests
        # Read and encode the input image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            
        payload = {
            "init_images": [image_data],
            "prompt": prompt,
            "negative_prompt": self.NEGATIVE_PROMPT,
            "steps": 30,
            "cfg_scale": 7,
            "width": 256,
            "height": 256,
            "denoising_strength": 0.7,
            "sampler_name": "DPM++ 2M Karras"
        }
        
        response = requests.post(
            f"{self.server_url}/sdapi/v1/img2img",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'images' in result and len(result['images']) > 0:
                image_base64 = result['images'][0]
                image_bytes = base64.b64decode(image_base64)
                return self._bytes_to_image(image_bytes)
                
        return None

    def _normalize_style(self, style):
        """Map style labels to canonical keys."""
        return normalize_style(style)
        
    def _generate_comfyui(self, image_path, prompt):
        """Generate using ComfyUI API."""
        # ComfyUI requires a workflow JSON
        # This is a simplified implementation - real usage would need proper workflow
        
        raise NotImplementedError(
            "ComfyUI support is coming soon. "
            "Please use Automatic1111 WebUI for now."
        )
        
    def _bytes_to_image(self, image_bytes):
        """Convert raw bytes to QImage."""
        image = QImage()
        image.loadFromData(image_bytes)
        return image
        
    @staticmethod
    def get_setup_instructions():
        """Return setup instructions for local Stable Diffusion."""
        return """
# Local Stable Diffusion Setup Guide

## Option 1: Automatic1111 WebUI (Recommended)

1. **Install Python 3.10.6** (required version)
   - Download from: https://www.python.org/downloads/release/python-3106/

2. **Clone the repository**
   ```
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ```

3. **Download a model**
   - Recommended: SD 1.5 or SDXL
   - Place .safetensors file in `models/Stable-diffusion/`
   
4. **Run with API enabled**
   ```
   webui-user.bat --api
   ```
   Or edit webui-user.bat to add `--api` to COMMANDLINE_ARGS

5. **Configure in ArcheoGlyph**
   - Server URL: http://127.0.0.1:7860
   - Backend: Automatic1111

## Option 2: ComfyUI

Coming soon...

## Recommended Models for Icon Generation

1. **For Colored style:**
   - Anything V5
   - Counterfeit V3

2. **For Line style:**
   - Deliberate V2
   - SD 1.5 with LoRA

3. **For Measured style:**
   - Realistic Vision V5
   - SDXL Base

## Troubleshooting

- **Connection refused**: Make sure the server is running
- **Slow generation**: Use a GPU with at least 6GB VRAM
- **Out of memory**: Reduce image size or use --lowvram flag
"""
