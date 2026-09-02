# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Local Stable Diffusion Generator
Generates stylized archaeological symbols using local Stable Diffusion.
Uses Automatic1111 WebUI API backend.
"""

import base64
from qgis.PyQt.QtGui import QImage
from qgis.PyQt.QtCore import QSettings

from ..log import log_exception
from .style_control_utils import resolve_style_controls, style_controls_prompt_hint
from .symbol_result import SymbolResult
from .style_utils import (
    STYLE_COLORED,
    STYLE_LINE,
    STYLE_MEASURED,
    STYLE_TYPOLOGY,
    normalize_style,
)


class LocalGenerator:
    """Generator using local Stable Diffusion for symbol creation."""
    
    BACKENDS = {
        'automatic1111': {
            'txt2img': '/sdapi/v1/txt2img',
            'img2img': '/sdapi/v1/img2img',
            'default_port': 7860
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
        STYLE_TYPOLOGY: (
            "archaeological typology symbol, standardized silhouette, "
            "bold contour, central axis cue, 1-3 structural bands, "
            "muted flat palette, avoid single flat fill color, no texture, "
            "transparent background, centered"
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
        stored_backend = str(self.settings.value('ArcheoGlyph/sd_backend', 'automatic1111')).strip().lower()
        if stored_backend not in self.BACKENDS:
            stored_backend = 'automatic1111'
        self.backend = stored_backend

        self.server_url = str(
            self.settings.value('ArcheoGlyph/sd_server', 'http://127.0.0.1:7860')
        ).strip() or 'http://127.0.0.1:7860'

        # Persist normalized values only.
        self.settings.setValue('ArcheoGlyph/sd_backend', self.backend)
        self.settings.setValue('ArcheoGlyph/sd_server', self.server_url)
        
    def set_server(self, url, backend='automatic1111'):
        """Save server settings."""
        normalized_backend = str(backend or 'automatic1111').strip().lower()
        if normalized_backend not in self.BACKENDS:
            normalized_backend = 'automatic1111'
        self.backend = normalized_backend

        normalized_url = str(url or '').strip() or 'http://127.0.0.1:7860'
        self.server_url = normalized_url

        self.settings.setValue('ArcheoGlyph/sd_server', normalized_url)
        self.settings.setValue('ArcheoGlyph/sd_backend', normalized_backend)
        
    def test_connection(self):
        """Test connection to the local SD server."""
        try:
            import requests
            response = requests.get(f"{self.server_url}/sdapi/v1/sd-models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            log_exception("Stable Diffusion server is not reachable", e)
            return False
            
    def generate(
        self,
        image_path,
        style,
        prompt="",
        color=None,
        factuality=None,
        symbolic_looseness=None,
        exaggeration=None,
    ):
        """
        Generate a symbol from the input image using local Stable Diffusion.
        
        :param image_path: Path to the input artifact image
        :param style: Style preset name
        :param color: Optional hex color for the symbol
        :return: SymbolResult carrying the generated raster
        """
        if not self.test_connection():
            raise ConnectionError(
                f"Cannot connect to Stable Diffusion server at {self.server_url}. "
                "Please ensure the server is running."
            )
            
        base_prompt = self.STYLE_PROMPTS.get(self._normalize_style(style), self.STYLE_PROMPTS[STYLE_COLORED])
        base_prompt += ", " + self._style_control_hint(
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        
        if color:
            base_prompt += f", {color} color scheme"

        text_hint = str(prompt or "").strip()
        if text_hint:
            base_prompt += f", user note: {text_hint}"

        if self.backend != 'automatic1111':
            raise NotImplementedError(
                "Only Automatic1111 backend is currently supported. "
                "Please use an Automatic1111 server URL in settings."
            )

        # Send the artifact on a clean ground with its silhouette as an inpaint
        # mask, so the model restyles the object instead of repainting the
        # photo's background along with it.
        mask_b64 = None
        try:
            from .contour_generator import ContourGenerator

            silhouette = ContourGenerator().get_silhouette_bytes(image_path)
            if silhouette:
                # The A1111 mask marks the region to change in white.
                import cv2
                import numpy as np

                buf = np.frombuffer(silhouette, dtype=np.uint8)
                image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    inverted = cv2.bitwise_not(image)
                    ok, encoded = cv2.imencode(".png", inverted)
                    if ok:
                        mask_b64 = base64.b64encode(bytes(encoded)).decode("utf-8")
        except Exception as e:
            log_exception("Silhouette mask for local Stable Diffusion failed", e)

        image = self._generate_a1111(image_path, base_prompt, mask_b64=mask_b64)
        return SymbolResult.coerce(image, source="local-sd", style=str(style or ""))
            
    def _generate_a1111(self, image_path, prompt, mask_b64=None):
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
            # 256 px was below the training resolution of every SD model and
            # produced mushy output; 512 is the smallest sensible size.
            "width": 512,
            "height": 512,
            "denoising_strength": 0.7,
            "sampler_name": "DPM++ 2M Karras",
        }
        if mask_b64:
            payload.update({
                "mask": mask_b64,
                "inpainting_fill": 1,          # keep the original pixels as the base
                "inpaint_full_res": False,
                "mask_blur": 4,
                "denoising_strength": 0.55,
            })
        
        api_url = f"{self.server_url}/sdapi/v1/img2img"
        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=120
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"Automatic1111 request failed at {api_url}: {exc}")

        if response.status_code != 200:
            detail = (response.text or "").strip()
            if len(detail) > 220:
                detail = detail[:220] + "..."
            raise RuntimeError(
                f"Automatic1111 returned HTTP {response.status_code}"
                + (f": {detail}" if detail else "")
            )

        try:
            result = response.json()
        except Exception:
            raise RuntimeError("Automatic1111 returned non-JSON response.")

        images = result.get('images', [])
        if not images:
            raise RuntimeError("Automatic1111 response contained no generated images.")

        image_base64 = images[0]
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            raise RuntimeError("Automatic1111 returned invalid base64 image payload.")
        return self._bytes_to_image(image_bytes)

    def _normalize_style(self, style):
        """Map style labels to canonical keys."""
        return normalize_style(style)

    def _style_control_hint(self, factuality=None, symbolic_looseness=None, exaggeration=None):
        """Read style sliders and convert to local prompt hints."""
        controls = resolve_style_controls(
            settings=self.settings,
            factuality=factuality,
            symbolic_looseness=symbolic_looseness,
            exaggeration=exaggeration,
        )
        return style_controls_prompt_hint(controls, prefix="style controls")
        
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
