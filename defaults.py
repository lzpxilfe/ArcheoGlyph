# -*- coding: utf-8 -*-
"""
ArcheoGlyph shared defaults.
"""

PLUGIN_VERSION = "0.1.0"

DEFAULT_LIBRARY_SYMBOL_SIZE_MM = 10.0
DEFAULT_MIN_SYMBOL_SIZE_MM = 10.0
DEFAULT_MAX_SYMBOL_SIZE_MM = 24.0
DEFAULT_GRADUATED_CLASSES = 5

HF_DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
HF_FALLBACK_MODEL_IDS = (
    HF_DEFAULT_MODEL_ID,
    "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image",
    "black-forest-labs/FLUX.2-dev",
    "black-forest-labs/FLUX.1-Kontext-dev",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-Krea-dev",
    "stabilityai/stable-diffusion-3.5-large",
)

# Legacy IDs that should be normalized to HF_DEFAULT_MODEL_ID
HF_LEGACY_MODEL_ALIASES = {
    "stabilityai/stable-diffusion-2-1": HF_DEFAULT_MODEL_ID,
    "runwayml/stable-diffusion-v1-5": HF_DEFAULT_MODEL_ID,
    "stable-diffusion-v1-5/stable-diffusion-v1-5": HF_DEFAULT_MODEL_ID,
    "stabilityai/stable-diffusion-xl-base-1.0": HF_DEFAULT_MODEL_ID,
    "CompVis/stable-diffusion-v1-4": HF_DEFAULT_MODEL_ID,
    "prompthero/openjourney": HF_DEFAULT_MODEL_ID,
}
