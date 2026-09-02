# -*- coding: utf-8 -*-
"""
ArcheoGlyph shared defaults.
"""

PLUGIN_VERSION = "0.2.0"

DEFAULT_LIBRARY_SYMBOL_SIZE_MM = 10.0
DEFAULT_MIN_SYMBOL_SIZE_MM = 10.0
DEFAULT_MAX_SYMBOL_SIZE_MM = 24.0
DEFAULT_GRADUATED_CLASSES = 5

HF_DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

# Image-to-image / instruction-edit models: they take the reference image plus
# an instruction. These are what reference-first generation should use.
HF_IMG2IMG_MODELS = (
    "black-forest-labs/FLUX.1-Kontext-dev",
    "Qwen/Qwen-Image-Edit-2511",
    "Qwen/Qwen-Image-Edit-2509",
    "Qwen/Qwen-Image-Edit",
)

# Text-to-image models: used only when there is no reference photo. Sending
# them an image payload is what produced the old 404/422 failures.
HF_TXT2IMG_MODELS = (
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-Krea-dev",
    "Qwen/Qwen-Image",
    "stabilityai/stable-diffusion-3.5-large",
)

# Combined list, kept for settings/UI code that offers every known model.
HF_FALLBACK_MODEL_IDS = HF_IMG2IMG_MODELS + HF_TXT2IMG_MODELS

# Models that accept classifier-free guidance and negative prompts. Guidance
# distilled and instruction-edit models reject or ignore those parameters.
HF_GUIDANCE_MODELS = (
    "stabilityai/stable-diffusion-3.5-large",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-Krea-dev",
)

HF_MAX_MODELS_PER_ROUTE = 4

HF_LEGACY_COMPAT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

# Legacy IDs that should be normalized to a broadly accessible compatibility model
HF_LEGACY_MODEL_ALIASES = {
    "stabilityai/stable-diffusion-2-1": HF_LEGACY_COMPAT_MODEL_ID,
    "runwayml/stable-diffusion-v1-5": HF_LEGACY_COMPAT_MODEL_ID,
    "stable-diffusion-v1-5/stable-diffusion-v1-5": HF_LEGACY_COMPAT_MODEL_ID,
    "stabilityai/stable-diffusion-xl-base-1.0": HF_LEGACY_COMPAT_MODEL_ID,
    "Qwen/Qwen-Image-Edit-2509": HF_LEGACY_COMPAT_MODEL_ID,
    "CompVis/stable-diffusion-v1-4": HF_LEGACY_COMPAT_MODEL_ID,
    "prompthero/openjourney": HF_LEGACY_COMPAT_MODEL_ID,
}

GEMINI_INSTALL_PACKAGE = "google-genai"
GEMINI_AI_STUDIO_URL = "https://aistudio.google.com/apikey"

GEMINI_TEXT_MODEL_CANDIDATES = (
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

GEMINI_IMAGE_MODEL_CANDIDATES = (
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
)

GEMINI_EXCLUDED_KEYWORDS = (
    "deep-research",
    "experimental",
    "tts",
    "computer-use",
    "audio",
)
