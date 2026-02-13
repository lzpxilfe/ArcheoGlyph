# ArcheoGlyph

Archaeological Symbol Generator for QGIS.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![QGIS](https://img.shields.io/badge/QGIS-3.0+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Overview
ArcheoGlyph generates factual, map-ready symbols from archaeological artifact and feature images.
It supports deterministic tracing and AI-assisted generation, then applies output directly to QGIS workflows.

## Key Features
- English-first UI for international use
- Styles: `Colored`, `Typology`, `Line`, `Measured`
- Auto Trace with contour + internal feature-line extraction
- Optional SAM backend for segmentation (`OpenCV` fallback included)
- Hugging Face reference-first image generation with modern model fallback
- Google Gemini factual-mode generation with safety checks and deterministic fallback
- Expanded template catalog: 79 built-in archaeological templates
- Save to QGIS symbol library or apply directly to a selected point layer
- Style parameter tab with expression sliders (Factuality, Symbol Looseness, Exaggeration)
- Centralized style-control settings (shared across Auto Trace, HF, Gemini, and Local SD)
- Default fixed symbol size tuned to `10 mm` for better map readability
- Template browser with category filter + search
- Graduated sizing controls with explicit numeric field + class count

## Version
Current plugin code version: `0.1.0`

### 0.1.0 includes
- Migration guard for old settings (legacy HF model IDs are auto-upgraded)
- Stable default HF model: `Qwen/Qwen-Image-Edit-2511`
- English style/template naming
- SAM beginner quick setup controls in settings
- Improved template coverage across artifacts, structures, remains, features, and survey symbols
- Research-inspired typology templates (bronze dagger classes, projectile-point classes, keyhole-tomb variants)

## Installation
1. Copy `ArcheoGlyph` folder to the QGIS plugins directory:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Restart QGIS.
3. Enable: `Plugins > Manage and Install Plugins > ArcheoGlyph`.

## Quick Start
1. Open ArcheoGlyph from the toolbar.
2. Drop an input image (artifact/feature) or select `Use Template`.
3. Choose generation mode and style (`Colored`, `Typology`, `Line`, `Measured`).
4. Click `Generate`.
5. Choose a point layer in `Target Layer`.
6. Click `Save to Library` or `Apply to Layer`.

## AI Configuration

### Hugging Face
1. Create a token at: https://huggingface.co/settings/tokens
2. Add the token in plugin settings.
3. Recommended models:
   - `Qwen/Qwen-Image-Edit-2511`
   - `Qwen/Qwen-Image-Edit-2509`
   - `black-forest-labs/FLUX.2-dev`
   - `black-forest-labs/FLUX.1-Kontext-dev`

### Google Gemini
1. Create an API key at: https://makersuite.google.com/app/apikey
2. Install dependency:
   ```bash
   pip install google-generativeai
   ```
3. Add key in plugin settings.

### Optional SAM for Auto Trace
Use `Settings > Hugging Face > Advanced`:
1. Click `Install SAM Package`
2. Click `Download ViT-B Checkpoint`
3. Click `Auto-Find Downloaded File`
4. Set backend to `SAM (Optional)` and save settings

## Documentation
- `docs/ai_setup_guide.md`
- `docs/local_model_setup.md`

## Template References
Template families in `Use Template` mode are shaped by typology/documentation conventions from:
- National Museum of Korea, *The Korean-type Bronze Daggers and Swords*  
  https://www.nmk.kr/english/research/research_jka_view.do?ar_idx=6
- Utah BLM report archive, *Projectile Point typology examples*  
  https://www.blm.gov/sites/default/files/documents/files/Context_Cultural-Resource-Series_Point-Typologies_Updated-Combined.pdf
- Mozu-Furuichi Kofun Group (UNESCO context), keyhole-tomb morphology references  
  https://www.mozu-furuichi.com/en/about/
- Historic England, archaeological illustration guidance  
  https://historicengland.org.uk/images-books/publications/drawing-archaeological-artefacts/heag044-understanding-archaeological-illustration/

## License
MIT License. See `LICENSE`.

## Contributing
Issues and pull requests are welcome:
- Repository: https://github.com/lzpxilfe/ArcheoGlyph
- Issues: https://github.com/lzpxilfe/ArcheoGlyph/issues

## Citation & Support
If ArcheoGlyph was useful in your research or production workflow, please cite:

> Hwang, J. (2026). ArcheoGlyph v0.1.0: Archaeological Symbol Generator for QGIS. https://github.com/lzpxilfe/ArcheoGlyph

If you like the project, please star the repository:
- https://github.com/lzpxilfe/ArcheoGlyph
