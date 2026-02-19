# ArchaeoGlyph

Archaeological Symbol Generator for QGIS.

![Version](https://img.shields.io/badge/version-0.1.1-blue)
![QGIS](https://img.shields.io/badge/QGIS-3.0+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

ArchaeoGlyph generates factual, map-ready symbols from archaeological artifact and feature images, then applies them directly to QGIS layers or the symbol library.

Note:
- Display name is `ArchaeoGlyph`.
- Plugin folder/repository path remains `ArcheoGlyph` for compatibility.

## Core Features

- `Auto Trace` factual contour + internal line extraction (offline)
- AI generation:
  - Hugging Face (reference-first, model fallback)
  - Google Gemini (factual guardrails + deterministic fallback)
  - Local Stable Diffusion (Automatic1111)
- Styles:
  - `Colored`
  - `Typology`
  - `Line`
  - `Measured`
- Optional SAM segmentation backend with OpenCV fallback
- Built-in archaeological template catalog (artifacts, structures, remains, features, survey)
- Style parameter controls (`Factuality`, `Symbolic Looseness`, `Exaggeration`)
- Save symbol to QGIS style library or apply directly to point layers
- Graduated symbol sizing (field + class controls)
- Latest-model management in Settings:
  - `Check Latest Models` (preview)
  - `Apply Latest Recommended Models`

## Version

Current plugin code version: `0.1.1`

### 0.1.1 highlights

- Branding unified to `ArchaeoGlyph` (display/UI/metadata)
- Metadata author updated to `lzpxilfe (balguljang2)`
- Latest-model UX improved (`Check` vs `Apply`)
- Stable defaults and model-refresh behavior improvements
- Low-quality round artifact extraction improvements in Auto Trace

## Installation

1. Copy the `ArcheoGlyph` plugin folder into:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Restart QGIS.
3. Enable from `Plugins > Manage and Install Plugins > ArchaeoGlyph`.

## Quick Start

1. Open `ArchaeoGlyph` from toolbar/menu.
2. Drop an input image or choose `Use Template`.
3. Pick generation mode + style.
4. Click `Generate`.
5. Click `Save to Library` or `Apply to Layer`.

## AI Setup

### Hugging Face

1. Create token: https://huggingface.co/settings/tokens
2. Enter token in plugin Settings.
3. Use `Check Latest Models` then `Apply Latest Recommended Models`.

### Google Gemini

1. Create key: https://makersuite.google.com/app/apikey
2. Install dependency:

```bash
pip install google-generativeai
```

3. Enter key in plugin Settings.

### Local Stable Diffusion (Automatic1111)

1. Start WebUI with API enabled.
2. Set server URL in Settings (`http://127.0.0.1:7860` by default).

## Repository

- GitHub: https://github.com/lzpxilfe/ArcheoGlyph.git
- Issues: https://github.com/lzpxilfe/ArcheoGlyph/issues

## Citation & Star

If you find this repository useful, please consider giving it a star ⭐ and citing it in your work:

```bibtex
@software{hwang2026archaeoglyph,
  author = {lzpxilfe (balguljang2)},
  title = {ArchaeoGlyph: Archaeological Symbol Generator for QGIS},
  year = {2026},
  url = {https://github.com/lzpxilfe/ArcheoGlyph.git}
}
```

## Support (Ko-fi)

If ArchaeoGlyph helps your research or workflow, you can support development here:  
https://ko-fi.com/lzpxilfe

## License

MIT License.
