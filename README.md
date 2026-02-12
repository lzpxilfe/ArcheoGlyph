# ArcheoGlyph

**Archaeological Symbol Generator for QGIS**

Generate cute, visually accessible symbols from archaeological artifact and feature images.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![QGIS](https://img.shields.io/badge/QGIS-3.0+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Features

- 🎨 **AI Symbol Generation** - Google Gemini (Vector Game Style) or Hugging Face (Icon Style)
- 📐 **Multiple Styles** - colored Silhouette (Game Asset), Line Drawing, Publication (Stippling)
- 🖌️ **Auto-Color** - Automatically detects artifact color from image
- 🖼️ **Template Library** - Built-in templates for common artifact types
- 🎯 **Color Adjustment** - HSL-based color customization
- 📊 **Proportional Sizing** - Data-driven symbol scaling
- 💾 **QGIS Integration** - Save to symbol library, apply to layers

## Installation

1. Copy `ArcheoGlyph` folder to QGIS plugins directory:
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

2. Restart QGIS

3. Enable: `Plugins > Manage and Install Plugins > ArcheoGlyph`

## Quick Start

1. Click ArcheoGlyph icon in toolbar
2. Drop an artifact image or select template
3. Choose style and color
4. Click **Generate**
5. **Save to Library** or **Apply to Layer**

## AI Configuration

### Google Gemini (Recommended)

1. Get free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Install package:
   ```bash
   pip install google-generativeai
   ```
3. Enter API key when prompted by plugin

### Local Stable Diffusion

For offline use, see [AI Setup Guide](docs/ai_setup_guide.md)

1. Install [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. Download a model (Anything V5 recommended)
3. Run with `--api` flag
4. Configure server URL in plugin

### Hugging Face (Free)
1. Get free Token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Enter Token in plugin settings
3. **Default Model**: `stabilityai/stable-diffusion-2-1` (Automatically configured)

### No AI Required

Select "Use Template" mode - works immediately with built-in SVG templates.

## Documentation

- [AI Setup Guide](docs/ai_setup_guide.md) - Detailed AI configuration
- [Local Model Setup](docs/local_model_setup.md) - Stable Diffusion installation

## License

MIT License - See [LICENSE](LICENSE)

## Author

**Jinseo Hwang** (황진서)

## Contributing

Issues and pull requests welcome on GitHub!
