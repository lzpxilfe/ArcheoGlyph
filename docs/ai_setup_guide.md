# AI Setup Guide for ArcheoGlyph

This guide explains how to configure AI-powered symbol generation in ArcheoGlyph.

## Quick Start

Choose one of the following options:

| Option | Requirement | Best For |
|--------|-------------|----------|
| Google Gemini | API Key (free tier available) | Beginners, online use |
| Local Stable Diffusion | GPU + 6GB VRAM | Offline use, customization |
| Templates | None | Quick start, no AI needed |

---

## Option 1: Google Gemini (Recommended)

### Step 1: Get API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key

### Step 2: Install Python Package

Open QGIS Python Console (`Plugins > Python Console`) and run:

```python
import subprocess
subprocess.check_call(['pip', 'install', 'google-generativeai'])
```

Or run in your system terminal:

```bash
pip install google-generativeai
```

### Step 3: Configure in Plugin

The plugin will prompt you for the API key on first use, or you can set it manually:

```python
from qgis.PyQt.QtCore import QSettings
QSettings().setValue('ArcheoGlyph/gemini_api_key', 'YOUR_API_KEY_HERE')
```

### Usage Limits (Free Tier)

- 60 requests per minute
- 1500 requests per day
- Sufficient for normal archaeological work

---

## Option 2: Local Stable Diffusion

### Prerequisites

- NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)
- Python 3.10.6
- ~10GB disk space

### Step 1: Install Automatic1111 WebUI

```bash
# Clone repository
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Windows
webui-user.bat

# Linux/Mac
./webui.sh
```

### Step 2: Download a Model

Place `.safetensors` model files in `models/Stable-diffusion/`

**Recommended models for icon generation:**

| Style | Model | Download |
|-------|-------|----------|
| Cute/Kawaii | Anything V5 | [Civitai](https://civitai.com/models/9409) |
| Minimal | Deliberate V2 | [Civitai](https://civitai.com/models/4823) |
| Classic | Realistic Vision V5 | [Civitai](https://civitai.com/models/4201) |

### Step 3: Enable API Mode

Edit `webui-user.bat` (Windows) or `webui-user.sh` (Linux/Mac):

```bash
# Add --api flag
set COMMANDLINE_ARGS=--api
```

Then restart the server.

### Step 4: Configure in Plugin

Default settings work if server runs on localhost:

- Server URL: `http://127.0.0.1:7860`
- Backend: `automatic1111`

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure server is running |
| Out of memory | Add `--lowvram` flag |
| Slow generation | Check GPU drivers, use `--xformers` |

---

## Option 3: Use Templates (No AI Required)

If you don't want to set up AI:

1. Select **"Use Template"** mode in the plugin
2. Choose artifact type (Pottery, Stone Tools, etc.)
3. Customize color
4. Generate!

Templates are built into the plugin and work offline immediately.

---

## Need Help?

- [ArcheoGlyph GitHub Issues](https://github.com/lzpxilfe/ArcheoGlyph/issues)
- [Google AI Studio Help](https://ai.google.dev/docs)
- [Automatic1111 Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
