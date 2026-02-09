# Local Stable Diffusion Setup Guide

This guide helps you set up a local Stable Diffusion server for offline symbol generation.

## Option 1: Automatic1111 WebUI (Recommended)

### Prerequisites

- **GPU**: NVIDIA GPU with at least 6GB VRAM (8GB+ recommended)
- **Python**: Version 3.10.6 (exact version recommended)
- **Git**: For cloning the repository

### Installation Steps

1. **Install Python 3.10.6**
   ```
   Download from: https://www.python.org/downloads/release/python-3106/
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ```

3. **Download a model**
   
   Recommended models for icon generation:
   
   | Style | Model | Link |
   |-------|-------|------|
   | Cute/Kawaii | Anything V5 | [Civitai](https://civitai.com/models/9409) |
   | Minimal | Deliberate V2 | [Civitai](https://civitai.com/models/4823) |
   | Classic | Realistic Vision | [Civitai](https://civitai.com/models/4201) |
   
   Place `.safetensors` file in `models/Stable-diffusion/`

4. **Enable API mode**
   
   Edit `webui-user.bat` (Windows) or `webui-user.sh` (Linux/Mac):
   ```
   set COMMANDLINE_ARGS=--api
   ```

5. **Run the server**
   ```bash
   # Windows
   webui-user.bat
   
   # Linux/Mac
   ./webui.sh
   ```

6. **Configure ArcheoGlyph**
   - Server URL: `http://127.0.0.1:7860`
   - Backend: `Automatic1111`

## Option 2: ComfyUI

> ⚠️ ComfyUI support is coming soon. Use Automatic1111 for now.

## Troubleshooting

### Connection Refused
- Ensure the server is running
- Check firewall settings
- Verify the port (default: 7860)

### Slow Generation
- Use a GPU with more VRAM
- Reduce image size in settings
- Try `--xformers` flag for memory optimization

### Out of Memory
- Add `--lowvram` or `--medvram` flag
- Reduce batch size to 1
- Close other GPU-intensive applications

## Performance Tips

1. **First run**: Initial setup downloads ~4GB of dependencies
2. **Model loading**: First generation is slower due to model loading
3. **Optimal settings**: 
   - Steps: 20-30
   - Size: 256x256
   - CFG Scale: 7

## Need Help?

- [Automatic1111 Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
- [Stable Diffusion Reddit](https://www.reddit.com/r/StableDiffusion/)
