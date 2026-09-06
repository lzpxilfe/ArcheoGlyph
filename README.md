# ArchaeoGlyph

Archaeological Symbol Generator for QGIS.

![Version](https://img.shields.io/badge/version-0.2.0-blue)
![QGIS](https://img.shields.io/badge/QGIS-3.0+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

ArchaeoGlyph converts archaeological photos into factual, map-ready symbols and applies them directly to QGIS layers or the symbol library.

Notes:
- Display/plugin name is `ArchaeoGlyph`.
- Repository/folder path remains `ArcheoGlyph` for compatibility.

## 한국어 안내

ArchaeoGlyph의 화면은 한국어를 지원합니다. QGIS 언어 설정이 한국어이면 자동으로 한국어로 표시되고,
`Settings` 창 위쪽의 `언어` 항목에서 `자동 / English / 한국어`를 직접 고를 수도 있습니다.
언어 변경은 창을 다시 열 때 적용되며, 메뉴와 도구 모음은 QGIS를 다시 시작한 뒤 바뀝니다.

번역 문구는 `i18n_ko.py`에 원문-번역 사전으로 들어 있습니다. 템플릿 이름은 화면에만 번역해서 보여 주고,
내부 식별자와 저장 값은 영어를 그대로 쓰므로 기존에 저장한 프로젝트와 설정이 그대로 동작합니다.
AI에 보내는 프롬프트는 모델 품질 때문에 영어를 유지합니다.

## What It Does

- Offline `Auto Trace` contour and structure extraction
- AI generation backends:
  - Hugging Face (reference-first + model fallback)
  - Google Gemini (factual safeguards + deterministic fallback)
  - Local Stable Diffusion via Automatic1111
- Built-in archaeological templates (artifacts, structures, remains, features, survey)
- Vector (SVG) output with QGIS `param(fill)` / `param(outline)` recolouring
- Direct integration with QGIS symbol library and point-layer rendering
- Graduated size rendering by numeric field (natural breaks, equal interval, quantile)

## Style System (Consolidated)

The UI now focuses on 3 styles:

1. `Simple Symbol`
- Best for distribution maps and legends
- Two-tone fill + bold outline + minimal structural lines

2. `Line`
- Monochrome contour and major internal lines
- Good for clear black/white iconography

3. `Measured`
- Monochrome report-oriented drawing style
- Preserves more documentary line detail

Compatibility note:
- Legacy labels like `Colored` and `Typology` are normalized to `Simple Symbol` behavior in current workflows.

## Quick Presets

Two one-click presets are available in the main dialog:

1. `Simple Symbol Quick Setup`
- Applies stable, map-friendly symbol defaults

2. `Fast Convert Setup`
- Applies speed-priority settings:
  - Auto Trace quality: `Fast`
  - Round mode: `Image-first`
  - Low-res detail boost: `Off`

## Generation Modes

1. `Auto Trace`
- Fastest local path (no API key required)
- Photographs and line drawings (measured drawings, rubbings) are detected
  automatically, or you can force either in `Input type`
- Optional ONNX background removal for difficult photographs

2. `AI (Google Gemini)`
- SVG-oriented factual generation, guided by the measured silhouette and
  stroke paths, with the result validated and sanitised before use

3. `AI (Hugging Face)`
- Reference-first generation through the Hugging Face inference providers

4. `AI (Local Stable Diffusion)`
- Automatic1111 API workflow for local/offline control

5. `Use Template`
- Uses built-in template names and categories

Every mode produces an SVG marker where possible, so symbols stay sharp when
printed and can be recoloured in QGIS through `param(fill)` and
`param(outline)`. Symbols are stored in your QGIS profile under
`archeoglyph/symbols/`, so saved projects keep working.

## Optional Dependencies

Install into the Python that QGIS uses:

```bash
python -m pip install "opencv-python-headless>=4.8,<4.13"   # required for Auto Trace
python -m pip install scikit-image scipy                    # better ink centrelines
python -m pip install onnxruntime                           # background-removal models
python -m pip install vtracer                               # smoother raster tracing
python -m pip install huggingface_hub                       # Hugging Face backend
python -m pip install google-genai                          # Gemini backend
```

## Template Coverage

Programmatic fallback templates include:
- Bronze weapon symbols (`Bronze Sword`, `Bronze Dagger-axe`, `Bronze Spear`)
- Kofun variants (including `Enpun`, `Zenpokouen`, `Hotategai`, `Hofun`, `Yosumi`, etc.)
- Paper-style map/report symbols (`North Arrow`, `Scale Bar`, `Harris Matrix Context`, `Stratigraphic Unit`)
- Pottery section snippets (`Rim/Base/Body Sherd`)

## Performance Tips

If conversion feels slow:

1. Click `Fast Convert Setup`
2. Keep image crop tight around the object
3. Prefer short side around 900-1200 px for routine work
4. Keep `Low-res detail boost` off unless needed
5. Use `Auto Trace` first, then AI only when necessary

## Installation

1. Copy the `ArcheoGlyph` folder into your QGIS plugins directory:
- Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
- Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
- macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

2. Restart QGIS.
3. Enable from `Plugins > Manage and Install Plugins > ArchaeoGlyph`.

## Quick Start

1. Open `ArchaeoGlyph` from the toolbar/menu.
2. Drop an image (or switch to `Use Template`).
3. Select style (`Simple Symbol` / `Line` / `Measured`).
4. Optionally run `Simple Symbol Quick Setup` or `Fast Convert Setup`.
5. Click `Generate`.
6. Click `Save to Library` or `Apply to Layer`.

## AI Setup

### Hugging Face

1. Create a token: https://huggingface.co/settings/tokens
2. Enter token in plugin Settings
3. Use:
- `Check Latest Models` (preview)
- `Apply Latest Recommended Models`

### Google Gemini

1. Create API key: https://aistudio.google.com/apikey
2. Install dependency:

```bash
pip install google-genai
```

3. Enter API key in plugin Settings

### Local Stable Diffusion (Automatic1111)

1. Start WebUI with API enabled
2. Set server URL in Settings (default: `http://127.0.0.1:7860`)

## Repository

- GitHub: https://github.com/lzpxilfe/ArcheoGlyph.git
- Issues: https://github.com/lzpxilfe/ArcheoGlyph/issues

## Citation
[![Cite this repository](https://img.shields.io/badge/Cite_this-repository-2ea44f?logo=github)](https://github.com/lzpxilfe/ArcheoGlyph)
[![Star this repository](https://img.shields.io/github/stars/lzpxilfe/ArcheoGlyph?style=social)](https://github.com/lzpxilfe/ArcheoGlyph)

인용 메타데이터는 [CITATION.cff](CITATION.cff)에 보관합니다.


```bibtex
@software{hwang2026archaeoglyph,
  author = {lzpxilfe (balguljang2)},
  title = {ArchaeoGlyph: Archaeological Symbol Generator for QGIS},
  year = {2026},
  url = {https://github.com/lzpxilfe/ArcheoGlyph.git}
}
```

## Support

- Ko-fi: https://ko-fi.com/lzpxilfe

## License

MIT License.
