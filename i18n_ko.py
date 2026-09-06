# -*- coding: utf-8 -*-
"""
Korean catalogue.

Keys are the exact English source strings passed to ``tr()``; values are the
Korean translations. ``tests/test_i18n.py`` checks that every string the code
translates has an entry here, that no entry is empty or orphaned, and that
``{placeholder}`` names match on both sides.

Terminology follows 한국고고학사전 (국립문화유산연구원) usage where the plugin has an
equivalent concept: artifact 유물, feature 유구, rubbing 탁본, measured drawing 실측도.

A few entries are deliberately identical to their English source: model IDs,
URLs, unit suffixes and placeholder examples that would be wrong to localise.
"""

CATALOG = {
    # -- Plugin menu and window ------------------------------------------
    "&ArchaeoGlyph": "&ArchaeoGlyph",
    "ArchaeoGlyph Symbol Generator": "ArchaeoGlyph 심볼 생성기",
    "ArchaeoGlyph v{version} - Symbol Generator": "ArchaeoGlyph v{version} - 심볼 생성기",

    # -- Main dialog: input and preview ----------------------------------
    "Drop Image Here\nor Click to Browse": "이미지를 끌어다 놓거나\n클릭해서 선택하세요",
    "Select Artifact Image": "유물 이미지 선택",
    "Preview": "미리보기",
    "Input Image": "입력 이미지",
    "Use a representative photo of the artifact or archaeological feature.\n"
    "Clean backgrounds produce better silhouettes and internal detail lines.":
        "유물이나 유구를 대표하는 사진을 사용하세요.\n"
        "배경이 깨끗할수록 외곽선과 내부 세부선이 잘 나옵니다.",
    "<i>Tip: Use a clear photo with a clean background for best results.</i>":
        "<i>도움말: 배경이 깨끗하고 선명한 사진일수록 결과가 좋습니다.</i>",
    "Clear": "지우기",
    "Generated Symbol": "생성된 심볼",

    # -- Main dialog: generation mode ------------------------------------
    "Generation Mode": "생성 방식",
    "Auto Trace": "자동 추적",
    "AI (Google Gemini)": "AI (Google Gemini)",
    "AI (Hugging Face)": "AI (Hugging Face)",
    "AI (Local Stable Diffusion)": "AI (로컬 Stable Diffusion)",
    "Use Template": "템플릿 사용",

    # -- Main dialog: style ----------------------------------------------
    "Style": "표현 방식",
    "Simple Symbol": "단순 심볼",
    "Line": "선묘",
    "Measured": "실측도",
    "Simple Symbol uses a two-tone fill with bold outlines for readable distribution maps.":
        "단순 심볼은 두 가지 톤의 면과 굵은 외곽선으로 분포도에서 잘 읽히도록 만듭니다.",
    "Simple Symbol Quick Setup": "단순 심볼 빠른 설정",
    "Applies a stable preset to turn photos into simple map symbols.":
        "사진을 단순한 지도 심볼로 바꾸는 안정적인 설정을 적용합니다.",
    "Fast Convert Setup": "빠른 변환 설정",
    "Applies speed-priority settings for quick conversion.":
        "속도를 우선하는 설정을 적용합니다.",
    "Mirror symmetry": "좌우 대칭",
    "Produces a bilaterally symmetrical symbol by mirroring the contour.":
        "윤곽을 반사해 좌우 대칭인 심볼을 만듭니다.",
    "Input type:": "입력 종류:",
    "Auto detect": "자동 판별",
    "Photograph": "사진",
    "Drawing / rubbing": "실측도 / 탁본",
    "Drawings and rubbings are traced from their ink strokes; photographs\n"
    "go through background removal first.":
        "실측도와 탁본은 먹선을 따라 추적하고, 사진은 먼저 배경을 제거합니다.",
    "Add schematic structure lines": "도식 구조선 추가",
    "Off by default: only lines observed in the image are drawn.\n"
    "Enable to add conventional rim/shoulder, centre and terminal lines.":
        "기본값은 꺼짐이며, 이미지에서 실제로 확인되는 선만 그립니다.\n"
        "켜면 구연부·동체부, 중심선, 끝선 같은 관용적인 선을 덧붙입니다.",
    "Low-res detail boost (upscale)": "저해상도 세부 보정 (확대)",
    "Auto Trace only. Aggressively upscales low-resolution images before contour analysis.":
        "자동 추적에만 적용됩니다. 윤곽 분석 전에 저해상도 이미지를 크게 확대합니다.",
    "Auto Trace quality:": "자동 추적 품질:",
    "Fast (speed priority)": "빠르게 (속도 우선)",
    "Precise (detail priority)": "정밀하게 (세부 우선)",
    "Round artifact mode:": "원형 유물 처리:",
    "Image-first (recommended)": "이미지 우선 (권장)",
    "Hybrid (rescue on failure)": "혼합 (실패 시 보완)",
    "Structure-first (stable)": "구조 우선 (안정적)",
    "Basic": "기본",
    "Parameters": "세부 조정",
    "Adjust expression balance for symbol output.": "심볼 표현의 균형을 조정합니다.",
    "Factuality:": "사실성:",
    "0 = expressive symbol, 100 = measured/documentary.":
        "0 = 표현적인 심볼, 100 = 실측·기록 중심.",
    "Symbol Looseness:": "단순화 정도:",
    "0 = tight measured shape, 100 = loose symbolic simplification.":
        "0 = 실측에 가까운 형태, 100 = 과감한 기호적 단순화.",
    "Exaggeration:": "강조:",
    "0 = none, 100 = strong stylization and simplified emphasis.":
        "0 = 없음, 100 = 강한 양식화와 단순화된 강조.",

    # -- Main dialog: templates ------------------------------------------
    "Template Type": "템플릿 종류",
    "Category:": "분류:",
    "Filter templates (e.g., dagger, tomb, survey)":
        "템플릿 검색 (예: 동검, 무덤, 지표조사)",
    "No templates match current filter": "조건에 맞는 템플릿이 없습니다",

    # -- Main dialog: colour ---------------------------------------------
    "Color": "색상",
    "Override Color": "색상 지정",
    "If unchecked, the symbol will use the artifact's natural colors.":
        "체크하지 않으면 유물 본래의 색을 사용합니다.",
    "Pick Color": "색상 선택",
    "Pick from Image": "이미지에서 추출",
    "Click Image to Pick": "이미지를 클릭하세요",

    # -- Main dialog: size scaling ---------------------------------------
    "Size Scaling": "크기 조정",
    "Mode:": "방식:",
    "Fixed Size": "고정 크기",
    "By Data Count (Natural Breaks)": "데이터 값 기준 (자연 분류)",
    "By Data Count (Equal Interval)": "데이터 값 기준 (등간격)",
    "By Data Count (Quantile)": "데이터 값 기준 (등개수)",
    "Min:": "최소:",
    "Max:": "최대:",
    "Size Field:": "크기 필드:",
    "Choose a numeric attribute for graduated size. Use Auto to pick the first numeric field.":
        "단계별 크기에 쓸 숫자 속성을 고르세요. 자동을 고르면 첫 번째 숫자 필드를 씁니다.",
    "Auto (first numeric field)": "자동 (첫 번째 숫자 필드)",
    "Classes:": "단계 수:",
    "Number of size classes for graduated rendering.": "단계별 표현에 사용할 크기 단계 수입니다.",

    # -- Main dialog: layer, prompt, buttons ------------------------------
    "Target Layer": "대상 레이어",
    "Choose the point layer that will receive the generated symbol.":
        "생성한 심볼을 적용할 포인트 레이어를 고르세요.",
    "Refresh": "새로 고침",
    "No point layers available": "사용할 수 있는 포인트 레이어가 없습니다",
    "Text Prompt": "텍스트 프롬프트",
    "Enter a description for the icon (e.g., 'ancient pottery shard')":
        "아이콘 설명을 입력하세요 (예: 'ancient pottery shard')",
    "Optional: style note (e.g., 'typology plate icon with clear shoulder line')":
        "선택 사항: 표현 지침 (예: 'typology plate icon with clear shoulder line')",
    "Optional: factual note (e.g., 'preserve chips and asymmetry, no decorative background')":
        "선택 사항: 사실성 지침 (예: 'preserve chips and asymmetry, no decorative background')",
    "Optional: local SD prompt hint (e.g., 'flat archaeological icon, muted tones')":
        "선택 사항: 로컬 SD 프롬프트 (예: 'flat archaeological icon, muted tones')",
    "Generate": "생성",
    "Cancel": "취소",
    "Stop the running generation.": "진행 중인 생성을 중단합니다.",
    "Save to Library": "라이브러리에 저장",
    "Apply to Layer": "레이어에 적용",
    "Settings": "설정",
    "Symbol name:": "심볼 이름:",

    # -- Main dialog: messages -------------------------------------------
    "HF custom prompt active: text guidance will influence stylization.":
        "Hugging Face 사용자 프롬프트가 적용되어 표현 방식에 반영됩니다.",
    "HF default factual guidance active.": "Hugging Face 기본 사실성 지침이 적용됩니다.",
    "No Image": "이미지 없음",
    "Please select an input image first.": "먼저 입력 이미지를 선택하세요.",
    "Quota Exceeded": "사용량 초과",
    "Google Gemini quota is currently exhausted for this API key/project.\n\n"
    "Actions:\n1. Wait for quota reset and retry.\n"
    "2. Use Auto Trace or Hugging Face in the meantime.\n"
    "3. Check quota/billing in Google AI Studio.":
        "이 API 키/프로젝트의 Google Gemini 사용량이 모두 소진되었습니다.\n\n"
        "해결 방법:\n1. 사용량이 초기화된 뒤 다시 시도하세요.\n"
        "2. 그동안 자동 추적이나 Hugging Face를 사용하세요.\n"
        "3. Google AI Studio에서 사용량과 결제 설정을 확인하세요.",
    "Error": "오류",
    "Generation failed: {message}": "생성 실패: {message}",
    "Failed": "실패",
    "Generation returned no result.": "생성 결과가 없습니다.",
    "Generated symbol could not be rendered.": "생성된 심볼을 그릴 수 없습니다.",
    "No Template": "템플릿 없음",
    "Adjust template filters and select a valid template.":
        "템플릿 검색 조건을 조정하고 사용할 템플릿을 선택하세요.",
    "No Symbol": "심볼 없음",
    "Please generate a symbol first.": "먼저 심볼을 생성하세요.",
    "Saved": "저장됨",
    "Symbol saved to QGIS library as '{name}'.":
        "심볼을 '{name}' 이름으로 QGIS 라이브러리에 저장했습니다.",
    "Failed to save symbol. See the ArchaeoGlyph message log.":
        "심볼을 저장하지 못했습니다. ArchaeoGlyph 로그 메시지를 확인하세요.",
    "No Layer": "레이어 없음",
    "Please choose a point layer in Target Layer.": "대상 레이어에서 포인트 레이어를 고르세요.",
    "Applied": "적용됨",
    "Symbol applied to layer: {layer}": "심볼을 레이어에 적용했습니다: {layer}",
    "Failed to apply symbol to layer. See the ArchaeoGlyph message log.":
        "심볼을 레이어에 적용하지 못했습니다. ArchaeoGlyph 로그 메시지를 확인하세요.",

    # -- Settings dialog: frame ------------------------------------------
    "ArchaeoGlyph Settings & Help": "ArchaeoGlyph 설정 및 도움말",
    "<h2>ArchaeoGlyph Settings</h2>": "<h2>ArchaeoGlyph 설정</h2>",
    "Google Gemini": "Google Gemini",
    "Hugging Face": "Hugging Face",
    "Local SD": "로컬 SD",
    "Quick Start": "빠른 시작",
    "Help": "도움말",
    "Save Settings": "설정 저장",
    "Close": "닫기",
    "Warning: {text}": "경고: {text}",

    # -- Settings dialog: Hugging Face -----------------------------------
    "<h3>Hugging Face Inference API</h3><p>Use open-source AI models through Hugging Face "
    "inference.Requires a Hugging Face account and token.</p>":
        "<h3>Hugging Face 추론 API</h3><p>Hugging Face 추론으로 오픈소스 AI 모델을 사용합니다. "
        "Hugging Face 계정과 토큰이 필요합니다.</p>",
    "API Token": "API 토큰",
    '1. Get a token from: <a href="https://huggingface.co/settings/tokens">'
    'huggingface.co/settings/tokens</a>':
        '1. 토큰 발급: <a href="https://huggingface.co/settings/tokens">'
        'huggingface.co/settings/tokens</a>',
    "hf_...": "hf_...",
    "Show Token": "토큰 보기",
    "Model Selection": "모델 선택",
    "Specify the Model ID to use (e.g., '{model}' or 'Qwen/Qwen-Image'). If a model returns "
    "403/404/503, the plugin automatically tries modern fallback models.\n"
    "Use 'Check Latest Models' to preview recommendations, then 'Apply Latest Recommended "
    "Models' to apply without Python console checks.":
        "사용할 모델 ID를 입력하세요 (예: '{model}' 또는 'Qwen/Qwen-Image'). 모델이 "
        "403/404/503을 반환하면 최신 대체 모델을 자동으로 시도합니다.\n"
        "'최신 모델 확인'으로 추천 목록을 본 뒤 '최신 추천 모델 적용'을 누르면 파이썬 콘솔 "
        "없이 바로 적용됩니다.",
    "organization/model-name": "organization/model-name",
    "Check Latest Models": "최신 모델 확인",
    "Apply Latest Recommended Models": "최신 추천 모델 적용",
    "Auto-refresh model recommendations weekly": "매주 추천 모델 자동 갱신",
    "Automatic refresh is disabled.": "자동 갱신이 꺼져 있습니다.",
    "Checking latest model recommendations...": "최신 추천 모델을 확인하는 중...",
    "Latest-model check is up to date (last check: {when}).":
        "최신 모델 확인이 최신 상태입니다 (마지막 확인: {when}).",
    "Latest Models Applied": "최신 모델 적용됨",
    "Latest Model Refresh Failed": "최신 모델 갱신 실패",
    "Test Hugging Face Connection": "Hugging Face 연결 테스트",
    "No Token": "토큰 없음",
    "Please enter Hugging Face token.": "Hugging Face 토큰을 입력하세요.",
    "Connected": "연결됨",
    "Success": "성공",
    "Connected with model: {model}": "모델에 연결했습니다: {model}",
    "Loading model...": "모델을 불러오는 중...",
    "Loading": "불러오는 중",
    "Connected, but model is initializing: {model}":
        "연결되었지만 모델을 준비하는 중입니다: {model}",
    "Invalid token": "잘못된 토큰",
    "Invalid Token": "잘못된 토큰",
    "Please check your Hugging Face token.": "Hugging Face 토큰을 확인하세요.",
    "Model access denied (403)": "모델 접근 거부 (403)",
    "Model Access Denied": "모델 접근 거부",
    "Model terms may need acceptance on Hugging Face, or the model is restricted.":
        "Hugging Face에서 모델 이용 약관에 동의해야 하거나, 접근이 제한된 모델입니다.",
    "Model not found (404)": "모델을 찾을 수 없음 (404)",
    "Model Not Found": "모델을 찾을 수 없음",
    "No candidate model was found.\nTry '{model}' or 'Qwen/Qwen-Image'.":
        "사용할 수 있는 모델을 찾지 못했습니다.\n'{model}' 또는 'Qwen/Qwen-Image'를 시도해 보세요.",
    "Connection Failed": "연결 실패",
    "Unexpected test result.": "예상하지 못한 테스트 결과입니다.",
    "HF: Overlay factual linework (stricter, may look similar to Auto Trace)":
        "Hugging Face: 사실적인 선묘 덧그리기 (엄격함, 자동 추적과 비슷해질 수 있음)",

    # -- Settings dialog: Auto Trace backend -----------------------------
    "Advanced (Optional)": "고급 설정 (선택)",
    "Auto Trace separates the artifact from its background. OpenCV needs no download but "
    "struggles with gradients, shadows and grey-on-grey photos; a background-removal model "
    "handles those. SAM is also supported.":
        "자동 추적은 유물을 배경에서 분리합니다. OpenCV는 내려받을 것이 없지만 그러데이션, "
        "그림자, 회색 배경 위의 회색 유물에 약합니다. 배경 제거 모델은 이런 경우를 잘 "
        "처리합니다. SAM도 사용할 수 있습니다.",
    "Auto Trace Backend:": "자동 추적 방식:",
    "Auto (recommended: best available model, else OpenCV)":
        "자동 (권장: 사용 가능한 최적 모델, 없으면 OpenCV)",
    "OpenCV only (no extra download)": "OpenCV만 사용 (추가 내려받기 없음)",
    "Background-removal model (ONNX)": "배경 제거 모델 (ONNX)",
    "SAM (optional)": "SAM (선택)",
    "Background-removal model (recommended for photographs)":
        "배경 제거 모델 (사진에 권장)",
    "Downloaded once, verified by size and SHA-256, and stored in your QGIS profile. "
    "Runs on the CPU; no image ever leaves your machine.":
        "한 번만 내려받아 크기와 SHA-256으로 검증한 뒤 QGIS 프로필에 저장합니다. "
        "CPU에서 실행되며 이미지는 컴퓨터 밖으로 나가지 않습니다.",
    "Model:": "모델:",
    "Install onnxruntime": "onnxruntime 설치",
    "Installs the CPU inference runtime with pip.": "pip으로 CPU 추론 런타임을 설치합니다.",
    "Download model": "모델 내려받기",
    "Download {label}?\n\nAbout {size} MB, stored in your QGIS profile and verified by SHA-256.":
        "{label}을(를) 내려받을까요?\n\n약 {size} MB이며, QGIS 프로필에 저장하고 SHA-256으로 "
        "검증합니다.",
    "Downloading {filename}...": "{filename} 내려받는 중...",
    "Verify": "검증",
    "Re-check the stored file against its published SHA-256.":
        "저장된 파일을 공개된 SHA-256과 다시 대조합니다.",
    "Model ready": "모델 준비됨",
    "Download failed": "내려받기 실패",
    "Model verified": "모델 검증됨",
    "{filename} matches its published SHA-256.\n\n{path}":
        "{filename}이(가) 공개된 SHA-256과 일치합니다.\n\n{path}",
    "Model does not match": "모델이 일치하지 않음",
    "The stored file does not match its published checksum. Delete it and download again.":
        "저장된 파일이 공개된 체크섬과 다릅니다. 파일을 지우고 다시 내려받으세요.",
    "Verify failed": "검증 실패",
    "Background-removal model not ready": "배경 제거 모델이 준비되지 않음",
    "Installed": "설치됨",
    "Not installed": "설치되지 않음",
    "Legacy only": "구버전만 설치됨",
    "Copy diagnostics": "진단 정보 복사",
    "Copy a plain-text report of versions, installed packages and models.\n"
    "Paste it into a bug report when something does not work.":
        "버전, 설치된 패키지, 모델 정보를 텍스트로 복사합니다.\n"
        "문제가 생겼을 때 버그 신고에 붙여 넣으세요.",
    "Diagnostics copied": "진단 정보 복사됨",
    "The report was copied to the clipboard.": "보고서를 클립보드에 복사했습니다.",
    "Diagnostics failed": "진단 실패",

    # -- Settings dialog: SAM --------------------------------------------
    "SAM Model Type:": "SAM 모델 종류:",
    "SAM1 ViT-B (local checkpoint)": "SAM1 ViT-B (로컬 체크포인트)",
    "SAM1 ViT-L (local checkpoint)": "SAM1 ViT-L (로컬 체크포인트)",
    "SAM1 ViT-H (local checkpoint)": "SAM1 ViT-H (로컬 체크포인트)",
    "SAM3 Large (HF, latest, may be gated)": "SAM3 Large (HF, 최신, 접근 제한될 수 있음)",
    "SAM2.1 Large (HF)": "SAM2.1 Large (HF)",
    "SAM2.1 Small (HF)": "SAM2.1 Small (HF)",
    "Auto-detected ({model})": "자동 감지됨 ({model})",
    "SAM Checkpoint:": "SAM 체크포인트:",
    "Path to sam_vit_*.pth (SAM1 only)": "sam_vit_*.pth 경로 (SAM1 전용)",
    "Browse...": "찾아보기...",
    "SAM Quick Setup (Recommended for first-time users):":
        "SAM 빠른 설정 (처음 사용할 때 권장):",
    "Install SAM Packages": "SAM 패키지 설치",
    "Install SAM packages now?\n\nThis installs:\n- segment-anything (SAM1 local checkpoint)\n"
    "- transformers + huggingface_hub (SAM2/3 via HF)\n\nNote: SAM still needs 'torch'. "
    "If torch is missing, install it first (CPU build is okay for basic use).":
        "지금 SAM 패키지를 설치할까요?\n\n설치 항목:\n- segment-anything (SAM1 로컬 체크포인트)\n"
        "- transformers + huggingface_hub (HF 기반 SAM2/3)\n\n참고: SAM에는 'torch'도 "
        "필요합니다. torch가 없다면 먼저 설치하세요 (기본 사용에는 CPU 빌드로 충분합니다).",
    "Installing...": "설치하는 중...",
    "Installing SAM packages...": "SAM 패키지를 설치하는 중...",
    "Installing SAM: {line}": "SAM 설치 중: {line}",
    "SAM packages installed successfully.\nIf this is first-time setup, restart QGIS.":
        "SAM 패키지를 설치했습니다.\n처음 설정하는 경우 QGIS를 다시 시작하세요.",
    "Install Failed": "설치 실패",
    "Could not install SAM packages automatically.\n\nManual command:\n{command}":
        "SAM 패키지를 자동으로 설치하지 못했습니다.\n\n수동 설치 명령:\n{command}",
    "SAM install process error: {error}": "SAM 설치 과정 오류: {error}",
    "Download ViT-B Checkpoint": "ViT-B 체크포인트 내려받기",
    "Download Started": "내려받기 시작됨",
    "Browser download opened for sam_vit_b_01ec64.pth.\n"
    "After download, click 'Auto-Find Downloaded File'.":
        "브라우저에서 sam_vit_b_01ec64.pth 내려받기를 열었습니다.\n"
        "내려받은 뒤 '내려받은 파일 자동 찾기'를 누르세요.",
    "Auto-Find Downloaded File": "내려받은 파일 자동 찾기",
    "Checkpoint Found": "체크포인트 찾음",
    "SAM checkpoint found and selected:\n{path}":
        "SAM 체크포인트를 찾아 선택했습니다:\n{path}",
    "Not Found": "찾을 수 없음",
    "No SAM checkpoint was found in common folders.\nClick 'Download ViT-B Checkpoint' first.":
        "일반적인 폴더에서 SAM 체크포인트를 찾지 못했습니다.\n"
        "먼저 'ViT-B 체크포인트 내려받기'를 누르세요.",
    "Open SAM2/3 Models": "SAM2/3 모델 열기",
    "SAM Setup Guide": "SAM 설정 안내",
    "SAM Quick Guide": "SAM 간단 안내",
    "SAM setup (beginner):\n\nTip: In Hugging Face tab, click 'Apply Latest Recommended "
    "Models' first.\n\nOption A: SAM2.1/SAM3 via Hugging Face (easiest)\n"
    "1. Keep 'Auto Trace Backend' = SAM (Optional)\n"
    "2. Choose a SAM2.1/SAM3 model in 'SAM Model Type'\n3. Click 'Install SAM Packages'\n"
    "4. Save Settings and restart QGIS\n\nOption B: SAM1 local checkpoint\n"
    "1. Keep 'Auto Trace Backend' = SAM (Optional)\n2. Choose SAM1 ViT model type\n"
    "3. Click 'Install SAM Packages'\n4. Click 'Download ViT-B Checkpoint'\n"
    "5. Click 'Auto-Find Downloaded File'\n6. Save Settings and restart QGIS\n\n"
    "If SAM is not ready, ArcheoGlyph automatically falls back to OpenCV.":
        "SAM 설정 (처음 사용자용):\n\n도움말: Hugging Face 탭에서 '최신 추천 모델 적용'을 "
        "먼저 누르세요.\n\n방법 A: Hugging Face 기반 SAM2.1/SAM3 (가장 간단)\n"
        "1. '자동 추적 방식'을 SAM (선택)으로 둡니다\n"
        "2. 'SAM 모델 종류'에서 SAM2.1/SAM3 모델을 고릅니다\n3. 'SAM 패키지 설치'를 누릅니다\n"
        "4. 설정을 저장하고 QGIS를 다시 시작합니다\n\n방법 B: SAM1 로컬 체크포인트\n"
        "1. '자동 추적 방식'을 SAM (선택)으로 둡니다\n2. SAM1 ViT 모델 종류를 고릅니다\n"
        "3. 'SAM 패키지 설치'를 누릅니다\n4. 'ViT-B 체크포인트 내려받기'를 누릅니다\n"
        "5. '내려받은 파일 자동 찾기'를 누릅니다\n6. 설정을 저장하고 QGIS를 다시 시작합니다\n\n"
        "SAM을 쓸 수 없으면 ArcheoGlyph가 자동으로 OpenCV로 되돌아갑니다.",
    "Select SAM Checkpoint": "SAM 체크포인트 선택",
    "SAM ready: dependencies and checkpoint detected.":
        "SAM 준비 완료: 필요한 패키지와 체크포인트를 확인했습니다.",
    "SAM ready (HF): {model} (checkpoint not required).":
        "SAM 준비 완료 (HF): {model} (체크포인트 불필요).",
    "SAM Package Missing": "SAM 패키지 없음",
    "SAM2/3 mode needs torch + transformers.\nSwitching backend to OpenCV for now.\n\n"
    "Use 'Install SAM Packages' first.":
        "SAM2/3에는 torch와 transformers가 필요합니다.\n지금은 OpenCV로 전환합니다.\n\n"
        "먼저 'SAM 패키지 설치'를 사용하세요.",
    "SAM checkpoint exists, but required packages are missing.\n"
    "Switching backend to OpenCV for now.\n\nUse 'Install SAM Packages' first.":
        "SAM 체크포인트는 있지만 필요한 패키지가 없습니다.\n지금은 OpenCV로 전환합니다.\n\n"
        "먼저 'SAM 패키지 설치'를 사용하세요.",
    "SAM Not Ready": "SAM 준비되지 않음",
    "SAM backend was selected, but checkpoint file is missing.\n"
    "Switching backend to OpenCV for now.":
        "SAM을 선택했지만 체크포인트 파일이 없습니다.\n지금은 OpenCV로 전환합니다.",

    # -- Settings dialog: Auto Trace quality -----------------------------
    "Auto Trace Quality Assist": "자동 추적 품질 도우미",
    "Control Auto Trace speed/detail profile and low-quality warning thresholds shown in the "
    "main generator window.":
        "자동 추적의 속도·세부 설정과, 생성 창에 표시되는 저품질 경고 기준을 조정합니다.",
    "Auto Trace detail mode:": "자동 추적 세부 모드:",
    "Warning threshold (minimum):": "경고 기준 (최소):",
    " px": " px",
    "Recommended threshold:": "권장 기준:",
    "Minimum sharpness:": "최소 선명도:",
    "Variance of the Laplacian. Lower values accept softer images; 0 disables the check.":
        "라플라시안 분산 값입니다. 값이 낮을수록 흐린 이미지도 허용하며, 0이면 검사하지 않습니다.",
    "Resolution and sharpness decide how much detail can be traced; file size does not. "
    "A practical floor is a 700px short side, with 900px recommended, and a sharpness of "
    "about 60 for an in-focus photo.":
        "추적 가능한 세부의 양은 파일 크기가 아니라 해상도와 선명도가 결정합니다. "
        "실용적인 하한은 짧은 변 700px이고 900px을 권장하며, 초점이 맞은 사진의 선명도는 "
        "60 정도입니다.",

    # -- Settings dialog: Gemini -----------------------------------------
    "Step 1: Install Required Package": "1단계: 필요한 패키지 설치",
    "<b>What is this?</b><br>The modern 'google-genai' package allows Python to communicate "
    "with Gemini 3.1 and Nano Banana image models.<br><br><b>How to install:</b><br>"
    "Click the button below. Installation takes 1-2 minutes.<br>If it fails, you can install "
    "manually by opening Command Prompt and typing:<br><code>pip install {package}</code>":
        "<b>무엇인가요?</b><br>'google-genai' 패키지는 파이썬에서 Gemini 3.1과 Nano Banana "
        "이미지 모델을 사용할 수 있게 해 줍니다.<br><br><b>설치 방법</b><br>"
        "아래 버튼을 누르세요. 설치에는 1-2분이 걸립니다.<br>실패하면 명령 프롬프트를 열고 "
        "다음을 입력해 직접 설치할 수 있습니다:<br><code>pip install {package}</code>",
    "Install {package}": "{package} 설치",
    "Click to automatically install the required Python package":
        "필요한 파이썬 패키지를 자동으로 설치합니다",
    "Step 2: Get Your Free API Key": "2단계: 무료 API 키 발급",
    "<b>What is an API key?</b><br>An API key is like a password that allows ArchaeoGlyph to "
    "use Google's AI service.<br><br><b>How to get one (FREE!):</b><br>1. Click the button "
    "below to open Google AI Studio<br>2. Sign in with your Google account<br>"
    "3. Click 'Create API Key'<br>4. Copy the generated key (starts with 'AIza...')":
        "<b>API 키란?</b><br>API 키는 ArchaeoGlyph가 Google의 AI 서비스를 사용할 수 있게 "
        "해 주는 암호와 같습니다.<br><br><b>발급 방법 (무료)</b><br>1. 아래 버튼을 눌러 "
        "Google AI Studio를 엽니다<br>2. Google 계정으로 로그인합니다<br>"
        "3. 'Create API Key'를 누릅니다<br>4. 만들어진 키를 복사합니다 ('AIza...'로 시작)",
    "Open Google AI Studio": "Google AI Studio 열기",
    "Opens Google AI Studio in your web browser": "웹 브라우저에서 Google AI Studio를 엽니다",
    "Step 3: Enter Your API Key": "3단계: API 키 입력",
    "<b>Paste your API key below:</b><br>Your key is stored locally and never sent anywhere "
    "except Google. It looks like: AIza...":
        "<b>아래에 API 키를 붙여 넣으세요.</b><br>키는 이 컴퓨터에만 저장되며 Google 외의 "
        "어디에도 전송되지 않습니다. 'AIza...' 형태입니다.",
    "Paste your API key here (AIza...)": "여기에 API 키를 붙여 넣으세요 (AIza...)",
    "Your Google Gemini API key": "Google Gemini API 키",
    "Show": "보기",
    "Show/Hide API key": "API 키 보기/숨기기",
    "Step 4: Test Your Connection": "4단계: 연결 확인",
    "<b>Verify everything works:</b><br>Click the test button to make sure your API key is "
    "valid and the connection works.":
        "<b>정상 동작을 확인하세요.</b><br>테스트 버튼을 눌러 API 키가 유효하고 연결이 "
        "되는지 확인합니다.",
    "Test Gemini Connection": "Gemini 연결 테스트",
    "Test if your API key works correctly": "API 키가 제대로 동작하는지 확인합니다",
    "No API Key": "API 키 없음",
    "Please enter your API key first!\n\nIf you don't have one:\n"
    "1. Click 'Open Google AI Studio'\n2. Sign in with Google\n3. Create a new key":
        "먼저 API 키를 입력하세요.\n\n키가 없다면:\n"
        "1. 'Google AI Studio 열기'를 누릅니다\n2. Google 계정으로 로그인합니다\n"
        "3. 새 키를 만듭니다",
    "Package missing": "패키지 없음",
    "Package Not Installed": "패키지가 설치되지 않음",
    "The {package} package is not installed.\n\nPlease:\n1. Complete Step 1 (Install Package)\n"
    "2. Restart QGIS\n3. Try again":
        "{package} 패키지가 설치되어 있지 않습니다.\n\n다음을 확인하세요:\n"
        "1. 1단계(패키지 설치)를 마칩니다\n2. QGIS를 다시 시작합니다\n3. 다시 시도합니다",
    "Connection successful!\n\nAI Response: {response}\n\nYou're all set! Click 'Save Settings' "
    "and start generating symbols!":
        "연결에 성공했습니다.\n\nAI 응답: {response}\n\n준비가 끝났습니다. '설정 저장'을 누르고 "
        "심볼을 만들어 보세요.",
    "Invalid key": "잘못된 키",
    "Invalid API Key": "잘못된 API 키",
    "Your API key appears to be invalid.\n\nPlease:\n1. Go to Google AI Studio\n"
    "2. Create a NEW API key\n3. Copy and paste it here":
        "API 키가 유효하지 않은 것으로 보입니다.\n\n다음을 확인하세요:\n"
        "1. Google AI Studio로 이동합니다\n2. 새 API 키를 만듭니다\n3. 복사해서 여기에 붙여 넣습니다",
    "Error: {message}": "오류: {message}",

    # -- Settings dialog: local Stable Diffusion -------------------------
    "Server Configuration": "서버 설정",
    "<b>Server URL:</b><br>Enter the URL where your Stable Diffusion server is running.<br>"
    "Default is <code>http://127.0.0.1:7860</code> (localhost).":
        "<b>서버 주소</b><br>Stable Diffusion 서버가 실행 중인 주소를 입력하세요.<br>"
        "기본값은 <code>http://127.0.0.1:7860</code> (로컬)입니다.",
    "URL:": "주소:",
    "http://127.0.0.1:7860": "http://127.0.0.1:7860",
    "The URL of your local Stable Diffusion API server":
        "로컬 Stable Diffusion API 서버 주소입니다",
    "Test Connection": "연결 테스트",
    "How to Set Up Local Stable Diffusion": "로컬 Stable Diffusion 설정 방법",
    "Open Full Setup Guide (GitHub)": "전체 설정 안내 열기 (GitHub)",
    "Not connected": "연결되지 않음",
    "Connected ({count} models)": "연결됨 (모델 {count}개)",
    "Connected to Stable Diffusion!\n\nFound {count} model(s).\n\n"
    "Don't forget to click 'Save Settings'!":
        "Stable Diffusion에 연결했습니다.\n\n모델 {count}개를 찾았습니다.\n\n"
        "'설정 저장'을 잊지 마세요.",
    "Cannot connect to: {url}\n\nMake sure:\n1. Stable Diffusion WebUI is running\n"
    "2. It was started with --api flag\n3. The URL is correct\n\nError: {error}":
        "다음 주소에 연결할 수 없습니다: {url}\n\n확인할 점:\n"
        "1. Stable Diffusion WebUI가 실행 중인지\n2. --api 옵션으로 실행했는지\n"
        "3. 주소가 올바른지\n\n오류: {error}",

    # -- Settings dialog: quick start ------------------------------------
    "<h3>Get Started in 30 Seconds</h3>": "<h3>30초 만에 시작하기</h3>",
    "Option 1: Use Templates (NO Setup Required!)": "방법 1: 템플릿 사용 (설정 불필요)",
    "<ol><li>Open ArchaeoGlyph from the toolbar</li><li>Select <b>'Use Template'</b> mode</li>"
    "<li>Choose artifact type (Pottery, Stone Tools, etc.)</li><li>Pick your color</li>"
    "<li>Click <b>Generate</b>!</li></ol><p><i>That's it. No API key or installation "
    "needed.</i></p>":
        "<ol><li>도구 모음에서 ArchaeoGlyph를 엽니다</li><li><b>'템플릿 사용'</b>을 고릅니다</li>"
        "<li>유물 종류를 고릅니다 (토기, 석기 등)</li><li>색을 고릅니다</li>"
        "<li><b>생성</b>을 누릅니다</li></ol><p><i>끝입니다. API 키도 설치도 필요 없습니다.</i></p>",
    "Option 2: Use AI (Hugging Face)": "방법 2: AI 사용 (Hugging Face)",
    "<ol><li>Go to the <b>Hugging Face</b> tab</li><li>Click link to get a <b>token</b></li>"
    "<li>Paste key and click <b>Save Settings</b></li><li>Restart QGIS</li></ol>"
    "<p><i>Generate symbols with online inference models.</i></p>":
        "<ol><li><b>Hugging Face</b> 탭으로 이동합니다</li><li>링크를 눌러 <b>토큰</b>을 "
        "발급받습니다</li><li>키를 붙여 넣고 <b>설정 저장</b>을 누릅니다</li>"
        "<li>QGIS를 다시 시작합니다</li></ol><p><i>온라인 추론 모델로 심볼을 만듭니다.</i></p>",
    "Option 3: Use AI (Google Gemini)": "방법 3: AI 사용 (Google Gemini)",
    "<ol><li>Go to the <b>Google Gemini</b> tab</li><li>Click <b>Install Package</b> "
    "(wait 1-2 min)</li><li>Click link to get <b>free API key</b></li><li>Paste key and click "
    "<b>Save Settings</b></li><li>Restart QGIS</li></ol><p><i>Now you can upload any image "
    "and generate custom symbols.</i></p>":
        "<ol><li><b>Google Gemini</b> 탭으로 이동합니다</li><li><b>패키지 설치</b>를 누릅니다 "
        "(1-2분 소요)</li><li>링크를 눌러 <b>무료 API 키</b>를 발급받습니다</li>"
        "<li>키를 붙여 넣고 <b>설정 저장</b>을 누릅니다</li><li>QGIS를 다시 시작합니다</li></ol>"
        "<p><i>이제 어떤 이미지든 올려서 원하는 심볼을 만들 수 있습니다.</i></p>",

    # -- Settings dialog: saving and package installation ----------------
    "Settings Saved": "설정 저장됨",
    "Your settings have been saved!\n\nIf you installed a new package, please restart QGIS.":
        "설정을 저장했습니다.\n\n새 패키지를 설치했다면 QGIS를 다시 시작하세요.",
    "Testing...": "확인하는 중...",
    "Starting...": "시작하는 중...",
    "Installing: {line}": "설치 중: {line}",
    "Install package": "패키지 설치",
    "Install Package": "패키지 설치",
    "Install '{package}' into the Python that QGIS uses?\n\n"
    "The installer runs in the background; you can keep using QGIS.":
        "QGIS가 사용하는 파이썬에 '{package}'을(를) 설치할까요?\n\n"
        "설치는 백그라운드에서 진행되며, QGIS는 계속 사용할 수 있습니다.",
    "This will install '{package}' package.\n\nThe installer will run in the background.\n"
    "You can continue using QGIS while it installs.\n\nContinue?":
        "'{package}' 패키지를 설치합니다.\n\n설치는 백그라운드에서 진행됩니다.\n"
        "설치하는 동안 QGIS를 계속 사용할 수 있습니다.\n\n계속할까요?",
    "'{package}' was installed.\n\nRestart QGIS if it is not picked up immediately.":
        "'{package}'을(를) 설치했습니다.\n\n바로 인식되지 않으면 QGIS를 다시 시작하세요.",
    "Installation failed": "설치 실패",
    "Installing '{package}' failed (exit code {code}).":
        "'{package}' 설치에 실패했습니다 (종료 코드 {code}).",
    "Package installed successfully!\n\nPlease RESTART QGIS to apply changes.":
        "패키지를 설치했습니다.\n\n변경 사항을 적용하려면 QGIS를 다시 시작하세요.",
    "Installation Failed": "설치 실패",
    "Installation failed (Exit Code: {code}).": "설치에 실패했습니다 (종료 코드: {code}).",
    "Check the 'ArcheoGlyph' tab in QGIS Log Messages panel for full details.":
        "자세한 내용은 QGIS 로그 메시지 패널의 'ArcheoGlyph' 탭을 확인하세요.",
    "Copied": "복사됨",
    "Command copied to clipboard!\nPaste it in your terminal.":
        "명령을 클립보드에 복사했습니다.\n터미널에 붙여 넣으세요.",
    "Process Error": "프로세스 오류",
    "Failed to start installer.\nError code: {error}":
        "설치 프로그램을 시작하지 못했습니다.\n오류 코드: {error}",
    # -- Template names ---------------------------------------------------
    # The English name stays the identifier (it keys TEMPLATE_INFO and travels
    # with saved projects); only the label shown in the combo is translated.
    # Artefacts (유물)
    "Pottery": "토기",
    "Stone Tool": "석기",
    "Bronze Artifact": "청동기",
    "Iron Artifact": "철기",
    "Ornament": "장신구",
    "Coin": "화폐",
    "Bone Tool": "골각기",
    "Weapon": "무기",
    "Arrowhead": "화살촉",
    "Blade": "날붙이",
    "Scraper": "긁개",
    "Needle / Pin": "바늘 / 침",
    "Bead": "구슬",
    "Bracelet / Ring": "팔찌 / 반지",
    "Seal / Stamp": "인장",
    "Spindle Whorl": "가락바퀴",
    "Chisel": "끌",
    "Bronze Dagger (Liaoning-style)": "비파형동검",
    "Bronze Dagger (Ordos-style)": "오르도스식 동검",
    "Bronze Dagger (Antenna-style)": "촉각식 동검",
    "Bronze Dagger (Slender)": "세형동검",
    "Bronze Dagger (Tao type)": "도씨검",
    "Bronze Dagger (Medium-fine)": "중세형동검",
    "Bronze Dagger (Flat bladed)": "평인 동검",
    "Bronze Dagger (Type IA)": "동검 IA식",
    "Bronze Dagger (Type IB)": "동검 IB식",
    "Bronze Dagger (Other)": "기타 동검",
    "Bronze Sword": "동검(장검)",
    "Bronze Dagger-axe": "동과",
    "Bronze Spear": "동모",
    "Pottery Rim Sherd (Section)": "토기 구연부편 (단면)",
    "Pottery Base Sherd (Section)": "토기 저부편 (단면)",
    "Pottery Body Sherd (Section)": "토기 동체부편 (단면)",
    "Projectile Point (Leaf-shaped)": "첨두기 (나뭇잎형)",
    "Projectile Point (Side-notched)": "첨두기 (측면 홈형)",
    "Projectile Point (Corner-notched)": "첨두기 (모서리 홈형)",
    "Projectile Point (Stemmed)": "첨두기 (슴베형)",
    "Projectile Point (Triangular)": "첨두기 (삼각형)",

    # Structures (유구)
    "Fortress / Castle": "성곽",
    "Dwelling / House": "주거지",
    "Tomb": "무덤",
    "Keyhole Tomb (Normal)": "전방후원분 (기본)",
    "Keyhole Tomb (With Moat)": "전방후원분 (주구)",
    "Keyhole Tomb (Stepped)": "전방후원분 (단축성)",
    "Keyhole Tomb (With Fukiishi)": "전방후원분 (즙석)",
    "Keyhole Tomb (Tsumishizuka)": "전방후원분 (적석총형)",
    "Keyhole Tomb (Makinokuchi)": "전방후원분 (마키노쿠치형)",
    "Kofun (Normal)": "고분 (기본)",
    "Kofun (With Shugo)": "고분 (주호)",
    "Kofun (With Fukiishi)": "고분 (즙석)",
    "Kofun (Tsumiishizuka)": "고분 (적석총)",
    "Kofun (Enpun)": "고분 (원분)",
    "Kofun (Zenpokouen)": "고분 (전방후원분)",
    "Kofun (Makimuku-en)": "고분 (마키무쿠형 원분)",
    "Kofun (Hotategai)": "고분 (가리비형)",
    "Kofun (Sohochuen)": "고분 (쌍방중원분)",
    "Kofun (Hofun)": "고분 (방분)",
    "Kofun (Zenpokoho)": "고분 (전방후방분)",
    "Kofun (Makimuku-ho)": "고분 (마키무쿠형 방분)",
    "Kofun (Yosumi)": "고분 (사우돌출형)",
    "Kofun (Daijobo)": "고분 (대상묘)",
    "Temple / Shrine": "사찰 / 사당",
    "Kiln / Furnace": "가마 / 노",
    "Well": "우물",
    "Wall / Rampart": "성벽",
    "Pit": "수혈",
    "Gate": "문지",
    "Road / Pavement": "도로 / 포장면",
    "Bridge": "교량",
    "Storage Pit": "저장혈",
    "Posthole": "주혈",
    "Workshop": "공방지",
    "Tower": "망루",

    # Human and animal remains (인골·동물유체)
    "Human Remains": "인골",
    "Burial": "매장 유구",
    "Skeleton": "전신 인골",
    "Cremation Burial": "화장묘",
    "Animal Remains": "동물유체",

    # Features (유구·흔적)
    "Hearth / Fire Pit": "노지",
    "Midden / Shell Mound": "패총",
    "Ditch / Moat": "구 / 해자",
    "Stone Alignment": "열석",
    "Dolmen": "지석묘",
    "Rock Art": "암각화",
    "Canal / Water Channel": "수로",
    "Terrace": "단",
    "Ash Layer": "재층",
    "Burnt Area": "소토 범위",
    "Mound / Barrow": "봉토분",
    "Standing Stone": "입석",

    # Survey and recording (조사·기록)
    "Excavation Area": "발굴 구역",
    "Survey Point": "조사 지점",
    "Find Spot": "유물 출토 지점",
    "Trench": "트렌치",
    "Datum Point": "기준점",
    "Sample Location": "시료 채취 지점",
    "Photo Point": "사진 촬영 지점",
    "Grid Corner": "그리드 모서리",
    "Test Pit": "시굴 피트",
    "North Arrow (Map Standard)": "방위표",
    "Scale Bar (Map Standard)": "축척 막대",
    "Harris Matrix Context": "해리스 매트릭스 단위",
    "Stratigraphic Unit": "층위 단위",
    # -- Language selector -------------------------------------------------
    "Language:": "언어:",
    "Automatic (follow QGIS)": "자동 (QGIS 설정 따름)",
    "English": "English",
    "한국어": "한국어",
    "Takes effect the next time you open a window.": "다음에 창을 열 때 적용됩니다.",
    "Auto Trace needs {missing}.\nSwitching to Auto for now; the model is used automatically "
    "once installed.":
        "자동 추적에는 {missing}이(가) 필요합니다.\n지금은 자동으로 전환합니다. 설치하면 "
        "모델이 자동으로 사용됩니다.",
    "the onnxruntime package": "onnxruntime 패키지",
    "the model file": "모델 파일",
    " and ": "와(과) ",
}
