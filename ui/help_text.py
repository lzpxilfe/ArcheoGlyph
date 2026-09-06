# -*- coding: utf-8 -*-
"""
The long help documents shown in the settings dialog.

These are whole HTML pages rather than short strings, so they live here as
per-language functions instead of as keys in the i18n catalogue: a multi-line
document makes a brittle dictionary key, and translating it as one unit lets
the Korean text be structured for Korean readers.
"""

from ..i18n import current_language


def local_sd_setup_html():
    """Setup instructions for a local Stable Diffusion server."""
    if current_language() == "ko":
        return LOCAL_SD_SETUP_KO
    return LOCAL_SD_SETUP_EN


def help_html():
    """The plugin's own help page."""
    if current_language() == "ko":
        return HELP_KO
    return HELP_EN


LOCAL_SD_SETUP_EN = """
        <h4>Prerequisites</h4>
        <ul>
            <li>NVIDIA GPU with 6GB+ VRAM (RTX 2060 or better recommended)</li>
            <li>Windows 10/11 with updated drivers</li>
            <li>~15 GB free disk space</li>
        </ul>
        
        <h4>Installation Steps</h4>
        <ol>
            <li><b>Install Python 3.10.6</b> from <a href="https://www.python.org/downloads/release/python-3106/">python.org</a></li>
            <li><b>Download Automatic1111 WebUI</b>:
                <br><code>git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git</code></li>
            <li><b>Download a model</b> from <a href="https://civitai.com">Civitai</a>
                <br>Recommended: "Anything V5" or "Deliberate V2"</li>
            <li><b>Place the model</b> (.safetensors file) in <code>models/Stable-diffusion/</code></li>
            <li><b>Edit webui-user.bat</b> and add: <code>set COMMANDLINE_ARGS=--api</code></li>
            <li><b>Run webui-user.bat</b> and wait for it to start</li>
            <li><b>Enter URL above</b> and test the connection</li>
        </ol>
        """

HELP_EN = """
        <h2>ArchaeoGlyph Help</h2>

        <h3>What is ArchaeoGlyph?</h3>
        <p>ArchaeoGlyph helps archaeologists create accurate, standardized symbols for GIS maps. 
        Upload an artifact photo or select a template, and the plugin generates a precise, 
        recognizable symbol perfect for archaeological documentation.</p>

        <h3>Generation Modes</h3>
        <table border="1" cellpadding="8" style="border-collapse: collapse;">
            <tr style="background: #f0f0f0;">
                <th>Mode</th>
                <th>Requires</th>
                <th>Best For</th>
            </tr>
            <tr>
                <td><b>Auto Trace</b></td>
                <td>Nothing!</td>
                <td>Fast & accurate silhouette from photo</td>
            </tr>
            <tr>
                <td><b>AI (Hugging Face)</b></td>
                <td>HF Token</td>
                <td>Icon generation</td>
            </tr>
            <tr>
                <td><b>AI (Gemini)</b></td>
                <td>API Key + Internet</td>
                <td>Custom stylized symbols (Smart)</td>
            </tr>
            <tr>
                <td><b>AI (Local SD)</b></td>
                <td>GPU + Setup</td>
                <td>Offline use, sensitive data</td>
            </tr>
            <tr>
                <td><b>Template</b></td>
                <td>Nothing!</td>
                <td>Standardized category symbols</td>
            </tr>
        </table>
        
        <h3>Symbol Styles</h3>
        <ul>
            <li><b>Simple Symbol</b> - map-friendly icon with bold contour, 2-tone fill, and minimal structure lines</li>
            <li><b>Line</b> - contour and major internal lines, monochrome</li>
            <li><b>Measured</b> - monochrome measured drawing style for reports</li>
        </ul>
        
        <h3>Size Scaling Options</h3>
        <ul>
            <li><b>Fixed Size</b> - All symbols same size</li>
            <li><b>Natural Breaks</b> - Sizes based on data clustering</li>
            <li><b>Equal Interval</b> - Evenly distributed size ranges</li>
            <li><b>Quantile</b> - Equal number of features per size class</li>
        </ul>
        
        <h3>Saving Symbols</h3>
        <ul>
            <li><b>Save to Library</b> - Stores in QGIS symbol library for reuse</li>
            <li><b>Apply to Layer</b> - Immediately applies to selected vector layer</li>
        </ul>
        
        <h3>Links</h3>
        <ul>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph">GitHub Repository</a></li>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph/issues">Report Issues / Request Features</a></li>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph/blob/main/docs/ai_setup_guide.md">Full AI Setup Guide</a></li>
        </ul>
        
        <h3>Author</h3>
        <p>Created by <b>Jinseo Hwang</b></p>
        """

LOCAL_SD_SETUP_KO = """
        <h4>필요 사양</h4>
        <ul>
            <li>VRAM 6GB 이상 NVIDIA GPU (RTX 2060 이상 권장)</li>
            <li>드라이버를 최신으로 갱신한 Windows 10/11</li>
            <li>여유 디스크 공간 약 15 GB</li>
        </ul>

        <h4>설치 순서</h4>
        <ol>
            <li><b>Python 3.10.6 설치</b> - <a href="https://www.python.org/downloads/release/python-3106/">python.org</a></li>
            <li><b>Automatic1111 WebUI 내려받기</b>:
                <br><code>git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git</code></li>
            <li><b>모델 내려받기</b> - <a href="https://civitai.com">Civitai</a>
                <br>권장: "Anything V5" 또는 "Deliberate V2"</li>
            <li><b>모델 파일</b>(.safetensors)을 <code>models/Stable-diffusion/</code>에 넣기</li>
            <li><b>webui-user.bat 편집</b> 후 추가: <code>set COMMANDLINE_ARGS=--api</code></li>
            <li><b>webui-user.bat 실행</b> 후 시작될 때까지 대기</li>
            <li><b>위에 주소 입력</b> 후 연결 테스트</li>
        </ol>
        """

HELP_KO = """
        <h2>ArchaeoGlyph 도움말</h2>

        <h3>ArchaeoGlyph는 무엇인가요?</h3>
        <p>ArchaeoGlyph는 고고학 조사 결과를 GIS 지도에 표현할 때 쓰는 정확하고 표준화된
        심볼을 만들어 줍니다. 유물 사진을 올리거나 템플릿을 고르면, 보고서와 지도에 바로
        쓸 수 있는 심볼을 생성합니다.</p>

        <h3>생성 방식</h3>
        <table border="1" cellpadding="8" style="border-collapse: collapse;">
            <tr style="background: #f0f0f0;">
                <th>방식</th>
                <th>필요한 것</th>
                <th>알맞은 용도</th>
            </tr>
            <tr>
                <td><b>자동 추적</b></td>
                <td>없음</td>
                <td>사진에서 빠르고 정확한 실루엣</td>
            </tr>
            <tr>
                <td><b>AI (Hugging Face)</b></td>
                <td>HF 토큰</td>
                <td>아이콘 생성</td>
            </tr>
            <tr>
                <td><b>AI (Gemini)</b></td>
                <td>API 키 + 인터넷</td>
                <td>맞춤 양식화 심볼</td>
            </tr>
            <tr>
                <td><b>AI (로컬 SD)</b></td>
                <td>GPU + 설정</td>
                <td>오프라인 작업, 민감한 자료</td>
            </tr>
            <tr>
                <td><b>템플릿</b></td>
                <td>없음</td>
                <td>분류별 표준 심볼</td>
            </tr>
        </table>

        <h3>표현 방식</h3>
        <ul>
            <li><b>단순 심볼</b> - 굵은 윤곽, 두 가지 톤의 면, 최소한의 구조선으로 지도에 알맞은 아이콘</li>
            <li><b>선묘</b> - 윤곽선과 주요 내부선만, 단색</li>
            <li><b>실측도</b> - 보고서용 단색 실측 도면 방식</li>
        </ul>

        <h3>크기 조정</h3>
        <ul>
            <li><b>고정 크기</b> - 모든 심볼을 같은 크기로</li>
            <li><b>자연 분류</b> - 데이터가 몰린 구간을 기준으로 크기 구분</li>
            <li><b>등간격</b> - 값 범위를 균등하게 나눔</li>
            <li><b>등개수</b> - 단계마다 같은 개수의 지형지물</li>
        </ul>

        <h3>심볼 저장</h3>
        <ul>
            <li><b>라이브러리에 저장</b> - QGIS 심볼 라이브러리에 넣어 다시 사용</li>
            <li><b>레이어에 적용</b> - 선택한 벡터 레이어에 바로 적용</li>
        </ul>

        <h3>한국 고고학 템플릿</h3>
        <p>무덤, 주거·생산·방어 유구, 토기, 석기·청동기·철기, 장신구·기와 등 한국 고고학
        분류에 맞춘 템플릿이 들어 있습니다. 템플릿 검색창에 한국어로 입력해 찾을 수 있습니다.
        심볼은 모두 코드로 직접 그린 원본 도형입니다.</p>

        <h3>링크</h3>
        <ul>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph">GitHub 저장소</a></li>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph/issues">문제 신고 / 기능 요청</a></li>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph/blob/main/docs/ai_setup_guide.md">AI 설정 전체 안내</a></li>
        </ul>

        <h3>만든 사람</h3>
        <p><b>황진서</b> (Jinseo Hwang)</p>
        """
