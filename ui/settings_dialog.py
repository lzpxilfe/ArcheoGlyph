# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Settings Dialog
Configure AI API keys and view setup instructions.
"""

import os
import sys
import importlib.util
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from qgis.PyQt.QtCore import Qt, QSettings, QUrl, QProcess, QThread, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QTabWidget, QWidget, QTextBrowser,
    QMessageBox, QScrollArea, QFrame, QApplication,
    QCheckBox, QComboBox, QFileDialog
)
from ..defaults import HF_DEFAULT_MODEL_ID, HF_FALLBACK_MODEL_IDS, HF_LEGACY_MODEL_ALIASES


class InfoLabel(QLabel):
    """A styled info label with icon."""
    
    def __init__(self, text, icon="Info", parent=None):
        super().__init__(parent)
        self.setText(f"{icon} {text}")
        self.setWordWrap(True)
        self.setStyleSheet("""
            QLabel {
                background-color: #e8f4fc;
                border: 1px solid #b8daef;
                border-radius: 5px;
                padding: 10px;
                color: #0c5460;
            }
        """)


class WarningLabel(QLabel):
    """A styled warning label."""
    
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setText(f"Warning: {text}")
        self.setWordWrap(True)
        self.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
                padding: 10px;
                color: #856404;
            }
        """)


class SettingsDialog(QDialog):
    """Settings dialog for API configuration and help."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Initialize the settings UI."""
        self.setWindowTitle("ArcheoGlyph Settings & Help")
        self.setMinimumSize(650, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("<h2>ArcheoGlyph Settings</h2>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #4a90d9;
                color: white;
                border-radius: 5px 5px 0 0;
            }
        """)
        
        # Tab 1: Google Gemini
        gemini_tab = self._create_gemini_tab()
        tabs.addTab(gemini_tab, "Google Gemini")
        
        # Tab 2: Hugging Face (New)
        hf_tab = self._create_huggingface_tab()
        tabs.addTab(hf_tab, "Hugging Face")
        
        # Tab 3: Local Stable Diffusion
        local_tab = self._create_local_sd_tab()
        tabs.addTab(local_tab, "Local SD")
        
        # Tab 4: Quick Start
        quickstart_tab = self._create_quickstart_tab()
        tabs.addTab(quickstart_tab, "Quick Start")
        
        # Tab 5: Help
        help_tab = self._create_help_tab()
        tabs.addTab(help_tab, "Help")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(save_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
    def _create_huggingface_tab(self):
        """Create the Hugging Face settings tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Introduction
        info_label = QLabel(
            "<h3>Hugging Face Inference API</h3>"
            "<p>Use open-source AI models through Hugging Face inference."
            "Requires a Hugging Face account and token.</p>"
        )
        info_label.setTextFormat(Qt.RichText)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Token Input
        key_group = QGroupBox("API Token")
        key_layout = QVBoxLayout(key_group)
        
        link_label = QLabel(
            '1. Get a token from: <a href="https://huggingface.co/settings/tokens">huggingface.co/settings/tokens</a>'
        )
        link_label.setOpenExternalLinks(True)
        key_layout.addWidget(link_label)
        
        self.hf_key_input = QLineEdit()
        self.hf_key_input.setEchoMode(QLineEdit.Password)
        self.hf_key_input.setPlaceholderText("hf_...")
        key_layout.addWidget(self.hf_key_input)
        
        # Show/Hide Checkbox
        show_cb = QCheckBox("Show Token")
        show_cb.stateChanged.connect(
            lambda state: self.hf_key_input.setEchoMode(
                QLineEdit.Normal if state == Qt.Checked else QLineEdit.Password
            )
        )
        key_layout.addWidget(show_cb)
        layout.addWidget(key_group)

        # Model Selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)

        model_help = QLabel(
            f"Specify the Model ID to use (e.g., '{HF_DEFAULT_MODEL_ID}' or "
            "'Qwen/Qwen-Image'). If a model returns 403/404/503, the plugin "
            "automatically tries modern fallback models.\n"
            "Use 'Apply Latest Recommended Models' to refresh HF/SAM/Gemini recommendations without Python console checks."
        )
        model_help.setWordWrap(True)
        model_help.setStyleSheet("color: #666; font-size: 11px;")
        model_layout.addWidget(model_help)

        self.hf_model_input = QLineEdit()
        self.hf_model_input.setText(HF_DEFAULT_MODEL_ID)
        self.hf_model_input.setPlaceholderText("organization/model-name")
        model_layout.addWidget(self.hf_model_input)

        model_actions = QHBoxLayout()
        self.refresh_models_btn = QPushButton("Apply Latest Recommended Models")
        self.refresh_models_btn.clicked.connect(lambda: self.refresh_latest_model_recommendations(manual=True))
        model_actions.addWidget(self.refresh_models_btn)

        self.auto_refresh_models_check = QCheckBox("Auto-refresh model recommendations weekly")
        self.auto_refresh_models_check.setChecked(True)
        model_actions.addWidget(self.auto_refresh_models_check)
        model_actions.addStretch()
        model_layout.addLayout(model_actions)

        self.model_refresh_status = QLabel("")
        self.model_refresh_status.setWordWrap(True)
        self.model_refresh_status.setStyleSheet("color: #666; font-size: 11px;")
        model_layout.addWidget(self.model_refresh_status)
        
        layout.addWidget(model_group)

        # Optional advanced controls
        advanced_group = QGroupBox("Advanced (Optional)")
        advanced_layout = QVBoxLayout(advanced_group)

        advanced_layout.addWidget(QLabel(
            "SAM segmentation is optional. Use SAM1 with a local checkpoint, or use "
            "SAM2.1/SAM3 from Hugging Face (no local checkpoint file needed)."
        ))

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Auto Trace Backend:"))
        self.mask_backend_combo = QComboBox()
        self.mask_backend_combo.addItem("Auto (Recommended: SAM -> OpenCV fallback)", "auto")
        self.mask_backend_combo.addItem("OpenCV", "opencv")
        self.mask_backend_combo.addItem("SAM (Optional)", "sam")
        backend_row.addWidget(self.mask_backend_combo)
        advanced_layout.addLayout(backend_row)

        sam_type_row = QHBoxLayout()
        sam_type_row.addWidget(QLabel("SAM Model Type:"))
        self.sam_model_type_combo = QComboBox()
        self.sam_model_type_combo.addItem("SAM1 ViT-B (local checkpoint)", "vit_b")
        self.sam_model_type_combo.addItem("SAM1 ViT-L (local checkpoint)", "vit_l")
        self.sam_model_type_combo.addItem("SAM1 ViT-H (local checkpoint)", "vit_h")
        self.sam_model_type_combo.addItem("SAM3 Large (HF, latest, may be gated)", "hf:facebook/sam3-hiera-large")
        self.sam_model_type_combo.addItem("SAM2.1 Large (HF)", "hf:facebook/sam2.1-hiera-large")
        self.sam_model_type_combo.addItem("SAM2.1 Small (HF)", "hf:facebook/sam2.1-hiera-small")
        sam_type_row.addWidget(self.sam_model_type_combo)
        advanced_layout.addLayout(sam_type_row)

        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(QLabel("SAM Checkpoint:"))
        self.sam_checkpoint_input = QLineEdit()
        self.sam_checkpoint_input.setPlaceholderText("Path to sam_vit_*.pth (SAM1 only)")
        checkpoint_row.addWidget(self.sam_checkpoint_input)
        sam_browse_btn = QPushButton("Browse...")
        sam_browse_btn.clicked.connect(self._browse_sam_checkpoint)
        checkpoint_row.addWidget(sam_browse_btn)
        advanced_layout.addLayout(checkpoint_row)

        advanced_layout.addWidget(QLabel("SAM Quick Setup (Recommended for first-time users):"))

        sam_actions_row = QHBoxLayout()
        self.sam_install_btn = QPushButton("Install SAM Packages")
        self.sam_install_btn.clicked.connect(self.install_sam_package)
        sam_actions_row.addWidget(self.sam_install_btn)

        sam_download_btn = QPushButton("Download ViT-B Checkpoint")
        sam_download_btn.clicked.connect(self._open_sam_checkpoint_download)
        sam_actions_row.addWidget(sam_download_btn)

        sam_find_btn = QPushButton("Auto-Find Downloaded File")
        sam_find_btn.clicked.connect(self._autofind_sam_checkpoint)
        sam_actions_row.addWidget(sam_find_btn)

        sam_hf_models_btn = QPushButton("Open SAM2/3 Models")
        sam_hf_models_btn.clicked.connect(self._open_sam_hf_models)
        sam_actions_row.addWidget(sam_hf_models_btn)
        advanced_layout.addLayout(sam_actions_row)

        sam_guide_btn = QPushButton("SAM Setup Guide")
        sam_guide_btn.clicked.connect(self._show_sam_quick_guide)
        advanced_layout.addWidget(sam_guide_btn)

        self.sam_status_label = QLabel("")
        self.sam_status_label.setWordWrap(True)
        self.sam_status_label.setStyleSheet("color: #666; font-size: 11px;")
        advanced_layout.addWidget(self.sam_status_label)
        self.sam_checkpoint_input.textChanged.connect(lambda _text: self._refresh_sam_status())
        self.mask_backend_combo.currentIndexChanged.connect(lambda _idx: self._refresh_sam_status())
        self.sam_model_type_combo.currentIndexChanged.connect(lambda _idx: self._refresh_sam_status())

        self.hf_overlay_linework_check = QCheckBox(
            "HF: Overlay factual linework (stricter, may look similar to Auto Trace)"
        )
        self.hf_overlay_linework_check.setChecked(False)
        advanced_layout.addWidget(self.hf_overlay_linework_check)

        layout.addWidget(advanced_group)
        
        # Connection Test
        test_btn = QPushButton("Test Hugging Face Connection")
        test_btn.clicked.connect(self.test_huggingface_connection)
        layout.addWidget(test_btn)
        
        self.hf_test_result = QLabel("")
        layout.addWidget(self.hf_test_result)
        
        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _create_gemini_tab(self):
        """Create Google Gemini settings tab with detailed instructions."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Introduction
        intro = InfoLabel(
            "Google Gemini can generate archaeological symbols from reference images. "
            "API availability depends on your project quota and billing state.",
            "Info"
        )
        layout.addWidget(intro)
        
        # Step 1: Install package
        install_group = QGroupBox("Step 1: Install Required Package")
        install_layout = QVBoxLayout(install_group)
        
        install_desc = QLabel(
            "<b>What is this?</b><br>"
            "The 'google-generativeai' package allows Python to communicate with Google's AI.<br><br>"
            "<b>How to install:</b><br>"
            "Click the button below. Installation takes 1-2 minutes.<br>"
            "If it fails, you can install manually by opening Command Prompt and typing:<br>"
            "<code>pip install google-generativeai</code>"
        )
        install_desc.setWordWrap(True)
        install_desc.setTextFormat(Qt.RichText)
        install_layout.addWidget(install_desc)
        
        btn_layout = QHBoxLayout()
        self.install_btn = QPushButton("Install google-generativeai")
        self.install_btn.setMinimumHeight(40)
        self.install_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.install_btn.setToolTip("Click to automatically install the required Python package")
        self.install_btn.clicked.connect(self.install_gemini_package)
        btn_layout.addWidget(self.install_btn)
        
        self.install_status = QLabel("")
        self.install_status.setMinimumWidth(120)
        btn_layout.addWidget(self.install_status)
        btn_layout.addStretch()
        install_layout.addLayout(btn_layout)
        
        layout.addWidget(install_group)
        
        # Step 2: Get API Key
        apikey_group = QGroupBox("Step 2: Get Your Free API Key")
        apikey_layout = QVBoxLayout(apikey_group)
        
        apikey_desc = QLabel(
            "<b>What is an API key?</b><br>"
            "An API key is like a password that allows ArcheoGlyph to use Google's AI service.<br><br>"
            "<b>How to get one (FREE!):</b><br>"
            "1. Click the button below to open Google AI Studio<br>"
            "2. Sign in with your Google account<br>"
            "3. Click 'Create API Key'<br>"
            "4. Copy the generated key (starts with 'AIza...')"
        )
        apikey_desc.setWordWrap(True)
        apikey_desc.setTextFormat(Qt.RichText)
        apikey_layout.addWidget(apikey_desc)
        
        link_btn = QPushButton("Open Google AI Studio")
        link_btn.setMinimumHeight(40)
        link_btn.setStyleSheet("""
            QPushButton {
                background-color: #ea4335;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d33426;
            }
        """)
        link_btn.setToolTip("Opens Google AI Studio in your web browser")
        link_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://makersuite.google.com/app/apikey")
            )
        )
        apikey_layout.addWidget(link_btn)
        
        layout.addWidget(apikey_group)
        
        # Step 3: Enter API Key
        key_group = QGroupBox("Step 3: Enter Your API Key")
        key_layout = QVBoxLayout(key_group)
        
        key_desc = QLabel(
            "<b>Paste your API key below:</b><br>"
            "Your key is stored locally and never shared. It looks like: AIza..."
        )
        key_desc.setWordWrap(True)
        key_desc.setTextFormat(Qt.RichText)
        key_layout.addWidget(key_desc)
        
        key_input_layout = QHBoxLayout()
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.Password)
        self.gemini_key_input.setPlaceholderText("Paste your API key here (AIza...)")
        self.gemini_key_input.setMinimumHeight(35)
        self.gemini_key_input.setToolTip("Your Google Gemini API key - kept secure and private")
        key_input_layout.addWidget(self.gemini_key_input)
        
        show_key_btn = QPushButton("Show")
        show_key_btn.setFixedWidth(40)
        show_key_btn.setToolTip("Show/Hide API key")
        show_key_btn.clicked.connect(self._toggle_key_visibility)
        key_input_layout.addWidget(show_key_btn)
        key_layout.addLayout(key_input_layout)
        
        layout.addWidget(key_group)
        
        # Step 4: Test connection
        test_group = QGroupBox("Step 4: Test Your Connection")
        test_layout = QVBoxLayout(test_group)
        
        test_desc = QLabel(
            "<b>Verify everything works:</b><br>"
            "Click the test button to make sure your API key is valid and the connection works."
        )
        test_desc.setWordWrap(True)
        test_desc.setTextFormat(Qt.RichText)
        test_layout.addWidget(test_desc)
        
        test_btn_layout = QHBoxLayout()
        test_btn = QPushButton("Test Gemini Connection")
        test_btn.setMinimumHeight(40)
        test_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        test_btn.setToolTip("Test if your API key works correctly")
        test_btn.clicked.connect(self.test_gemini_connection)
        test_btn_layout.addWidget(test_btn)
        
        self.gemini_test_result = QLabel("")
        test_btn_layout.addWidget(self.gemini_test_result)
        test_btn_layout.addStretch()
        test_layout.addLayout(test_btn_layout)
        
        layout.addWidget(test_group)
        
        # Usage info
        usage_info = InfoLabel(
            "If you encounter HTTP 429, check Gemini quota limits and retry after cooldown.",
            "Tip"
        )
        layout.addWidget(usage_info)
        
        layout.addStretch()
        scroll.setWidget(tab)
        return scroll
        
    def _create_local_sd_tab(self):
        """Create Local Stable Diffusion settings tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Introduction
        intro = InfoLabel(
            "Local Stable Diffusion runs AI on YOUR computer - no internet required! "
            "Great for offline field work or sensitive data. Requires a GPU with 6GB+ VRAM.",
            "Info"
        )
        layout.addWidget(intro)
        
        # Warning
        warning = WarningLabel(
            "Advanced Setup Required: This option requires installing additional software "
            "and downloading large model files (4-8 GB). If you're not comfortable with this, "
            "use Google Gemini or Templates instead."
        )
        layout.addWidget(warning)
        
        # Server URL
        server_group = QGroupBox("Server Configuration")
        server_layout = QVBoxLayout(server_group)
        
        server_desc = QLabel(
            "<b>Server URL:</b><br>"
            "Enter the URL where your Stable Diffusion server is running.<br>"
            "Default is <code>http://127.0.0.1:7860</code> (localhost)."
        )
        server_desc.setWordWrap(True)
        server_desc.setTextFormat(Qt.RichText)
        server_layout.addWidget(server_desc)
        
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        self.sd_url_input = QLineEdit()
        self.sd_url_input.setPlaceholderText("http://127.0.0.1:7860")
        self.sd_url_input.setMinimumHeight(35)
        self.sd_url_input.setToolTip("The URL of your local Stable Diffusion API server")
        url_layout.addWidget(self.sd_url_input)
        server_layout.addLayout(url_layout)
        
        test_layout = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        test_btn.setMinimumHeight(35)
        test_btn.clicked.connect(self.test_sd_connection)
        test_layout.addWidget(test_btn)
        
        self.sd_test_result = QLabel("")
        test_layout.addWidget(self.sd_test_result)
        test_layout.addStretch()
        server_layout.addLayout(test_layout)
        
        layout.addWidget(server_group)
        
        # Setup instructions
        setup_group = QGroupBox("How to Set Up Local Stable Diffusion")
        setup_layout = QVBoxLayout(setup_group)
        
        setup_text = QTextBrowser()
        setup_text.setOpenExternalLinks(True)
        setup_text.setMaximumHeight(250)
        setup_text.setHtml("""
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
        """)
        setup_layout.addWidget(setup_text)
        
        guide_btn = QPushButton("Open Full Setup Guide (GitHub)")
        guide_btn.clicked.connect(self._open_sd_guide)
        setup_layout.addWidget(guide_btn)
        
        layout.addWidget(setup_group)
        
        layout.addStretch()
        scroll.setWidget(tab)
        return scroll
        
    def _create_quickstart_tab(self):
        """Create quick start guide tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("<h3>Get Started in 30 Seconds</h3>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # No setup option
        no_setup = QGroupBox("Option 1: Use Templates (NO Setup Required!)")
        no_setup_layout = QVBoxLayout(no_setup)
        no_setup_layout.addWidget(QLabel(
            "<ol>"
            "<li>Open ArcheoGlyph from the toolbar</li>"
            "<li>Select <b>'Use Template'</b> mode</li>"
            "<li>Choose artifact type (Pottery, Stone Tools, etc.)</li>"
            "<li>Pick your color</li>"
            "<li>Click <b>Generate</b>!</li>"
            "</ol>"
            "<p><i>That's it. No API key or installation needed.</i></p>"
        ))
        layout.addWidget(no_setup)
        
        # Hugging Face option
        hf_opt = QGroupBox("Option 2: Use AI (Hugging Face)")
        hf_layout = QVBoxLayout(hf_opt)
        hf_layout.addWidget(QLabel(
            "<ol>"
            "<li>Go to the <b>Hugging Face</b> tab</li>"
            "<li>Click link to get a <b>token</b></li>"
            "<li>Paste key and click <b>Save Settings</b></li>"
            "<li>Restart QGIS</li>"
            "</ol>"
            "<p><i>Generate symbols with online inference models.</i></p>"
        ))
        layout.addWidget(hf_opt)

        # Gemini option
        gemini_opt = QGroupBox("Option 3: Use AI (Google Gemini)")
        gemini_layout = QVBoxLayout(gemini_opt)
        gemini_layout.addWidget(QLabel(
            "<ol>"
            "<li>Go to the <b>Google Gemini</b> tab</li>"
            "<li>Click <b>Install Package</b> (wait 1-2 min)</li>"
            "<li>Click link to get <b>free API key</b></li>"
            "<li>Paste key and click <b>Save Settings</b></li>"
            "<li>Restart QGIS</li>"
            "</ol>"
            "<p><i>Now you can upload any image and generate custom symbols.</i></p>"
        ))
        layout.addWidget(gemini_opt)
        
        # Tips
        tips = InfoLabel(
            "Tip: Start with Templates to try the plugin, then add AI features later!",
            "Tip"
        )
        layout.addWidget(tips)
        
        layout.addStretch()
        scroll.setWidget(tab)
        return scroll
        
    def _create_help_tab(self):
        """Create help tab with documentation."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(True)
        help_text.setHtml("""
        <h2>ArcheoGlyph Help</h2>

        <h3>What is ArcheoGlyph?</h3>
        <p>ArcheoGlyph helps archaeologists create accurate, standardized symbols for GIS maps. 
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
            <li><b>Colored</b> - fact-based color symbol with clear readability</li>
            <li><b>Typology</b> - catalog-like icon with bold contour and 1-3 structural lines</li>
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
        """)
        layout.addWidget(help_text)
        
        return tab
        
    def _toggle_key_visibility(self):
        """Toggle API key visibility."""
        if self.gemini_key_input.echoMode() == QLineEdit.Password:
            self.gemini_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.gemini_key_input.setEchoMode(QLineEdit.Password)

    def _toggle_hf_key_visibility(self):
        """Toggle Hugging Face Key visibility."""
        if self.hf_key_input.echoMode() == QLineEdit.Password:
            self.hf_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.hf_key_input.setEchoMode(QLineEdit.Password)

    def _normalize_hf_model_id(self, model_id):
        """Normalize model ID into 'organization/model-name' format."""
        default = HF_DEFAULT_MODEL_ID
        value = (model_id or "").strip().replace("\\", "/")
        if not value:
            return default

        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc and "huggingface.co" in parsed.netloc:
            value = parsed.path.strip("/")

        for prefix in ("hf-inference/models/", "models/"):
            if value.startswith(prefix):
                value = value[len(prefix):]

        value = "/".join([part.strip() for part in value.strip("/").split("/") if part.strip()])

        value = HF_LEGACY_MODEL_ALIASES.get(value, value)

        if "/" not in value:
            return default
        return value

    def _parse_bool_setting(self, value, default=False):
        """Parse permissive boolean string/flag settings."""
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return bool(default)

    def _parse_utc_iso(self, value):
        """Parse ISO UTC timestamp from settings."""
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

    def _format_utc(self, dt_obj):
        """Format datetime object as stable UTC ISO text."""
        return dt_obj.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _maybe_auto_refresh_latest_models(self):
        """Auto-refresh latest model recommendations on a weekly cadence."""
        enabled = self.auto_refresh_models_check.isChecked()
        if not enabled:
            self.model_refresh_status.setText("Automatic refresh is disabled.")
            self.model_refresh_status.setStyleSheet("color: #666; font-size: 11px;")
            return

        now_utc = datetime.now(timezone.utc)
        last_checked_raw = self.settings.value("ArcheoGlyph/model_refresh_last_checked_utc", "")
        last_checked = self._parse_utc_iso(last_checked_raw)
        refresh_interval = timedelta(days=7)

        if last_checked is None or (now_utc - last_checked) >= refresh_interval:
            self.refresh_latest_model_recommendations(manual=False)
            return

        self.model_refresh_status.setText(
            f"Latest-model check is up to date (last check: {self._format_utc(last_checked)})."
        )
        self.model_refresh_status.setStyleSheet("color: #2f6f44; font-size: 11px;")

    def refresh_latest_model_recommendations(self, manual=False):
        """Resolve and apply latest practical HF/SAM recommendations asynchronously."""
        running = getattr(self, "model_refresh_thread", None)
        if running is not None and running.isRunning():
            return

        self.refresh_models_btn.setEnabled(False)
        self.model_refresh_status.setText("Checking latest model recommendations...")
        self.model_refresh_status.setStyleSheet("color: orange; font-size: 11px;")

        hf_api_key = self.hf_key_input.text().strip()
        gemini_api_key = self.gemini_key_input.text().strip()
        hf_candidates = list(HF_FALLBACK_MODEL_IDS)
        current_model = self._normalize_hf_model_id(self.hf_model_input.text().strip())
        if current_model and current_model not in hf_candidates:
            hf_candidates.insert(0, current_model)

        sam_candidates = [
            "facebook/sam3-hiera-large",
            "facebook/sam2.1-hiera-large",
            "facebook/sam2.1-hiera-small",
        ]

        self.model_refresh_thread = LatestModelRefreshThread(
            hf_api_key=hf_api_key,
            gemini_api_key=gemini_api_key,
            hf_candidates=hf_candidates,
            sam_candidates=sam_candidates,
        )
        self.model_refresh_thread.finished.connect(
            lambda result: self._handle_latest_model_refresh_result(result, manual)
        )
        self.model_refresh_thread.start()

    def _handle_latest_model_refresh_result(self, result, manual):
        """Apply latest-model recommendations from background resolver."""
        self.refresh_models_btn.setEnabled(True)

        if not isinstance(result, dict):
            result = {"status": "error", "message": "Invalid model refresh result payload."}

        now_utc = datetime.now(timezone.utc)
        self.settings.setValue("ArcheoGlyph/model_refresh_last_checked_utc", self._format_utc(now_utc))

        status = str(result.get("status", "")).strip().lower()
        message = str(result.get("message", "")).strip()
        recommended_hf = str(result.get("hf_model", "")).strip()
        recommended_sam = str(result.get("sam_model_type", "")).strip()
        recommended_gemini = str(result.get("gemini_model", "")).strip()

        changed = []

        if recommended_hf:
            normalized = self._normalize_hf_model_id(recommended_hf)
            if normalized and normalized != self._normalize_hf_model_id(self.hf_model_input.text()):
                self.hf_model_input.setText(normalized)
                changed.append(f"HF model -> {normalized}")
            self.settings.setValue("ArcheoGlyph/hf_model_id", normalized)

        if recommended_sam:
            idx = self.sam_model_type_combo.findData(recommended_sam)
            if idx < 0:
                self.sam_model_type_combo.addItem(f"Auto-detected ({recommended_sam})", recommended_sam)
                idx = self.sam_model_type_combo.findData(recommended_sam)
            if idx >= 0:
                current_sam = str(self.sam_model_type_combo.currentData() or "").strip()
                if current_sam != recommended_sam:
                    self.sam_model_type_combo.setCurrentIndex(idx)
                    changed.append(f"SAM model -> {recommended_sam}")
                self.settings.setValue("ArcheoGlyph/sam_model_type", recommended_sam)

        if self.auto_refresh_models_check.isChecked():
            self.settings.setValue("ArcheoGlyph/auto_update_models", "true")
        else:
            self.settings.setValue("ArcheoGlyph/auto_update_models", "false")

        if status == "ok":
            summary = []
            if changed:
                summary.extend(changed)
            else:
                summary.append("No setting changes were needed (already up to date).")
            if recommended_gemini:
                summary.append(f"Gemini best available: {recommended_gemini}")
            self.model_refresh_status.setText(" | ".join(summary))
            self.model_refresh_status.setStyleSheet("color: #2f6f44; font-size: 11px;")

            if manual:
                QMessageBox.information(
                    self,
                    "Latest Models Applied",
                    "\n".join(summary)
                )
        else:
            fallback_msg = message or "Latest model refresh failed."
            self.model_refresh_status.setText(fallback_msg)
            self.model_refresh_status.setStyleSheet("color: #b00020; font-size: 11px;")
            if manual:
                QMessageBox.warning(self, "Latest Model Refresh Failed", fallback_msg)

    def _open_sd_guide(self):
        """Open local SD setup guide."""
        QDesktopServices.openUrl(
            QUrl("https://github.com/lzpxilfe/ArcheoGlyph/blob/main/docs/ai_setup_guide.md")
        )

    def _get_python_executable(self):
        """Return Python interpreter path compatible with QGIS environment."""
        if sys.platform == 'win32':
            python_path = os.path.join(sys.exec_prefix, 'python.exe')
            if os.path.exists(python_path):
                return python_path
        return sys.executable

    def install_sam_package(self):
        """Install SAM-related packages (SAM1 + SAM2/3 path)."""
        reply = QMessageBox.question(
            self,
            "Install SAM Packages",
            "Install SAM packages now?\n\n"
            "This installs:\n"
            "- segment-anything (SAM1 local checkpoint)\n"
            "- transformers + huggingface_hub (SAM2/3 via HF)\n\n"
            "Note: SAM still needs 'torch'. If torch is missing, install it first "
            "(CPU build is okay for basic use).",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        python_path = self._get_python_executable()
        self.sam_install_btn.setEnabled(False)
        self.sam_install_btn.setText("Installing...")
        self.sam_status_label.setText("Installing SAM packages...")
        self.sam_status_label.setStyleSheet("color: orange; font-size: 11px;")

        self.sam_process = QProcess(self)
        self.sam_process.readyReadStandardOutput.connect(self._handle_sam_install_output)
        self.sam_process.readyReadStandardError.connect(self._handle_sam_install_output)
        self.sam_process.finished.connect(self._handle_sam_install_finished)
        self.sam_process.errorOccurred.connect(self._handle_sam_install_error)
        self.sam_process.start(
            python_path,
            [
                "-m", "pip", "install", "--user",
                "segment-anything",
                "transformers",
                "huggingface_hub",
                "pillow",
            ],
        )

    def _handle_sam_install_output(self):
        """Handle SAM installer output."""
        if not hasattr(self, "sam_process") or self.sam_process is None:
            return
        out = bytes(self.sam_process.readAllStandardOutput()).decode('utf-8', errors='replace').strip()
        err = bytes(self.sam_process.readAllStandardError()).decode('utf-8', errors='replace').strip()
        msg = out or err
        if msg:
            last_line = msg.splitlines()[-1][:120]
            self.sam_status_label.setText(f"Installing SAM: {last_line}")
            self.sam_status_label.setStyleSheet("color: orange; font-size: 11px;")

    def _handle_sam_install_finished(self, exit_code, exit_status):
        """Handle SAM installer completion."""
        self.sam_install_btn.setEnabled(True)
        self.sam_install_btn.setText("Install SAM Packages")
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            QMessageBox.information(
                self,
                "Installed",
                "SAM packages installed successfully.\n"
                "If this is first-time setup, restart QGIS."
            )
        else:
            python_path = self._get_python_executable()
            QMessageBox.warning(
                self,
                "Install Failed",
                "Could not install SAM packages automatically.\n\n"
                "Manual command:\n"
                f"{python_path} -m pip install --user segment-anything transformers huggingface_hub pillow"
            )
        self._refresh_sam_status()

    def _handle_sam_install_error(self, error):
        """Handle SAM installer process errors."""
        self.sam_install_btn.setEnabled(True)
        self.sam_install_btn.setText("Install SAM Packages")
        self.sam_status_label.setText(f"SAM install process error: {error}")
        self.sam_status_label.setStyleSheet("color: red; font-size: 11px;")
        self._refresh_sam_status()

    def _open_sam_checkpoint_download(self):
        """Open official SAM ViT-B checkpoint download URL."""
        QDesktopServices.openUrl(
            QUrl("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        )
        QMessageBox.information(
            self,
            "Download Started",
            "Browser download opened for sam_vit_b_01ec64.pth.\n"
            "After download, click 'Auto-Find Downloaded File'."
        )

    def _open_sam_hf_models(self):
        """Open official HF SAM3 model page."""
        QDesktopServices.openUrl(QUrl("https://huggingface.co/facebook/sam3-hiera-large"))

    def _get_candidate_sam_paths(self):
        """Return common paths where SAM checkpoints are likely located."""
        candidates = []
        home = os.path.expanduser("~")
        downloads = os.path.join(home, "Downloads")
        desktop = os.path.join(home, "Desktop")

        names = [
            "sam_vit_b_01ec64.pth",
            "sam_vit_l_0b3195.pth",
            "sam_vit_h_4b8939.pth",
        ]

        for folder in [downloads, desktop, home]:
            for name in names:
                candidates.append(os.path.join(folder, name))

        plugin_root = os.path.dirname(os.path.dirname(__file__))
        for name in names:
            candidates.append(os.path.join(plugin_root, "models", "sam", name))

        return candidates

    def _autofind_sam_checkpoint(self):
        """Find SAM checkpoint automatically in common folders."""
        for path in self._get_candidate_sam_paths():
            if os.path.exists(path):
                self.sam_checkpoint_input.setText(path)
                if "vit_l" in os.path.basename(path):
                    idx = self.sam_model_type_combo.findData("vit_l")
                elif "vit_h" in os.path.basename(path):
                    idx = self.sam_model_type_combo.findData("vit_h")
                else:
                    idx = self.sam_model_type_combo.findData("vit_b")
                if idx >= 0:
                    self.sam_model_type_combo.setCurrentIndex(idx)
                QMessageBox.information(
                    self,
                    "Checkpoint Found",
                    f"SAM checkpoint found and selected:\n{path}"
                )
                self._refresh_sam_status()
                return

        QMessageBox.information(
            self,
            "Not Found",
            "No SAM checkpoint was found in common folders.\n"
            "Click 'Download ViT-B Checkpoint' first."
        )
        self._refresh_sam_status()

    def _show_sam_quick_guide(self):
        """Show beginner-friendly SAM setup instructions."""
        QMessageBox.information(
            self,
            "SAM Quick Guide",
            "SAM setup (beginner):\n\n"
            "Tip: In Hugging Face tab, click 'Apply Latest Recommended Models' first.\n\n"
            "Option A: SAM2.1/SAM3 via Hugging Face (easiest)\n"
            "1. Keep 'Auto Trace Backend' = SAM (Optional)\n"
            "2. Choose a SAM2.1/SAM3 model in 'SAM Model Type'\n"
            "3. Click 'Install SAM Packages'\n"
            "4. Save Settings and restart QGIS\n\n"
            "Option B: SAM1 local checkpoint\n"
            "1. Keep 'Auto Trace Backend' = SAM (Optional)\n"
            "2. Choose SAM1 ViT model type\n"
            "3. Click 'Install SAM Packages'\n"
            "4. Click 'Download ViT-B Checkpoint'\n"
            "5. Click 'Auto-Find Downloaded File'\n"
            "6. Save Settings and restart QGIS\n\n"
            "If SAM is not ready, ArcheoGlyph automatically falls back to OpenCV."
        )

    def _refresh_sam_status(self):
        """Update SAM readiness status text."""
        model_choice = str(self.sam_model_type_combo.currentData() or self.sam_model_type_combo.currentText()).strip()
        uses_hf_sam = model_choice.lower().startswith("hf:")
        checkpoint = self.sam_checkpoint_input.text().strip()
        checkpoint_ok = bool(checkpoint and os.path.exists(checkpoint))

        dep_missing = []
        try:
            import torch  # noqa: F401
        except Exception:
            dep_missing.append("torch")
        if uses_hf_sam:
            try:
                import transformers  # noqa: F401
            except Exception:
                dep_missing.append("transformers")
            if dep_missing:
                self.sam_status_label.setText(
                    "SAM2/3 not ready (missing package(s): " + ", ".join(dep_missing) + "). "
                    "OpenCV backend will be used until setup is complete."
                )
                self.sam_status_label.setStyleSheet("color: #9a6700; font-size: 11px;")
                return
            model_id = model_choice[3:] if model_choice.lower().startswith("hf:") else model_choice
            self.sam_status_label.setText(
                f"SAM ready (HF): {model_id} (checkpoint not required)."
            )
            self.sam_status_label.setStyleSheet("color: green; font-size: 11px;")
            return

        try:
            import segment_anything  # noqa: F401
        except Exception:
            dep_missing.append("segment-anything")

        if checkpoint_ok and not dep_missing:
            self.sam_status_label.setText("SAM ready: dependencies and checkpoint detected.")
            self.sam_status_label.setStyleSheet("color: green; font-size: 11px;")
            return

        issues = []
        if not checkpoint_ok:
            issues.append("checkpoint missing")
        if dep_missing:
            issues.append("missing package(s): " + ", ".join(dep_missing))

        self.sam_status_label.setText(
            "SAM not ready (" + "; ".join(issues) + "). "
            "OpenCV backend will be used until SAM setup is complete."
        )
        self.sam_status_label.setStyleSheet("color: #9a6700; font-size: 11px;")

    def _browse_sam_checkpoint(self):
        """Browse SAM checkpoint file."""
        start_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM Checkpoint",
            start_dir,
            "SAM Checkpoint (sam_vit_*.pth *.pth *.pt);;PyTorch Checkpoint (*.pth *.pt);;All Files (*)"
        )
        if file_path:
            self.sam_checkpoint_input.setText(file_path)
            if "vit_l" in os.path.basename(file_path):
                idx = self.sam_model_type_combo.findData("vit_l")
            elif "vit_h" in os.path.basename(file_path):
                idx = self.sam_model_type_combo.findData("vit_h")
            else:
                idx = self.sam_model_type_combo.findData("vit_b")
            if idx >= 0:
                self.sam_model_type_combo.setCurrentIndex(idx)
        self._refresh_sam_status()
            
    def load_settings(self):
        """Load saved settings."""
        gemini_key = self.settings.value('ArcheoGlyph/gemini_api_key', '')
        hf_key = self.settings.value('ArcheoGlyph/huggingface_api_key', '')
        hf_model = self.settings.value(
            'ArcheoGlyph/hf_model_id',
            HF_DEFAULT_MODEL_ID
        )
        hf_model = self._normalize_hf_model_id(hf_model)
        self.settings.setValue('ArcheoGlyph/hf_model_id', hf_model)

        mask_backend = self.settings.value('ArcheoGlyph/mask_backend', 'auto')
        sam_checkpoint = self.settings.value('ArcheoGlyph/sam_checkpoint_path', '')
        sam_model_type = self.settings.value('ArcheoGlyph/sam_model_type', 'hf:facebook/sam2.1-hiera-small')
        hf_overlay_linework = str(
            self.settings.value('ArcheoGlyph/hf_overlay_linework', 'false')
        ).strip().lower() in ("1", "true", "yes", "on")
        auto_update_models = self._parse_bool_setting(
            self.settings.value('ArcheoGlyph/auto_update_models', 'true'),
            default=True,
        )
            
        sd_url = self.settings.value('ArcheoGlyph/sd_server', 'http://127.0.0.1:7860')
        
        self.gemini_key_input.setText(gemini_key)
        self.hf_key_input.setText(hf_key)
        self.hf_model_input.setText(hf_model)
        self.auto_refresh_models_check.setChecked(auto_update_models)
        self.sd_url_input.setText(sd_url)

        idx = self.mask_backend_combo.findData(str(mask_backend).strip().lower())
        if idx >= 0:
            self.mask_backend_combo.setCurrentIndex(idx)
        self.sam_checkpoint_input.setText(str(sam_checkpoint))
        if not str(sam_checkpoint).strip():
            for path in self._get_candidate_sam_paths():
                if os.path.exists(path):
                    self.sam_checkpoint_input.setText(path)
                    break
        sam_model_type = str(sam_model_type).strip() or "hf:facebook/sam2.1-hiera-small"
        type_idx = self.sam_model_type_combo.findData(sam_model_type)
        if type_idx < 0:
            type_idx = self.sam_model_type_combo.findText(sam_model_type)
        if type_idx >= 0:
            self.sam_model_type_combo.setCurrentIndex(type_idx)
        self.hf_overlay_linework_check.setChecked(hf_overlay_linework)
        self._refresh_sam_status()
        
        # Check if package is installed
        try:
            package_found = importlib.util.find_spec("google.generativeai") is not None
            if package_found:
                self.install_status.setText("Installed")
                self.install_status.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.install_status.setText("Not installed")
                self.install_status.setStyleSheet("color: red;")
        except Exception:
            self.install_status.setText("Not installed")
            self.install_status.setStyleSheet("color: red;")

        self._maybe_auto_refresh_latest_models()

    def save_settings(self):
        """Save settings."""
        self.settings.setValue('ArcheoGlyph/gemini_api_key', self.gemini_key_input.text())
        self.settings.setValue('ArcheoGlyph/huggingface_api_key', self.hf_key_input.text())
        self.settings.setValue('ArcheoGlyph/hf_model_id', self._normalize_hf_model_id(self.hf_model_input.text()))
        mask_backend = self.mask_backend_combo.currentData()
        sam_checkpoint = self.sam_checkpoint_input.text().strip()
        sam_model_type = str(self.sam_model_type_combo.currentData() or self.sam_model_type_combo.currentText()).strip()
        uses_hf_sam = sam_model_type.lower().startswith("hf:")

        # Safety: strict validation only when user forces SAM-only backend.
        if mask_backend == "sam":
            if uses_hf_sam:
                try:
                    import torch  # noqa: F401
                    import transformers  # noqa: F401
                except Exception:
                    QMessageBox.warning(
                        self,
                        "SAM Package Missing",
                        "SAM2/3 mode needs torch + transformers.\n"
                        "Switching backend to OpenCV for now.\n\n"
                        "Use 'Install SAM Packages' first."
                    )
                    mask_backend = "opencv"
                    idx = self.mask_backend_combo.findData("opencv")
                    if idx >= 0:
                        self.mask_backend_combo.setCurrentIndex(idx)
            else:
                if not sam_checkpoint or not os.path.exists(sam_checkpoint):
                    QMessageBox.warning(
                        self,
                        "SAM Not Ready",
                        "SAM backend was selected, but checkpoint file is missing.\n"
                        "Switching backend to OpenCV for now."
                    )
                    mask_backend = "opencv"
                    idx = self.mask_backend_combo.findData("opencv")
                    if idx >= 0:
                        self.mask_backend_combo.setCurrentIndex(idx)
                else:
                    try:
                        import torch  # noqa: F401
                        import segment_anything  # noqa: F401
                    except Exception:
                        QMessageBox.warning(
                            self,
                            "SAM Package Missing",
                            "SAM checkpoint exists, but required packages are missing.\n"
                            "Switching backend to OpenCV for now.\n\n"
                            "Use 'Install SAM Packages' first."
                        )
                        mask_backend = "opencv"
                        idx = self.mask_backend_combo.findData("opencv")
                        if idx >= 0:
                            self.mask_backend_combo.setCurrentIndex(idx)

        self.settings.setValue('ArcheoGlyph/mask_backend', mask_backend)
        self.settings.setValue('ArcheoGlyph/sam_checkpoint_path', sam_checkpoint)
        self.settings.setValue('ArcheoGlyph/sam_model_type', sam_model_type)
        self.settings.setValue(
            'ArcheoGlyph/hf_overlay_linework',
            'true' if self.hf_overlay_linework_check.isChecked() else 'false'
        )
        self.settings.setValue(
            'ArcheoGlyph/auto_update_models',
            'true' if self.auto_refresh_models_check.isChecked() else 'false'
        )
        self.settings.setValue('ArcheoGlyph/sd_server', self.sd_url_input.text())
        self._refresh_sam_status()
        
        QMessageBox.information(
            self, 
            "Settings Saved", 
            "Your settings have been saved!\n\n"
            "If you installed a new package, please restart QGIS."
        )

    def test_huggingface_connection(self):
        """Test Hugging Face connection asynchronously."""
        api_key = self.hf_key_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "No Token", "Please enter Hugging Face token.")
            return

        trigger_button = self.sender()
        if trigger_button:
            trigger_button.setEnabled(False)

        self.hf_test_result.setText("Testing...")
        self.hf_test_result.setStyleSheet("color: orange;")

        model_id = self._normalize_hf_model_id(self.hf_model_input.text().strip())
        self.hf_model_input.setText(model_id)

        candidate_models = []
        for mid in [model_id] + list(HF_FALLBACK_MODEL_IDS) + list(HF_LEGACY_MODEL_ALIASES.keys()):
            normalized = self._normalize_hf_model_id(mid)
            if normalized not in candidate_models:
                candidate_models.append(normalized)

        self.hf_test_thread = HfConnectionTestThread(api_key, candidate_models)
        self.hf_test_thread.finished.connect(
            lambda result: self._handle_hf_test_result(result, trigger_button, model_id)
        )
        self.hf_test_thread.start()

    def _handle_hf_test_result(self, result, trigger_button, requested_model_id):
        """Handle async Hugging Face connection test results."""
        if trigger_button:
            trigger_button.setEnabled(True)

        status = ""
        model = ""
        message = ""
        if isinstance(result, dict):
            status = str(result.get("status", "")).strip().lower()
            model = str(result.get("model", "")).strip()
            message = str(result.get("message", "")).strip()

        if status == "connected":
            if model and model != requested_model_id:
                self.hf_model_input.setText(model)
                self.settings.setValue('ArcheoGlyph/hf_model_id', model)
            self.hf_test_result.setText("Connected")
            self.hf_test_result.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(self, "Success", f"Connected with model: {model or requested_model_id}")
            return

        if status == "loading":
            if model and model != requested_model_id:
                self.hf_model_input.setText(model)
                self.settings.setValue('ArcheoGlyph/hf_model_id', model)
            self.hf_test_result.setText("Loading model...")
            self.hf_test_result.setStyleSheet("color: orange;")
            QMessageBox.information(
                self,
                "Loading",
                f"Connected, but model is initializing: {model or requested_model_id}"
            )
            return

        if status == "invalid_token":
            self.hf_test_result.setText("Invalid token")
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Invalid Token", "Please check your Hugging Face token.")
            return

        if status == "forbidden":
            self.hf_test_result.setText("Model access denied (403)")
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Model Access Denied",
                "Model terms may need acceptance on Hugging Face, or the model is restricted."
            )
            return

        if status == "not_found":
            self.hf_test_result.setText("Model not found (404)")
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Model Not Found",
                "No candidate model was found.\n"
                f"Try '{HF_DEFAULT_MODEL_ID}' or 'Qwen/Qwen-Image'."
            )
            return

        if status == "error":
            self.hf_test_result.setText("Failed")
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Connection Failed", message or "Unknown error")
            return

        self.hf_test_result.setText("Failed")
        self.hf_test_result.setStyleSheet("color: red;")
        QMessageBox.warning(self, "Connection Failed", "Unexpected test result.")
        
    def install_gemini_package(self):
        """Install google-generativeai package using QProcess (Async)."""
        reply = QMessageBox.question(
            self,
            "Install Package",
            "This will install 'google-generativeai' package.\n\n"
            "The installer will run in the background.\n"
            "You can continue using QGIS while it installs.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
            
        self.install_btn.setEnabled(False)
        self.install_btn.setText("Installing...")
        self.install_status.setText("Starting...")
        self.install_status.setStyleSheet("color: orange;")
        
        # Setup QProcess
        from qgis.PyQt.QtCore import QProcess
        
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._handle_process_output)
        self.process.readyReadStandardError.connect(self._handle_process_output)
        self.process.finished.connect(self._handle_process_finished)
        self.process.errorOccurred.connect(self._handle_process_error)
        
        # Fix: sys.executable in QGIS is 'qgis-bin.exe', which launches QGIS again!
        # We need the actual python interpreter.
        if sys.platform == 'win32':
            python_path = os.path.join(sys.exec_prefix, 'python.exe')
            if not os.path.exists(python_path):
                # Fallback to sys.executable if python.exe not found (unlikely)
                python_path = sys.executable
        else:
            python_path = sys.executable
            
        # Use --user flag to avoid permission issues
        args = ['-m', 'pip', 'install', '--user', 'google-generativeai']
        
        self.process.start(python_path, args)
        
    def _handle_process_output(self):
        """Handle process output."""
        data = self.process.readAllStandardOutput()
        self.process.readAllStandardError()
        
        if data:
            msg = bytes(data).decode('utf-8').strip()
            # Show last line in status if it's not too long
            if len(msg) < 50:
                self.install_status.setText(f"Installing: {msg}")
            else:
                self.install_status.setText("Installing...")
                
    def _handle_process_finished(self, exit_code, exit_status):
        """Handle install completion."""
        self.install_btn.setEnabled(True)
        self.install_btn.setText("Install google-generativeai")
        
        from qgis.core import QgsMessageLog, Qgis
        
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.install_status.setText("Installed")
            self.install_status.setStyleSheet("color: green; font-weight: bold;")
            QgsMessageLog.logMessage("ArcheoGlyph: Package installed successfully.", "ArcheoGlyph", Qgis.Success)
            
            QMessageBox.information(
                self, 
                "Success", 
                "Package installed successfully!\n\n"
                "Please RESTART QGIS to apply changes."
            )
        else:
            self.install_status.setText("Failed")
            self.install_status.setStyleSheet("color: red;")
            
            # Read all output for debugging
            stdout = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
            stderr = bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace')
            
            full_log = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            QgsMessageLog.logMessage(f"ArcheoGlyph Install Failed:\n{full_log}", "ArcheoGlyph", Qgis.Critical)
            
            # Show error details
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Installation Failed")
            msg.setText(f"Installation failed (Exit Code: {exit_code}).")
            msg.setInformativeText("Check the 'ArcheoGlyph' tab in QGIS Log Messages panel for full details.")
            msg.setDetailedText(full_log)
            msg.addButton("Copy Command", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Ok)
            
            msg.exec_()
            
            if msg.clickedButton().text() == "Copy Command":
                clipboard = QApplication.clipboard()
                cmd = f'"{sys.executable}" -m pip install --user google-generativeai'
                clipboard.setText(cmd)
                QMessageBox.information(self, "Copied", "Command copied to clipboard!\nPaste it in your terminal.")
            
    def _handle_process_error(self, error):
        """Handle process start error."""
        self.install_btn.setEnabled(True)
        self.install_btn.setText("Install google-generativeai")
        self.install_status.setText("Error")
        self.install_status.setStyleSheet("color: red;")
        
        QMessageBox.warning(
            self,
            "Process Error",
            f"Failed to start installer.\nError code: {error}"
        )
            
    def test_gemini_connection(self):
        """Test Gemini API connection (Async)."""
        api_key = self.gemini_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(
                self, 
                "No API Key", 
                "Please enter your API key first!\n\n"
                "If you don't have one:\n"
                "1. Click 'Open Google AI Studio'\n"
                "2. Sign in with Google\n"
                "3. Create a new key"
            )
            return
            
        self.gemini_test_result.setText("Testing...")
        self.gemini_test_result.setStyleSheet("color: orange;")
        
        # Disable button during test
        sender = self.sender()
        if sender:
            sender.setEnabled(False)
            
        self.test_thread = GeminiTestThread(api_key)
        self.test_thread.finished.connect(lambda s, m: self._handle_test_result(s, m, sender))
        self.test_thread.start()
        
    def _handle_test_result(self, success, message, button):
        """Handle API test result."""
        if button:
            button.setEnabled(True)
            
        if success:
            self.gemini_test_result.setText("Connected")
            self.gemini_test_result.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(
                self,
                "Success",
                f"Connection successful!\n\n"
                f"AI Response: {message[:100]}\n\n"
                f"You're all set! Click 'Save Settings' and start generating symbols!"
            )
        else:
            if "Package" in message:
                self.gemini_test_result.setText("Package missing")
                self.gemini_test_result.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self,
                    "Package Not Installed",
                    "The google-generativeai package is not installed.\n\n"
                    "Please:\n"
                    "1. Complete Step 1 (Install Package)\n"
                    "2. Restart QGIS\n"
                    "3. Try again"
                )
            elif "API_KEY_INVALID" in message or "invalid" in message.lower():
                self.gemini_test_result.setText("Invalid key")
                self.gemini_test_result.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self, 
                    "Invalid API Key", 
                    "Your API key appears to be invalid.\n\n"
                    "Please:\n"
                    "1. Go to Google AI Studio\n"
                    "2. Create a NEW API key\n"
                    "3. Copy and paste it here"
                )
            else:
                self.gemini_test_result.setText("Failed")
                self.gemini_test_result.setStyleSheet("color: red;")
                QMessageBox.warning(self, "Connection Failed", f"Error: {message}")

    def test_sd_connection(self):
        """Test Stable Diffusion server connection."""
        url = self.sd_url_input.text().strip()
        
        if not url:
            url = "http://127.0.0.1:7860"
            self.sd_url_input.setText(url)
            
        self.sd_test_result.setText("Testing...")
        self.sd_test_result.setStyleSheet("color: orange;")
        QApplication.processEvents()
        
        try:
            import urllib.request
            import json
            
            req = urllib.request.Request(f"{url}/sdapi/v1/sd-models", method='GET')
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    self.sd_test_result.setText(f"Connected ({len(data)} models)")
                    self.sd_test_result.setStyleSheet("color: green; font-weight: bold;")
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Connected to Stable Diffusion!\n\n"
                        f"Found {len(data)} model(s).\n\n"
                        f"Don't forget to click 'Save Settings'!"
                    )
                    
        except Exception as e:
            self.sd_test_result.setText("Not connected")
            self.sd_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Connection Failed",
                f"Cannot connect to: {url}\n\n"
                f"Make sure:\n"
                f"1. Stable Diffusion WebUI is running\n"
                f"2. It was started with --api flag\n"
                f"3. The URL is correct\n\n"
                f"Error: {str(e)}"
            )


class LatestModelRefreshThread(QThread):
    """Resolve latest practical model recommendations for HF/SAM/Gemini."""
    finished = pyqtSignal(object)  # dict result payload

    def __init__(self, hf_api_key, gemini_api_key, hf_candidates, sam_candidates):
        super().__init__()
        self.hf_api_key = str(hf_api_key or "").strip()
        self.gemini_api_key = str(gemini_api_key or "").strip()
        self.hf_candidates = [str(x).strip() for x in list(hf_candidates or []) if str(x).strip()]
        self.sam_candidates = [str(x).strip() for x in list(sam_candidates or []) if str(x).strip()]

    def _parse_last_modified(self, value):
        text = str(value or "").strip()
        if not text:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    def _hf_model_meta(self, model_id):
        import json
        import urllib.request
        import urllib.error

        headers = {"Accept": "application/json"}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"

        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{model_id}",
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                gated = payload.get("gated", None)
                ungated = gated in (None, False, "false", "False", "")
                return {
                    "ok": True,
                    "model_id": model_id,
                    "last_modified": payload.get("lastModified", ""),
                    "ungated": bool(ungated),
                    "private": bool(payload.get("private", False)),
                    "downloads": int(payload.get("downloads", 0) or 0),
                }
        except urllib.error.HTTPError as exc:
            return {"ok": False, "model_id": model_id, "status": exc.code}
        except Exception:
            return {"ok": False, "model_id": model_id, "status": None}

    def _select_best_hf(self):
        records = []
        for model_id in self.hf_candidates:
            meta = self._hf_model_meta(model_id)
            if not meta.get("ok"):
                continue
            if meta.get("private"):
                continue
            records.append(meta)

        if not records:
            return "", []

        ranked = sorted(
            records,
            key=lambda m: (
                1 if m.get("ungated") else 0,
                self._parse_last_modified(m.get("last_modified")),
                int(m.get("downloads", 0)),
            ),
            reverse=True,
        )
        return ranked[0].get("model_id", ""), records

    def _select_best_sam(self):
        sam_records = []
        for model_id in self.sam_candidates:
            meta = self._hf_model_meta(model_id)
            if not meta.get("ok"):
                continue
            if meta.get("private"):
                continue
            sam_records.append(meta)

        if not sam_records:
            return "", []

        # Prefer latest-listed ungated candidate. If none ungated, fallback to first reachable.
        record_by_id = {rec["model_id"]: rec for rec in sam_records}
        for model_id in self.sam_candidates:
            rec = record_by_id.get(model_id)
            if rec and rec.get("ungated"):
                return f"hf:{model_id}", sam_records
        for model_id in self.sam_candidates:
            rec = record_by_id.get(model_id)
            if rec:
                return f"hf:{model_id}", sam_records
        return "", sam_records

    def _select_best_gemini(self):
        if not self.gemini_api_key:
            return ""

        try:
            import re
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)

            available = []
            for model in genai.list_models():
                methods = list(getattr(model, "supported_generation_methods", []) or [])
                if "generateContent" not in methods:
                    continue
                name = str(getattr(model, "name", "") or "")
                if name.startswith("models/"):
                    name = name.replace("models/", "", 1)
                low = name.lower()
                if "gemini" not in low:
                    continue
                if "deep-research" in low or "experimental" in low:
                    continue
                available.append(name)

            if not available:
                return ""

            def rank(name):
                low = name.lower()
                major = 0
                minor = 0
                m = re.search(r"gemini-(\d+)(?:\.(\d+))?", low)
                if m:
                    major = int(m.group(1))
                    minor = int(m.group(2) or 0)
                score = (major * 1000) + (minor * 100)
                if "image" in low:
                    score += 80
                if "pro" in low:
                    score += 50
                if "flash" in low:
                    score += 40
                if "preview" in low:
                    score += 5
                if "lite" in low:
                    score -= 10
                if "exp" in low:
                    score -= 20
                return score

            available.sort(key=rank, reverse=True)
            return available[0]
        except Exception:
            return ""

    def run(self):
        try:
            hf_model, hf_records = self._select_best_hf()
            sam_model_type, sam_records = self._select_best_sam()
            gemini_model = self._select_best_gemini()

            if not hf_model and not sam_model_type and not gemini_model:
                self.finished.emit({
                    "status": "error",
                    "message": "Could not resolve latest model recommendations (check network/API keys).",
                })
                return

            summary = []
            if hf_model:
                summary.append(f"HF: {hf_model}")
            if sam_model_type:
                summary.append(f"SAM: {sam_model_type}")
            if gemini_model:
                summary.append(f"Gemini: {gemini_model}")

            self.finished.emit({
                "status": "ok",
                "hf_model": hf_model,
                "sam_model_type": sam_model_type,
                "gemini_model": gemini_model,
                "message": " | ".join(summary),
                "hf_candidates_found": len(hf_records),
                "sam_candidates_found": len(sam_records),
            })
        except Exception as exc:
            self.finished.emit({"status": "error", "message": str(exc)})


class GeminiTestThread(QThread):
    """Thread for testing Gemini API connection."""
    finished = pyqtSignal(bool, str) # success, message

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # List available models from the API
            available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        # Clean up model name (remove 'models/' prefix if present)
                        name = m.name
                        if name.startswith('models/'):
                            name = name.replace('models/', '')
                        available_models.append(name)
            except Exception as e:
                self.finished.emit(False, f"Connection/Auth Error: {str(e)}")
                return
            
            if not available_models:
                self.finished.emit(False, "No models available for your API key.")
                return

            # Prioritize models
            models_to_try = []
            
            # Exclusion list (same as generator)
            excluded_keywords = ['deep-research', 'experimental']
            
            def is_excluded(name):
                return any(keyword in name.lower() for keyword in excluded_keywords)
            
            # 1. Latest generation first, then robust fallbacks.
            preferred_models = [
                'gemini-3-pro-image-preview',
                'gemini-3-pro-preview',
                'gemini-3-flash-preview',
                'gemini-2.5-pro',
                'gemini-2.5-flash-image',
                'gemini-2.5-flash',
                'gemini-2.0-flash',
            ]
            
            for pref in preferred_models:
                for m in available_models:
                    if pref in m and not is_excluded(m):
                        models_to_try.append(m)
            
            # 2. Others (Fallback) - with strict filtering
            for m in available_models:
                if m not in models_to_try and not is_excluded(m):
                    if 'flash' in m.lower() or 'pro' in m.lower():
                        models_to_try.append(m)
            
            last_error = None
            success = False

            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Say 'Hello!'")
                    
                    if response and response.text:
                        self.finished.emit(True, f"[{model_name}] {response.text}")
                        success = True
                        break
                        
                except Exception as e:
                    last_error = e
                    continue
            
            if not success:
                error_msg = str(last_error) if last_error else "No suitable model found"
                self.finished.emit(False, error_msg)
                
        except ImportError:
            self.finished.emit(False, "Package 'google-generativeai' not installed")
        except Exception as e:
            self.finished.emit(False, str(e))


class HfConnectionTestThread(QThread):
    """Thread for testing Hugging Face model connectivity without blocking UI."""
    finished = pyqtSignal(object)  # dict result payload

    def __init__(self, api_key, candidate_models):
        super().__init__()
        self.api_key = str(api_key or "").strip()
        self.candidate_models = list(candidate_models or [])

    def run(self):
        try:
            import requests
        except Exception as exc:
            self.finished.emit({"status": "error", "message": str(exc)})
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }
        payload = {
            "inputs": "simple icon of an ancient pottery shard on white background",
            "parameters": {"num_inference_steps": 1},
        }

        saw_403 = False
        saw_404 = False
        last_status = None
        last_error = ""

        for candidate in self.candidate_models:
            if not candidate:
                continue
            api_url = f"https://router.huggingface.co/hf-inference/models/{candidate}"
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=12)
            except requests.RequestException as exc:
                last_error = str(exc)
                continue

            response_text = (response.text or "").lower()
            last_status = response.status_code

            if response.status_code == 200:
                self.finished.emit({"status": "connected", "model": candidate})
                return
            if response.status_code == 503 or "loading" in response_text:
                self.finished.emit({"status": "loading", "model": candidate})
                return
            if response.status_code == 401:
                self.finished.emit({"status": "invalid_token"})
                return
            if response.status_code == 403:
                saw_403 = True
                continue
            if response.status_code == 404:
                saw_404 = True
                continue

        if saw_403:
            self.finished.emit({"status": "forbidden"})
            return
        if saw_404:
            self.finished.emit({"status": "not_found"})
            return
        if last_error:
            self.finished.emit({"status": "error", "message": last_error})
            return
        self.finished.emit({"status": "error", "message": f"Error {last_status if last_status is not None else 'unknown'}"})



