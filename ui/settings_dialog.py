# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Settings Dialog
Configure AI API keys and view setup instructions.
"""

import os
import sys
import importlib.util
from importlib.util import find_spec
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from qgis.PyQt.QtCore import Qt, QSettings, QUrl, QProcess, QThread, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QTabWidget, QWidget, QTextBrowser,
    QMessageBox, QScrollArea, QFrame, QApplication,
    QCheckBox, QComboBox, QFileDialog, QSpinBox, QProgressBar
)
from .help_text import help_html, local_sd_setup_html
from ..i18n import (
    LANGUAGE_SETTING,
    apply_settings_language,
    available_languages,
    tr,
)
from ..auth import get_api_key, set_api_key, storage_description
from ..generators.autotrace.model_store import (
    DEFAULT_MODEL_KEY,
    MODEL_SPECS,
    download_model,
    is_installed,
    model_path,
    verify_model,
)
from ..defaults import (
    GEMINI_AI_STUDIO_URL,
    GEMINI_EXCLUDED_KEYWORDS,
    GEMINI_IMAGE_MODEL_CANDIDATES,
    GEMINI_INSTALL_PACKAGE,
    GEMINI_TEXT_MODEL_CANDIDATES,
    HF_DEFAULT_MODEL_ID,
    HF_FALLBACK_MODEL_IDS,
    HF_LEGACY_MODEL_ALIASES,
)


class InfoLabel(QLabel):
    """A styled info label with icon."""
    
    def __init__(self, text, icon="Info", parent=None):
        super().__init__(parent)
        self.setText("{icon} {text}".format(icon=icon, text=text))
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
        self.setText(tr("Warning: {text}").format(text=text))
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


def _normalize_gemini_model_name(model_name):
    """Normalize Gemini model name from SDK or saved setting."""
    normalized = str(model_name or "").strip()
    if normalized.startswith("models/"):
        normalized = normalized.replace("models/", "", 1)
    return normalized


def _is_excluded_gemini_model(model_name):
    """Filter out Gemini utility models not suitable for symbol generation."""
    low = str(model_name or "").strip().lower()
    return any(keyword in low for keyword in GEMINI_EXCLUDED_KEYWORDS)


def _is_image_gemini_model(model_name):
    """Detect Gemini image-generation/edit model names."""
    return "image" in str(model_name or "").strip().lower()


def _rank_gemini_model(model_name):
    """Rank Gemini models by family recency and modality utility."""
    import re

    low = str(model_name or "").strip().lower()
    major = 0
    minor = 0
    match = re.search(r"gemini-(\d+)(?:\.(\d+))?", low)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2) or 0)

    score = (major * 1000) + (minor * 100)
    if _is_image_gemini_model(low):
        score += 160
    if "pro" in low:
        score += 60
    if "flash" in low:
        score += 45
    if "preview" in low:
        score += 8
    if "lite" in low:
        score -= 12
    if "exp" in low:
        score -= 20
    return score


class SettingsDialog(QDialog):
    """Settings dialog for API configuration and help."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        apply_settings_language()
        self.settings = QSettings()
        # Worker threads are kept here so they are never garbage collected
        # while still running (which aborts the QGIS process).
        self.model_refresh_thread = None
        self.test_thread = None
        self.hf_test_thread = None
        self.onnx_download_thread = None
        self.setup_ui()
        self.load_settings()

    def _running_threads(self):
        return [
            thread for thread in (
                self.model_refresh_thread, self.test_thread,
                self.hf_test_thread, self.onnx_download_thread,
            )
            if thread is not None and thread.isRunning()
        ]

    def closeEvent(self, event):
        """Wait for worker threads before the dialog (and its threads) die."""
        for thread in self._running_threads():
            cancel = getattr(thread, "cancel", None)
            if callable(cancel):
                cancel()
            thread.wait(3000)
        super().closeEvent(event)
        
    def setup_ui(self):
        """Initialize the settings UI."""
        self.setWindowTitle(tr("ArchaeoGlyph Settings & Help"))
        self.setMinimumSize(650, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel(tr("<h2>ArchaeoGlyph Settings</h2>"))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Language: applies the next time a dialog is opened, so the widgets
        # already on screen are never left half-translated.
        language_row = QHBoxLayout()
        language_row.addWidget(QLabel(tr("Language:")))
        self.language_combo = QComboBox()
        for code, label in available_languages():
            self.language_combo.addItem(tr(label), code)
        language_row.addWidget(self.language_combo)
        language_note = QLabel(tr("Takes effect the next time you open a window."))
        language_note.setStyleSheet("color: #666;")
        language_row.addWidget(language_note)
        language_row.addStretch()
        layout.addLayout(language_row)
        
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
        tabs.addTab(gemini_tab, tr("Google Gemini"))
        
        # Tab 2: Hugging Face (New)
        hf_tab = self._create_huggingface_tab()
        tabs.addTab(hf_tab, tr("Hugging Face"))
        
        # Tab 3: Local Stable Diffusion
        local_tab = self._create_local_sd_tab()
        tabs.addTab(local_tab, tr("Local SD"))
        
        # Tab 4: Quick Start
        quickstart_tab = self._create_quickstart_tab()
        tabs.addTab(quickstart_tab, tr("Quick Start"))
        
        # Tab 5: Help
        help_tab = self._create_help_tab()
        tabs.addTab(help_tab, tr("Help"))
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_btn = QPushButton(tr("Save Settings"))
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
        
        close_btn = QPushButton(tr("Close"))
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
            tr("<h3>Hugging Face Inference API</h3>"
            "<p>Use open-source AI models through Hugging Face inference."
            "Requires a Hugging Face account and token.</p>")
        )
        info_label.setTextFormat(Qt.RichText)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Token Input
        key_group = QGroupBox(tr("API Token"))
        key_layout = QVBoxLayout(key_group)
        
        link_label = QLabel(
            tr('1. Get a token from: <a href="https://huggingface.co/settings/tokens">huggingface.co/settings/tokens</a>')
        )
        link_label.setOpenExternalLinks(True)
        key_layout.addWidget(link_label)
        
        self.hf_key_input = QLineEdit()
        self.hf_key_input.setEchoMode(QLineEdit.Password)
        self.hf_key_input.setPlaceholderText(tr("hf_..."))
        key_layout.addWidget(self.hf_key_input)
        
        # Show/Hide Checkbox
        show_cb = QCheckBox(tr("Show Token"))
        show_cb.stateChanged.connect(
            lambda state: self.hf_key_input.setEchoMode(
                QLineEdit.Normal if state == Qt.Checked else QLineEdit.Password
            )
        )
        key_layout.addWidget(show_cb)
        layout.addWidget(key_group)

        # Model Selection
        model_group = QGroupBox(tr("Model Selection"))
        model_layout = QVBoxLayout(model_group)

        model_help = QLabel(
            tr(
                "Specify the Model ID to use (e.g., '{model}' or "
                "'Qwen/Qwen-Image'). If a model returns 403/404/503, the plugin "
                "automatically tries modern fallback models.\n"
                "Use 'Check Latest Models' to preview recommendations, then "
                "'Apply Latest Recommended Models' to apply without Python console checks."
            ).format(model=HF_DEFAULT_MODEL_ID)
        )
        model_help.setWordWrap(True)
        model_help.setStyleSheet("color: #666; font-size: 11px;")
        model_layout.addWidget(model_help)

        self.hf_model_input = QLineEdit()
        self.hf_model_input.setText(HF_DEFAULT_MODEL_ID)
        self.hf_model_input.setPlaceholderText(tr("organization/model-name"))
        model_layout.addWidget(self.hf_model_input)

        model_actions = QHBoxLayout()
        self.check_models_btn = QPushButton(tr("Check Latest Models"))
        self.check_models_btn.clicked.connect(
            lambda: self.refresh_latest_model_recommendations(manual=True, apply_changes=False)
        )
        model_actions.addWidget(self.check_models_btn)

        self.refresh_models_btn = QPushButton(tr("Apply Latest Recommended Models"))
        self.refresh_models_btn.clicked.connect(
            lambda: self.refresh_latest_model_recommendations(manual=True, apply_changes=True)
        )
        model_actions.addWidget(self.refresh_models_btn)

        self.auto_refresh_models_check = QCheckBox(tr("Auto-refresh model recommendations weekly"))
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
        advanced_group = QGroupBox(tr("Advanced (Optional)"))
        advanced_layout = QVBoxLayout(advanced_group)

        advanced_layout.addWidget(QLabel(
            tr("Auto Trace separates the artifact from its background. OpenCV needs no "
            "download but struggles with gradients, shadows and grey-on-grey photos; "
            "a background-removal model handles those. SAM is also supported.")
        ))

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel(tr("Auto Trace Backend:")))
        self.mask_backend_combo = QComboBox()
        self.mask_backend_combo.addItem(tr("Auto (recommended: best available model, else OpenCV)"), "auto")
        self.mask_backend_combo.addItem(tr("OpenCV only (no extra download)"), "opencv")
        self.mask_backend_combo.addItem(tr("Background-removal model (ONNX)"), "onnx")
        self.mask_backend_combo.addItem(tr("SAM (optional)"), "sam")
        backend_row.addWidget(self.mask_backend_combo)
        advanced_layout.addLayout(backend_row)

        onnx_group = QGroupBox(tr("Background-removal model (recommended for photographs)"))
        onnx_layout = QVBoxLayout(onnx_group)
        onnx_layout.addWidget(QLabel(
            tr("Downloaded once, verified by size and SHA-256, and stored in your QGIS "
            "profile. Runs on the CPU; no image ever leaves your machine.")
        ))

        onnx_model_row = QHBoxLayout()
        onnx_model_row.addWidget(QLabel(tr("Model:")))
        self.onnx_model_combo = QComboBox()
        for key, spec in MODEL_SPECS.items():
            self.onnx_model_combo.addItem(spec.label, key)
        self.onnx_model_combo.currentIndexChanged.connect(lambda _idx: self._refresh_onnx_status())
        onnx_model_row.addWidget(self.onnx_model_combo, 1)
        onnx_layout.addLayout(onnx_model_row)

        onnx_actions = QHBoxLayout()
        self.onnx_install_runtime_btn = QPushButton(tr("Install onnxruntime"))
        self.onnx_install_runtime_btn.setToolTip(tr("Installs the CPU inference runtime with pip."))
        self.onnx_install_runtime_btn.clicked.connect(self.install_onnx_runtime)
        onnx_actions.addWidget(self.onnx_install_runtime_btn)

        self.onnx_download_btn = QPushButton(tr("Download model"))
        self.onnx_download_btn.clicked.connect(self.download_onnx_model)
        onnx_actions.addWidget(self.onnx_download_btn)

        self.onnx_verify_btn = QPushButton(tr("Verify"))
        self.onnx_verify_btn.setToolTip(tr("Re-check the stored file against its published SHA-256."))
        self.onnx_verify_btn.clicked.connect(self.verify_onnx_model)
        onnx_actions.addWidget(self.onnx_verify_btn)
        onnx_actions.addStretch()
        onnx_layout.addLayout(onnx_actions)

        self.onnx_progress = QProgressBar()
        self.onnx_progress.setVisible(False)
        onnx_layout.addWidget(self.onnx_progress)

        diagnostics_row = QHBoxLayout()
        self.diagnostics_btn = QPushButton(tr("Copy diagnostics"))
        self.diagnostics_btn.setToolTip(
            tr("Copy a plain-text report of versions, installed packages and models.\n"
            "Paste it into a bug report when something does not work.")
        )
        self.diagnostics_btn.clicked.connect(self.copy_diagnostics)
        diagnostics_row.addWidget(self.diagnostics_btn)
        diagnostics_row.addStretch()
        onnx_layout.addLayout(diagnostics_row)

        self.onnx_status_label = QLabel("")
        self.onnx_status_label.setWordWrap(True)
        self.onnx_status_label.setStyleSheet("color: #666; font-size: 11px;")
        onnx_layout.addWidget(self.onnx_status_label)
        advanced_layout.addWidget(onnx_group)

        sam_type_row = QHBoxLayout()
        sam_type_row.addWidget(QLabel(tr("SAM Model Type:")))
        self.sam_model_type_combo = QComboBox()
        self.sam_model_type_combo.addItem(tr("SAM1 ViT-B (local checkpoint)"), "vit_b")
        self.sam_model_type_combo.addItem(tr("SAM1 ViT-L (local checkpoint)"), "vit_l")
        self.sam_model_type_combo.addItem(tr("SAM1 ViT-H (local checkpoint)"), "vit_h")
        self.sam_model_type_combo.addItem(tr("SAM3 Large (HF, latest, may be gated)"), "hf:facebook/sam3-hiera-large")
        self.sam_model_type_combo.addItem(tr("SAM2.1 Large (HF)"), "hf:facebook/sam2.1-hiera-large")
        self.sam_model_type_combo.addItem(tr("SAM2.1 Small (HF)"), "hf:facebook/sam2.1-hiera-small")
        sam_type_row.addWidget(self.sam_model_type_combo)
        advanced_layout.addLayout(sam_type_row)

        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(QLabel(tr("SAM Checkpoint:")))
        self.sam_checkpoint_input = QLineEdit()
        self.sam_checkpoint_input.setPlaceholderText(tr("Path to sam_vit_*.pth (SAM1 only)"))
        checkpoint_row.addWidget(self.sam_checkpoint_input)
        sam_browse_btn = QPushButton(tr("Browse..."))
        sam_browse_btn.clicked.connect(self._browse_sam_checkpoint)
        checkpoint_row.addWidget(sam_browse_btn)
        advanced_layout.addLayout(checkpoint_row)

        advanced_layout.addWidget(QLabel(tr("SAM Quick Setup (Recommended for first-time users):")))

        sam_actions_row = QHBoxLayout()
        self.sam_install_btn = QPushButton(tr("Install SAM Packages"))
        self.sam_install_btn.clicked.connect(self.install_sam_package)
        sam_actions_row.addWidget(self.sam_install_btn)

        sam_download_btn = QPushButton(tr("Download ViT-B Checkpoint"))
        sam_download_btn.clicked.connect(self._open_sam_checkpoint_download)
        sam_actions_row.addWidget(sam_download_btn)

        sam_find_btn = QPushButton(tr("Auto-Find Downloaded File"))
        sam_find_btn.clicked.connect(self._autofind_sam_checkpoint)
        sam_actions_row.addWidget(sam_find_btn)

        sam_hf_models_btn = QPushButton(tr("Open SAM2/3 Models"))
        sam_hf_models_btn.clicked.connect(self._open_sam_hf_models)
        sam_actions_row.addWidget(sam_hf_models_btn)
        advanced_layout.addLayout(sam_actions_row)

        sam_guide_btn = QPushButton(tr("SAM Setup Guide"))
        sam_guide_btn.clicked.connect(self._show_sam_quick_guide)
        advanced_layout.addWidget(sam_guide_btn)

        self.sam_status_label = QLabel("")
        self.sam_status_label.setWordWrap(True)
        self.sam_status_label.setStyleSheet("color: #666; font-size: 11px;")
        advanced_layout.addWidget(self.sam_status_label)
        self.sam_checkpoint_input.textChanged.connect(lambda _text: self._refresh_sam_status())
        self.mask_backend_combo.currentIndexChanged.connect(lambda _idx: self._refresh_sam_status())
        self.mask_backend_combo.currentIndexChanged.connect(lambda _idx: self._refresh_onnx_status())
        self.sam_model_type_combo.currentIndexChanged.connect(lambda _idx: self._refresh_sam_status())

        self.hf_overlay_linework_check = QCheckBox(
            tr("HF: Overlay factual linework (stricter, may look similar to Auto Trace)")
        )
        self.hf_overlay_linework_check.setChecked(False)
        advanced_layout.addWidget(self.hf_overlay_linework_check)

        layout.addWidget(advanced_group)

        quality_group = QGroupBox(tr("Auto Trace Quality Assist"))
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.addWidget(QLabel(
            tr("Control Auto Trace speed/detail profile and low-quality warning thresholds "
            "shown in the main generator window.")
        ))

        detail_mode_row = QHBoxLayout()
        detail_mode_row.addWidget(QLabel(tr("Auto Trace detail mode:")))
        self.autotrace_detail_mode_combo = QComboBox()
        self.autotrace_detail_mode_combo.addItem(tr("Fast (speed priority)"), "fast")
        self.autotrace_detail_mode_combo.addItem(tr("Precise (detail priority)"), "precise")
        detail_mode_row.addWidget(self.autotrace_detail_mode_combo, 1)
        quality_layout.addLayout(detail_mode_row)

        weak_row = QHBoxLayout()
        weak_row.addWidget(QLabel(tr("Warning threshold (minimum):")))
        self.image_warn_min_short_px_spin = QSpinBox()
        self.image_warn_min_short_px_spin.setRange(256, 4096)
        self.image_warn_min_short_px_spin.setSuffix(tr(" px"))
        weak_row.addWidget(self.image_warn_min_short_px_spin)
        quality_layout.addLayout(weak_row)

        rec_row = QHBoxLayout()
        rec_row.addWidget(QLabel(tr("Recommended threshold:")))
        self.image_warn_recommended_short_px_spin = QSpinBox()
        self.image_warn_recommended_short_px_spin.setRange(256, 4096)
        self.image_warn_recommended_short_px_spin.setSuffix(tr(" px"))
        rec_row.addWidget(self.image_warn_recommended_short_px_spin)
        quality_layout.addLayout(rec_row)

        sharp_row = QHBoxLayout()
        sharp_row.addWidget(QLabel(tr("Minimum sharpness:")))
        self.image_warn_min_sharpness_spin = QSpinBox()
        self.image_warn_min_sharpness_spin.setRange(0, 2000)
        self.image_warn_min_sharpness_spin.setToolTip(
            tr("Variance of the Laplacian. Lower values accept softer images; 0 disables the check.")
        )
        sharp_row.addWidget(self.image_warn_min_sharpness_spin)
        sharp_row.addStretch()
        quality_layout.addLayout(sharp_row)

        quality_help = QLabel(
            tr("Resolution and sharpness decide how much detail can be traced; file size does not. "
            "A practical floor is a 700px short side, with 900px recommended, and a sharpness "
            "of about 60 for an in-focus photo.")
        )
        quality_help.setWordWrap(True)
        quality_help.setStyleSheet("color: #666; font-size: 11px;")
        quality_layout.addWidget(quality_help)

        layout.addWidget(quality_group)
        
        # Connection Test
        test_btn = QPushButton(tr("Test Hugging Face Connection"))
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
            tr("Google Gemini can generate archaeological symbols from reference images. "
               "API availability depends on your project quota and billing state."),
            "Info"
        )
        layout.addWidget(intro)
        
        # Step 1: Install package
        install_group = QGroupBox(tr("Step 1: Install Required Package"))
        install_layout = QVBoxLayout(install_group)
        
        install_desc = QLabel(
            tr(
                "<b>What is this?</b><br>"
                "The modern 'google-genai' package allows Python to communicate with Gemini 3.1 "
                "and Nano Banana image models.<br><br>"
                "<b>How to install:</b><br>"
                "Click the button below. Installation takes 1-2 minutes.<br>"
                "If it fails, you can install manually by opening Command Prompt and typing:<br>"
                "<code>pip install {package}</code>"
            ).format(package=GEMINI_INSTALL_PACKAGE)
        )
        install_desc.setWordWrap(True)
        install_desc.setTextFormat(Qt.RichText)
        install_layout.addWidget(install_desc)
        
        btn_layout = QHBoxLayout()
        self.install_btn = QPushButton(tr("Install {package}").format(package=GEMINI_INSTALL_PACKAGE))
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
        self.install_btn.setToolTip(tr("Click to automatically install the required Python package"))
        self.install_btn.clicked.connect(self.install_gemini_package)
        btn_layout.addWidget(self.install_btn)
        
        self.install_status = QLabel("")
        self.install_status.setMinimumWidth(120)
        btn_layout.addWidget(self.install_status)
        btn_layout.addStretch()
        install_layout.addLayout(btn_layout)
        
        layout.addWidget(install_group)
        
        # Step 2: Get API Key
        apikey_group = QGroupBox(tr("Step 2: Get Your Free API Key"))
        apikey_layout = QVBoxLayout(apikey_group)
        
        apikey_desc = QLabel(
            tr("<b>What is an API key?</b><br>"
            "An API key is like a password that allows ArchaeoGlyph to use Google's AI service.<br><br>"
            "<b>How to get one (FREE!):</b><br>"
            "1. Click the button below to open Google AI Studio<br>"
            "2. Sign in with your Google account<br>"
            "3. Click 'Create API Key'<br>"
            "4. Copy the generated key (starts with 'AIza...')")
        )
        apikey_desc.setWordWrap(True)
        apikey_desc.setTextFormat(Qt.RichText)
        apikey_layout.addWidget(apikey_desc)
        
        link_btn = QPushButton(tr("Open Google AI Studio"))
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
        link_btn.setToolTip(tr("Opens Google AI Studio in your web browser"))
        link_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl(GEMINI_AI_STUDIO_URL)
            )
        )
        apikey_layout.addWidget(link_btn)
        
        layout.addWidget(apikey_group)
        
        # Step 3: Enter API Key
        key_group = QGroupBox(tr("Step 3: Enter Your API Key"))
        key_layout = QVBoxLayout(key_group)
        
        key_desc = QLabel(
            tr("<b>Paste your API key below:</b><br>"
            "Your key is stored locally and never sent anywhere except Google. It looks like: AIza...")
        )
        key_desc.setWordWrap(True)
        key_desc.setTextFormat(Qt.RichText)
        key_layout.addWidget(key_desc)
        
        key_input_layout = QHBoxLayout()
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.Password)
        self.gemini_key_input.setPlaceholderText(tr("Paste your API key here (AIza...)"))
        self.gemini_key_input.setMinimumHeight(35)
        self.gemini_key_input.setToolTip(tr("Your Google Gemini API key"))
        key_input_layout.addWidget(self.gemini_key_input)
        
        show_key_btn = QPushButton(tr("Show"))
        show_key_btn.setFixedWidth(40)
        show_key_btn.setToolTip(tr("Show/Hide API key"))
        show_key_btn.clicked.connect(self._toggle_key_visibility)
        key_input_layout.addWidget(show_key_btn)
        key_layout.addLayout(key_input_layout)

        self.key_storage_label = QLabel("")
        self.key_storage_label.setWordWrap(True)
        self.key_storage_label.setStyleSheet("color: #555; font-size: 11px;")
        key_layout.addWidget(self.key_storage_label)

        layout.addWidget(key_group)
        
        # Step 4: Test connection
        test_group = QGroupBox(tr("Step 4: Test Your Connection"))
        test_layout = QVBoxLayout(test_group)
        
        test_desc = QLabel(
            tr("<b>Verify everything works:</b><br>"
            "Click the test button to make sure your API key is valid and the connection works.")
        )
        test_desc.setWordWrap(True)
        test_desc.setTextFormat(Qt.RichText)
        test_layout.addWidget(test_desc)
        
        test_btn_layout = QHBoxLayout()
        test_btn = QPushButton(tr("Test Gemini Connection"))
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
        test_btn.setToolTip(tr("Test if your API key works correctly"))
        test_btn.clicked.connect(self.test_gemini_connection)
        test_btn_layout.addWidget(test_btn)
        
        self.gemini_test_result = QLabel("")
        test_btn_layout.addWidget(self.gemini_test_result)
        test_btn_layout.addStretch()
        test_layout.addLayout(test_btn_layout)
        
        layout.addWidget(test_group)
        
        # Usage info
        usage_info = InfoLabel(
            tr("Gemini text models can return SVG. Image models such as Nano Banana return "
               "raster images that are post-processed into factual symbols. If you encounter "
               "HTTP 429, check Gemini quota limits and retry after cooldown."),
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
            tr("Local Stable Diffusion runs AI on YOUR computer - no internet required! "
               "Great for offline field work or sensitive data. "
               "Requires a GPU with 6GB+ VRAM."),
            "Info"
        )
        layout.addWidget(intro)
        
        # Warning
        warning = WarningLabel(
            tr("Advanced Setup Required: This option requires installing additional software "
               "and downloading large model files (4-8 GB). If you're not comfortable with "
               "this, use Google Gemini or Templates instead.")
        )
        layout.addWidget(warning)
        
        # Server URL
        server_group = QGroupBox(tr("Server Configuration"))
        server_layout = QVBoxLayout(server_group)
        
        server_desc = QLabel(
            tr("<b>Server URL:</b><br>"
            "Enter the URL where your Stable Diffusion server is running.<br>"
            "Default is <code>http://127.0.0.1:7860</code> (localhost).")
        )
        server_desc.setWordWrap(True)
        server_desc.setTextFormat(Qt.RichText)
        server_layout.addWidget(server_desc)
        
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel(tr("URL:")))
        self.sd_url_input = QLineEdit()
        self.sd_url_input.setPlaceholderText(tr("http://127.0.0.1:7860"))
        self.sd_url_input.setMinimumHeight(35)
        self.sd_url_input.setToolTip(tr("The URL of your local Stable Diffusion API server"))
        url_layout.addWidget(self.sd_url_input)
        server_layout.addLayout(url_layout)
        
        test_layout = QHBoxLayout()
        test_btn = QPushButton(tr("Test Connection"))
        test_btn.setMinimumHeight(35)
        test_btn.clicked.connect(self.test_sd_connection)
        test_layout.addWidget(test_btn)
        
        self.sd_test_result = QLabel("")
        test_layout.addWidget(self.sd_test_result)
        test_layout.addStretch()
        server_layout.addLayout(test_layout)
        
        layout.addWidget(server_group)
        
        # Setup instructions
        setup_group = QGroupBox(tr("How to Set Up Local Stable Diffusion"))
        setup_layout = QVBoxLayout(setup_group)
        
        setup_text = QTextBrowser()
        setup_text.setOpenExternalLinks(True)
        setup_text.setMaximumHeight(250)
        setup_text.setHtml(local_sd_setup_html())
        setup_layout.addWidget(setup_text)
        
        guide_btn = QPushButton(tr("Open Full Setup Guide (GitHub)"))
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
        header = QLabel(tr("<h3>Get Started in 30 Seconds</h3>"))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # No setup option
        no_setup = QGroupBox(tr("Option 1: Use Templates (NO Setup Required!)"))
        no_setup_layout = QVBoxLayout(no_setup)
        no_setup_layout.addWidget(QLabel(
            tr("<ol>"
            "<li>Open ArchaeoGlyph from the toolbar</li>"
            "<li>Select <b>'Use Template'</b> mode</li>"
            "<li>Choose artifact type (Pottery, Stone Tools, etc.)</li>"
            "<li>Pick your color</li>"
            "<li>Click <b>Generate</b>!</li>"
            "</ol>"
            "<p><i>That's it. No API key or installation needed.</i></p>")
        ))
        layout.addWidget(no_setup)
        
        # Hugging Face option
        hf_opt = QGroupBox(tr("Option 2: Use AI (Hugging Face)"))
        hf_layout = QVBoxLayout(hf_opt)
        hf_layout.addWidget(QLabel(
            tr("<ol>"
            "<li>Go to the <b>Hugging Face</b> tab</li>"
            "<li>Click link to get a <b>token</b></li>"
            "<li>Paste key and click <b>Save Settings</b></li>"
            "<li>Restart QGIS</li>"
            "</ol>"
            "<p><i>Generate symbols with online inference models.</i></p>")
        ))
        layout.addWidget(hf_opt)

        # Gemini option
        gemini_opt = QGroupBox(tr("Option 3: Use AI (Google Gemini)"))
        gemini_layout = QVBoxLayout(gemini_opt)
        gemini_layout.addWidget(QLabel(
            tr("<ol>"
            "<li>Go to the <b>Google Gemini</b> tab</li>"
            "<li>Click <b>Install Package</b> (wait 1-2 min)</li>"
            "<li>Click link to get <b>free API key</b></li>"
            "<li>Paste key and click <b>Save Settings</b></li>"
            "<li>Restart QGIS</li>"
            "</ol>"
            "<p><i>Now you can upload any image and generate custom symbols.</i></p>")
        ))
        layout.addWidget(gemini_opt)
        
        # Tips
        tips = InfoLabel(
            tr("Tip: Start with Templates to try the plugin, then add AI features later!"),
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
        help_text.setHtml(help_html())
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
            self.model_refresh_status.setText(tr("Automatic refresh is disabled."))
            self.model_refresh_status.setStyleSheet("color: #666; font-size: 11px;")
            return

        now_utc = datetime.now(timezone.utc)
        last_checked_raw = self.settings.value("ArcheoGlyph/model_refresh_last_checked_utc", "")
        last_checked = self._parse_utc_iso(last_checked_raw)
        refresh_interval = timedelta(days=7)

        if last_checked is None or (now_utc - last_checked) >= refresh_interval:
            self.refresh_latest_model_recommendations(manual=False, apply_changes=True)
            return

        self.model_refresh_status.setText(
            tr("Latest-model check is up to date (last check: {when}).").format(
                when=self._format_utc(last_checked)
            )
        )
        self.model_refresh_status.setStyleSheet("color: #2f6f44; font-size: 11px;")

    def refresh_latest_model_recommendations(self, manual=False, apply_changes=True):
        """Resolve and apply latest practical HF/SAM recommendations asynchronously."""
        running = getattr(self, "model_refresh_thread", None)
        if running is not None and running.isRunning():
            return

        self.refresh_models_btn.setEnabled(False)
        if hasattr(self, "check_models_btn"):
            self.check_models_btn.setEnabled(False)
        self.model_refresh_status.setText(tr("Checking latest model recommendations..."))
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
        self.model_refresh_thread.result_ready.connect(
            lambda result: self._handle_latest_model_refresh_result(result, manual, apply_changes)
        )
        self.model_refresh_thread.start()

    def _apply_model_recommendations(self, recommended_hf, recommended_sam, recommended_gemini):
        """Apply resolved recommendations to UI/settings and return change summary lines."""
        changed = []

        if recommended_hf:
            normalized = self._normalize_hf_model_id(recommended_hf)
            if normalized and normalized != self._normalize_hf_model_id(self.hf_model_input.text()):
                self.hf_model_input.setText(normalized)
                changed.append(tr("HF model -> {model}").format(model=normalized))
            if normalized:
                self.settings.setValue("ArcheoGlyph/hf_model_id", normalized)

        if recommended_sam:
            idx = self.sam_model_type_combo.findData(recommended_sam)
            if idx < 0:
                self.sam_model_type_combo.addItem(tr("Auto-detected ({model})").format(model=recommended_sam), recommended_sam)
                idx = self.sam_model_type_combo.findData(recommended_sam)
            if idx >= 0:
                current_sam = str(self.sam_model_type_combo.currentData() or "").strip()
                if current_sam != recommended_sam:
                    self.sam_model_type_combo.setCurrentIndex(idx)
                    changed.append(tr("SAM model -> {model}").format(model=recommended_sam))
                self.settings.setValue("ArcheoGlyph/sam_model_type", recommended_sam)

        if recommended_gemini:
            previous_gemini = str(self.settings.value("ArcheoGlyph/gemini_model_id", "") or "").strip()
            if recommended_gemini != previous_gemini:
                changed.append(tr("Gemini preferred -> {model}").format(model=recommended_gemini))
            self.settings.setValue("ArcheoGlyph/gemini_model_id", recommended_gemini)

        return changed

    def _preview_model_recommendations(self, recommended_hf, recommended_sam, recommended_gemini):
        """Return preview-only change lines without applying anything."""
        pending = []

        if recommended_hf:
            normalized = self._normalize_hf_model_id(recommended_hf)
            current_hf = self._normalize_hf_model_id(self.hf_model_input.text())
            if normalized and normalized != current_hf:
                pending.append(tr("HF model: {current} -> {new}").format(
                    current=current_hf or tr("(empty)"), new=normalized
                ))

        if recommended_sam:
            current_sam = str(self.sam_model_type_combo.currentData() or "").strip()
            if current_sam != recommended_sam:
                pending.append(tr("SAM model: {current} -> {new}").format(
                    current=current_sam or tr("(empty)"), new=recommended_sam
                ))

        if recommended_gemini:
            previous_gemini = str(self.settings.value("ArcheoGlyph/gemini_model_id", "") or "").strip()
            if previous_gemini != recommended_gemini:
                pending.append(tr("Gemini preferred: {current} -> {new}").format(
                    current=previous_gemini or tr("(empty)"), new=recommended_gemini
                ))

        return pending

    def _handle_latest_model_refresh_result(self, result, manual, apply_changes):
        """Handle latest-model recommendations from background resolver."""
        self.refresh_models_btn.setEnabled(True)
        if hasattr(self, "check_models_btn"):
            self.check_models_btn.setEnabled(True)
        # Keep the reference: this runs from the thread's own signal, and
        # dropping the last reference here can destroy a still-running QThread.

        if not isinstance(result, dict):
            result = {"status": "error", "message": tr("Invalid model refresh result payload.")}

        status = str(result.get("status", "")).strip().lower()
        message = str(result.get("message", "")).strip()
        recommended_hf = str(result.get("hf_model", "")).strip()
        recommended_sam = str(result.get("sam_model_type", "")).strip()
        recommended_gemini = str(result.get("gemini_model", "")).strip()

        if self.auto_refresh_models_check.isChecked():
            self.settings.setValue("ArcheoGlyph/auto_update_models", "true")
        else:
            self.settings.setValue("ArcheoGlyph/auto_update_models", "false")

        if status == "ok":
            now_utc = datetime.now(timezone.utc)
            self.settings.setValue("ArcheoGlyph/model_refresh_last_checked_utc", self._format_utc(now_utc))

            summary = []
            if apply_changes:
                changed = self._apply_model_recommendations(
                    recommended_hf=recommended_hf,
                    recommended_sam=recommended_sam,
                    recommended_gemini=recommended_gemini,
                )
                if changed:
                    summary.extend(changed)
                else:
                    summary.append(
                        tr("No setting changes were needed (already up to date).")
                    )
                if recommended_gemini:
                    summary.append(
                        tr("Gemini best available: {model}").format(model=recommended_gemini)
                    )
                title = tr("Latest Models Applied")
            else:
                preview = self._preview_model_recommendations(
                    recommended_hf=recommended_hf,
                    recommended_sam=recommended_sam,
                    recommended_gemini=recommended_gemini,
                )
                if preview:
                    summary.append(tr("Preview only (not applied):"))
                    summary.extend(preview)
                else:
                    summary.append(
                        tr("Preview only: current settings are already up to date.")
                    )
                title = tr("Latest Models Preview")

            self.model_refresh_status.setText(" | ".join(summary))
            self.model_refresh_status.setStyleSheet("color: #2f6f44; font-size: 11px;")

            if manual:
                if not apply_changes:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle(title)
                    msg.setText("\n".join(summary))
                    apply_now_btn = msg.addButton(tr("Apply Now"), QMessageBox.AcceptRole)
                    msg.addButton(QMessageBox.Close)
                    msg.exec_()

                    if msg.clickedButton() == apply_now_btn:
                        applied_lines = self._apply_model_recommendations(
                            recommended_hf=recommended_hf,
                            recommended_sam=recommended_sam,
                            recommended_gemini=recommended_gemini,
                        )
                        applied_summary = []
                        if applied_lines:
                            applied_summary.extend(applied_lines)
                        else:
                            applied_summary.append(
                                tr("No setting changes were needed (already up to date).")
                            )
                        if recommended_gemini:
                            applied_summary.append(
                                tr("Gemini best available: {model}").format(
                                    model=recommended_gemini
                                )
                            )

                        self.model_refresh_status.setText(" | ".join(applied_summary))
                        self.model_refresh_status.setStyleSheet("color: #2f6f44; font-size: 11px;")
                        QMessageBox.information(
                            self,
                            tr("Latest Models Applied"),
                            "\n".join(applied_summary),
                        )
                else:
                    QMessageBox.information(self, title, "\n".join(summary))
        else:
            fallback_msg = message or tr("Latest model refresh failed.")
            self.model_refresh_status.setText(fallback_msg)
            self.model_refresh_status.setStyleSheet("color: #b00020; font-size: 11px;")
            if manual:
                QMessageBox.warning(self, tr("Latest Model Refresh Failed"), fallback_msg)

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
            tr("Install SAM Packages"),
            tr("Install SAM packages now?\n\n"
            "This installs:\n"
            "- segment-anything (SAM1 local checkpoint)\n"
            "- transformers + huggingface_hub (SAM2/3 via HF)\n\n"
            "Note: SAM still needs 'torch'. If torch is missing, install it first "
            "(CPU build is okay for basic use)."),
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        python_path = self._get_python_executable()
        self.sam_install_btn.setEnabled(False)
        self.sam_install_btn.setText(tr("Installing..."))
        self.sam_status_label.setText(tr("Installing SAM packages..."))
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
            self.sam_status_label.setText(tr("Installing SAM: {line}").format(line=last_line))
            self.sam_status_label.setStyleSheet("color: orange; font-size: 11px;")

    def _handle_sam_install_finished(self, exit_code, exit_status):
        """Handle SAM installer completion."""
        self.sam_install_btn.setEnabled(True)
        self.sam_install_btn.setText(tr("Install SAM Packages"))
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            QMessageBox.information(
                self,
                tr("Installed"),
                tr("SAM packages installed successfully.\n"
                "If this is first-time setup, restart QGIS.")
            )
        else:
            python_path = self._get_python_executable()
            QMessageBox.warning(
                self,
                tr("Install Failed"),
                tr(
                    "Could not install SAM packages automatically.\n\n"
                    "Manual command:\n"
                    "{command}"
                ).format(
                    command=(
                        f"{python_path} -m pip install --user "
                        "segment-anything transformers huggingface_hub pillow"
                    )
                )
            )
        self._refresh_sam_status()

    def _handle_sam_install_error(self, error):
        """Handle SAM installer process errors."""
        self.sam_install_btn.setEnabled(True)
        self.sam_install_btn.setText(tr("Install SAM Packages"))
        self.sam_status_label.setText(tr("SAM install process error: {error}").format(error=error))
        self.sam_status_label.setStyleSheet("color: red; font-size: 11px;")
        self._refresh_sam_status()

    def _open_sam_checkpoint_download(self):
        """Open official SAM ViT-B checkpoint download URL."""
        QDesktopServices.openUrl(
            QUrl("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        )
        QMessageBox.information(
            self,
            tr("Download Started"),
            tr("Browser download opened for sam_vit_b_01ec64.pth.\n"
            "After download, click 'Auto-Find Downloaded File'.")
        )

    def _open_sam_hf_models(self):
        """Open Hugging Face SAM model search page (works even when specific models are gated)."""
        QDesktopServices.openUrl(QUrl("https://huggingface.co/models?search=facebook%2Fsam"))

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
                    tr("Checkpoint Found"),
                    tr("SAM checkpoint found and selected:\n{path}").format(path=path)
                )
                self._refresh_sam_status()
                return

        QMessageBox.information(
            self,
            tr("Not Found"),
            tr("No SAM checkpoint was found in common folders.\n"
            "Click 'Download ViT-B Checkpoint' first.")
        )
        self._refresh_sam_status()

    def _show_sam_quick_guide(self):
        """Show beginner-friendly SAM setup instructions."""
        QMessageBox.information(
            self,
            tr("SAM Quick Guide"),
            tr("SAM setup (beginner):\n\n"
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
            "If SAM is not ready, ArcheoGlyph automatically falls back to OpenCV.")
        )

    def _profile_dir(self):
        """QGIS profile directory that holds downloaded models."""
        from ..generators.contour_generator import profile_base_dir

        return profile_base_dir()

    def _selected_onnx_key(self):
        return str(self.onnx_model_combo.currentData() or DEFAULT_MODEL_KEY)

    def _refresh_onnx_status(self):
        """Describe runtime and model availability for the ONNX backend."""
        if not hasattr(self, "onnx_status_label"):
            return
        key = self._selected_onnx_key()
        spec = MODEL_SPECS.get(key)
        runtime = find_spec("onnxruntime") is not None
        ready = bool(spec) and is_installed(spec, self._profile_dir())

        self.onnx_download_btn.setEnabled(bool(spec) and not ready)
        self.onnx_verify_btn.setEnabled(ready)
        self.onnx_install_runtime_btn.setEnabled(not runtime)

        parts = []
        if runtime:
            parts.append(tr("onnxruntime installed"))
        else:
            parts.append(tr("onnxruntime missing - press 'Install onnxruntime'"))
        if ready:
            parts.append(tr("model downloaded ({size} MB)").format(
                size=spec.size // (1024 * 1024)
            ))
        elif spec:
            parts.append(tr("model not downloaded ({size} MB)").format(
                size=spec.size // (1024 * 1024)
            ))

        colour = "green" if (runtime and ready) else "#9a6700"
        if runtime and ready:
            parts.append(tr("Auto Trace will use it for photographs."))
        self.onnx_status_label.setText(" | ".join(parts))
        self.onnx_status_label.setStyleSheet(f"color: {colour}; font-size: 11px;")

    def install_onnx_runtime(self):
        """Install onnxruntime with pip, reusing the package installer."""
        self._start_pip_install(
            "onnxruntime",
            button=self.onnx_install_runtime_btn,
            done=self._refresh_onnx_status,
        )

    def download_onnx_model(self):
        """Download the selected model in the background, verifying it."""
        spec = MODEL_SPECS.get(self._selected_onnx_key())
        if spec is None:
            return
        size_mb = spec.size // (1024 * 1024)
        reply = QMessageBox.question(
            self,
            tr("Download model"),
            tr(
                "Download {label}?\n\n"
                "About {size} MB, stored in your QGIS profile and verified by SHA-256."
            ).format(label=spec.label, size=size_mb),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        self.onnx_download_btn.setEnabled(False)
        self.onnx_progress.setVisible(True)
        self.onnx_progress.setRange(0, 100)
        self.onnx_progress.setValue(0)
        self.onnx_status_label.setText(tr("Downloading {filename}...").format(filename=spec.filename))

        self.onnx_download_thread = ModelDownloadThread(spec, self._profile_dir())
        self.onnx_download_thread.progress.connect(self._on_onnx_progress)
        self.onnx_download_thread.result_ready.connect(self._on_onnx_download_finished)
        self.onnx_download_thread.start()

    def _on_onnx_progress(self, received, total):
        if total > 0:
            self.onnx_progress.setValue(int(100 * received / total))

    def _on_onnx_download_finished(self, result):
        self.onnx_progress.setVisible(False)
        self.onnx_download_btn.setEnabled(True)
        message = str((result or {}).get("message", ""))
        if (result or {}).get("ok"):
            QMessageBox.information(
                self, tr("Model ready"), message or tr("Download complete.")
            )
        else:
            QMessageBox.warning(
                self, tr("Download failed"),
                message or tr("The download did not complete."),
            )
        self._refresh_onnx_status()

    def verify_onnx_model(self):
        """Re-check the stored model against its published checksum."""
        spec = MODEL_SPECS.get(self._selected_onnx_key())
        if spec is None:
            return
        try:
            ok = verify_model(spec, self._profile_dir())
        except OSError as e:
            QMessageBox.warning(self, tr("Verify failed"), str(e))
            return
        if ok:
            QMessageBox.information(
                self, tr("Model verified"),
                tr("{filename} matches its published SHA-256.\n\n{path}").format(
                    filename=spec.filename, path=model_path(spec, self._profile_dir())
                ),
            )
        else:
            QMessageBox.warning(
                self, tr("Model does not match"),
                tr("The stored file does not match its published checksum. "
                "Delete it and download again."),
            )
        self._refresh_onnx_status()

    def copy_diagnostics(self):
        """Show the runtime report and put it on the clipboard."""
        from ..diagnostics import report_text

        try:
            text = report_text(self._profile_dir())
        except Exception as e:
            QMessageBox.warning(self, tr("Diagnostics failed"), str(e))
            return

        QApplication.clipboard().setText(text)
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle(tr("Diagnostics copied"))
        message.setText(tr("The report was copied to the clipboard."))
        message.setDetailedText(text)
        message.exec_()

    def _refresh_sam_status(self):
        """Update SAM readiness status text."""
        model_choice = str(self.sam_model_type_combo.currentData() or self.sam_model_type_combo.currentText()).strip()
        uses_hf_sam = model_choice.lower().startswith("hf:")
        checkpoint = self.sam_checkpoint_input.text().strip()
        checkpoint_ok = bool(checkpoint and os.path.exists(checkpoint))

        # Probe without importing: a real torch import freezes the dialog for
        # seconds, and this runs on every keystroke in the checkpoint field.
        dep_missing = []
        if not find_spec("torch"):
            dep_missing.append("torch")
        if uses_hf_sam:
            if not find_spec("transformers"):
                dep_missing.append("transformers")
            if dep_missing:
                self.sam_status_label.setText(
                    tr("SAM2/3 not ready (missing package(s): {packages}). "
                       "OpenCV backend will be used until setup is complete.").format(
                        packages=tr(", ").join(dep_missing)
                    )
                )
                self.sam_status_label.setStyleSheet("color: #9a6700; font-size: 11px;")
                return
            model_id = model_choice[3:] if model_choice.lower().startswith("hf:") else model_choice
            self.sam_status_label.setText(
                tr("SAM ready (HF): {model} (checkpoint not required).").format(
                    model=model_id
                )
            )
            self.sam_status_label.setStyleSheet("color: green; font-size: 11px;")
            return

        if not find_spec("segment_anything"):
            dep_missing.append("segment-anything")

        if checkpoint_ok and not dep_missing:
            self.sam_status_label.setText(tr("SAM ready: dependencies and checkpoint detected."))
            self.sam_status_label.setStyleSheet("color: green; font-size: 11px;")
            return

        issues = []
        if not checkpoint_ok:
            issues.append(tr("checkpoint missing"))
        if dep_missing:
            issues.append(tr("missing package(s): {packages}").format(
                packages=tr(", ").join(dep_missing)
            ))

        self.sam_status_label.setText(
            tr("SAM not ready ({issues}). "
               "OpenCV backend will be used until SAM setup is complete.").format(
                issues=tr("; ").join(issues)
            )
        )
        self.sam_status_label.setStyleSheet("color: #9a6700; font-size: 11px;")

    def _browse_sam_checkpoint(self):
        """Browse SAM checkpoint file."""
        start_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(start_dir):
            start_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select SAM Checkpoint"),
            start_dir,
            tr("SAM Checkpoint (sam_vit_*.pth *.pth *.pt);;"
               "PyTorch Checkpoint (*.pth *.pt);;All Files (*)")
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
            
    def _describe_key_storage(self):
        """Tell the user where API keys are kept."""
        if hasattr(self, "key_storage_label"):
            self.key_storage_label.setText(storage_description(self.settings))

    def load_settings(self):
        """Load saved settings."""
        language = str(self.settings.value(LANGUAGE_SETTING, "auto") or "auto")
        index = self.language_combo.findData(language)
        self.language_combo.setCurrentIndex(index if index >= 0 else 0)
        gemini_key = get_api_key("gemini", self.settings)
        hf_key = get_api_key("huggingface", self.settings)
        # Loading must not change stored settings; normalise for display only.
        hf_model = self._normalize_hf_model_id(
            self.settings.value('ArcheoGlyph/hf_model_id', HF_DEFAULT_MODEL_ID)
        )

        mask_backend = self.settings.value('ArcheoGlyph/mask_backend', 'auto')
        sam_checkpoint = self.settings.value('ArcheoGlyph/sam_checkpoint_path', '')
        sam_model_type = self.settings.value('ArcheoGlyph/sam_model_type', 'hf:facebook/sam2.1-hiera-large')
        hf_overlay_linework = str(
            self.settings.value('ArcheoGlyph/hf_overlay_linework', 'false')
        ).strip().lower() in ("1", "true", "yes", "on")
        auto_update_models = self._parse_bool_setting(
            self.settings.value('ArcheoGlyph/auto_update_models', 'true'),
            default=True,
        )
        autotrace_detail_mode = str(
            self.settings.value('ArcheoGlyph/autotrace_detail_mode', 'fast')
        ).strip().lower()
        if autotrace_detail_mode not in ("fast", "precise"):
            autotrace_detail_mode = "fast"
        image_warn_min_sharpness = self._parse_int_setting(
            self.settings.value('ArcheoGlyph/image_warn_min_sharpness', 60),
            default=60,
        )
        image_warn_min_short_px = self._parse_int_setting(
            self.settings.value('ArcheoGlyph/image_warn_min_short_px', 700),
            default=700,
        )
        image_warn_recommended_short_px = self._parse_int_setting(
            self.settings.value('ArcheoGlyph/image_warn_recommended_short_px', 900),
            default=900,
        )
            
        sd_url = self.settings.value('ArcheoGlyph/sd_server', 'http://127.0.0.1:7860')
        
        self.gemini_key_input.setText(gemini_key)
        self.hf_key_input.setText(hf_key)
        self.hf_model_input.setText(hf_model)
        self.auto_refresh_models_check.setChecked(auto_update_models)
        self.sd_url_input.setText(sd_url)
        mode_idx = self.autotrace_detail_mode_combo.findData(autotrace_detail_mode)
        if mode_idx < 0:
            mode_idx = self.autotrace_detail_mode_combo.findData("precise")
        if mode_idx >= 0:
            self.autotrace_detail_mode_combo.setCurrentIndex(mode_idx)

        image_warn_min_sharpness = max(0, min(2000, int(image_warn_min_sharpness)))
        image_warn_min_short_px = max(256, min(4096, int(image_warn_min_short_px)))
        image_warn_recommended_short_px = max(
            image_warn_min_short_px,
            min(4096, int(image_warn_recommended_short_px)),
        )
        self.image_warn_min_sharpness_spin.setValue(image_warn_min_sharpness)
        self.image_warn_min_short_px_spin.setValue(image_warn_min_short_px)
        self.image_warn_recommended_short_px_spin.setValue(image_warn_recommended_short_px)

        idx = self.mask_backend_combo.findData(str(mask_backend).strip().lower())
        if idx >= 0:
            self.mask_backend_combo.setCurrentIndex(idx)
        self.sam_checkpoint_input.setText(str(sam_checkpoint))
        if not str(sam_checkpoint).strip():
            for path in self._get_candidate_sam_paths():
                if os.path.exists(path):
                    self.sam_checkpoint_input.setText(path)
                    break
        sam_model_type = str(sam_model_type).strip() or "hf:facebook/sam2.1-hiera-large"
        if sam_model_type.lower() == "hf:facebook/sam3-hiera-large":
            sam_model_type = "hf:facebook/sam2.1-hiera-large"
        type_idx = self.sam_model_type_combo.findData(sam_model_type)
        if type_idx < 0:
            type_idx = self.sam_model_type_combo.findText(sam_model_type)
        if type_idx >= 0:
            self.sam_model_type_combo.setCurrentIndex(type_idx)
        self.hf_overlay_linework_check.setChecked(hf_overlay_linework)
        onnx_model_key = str(
            self.settings.value('ArcheoGlyph/onnx_bg_model', DEFAULT_MODEL_KEY) or DEFAULT_MODEL_KEY
        )
        onnx_idx = self.onnx_model_combo.findData(onnx_model_key)
        if onnx_idx < 0:
            onnx_idx = self.onnx_model_combo.findData(DEFAULT_MODEL_KEY)
        if onnx_idx >= 0:
            self.onnx_model_combo.setCurrentIndex(onnx_idx)
        self._refresh_onnx_status()
        self._refresh_sam_status()
        self._describe_key_storage()

        # Check if package is installed
        try:
            package_found = importlib.util.find_spec("google.genai") is not None
            legacy_found = importlib.util.find_spec("google.generativeai") is not None
            if package_found:
                self.install_status.setText(tr("Installed"))
                self.install_status.setStyleSheet("color: green; font-weight: bold;")
            elif legacy_found:
                self.install_status.setText(tr("Legacy only"))
                self.install_status.setStyleSheet("color: #8a4b00; font-weight: bold;")
            else:
                self.install_status.setText(tr("Not installed"))
                self.install_status.setStyleSheet("color: red;")
        except Exception:
            self.install_status.setText(tr("Not installed"))
            self.install_status.setStyleSheet("color: red;")

        self._maybe_auto_refresh_latest_models()

    def save_settings(self):
        """Save settings."""
        self.settings.setValue(
            LANGUAGE_SETTING, str(self.language_combo.currentData() or "auto")
        )
        set_api_key("gemini", self.gemini_key_input.text(), self.settings)
        set_api_key("huggingface", self.hf_key_input.text(), self.settings)
        self.settings.setValue('ArcheoGlyph/hf_model_id', self._normalize_hf_model_id(self.hf_model_input.text()))
        mask_backend = self.mask_backend_combo.currentData()
        onnx_model_key = self._selected_onnx_key()
        sam_checkpoint = self.sam_checkpoint_input.text().strip()

        if mask_backend == "onnx":
            spec = MODEL_SPECS.get(onnx_model_key)
            missing = []
            if not find_spec("onnxruntime"):
                missing.append(tr("the onnxruntime package"))
            if spec is None or not is_installed(spec, self._profile_dir()):
                missing.append(tr("the model file"))
            if missing:
                QMessageBox.warning(
                    self,
                    tr("Background-removal model not ready"),
                    tr(
                        "Auto Trace needs {missing}.\n"
                        "Switching to Auto for now; the model is used automatically "
                        "once installed."
                    ).format(missing=tr(" and ").join(missing)),
                )
                mask_backend = "auto"
                idx = self.mask_backend_combo.findData("auto")
                if idx >= 0:
                    self.mask_backend_combo.setCurrentIndex(idx)
        sam_model_type = str(self.sam_model_type_combo.currentData() or self.sam_model_type_combo.currentText()).strip()
        uses_hf_sam = sam_model_type.lower().startswith("hf:")

        # Safety: strict validation only when user forces SAM-only backend.
        if mask_backend == "sam":
            if uses_hf_sam:
                if not (find_spec("torch") and find_spec("transformers")):
                    QMessageBox.warning(
                        self,
                        tr("SAM Package Missing"),
                        tr("SAM2/3 mode needs torch + transformers.\n"
                        "Switching backend to OpenCV for now.\n\n"
                        "Use 'Install SAM Packages' first.")
                    )
                    mask_backend = "opencv"
                    idx = self.mask_backend_combo.findData("opencv")
                    if idx >= 0:
                        self.mask_backend_combo.setCurrentIndex(idx)
            else:
                if not sam_checkpoint or not os.path.exists(sam_checkpoint):
                    QMessageBox.warning(
                        self,
                        tr("SAM Not Ready"),
                        tr("SAM backend was selected, but checkpoint file is missing.\n"
                        "Switching backend to OpenCV for now.")
                    )
                    mask_backend = "opencv"
                    idx = self.mask_backend_combo.findData("opencv")
                    if idx >= 0:
                        self.mask_backend_combo.setCurrentIndex(idx)
                else:
                    if not (find_spec("torch") and find_spec("segment_anything")):
                        QMessageBox.warning(
                            self,
                            tr("SAM Package Missing"),
                            tr("SAM checkpoint exists, but required packages are missing.\n"
                            "Switching backend to OpenCV for now.\n\n"
                            "Use 'Install SAM Packages' first.")
                        )
                        mask_backend = "opencv"
                        idx = self.mask_backend_combo.findData("opencv")
                        if idx >= 0:
                            self.mask_backend_combo.setCurrentIndex(idx)

        self.settings.setValue('ArcheoGlyph/mask_backend', mask_backend)
        self.settings.setValue('ArcheoGlyph/onnx_bg_model', onnx_model_key)
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
        detail_mode = str(self.autotrace_detail_mode_combo.currentData() or "fast").strip().lower()
        if detail_mode not in ("fast", "precise"):
            detail_mode = "fast"
        warn_min_sharpness = int(self.image_warn_min_sharpness_spin.value())
        warn_min_short_px = int(self.image_warn_min_short_px_spin.value())
        warn_rec_short_px = max(
            warn_min_short_px,
            int(self.image_warn_recommended_short_px_spin.value()),
        )
        self.image_warn_recommended_short_px_spin.setValue(warn_rec_short_px)
        self.settings.setValue('ArcheoGlyph/autotrace_detail_mode', detail_mode)
        self.settings.setValue('ArcheoGlyph/image_warn_min_sharpness', warn_min_sharpness)
        self.settings.setValue('ArcheoGlyph/image_warn_min_short_px', warn_min_short_px)
        self.settings.setValue('ArcheoGlyph/image_warn_recommended_short_px', warn_rec_short_px)
        self.settings.setValue('ArcheoGlyph/sd_server', self.sd_url_input.text())
        self._refresh_sam_status()
        
        QMessageBox.information(
            self, 
            tr("Settings Saved"), 
            tr("Your settings have been saved!\n\n"
            "If you installed a new package, please restart QGIS.")
        )

    def _parse_int_setting(self, value, default=0):
        """Parse integer settings safely with fallback."""
        try:
            return int(str(value).strip())
        except Exception:
            return int(default)

    def test_huggingface_connection(self):
        """Test Hugging Face connection asynchronously."""
        api_key = self.hf_key_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, tr("No Token"), tr("Please enter Hugging Face token."))
            return

        trigger_button = self.sender()
        if trigger_button:
            trigger_button.setEnabled(False)

        self.hf_test_result.setText(tr("Testing..."))
        self.hf_test_result.setStyleSheet("color: orange;")

        model_id = self._normalize_hf_model_id(self.hf_model_input.text().strip())
        self.hf_model_input.setText(model_id)

        candidate_models = []
        for mid in [model_id] + list(HF_FALLBACK_MODEL_IDS) + list(HF_LEGACY_MODEL_ALIASES.keys()):
            normalized = self._normalize_hf_model_id(mid)
            if normalized not in candidate_models:
                candidate_models.append(normalized)

        self.hf_test_thread = HfConnectionTestThread(api_key, candidate_models)
        self.hf_test_thread.result_ready.connect(
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
            self.hf_test_result.setText(tr("Connected"))
            self.hf_test_result.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(self, tr("Success"), tr("Connected with model: {model}").format(
                                                             model=model or requested_model_id
                                                         ))
            return

        if status == "loading":
            if model and model != requested_model_id:
                self.hf_model_input.setText(model)
                self.settings.setValue('ArcheoGlyph/hf_model_id', model)
            self.hf_test_result.setText(tr("Loading model..."))
            self.hf_test_result.setStyleSheet("color: orange;")
            QMessageBox.information(
                self,
                tr("Loading"),
                tr("Connected, but model is initializing: {model}").format(
                    model=model or requested_model_id
                )
            )
            return

        if status == "invalid_token":
            self.hf_test_result.setText(tr("Invalid token"))
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(self, tr("Invalid Token"), tr("Please check your Hugging Face token."))
            return

        if status == "forbidden":
            self.hf_test_result.setText(tr("Model access denied (403)"))
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                tr("Model Access Denied"),
                tr("Model terms may need acceptance on Hugging Face, or the model is restricted.")
            )
            return

        if status == "not_found":
            self.hf_test_result.setText(tr("Model not found (404)"))
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                tr("Model Not Found"),
                tr(
                    "No candidate model was found.\n"
                    "Try '{model}' or 'Qwen/Qwen-Image'."
                ).format(model=HF_DEFAULT_MODEL_ID)
            )
            return

        if status == "error":
            self.hf_test_result.setText(tr("Failed"))
            self.hf_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self, tr("Connection Failed"), message or tr("Unknown error")
            )
            return

        self.hf_test_result.setText(tr("Failed"))
        self.hf_test_result.setStyleSheet("color: red;")
        QMessageBox.warning(self, tr("Connection Failed"), tr("Unexpected test result."))
        
    def _start_pip_install(self, package, button=None, done=None):
        """
        Install a package with pip in the background, reusing one QProcess and
        the accumulated-output handling used by the Gemini installer.
        """
        reply = QMessageBox.question(
            self,
            tr("Install package"),
            tr(
                "Install '{package}' into the Python that QGIS uses?\n\n"
                "The installer runs in the background; you can keep using QGIS."
            ).format(package=package),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        if button is not None:
            button.setEnabled(False)
            button.setText(tr("Installing..."))

        self._pip_target = {"package": package, "button": button, "done": done, "label": button.text() if button else ""}
        self._pip_log = {}
        self.pip_process = QProcess(self)
        self.pip_process.readyReadStandardOutput.connect(self._handle_pip_output)
        self.pip_process.readyReadStandardError.connect(self._handle_pip_output)
        self.pip_process.finished.connect(self._handle_pip_finished)
        self.pip_process.start(self._get_python_executable(), ["-m", "pip", "install", "--user", package])

    def _handle_pip_output(self):
        for stream, reader in (
            ("stdout", self.pip_process.readAllStandardOutput),
            ("stderr", self.pip_process.readAllStandardError),
        ):
            text = bytes(reader()).decode("utf-8", errors="replace").strip()
            if text:
                self._pip_log.setdefault(stream, []).append(text)

    def _handle_pip_finished(self, exit_code, _exit_status):
        self._handle_pip_output()
        target = getattr(self, "_pip_target", {})
        button = target.get("button")
        package = target.get("package", "package")
        if button is not None:
            button.setEnabled(True)
            button.setText(tr("Install {package}").format(package=package))

        if exit_code == 0:
            QMessageBox.information(
                self, tr("Installed"),
                tr(
                    "'{package}' was installed.\n\n"
                    "Restart QGIS if it is not picked up immediately."
                ).format(package=package),
            )
        else:
            log_text = "STDOUT:\n" + "\n".join(self._pip_log.get("stdout", []))
            log_text += "\n\nSTDERR:\n" + "\n".join(self._pip_log.get("stderr", []))
            message = QMessageBox(self)
            message.setIcon(QMessageBox.Warning)
            message.setWindowTitle(tr("Installation failed"))
            message.setText(tr("Installing '{package}' failed (exit code {code}).").format(
                                package=package, code=exit_code
                            ))
            message.setDetailedText(log_text)
            message.exec_()

        callback = target.get("done")
        if callable(callback):
            callback()

    def install_gemini_package(self):
        """Install Google GenAI SDK using QProcess (Async)."""
        reply = QMessageBox.question(
            self,
            tr("Install Package"),
            tr(
                "This will install '{package}' package.\n\n"
                "The installer will run in the background.\n"
                "You can continue using QGIS while it installs.\n\n"
                "Continue?"
            ).format(package=GEMINI_INSTALL_PACKAGE),
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
            
        self.install_btn.setEnabled(False)
        self.install_btn.setText(tr("Installing..."))
        self.install_status.setText(tr("Starting..."))
        self.install_status.setStyleSheet("color: orange;")
        
        # Setup QProcess
        from qgis.PyQt.QtCore import QProcess
        
        self._install_log = {}
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
        args = ['-m', 'pip', 'install', '--user', GEMINI_INSTALL_PACKAGE]
        
        self.process.start(python_path, args)
        
    def _handle_process_output(self):
        """Handle process output (kept, since reading it drains the buffer)."""
        stdout = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace').strip()
        stderr = bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace').strip()
        for stream, text in (("stdout", stdout), ("stderr", stderr)):
            if text:
                self._install_log.setdefault(stream, []).append(text)
        msg = stdout or stderr

        if msg:
            last_line = msg.splitlines()[-1] if "\n" in msg else msg
            # Show last line in status if it's not too long
            if len(last_line) < 50:
                self.install_status.setText(tr("Installing: {line}").format(line=last_line))
            else:
                self.install_status.setText(tr("Installing..."))
                
    def _handle_process_finished(self, exit_code, exit_status):
        """Handle install completion."""
        self.install_btn.setEnabled(True)
        self.install_btn.setText(tr("Install {package}").format(package=GEMINI_INSTALL_PACKAGE))
        
        from qgis.core import QgsMessageLog, Qgis
        
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.install_status.setText(tr("Installed"))
            self.install_status.setStyleSheet("color: green; font-weight: bold;")
            QgsMessageLog.logMessage("ArcheoGlyph: Package installed successfully.", "ArcheoGlyph", Qgis.Success)
            
            QMessageBox.information(
                self, 
                tr("Success"), 
                tr("Package installed successfully!\n\n"
                "Please RESTART QGIS to apply changes.")
            )
        else:
            self.install_status.setText(tr("Failed"))
            self.install_status.setStyleSheet("color: red;")
            
            # Use the accumulated log: the streams were already drained while
            # the installer was running, so reading them here returns nothing.
            self._handle_process_output()
            log_data = getattr(self, "_install_log", {})
            stdout = "\n".join(log_data.get("stdout", []))
            stderr = "\n".join(log_data.get("stderr", []))
            full_log = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            QgsMessageLog.logMessage(f"ArcheoGlyph Install Failed:\n{full_log}", "ArcheoGlyph", Qgis.Critical)
            
            # Show error details
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle(tr("Installation Failed"))
            msg.setText(tr("Installation failed (Exit Code: {code}).").format(code=exit_code))
            msg.setInformativeText(tr("Check the 'ArcheoGlyph' tab in QGIS Log Messages panel for full details."))
            msg.setDetailedText(full_log)
            copy_button = msg.addButton(tr("Copy Command"), QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Ok)

            msg.exec_()

            if msg.clickedButton() is copy_button:
                clipboard = QApplication.clipboard()
                cmd = f'"{self._get_python_executable()}" -m pip install --user {GEMINI_INSTALL_PACKAGE}'
                clipboard.setText(cmd)
                QMessageBox.information(self, tr("Copied"), tr("Command copied to clipboard!\nPaste it in your terminal."))
            
    def _handle_process_error(self, error):
        """Handle process start error."""
        self.install_btn.setEnabled(True)
        self.install_btn.setText(tr("Install {package}").format(package=GEMINI_INSTALL_PACKAGE))
        self.install_status.setText(tr("Error"))
        self.install_status.setStyleSheet("color: red;")
        
        QMessageBox.warning(
            self,
            tr("Process Error"),
            tr("Failed to start installer.\nError code: {error}").format(error=error)
        )
            
    def test_gemini_connection(self):
        """Test Gemini API connection (Async)."""
        api_key = self.gemini_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(
                self, 
                tr("No API Key"), 
                tr("Please enter your API key first!\n\n"
                "If you don't have one:\n"
                "1. Click 'Open Google AI Studio'\n"
                "2. Sign in with Google\n"
                "3. Create a new key")
            )
            return
            
        self.gemini_test_result.setText(tr("Testing..."))
        self.gemini_test_result.setStyleSheet("color: orange;")
        
        # Disable button during test
        sender = self.sender()
        if sender:
            sender.setEnabled(False)
            
        self.test_thread = GeminiTestThread(api_key)
        self.test_thread.result_ready.connect(lambda s, m: self._handle_test_result(s, m, sender))
        self.test_thread.start()
        
    def _handle_test_result(self, success, message, button):
        """Handle API test result."""
        if button:
            button.setEnabled(True)
            
        if success:
            self.gemini_test_result.setText(tr("Connected"))
            self.gemini_test_result.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(
                self,
                tr("Success"),
                tr(
                    "Connection successful!\n\n"
                    "AI Response: {response}\n\n"
                    "You're all set! Click 'Save Settings' and start generating symbols!"
                ).format(response=message[:100])
            )
        else:
            if "Package" in message:
                self.gemini_test_result.setText(tr("Package missing"))
                self.gemini_test_result.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self,
                    tr("Package Not Installed"),
                    tr(
                        "The {package} package is not installed.\n\n"
                        "Please:\n"
                        "1. Complete Step 1 (Install Package)\n"
                        "2. Restart QGIS\n"
                        "3. Try again"
                    ).format(package=GEMINI_INSTALL_PACKAGE)
                )
            elif "API_KEY_INVALID" in message or "invalid" in message.lower():
                self.gemini_test_result.setText(tr("Invalid key"))
                self.gemini_test_result.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self, 
                    tr("Invalid API Key"), 
                    tr("Your API key appears to be invalid.\n\n"
                    "Please:\n"
                    "1. Go to Google AI Studio\n"
                    "2. Create a NEW API key\n"
                    "3. Copy and paste it here")
                )
            else:
                self.gemini_test_result.setText(tr("Failed"))
                self.gemini_test_result.setStyleSheet("color: red;")
                QMessageBox.warning(self, tr("Connection Failed"), tr("Error: {message}").format(message=message))

    def test_sd_connection(self):
        """Test Stable Diffusion server connection."""
        url = self.sd_url_input.text().strip()
        
        if not url:
            url = "http://127.0.0.1:7860"
            self.sd_url_input.setText(url)
            
        self.sd_test_result.setText(tr("Testing..."))
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
                    self.sd_test_result.setText(tr("Connected ({count} models)").format(count=len(data)))
                    self.sd_test_result.setStyleSheet("color: green; font-weight: bold;")
                    QMessageBox.information(
                        self,
                        tr("Success"),
                        tr(
                            "Connected to Stable Diffusion!\n\n"
                            "Found {count} model(s).\n\n"
                            "Don't forget to click 'Save Settings'!"
                        ).format(count=len(data))
                    )
                    
        except Exception as e:
            self.sd_test_result.setText(tr("Not connected"))
            self.sd_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                tr("Connection Failed"),
                tr(
                    "Cannot connect to: {url}\n\n"
                    "Make sure:\n"
                    "1. Stable Diffusion WebUI is running\n"
                    "2. It was started with --api flag\n"
                    "3. The URL is correct\n\n"
                    "Error: {error}"
                ).format(url=url, error=str(e))
            )


class LatestModelRefreshThread(QThread):
    """Resolve latest practical model recommendations for HF/SAM/Gemini."""
    # Not named "finished": QThread already defines that signal, and shadowing
    # it breaks the idiomatic finished -> deleteLater connection.
    result_ready = pyqtSignal(object)  # dict result payload

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
                tags = payload.get("tags", []) or []
                if not isinstance(tags, list):
                    tags = []
                return {
                    "ok": True,
                    "model_id": model_id,
                    "last_modified": payload.get("lastModified", ""),
                    "ungated": bool(ungated),
                    "private": bool(payload.get("private", False)),
                    "downloads": int(payload.get("downloads", 0) or 0),
                    "pipeline_tag": str(payload.get("pipeline_tag", "") or "").strip().lower(),
                    "tags": [str(t).strip().lower() for t in tags if str(t).strip()],
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

        def _prefers_reference_image_flow(meta):
            pipeline_tag = str(meta.get("pipeline_tag", "") or "").strip().lower()
            if pipeline_tag == "image-to-image":
                return True
            tags = list(meta.get("tags", []) or [])
            return (
                "image-to-image" in tags or
                "img2img" in tags or
                "inpainting" in tags
            )

        ranked = sorted(
            records,
            key=lambda m: (
                1 if m.get("ungated") else 0,
                1 if _prefers_reference_image_flow(m) else 0,
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

        ranked = sorted(
            sam_records,
            key=lambda m: (
                1 if m.get("ungated") else 0,
                self._parse_last_modified(m.get("last_modified")),
                int(m.get("downloads", 0)),
            ),
            reverse=True,
        )
        best = ranked[0]
        return f"hf:{best.get('model_id', '')}", sam_records

    def _select_best_gemini(self):
        if not self.gemini_api_key:
            return ""

        try:
            from google import genai

            client = genai.Client(api_key=self.gemini_api_key)

            available = []
            for model in client.models.list():
                name = _normalize_gemini_model_name(getattr(model, "name", ""))
                if not name:
                    continue
                low = name.lower()
                if "gemini" not in low or _is_excluded_gemini_model(name):
                    continue
                available.append(name)

            if not available:
                return ""

            preferred = []
            for alias in list(GEMINI_IMAGE_MODEL_CANDIDATES) + list(GEMINI_TEXT_MODEL_CANDIDATES):
                normalized_alias = _normalize_gemini_model_name(alias)
                exact = [name for name in available if name == normalized_alias]
                if exact:
                    preferred.extend(exact)
                    continue
                prefix_matches = [name for name in available if name.startswith(normalized_alias)]
                prefix_matches.sort(key=_rank_gemini_model, reverse=True)
                if prefix_matches:
                    preferred.append(prefix_matches[0])

            ordered = []
            for name in preferred:
                if name not in ordered:
                    ordered.append(name)

            remaining = sorted(
                [name for name in available if name not in ordered],
                key=_rank_gemini_model,
                reverse=True,
            )
            ordered.extend(remaining)

            if not ordered:
                return ""

            available = ordered
            return available[0]
        except Exception:
            return ""

    def run(self):
        try:
            hf_model, hf_records = self._select_best_hf()
            sam_model_type, sam_records = self._select_best_sam()
            gemini_model = self._select_best_gemini()

            if not hf_model and not sam_model_type and not gemini_model:
                self.result_ready.emit({
                    "status": "error",
                    "message": tr(
                        "Could not resolve latest model recommendations "
                        "(check network/API keys)."
                    ),
                })
                return

            summary = []
            if hf_model:
                summary.append(f"HF: {hf_model}")
            if sam_model_type:
                summary.append(f"SAM: {sam_model_type}")
            if gemini_model:
                summary.append(f"Gemini: {gemini_model}")

            self.result_ready.emit({
                "status": "ok",
                "hf_model": hf_model,
                "sam_model_type": sam_model_type,
                "gemini_model": gemini_model,
                "message": " | ".join(summary),
                "hf_candidates_found": len(hf_records),
                "sam_candidates_found": len(sam_records),
            })
        except Exception as exc:
            self.result_ready.emit({"status": "error", "message": str(exc)})


class ModelDownloadThread(QThread):
    """Download and verify an ONNX model without blocking the dialog."""
    progress = pyqtSignal(int, int)      # received, total
    result_ready = pyqtSignal(object)    # {"ok": bool, "message": str}

    def __init__(self, spec, base_dir):
        super().__init__()
        self.spec = spec
        self.base_dir = base_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            path = download_model(
                self.spec,
                self.base_dir,
                progress=lambda received, total: self.progress.emit(int(received), int(total)),
                cancel_check=lambda: self._cancel,
            )
            self.result_ready.emit({
                "ok": True,
                "message": tr("Model ready:\n{path}").format(path=path),
            })
        except Exception as e:
            self.result_ready.emit({"ok": False, "message": str(e)})


class GeminiTestThread(QThread):
    """
    Check the Gemini API key without generating anything.

    Listing models proves the key is valid and shows what it can reach; the
    old test ran real image generations against model after model, which cost
    quota every time the user pressed the button.
    """
    result_ready = pyqtSignal(bool, str)  # success, message

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        try:
            from google import genai
        except ImportError:
            self.result_ready.emit(
                False, tr("Package '{package}' not installed").format(
                    package=GEMINI_INSTALL_PACKAGE
                )
            )
            return

        try:
            client = genai.Client(api_key=self.api_key)
            available = []
            for model in client.models.list():
                name = _normalize_gemini_model_name(getattr(model, "name", ""))
                if not name or "gemini" not in name.lower() or _is_excluded_gemini_model(name):
                    continue
                available.append(name)
        except Exception as e:
            self.result_ready.emit(
                False, tr("Connection/Auth Error: {error}").format(error=e)
            )
            return

        if not available:
            self.result_ready.emit(
                False, tr("The key works, but no Gemini models are available for it.")
            )
            return

        available.sort(key=_rank_gemini_model, reverse=True)
        image_models = [name for name in available if _is_image_gemini_model(name)]
        summary = tr("Key valid. {count} models available; best: {model}").format(
            count=len(available), model=available[0]
        )
        if image_models:
            summary += tr("; image model: {model}").format(model=image_models[0])
        self.result_ready.emit(True, summary)


class HfConnectionTestThread(QThread):
    """
    Check the Hugging Face token and model availability cheaply.

    Model metadata says whether the id exists and whether it is gated; the old
    test submitted a real generation job to up to seventeen models in turn.
    """
    result_ready = pyqtSignal(object)  # dict result payload

    def __init__(self, api_key, candidate_models):
        super().__init__()
        self.api_key = str(api_key or "").strip()
        self.candidate_models = list(candidate_models or [])

    def run(self):
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
        except Exception:
            self.result_ready.emit({
                "status": "error",
                "message": tr(
                    "The 'huggingface_hub' package is required. "
                    "Install it with: pip install huggingface_hub"
                ),
            })
            return

        api = HfApi(token=self.api_key or None)
        try:
            who = api.whoami()
        except Exception as exc:
            message = str(exc)
            if "401" in message or "invalid" in message.lower():
                self.result_ready.emit({"status": "invalid_token"})
            else:
                self.result_ready.emit({"status": "error", "message": message})
            return

        saw_gated = False
        saw_missing = False
        last_error = ""
        for candidate in self.candidate_models[:6]:
            if not candidate:
                continue
            try:
                info = api.model_info(candidate)
            except GatedRepoError:
                saw_gated = True
                continue
            except RepositoryNotFoundError:
                saw_missing = True
                continue
            except (HfHubHTTPError, Exception) as exc:
                last_error = str(exc)
                continue

            self.result_ready.emit({
                "status": "connected",
                "model": candidate,
                "user": str(who.get("name", "")) if isinstance(who, dict) else "",
                "pipeline": str(getattr(info, "pipeline_tag", "") or ""),
            })
            return

        if saw_gated:
            self.result_ready.emit({"status": "forbidden"})
            return
        if saw_missing:
            self.result_ready.emit({"status": "not_found"})
            return
        self.result_ready.emit({
            "status": "error",
            "message": last_error or tr("No model could be reached."),
        })
