# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Settings Dialog
Configure AI API keys and view setup instructions.
"""

import os
import subprocess
import sys
from qgis.PyQt.QtCore import Qt, QSettings, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QFont
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QTabWidget, QWidget, QTextBrowser,
    QMessageBox, QProgressDialog
)


class SettingsDialog(QDialog):
    """Settings dialog for API configuration and help."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Initialize the settings UI."""
        self.setWindowTitle("ArcheoGlyph Settings")
        self.setMinimumSize(550, 500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout(self)
        
        # Tab widget
        tabs = QTabWidget()
        
        # Tab 1: Google Gemini
        gemini_tab = self._create_gemini_tab()
        tabs.addTab(gemini_tab, "🌐 Google Gemini")
        
        # Tab 2: Local Stable Diffusion
        local_tab = self._create_local_sd_tab()
        tabs.addTab(local_tab, "💻 Local SD")
        
        # Tab 3: Help
        help_tab = self._create_help_tab()
        tabs.addTab(help_tab, "❓ Help")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(save_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
    def _create_gemini_tab(self):
        """Create Google Gemini settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Step 1: Install package
        install_group = QGroupBox("Step 1: Install Package")
        install_layout = QVBoxLayout(install_group)
        
        install_info = QLabel(
            "Install the Google Generative AI package to use Gemini.\n"
            "Click the button below to install automatically."
        )
        install_info.setWordWrap(True)
        install_layout.addWidget(install_info)
        
        btn_layout = QHBoxLayout()
        self.install_btn = QPushButton("📦 Install google-generativeai")
        self.install_btn.clicked.connect(self.install_gemini_package)
        btn_layout.addWidget(self.install_btn)
        
        self.install_status = QLabel("")
        btn_layout.addWidget(self.install_status)
        btn_layout.addStretch()
        install_layout.addLayout(btn_layout)
        
        layout.addWidget(install_group)
        
        # Step 2: Get API Key
        apikey_group = QGroupBox("Step 2: Get API Key")
        apikey_layout = QVBoxLayout(apikey_group)
        
        apikey_info = QLabel(
            "Get your free API key from Google AI Studio.\n"
            "Click the link below, sign in, and create a new key."
        )
        apikey_info.setWordWrap(True)
        apikey_layout.addWidget(apikey_info)
        
        link_btn = QPushButton("🔑 Open Google AI Studio →")
        link_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://makersuite.google.com/app/apikey")
            )
        )
        apikey_layout.addWidget(link_btn)
        
        layout.addWidget(apikey_group)
        
        # Step 3: Enter API Key
        key_group = QGroupBox("Step 3: Enter API Key")
        key_layout = QVBoxLayout(key_group)
        
        key_info = QLabel("Paste your API key below:")
        key_layout.addWidget(key_info)
        
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.Password)
        self.gemini_key_input.setPlaceholderText("AIza...")
        key_layout.addWidget(self.gemini_key_input)
        
        show_key_btn = QPushButton("👁 Show/Hide Key")
        show_key_btn.clicked.connect(self._toggle_key_visibility)
        key_layout.addWidget(show_key_btn)
        
        layout.addWidget(key_group)
        
        # Test connection
        test_group = QGroupBox("Step 4: Test Connection")
        test_layout = QVBoxLayout(test_group)
        
        test_btn = QPushButton("✅ Test Gemini Connection")
        test_btn.clicked.connect(self.test_gemini_connection)
        test_layout.addWidget(test_btn)
        
        self.gemini_test_result = QLabel("")
        test_layout.addWidget(self.gemini_test_result)
        
        layout.addWidget(test_group)
        
        layout.addStretch()
        return tab
        
    def _create_local_sd_tab(self):
        """Create Local Stable Diffusion settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Server URL
        server_group = QGroupBox("Server Configuration")
        server_layout = QVBoxLayout(server_group)
        
        server_info = QLabel(
            "Enter the URL of your local Stable Diffusion server.\n"
            "Default: http://127.0.0.1:7860 (Automatic1111 WebUI)"
        )
        server_info.setWordWrap(True)
        server_layout.addWidget(server_info)
        
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("Server URL:"))
        self.sd_url_input = QLineEdit()
        self.sd_url_input.setPlaceholderText("http://127.0.0.1:7860")
        url_layout.addWidget(self.sd_url_input)
        server_layout.addLayout(url_layout)
        
        test_btn = QPushButton("🔌 Test Connection")
        test_btn.clicked.connect(self.test_sd_connection)
        server_layout.addWidget(test_btn)
        
        self.sd_test_result = QLabel("")
        server_layout.addWidget(self.sd_test_result)
        
        layout.addWidget(server_group)
        
        # Setup instructions
        setup_group = QGroupBox("Setup Instructions")
        setup_layout = QVBoxLayout(setup_group)
        
        setup_info = QLabel(
            "To use Local Stable Diffusion:\n\n"
            "1. Install Automatic1111 WebUI\n"
            "2. Download a model (Anything V5 recommended)\n"
            "3. Run with --api flag enabled\n"
            "4. Enter the server URL above"
        )
        setup_info.setWordWrap(True)
        setup_layout.addWidget(setup_info)
        
        guide_btn = QPushButton("📖 Open Full Setup Guide")
        guide_btn.clicked.connect(self._open_sd_guide)
        setup_layout.addWidget(guide_btn)
        
        layout.addWidget(setup_group)
        
        layout.addStretch()
        return tab
        
    def _create_help_tab(self):
        """Create help tab with documentation."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(True)
        help_text.setHtml("""
        <h2>ArcheoGlyph Help</h2>
        
        <h3>🎨 How to Use</h3>
        <ol>
            <li>Drop an artifact image or select a template</li>
            <li>Choose generation mode:
                <ul>
                    <li><b>AI (Gemini)</b> - Uses Google's AI (requires API key)</li>
                    <li><b>AI (Local SD)</b> - Uses local Stable Diffusion (requires setup)</li>
                    <li><b>Template</b> - Uses built-in symbols (no setup needed!)</li>
                </ul>
            </li>
            <li>Select style and color</li>
            <li>Click <b>Generate</b></li>
            <li>Save to library or apply to layer</li>
        </ol>
        
        <h3>🚀 Quick Start (No AI Required)</h3>
        <p>Select <b>"Use Template"</b> mode to instantly generate symbols 
        without any AI configuration!</p>
        
        <h3>🔑 Setting Up Google Gemini</h3>
        <ol>
            <li>Go to the <b>Google Gemini</b> tab</li>
            <li>Click <b>Install Package</b></li>
            <li>Get API key from Google AI Studio</li>
            <li>Paste key and test connection</li>
        </ol>
        
        <h3>💻 Setting Up Local Stable Diffusion</h3>
        <ol>
            <li>Install <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Automatic1111 WebUI</a></li>
            <li>Download a model from <a href="https://civitai.com">Civitai</a></li>
            <li>Run with <code>--api</code> flag</li>
            <li>Enter server URL in <b>Local SD</b> tab</li>
        </ol>
        
        <h3>📚 More Help</h3>
        <ul>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph">GitHub Repository</a></li>
            <li><a href="https://github.com/lzpxilfe/ArcheoGlyph/issues">Report Issues</a></li>
        </ul>
        """)
        layout.addWidget(help_text)
        
        return tab
        
    def _toggle_key_visibility(self):
        """Toggle API key visibility."""
        if self.gemini_key_input.echoMode() == QLineEdit.Password:
            self.gemini_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.gemini_key_input.setEchoMode(QLineEdit.Password)
            
    def _open_sd_guide(self):
        """Open local SD setup guide."""
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        guide_path = os.path.join(plugin_dir, 'docs', 'local_model_setup.md')
        
        if os.path.exists(guide_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(guide_path))
        else:
            QDesktopServices.openUrl(
                QUrl("https://github.com/lzpxilfe/ArcheoGlyph/blob/main/docs/local_model_setup.md")
            )
            
    def load_settings(self):
        """Load saved settings."""
        gemini_key = self.settings.value('ArcheoGlyph/gemini_api_key', '')
        sd_url = self.settings.value('ArcheoGlyph/sd_server', 'http://127.0.0.1:7860')
        
        self.gemini_key_input.setText(gemini_key)
        self.sd_url_input.setText(sd_url)
        
        # Check if package is installed
        try:
            import google.generativeai
            self.install_status.setText("✅ Installed")
            self.install_status.setStyleSheet("color: green;")
        except ImportError:
            self.install_status.setText("❌ Not installed")
            self.install_status.setStyleSheet("color: red;")
            
    def save_settings(self):
        """Save settings."""
        self.settings.setValue('ArcheoGlyph/gemini_api_key', self.gemini_key_input.text())
        self.settings.setValue('ArcheoGlyph/sd_server', self.sd_url_input.text())
        
        QMessageBox.information(self, "Saved", "Settings saved successfully!")
        
    def install_gemini_package(self):
        """Install google-generativeai package."""
        reply = QMessageBox.question(
            self,
            "Install Package",
            "This will install 'google-generativeai' package using pip.\n\n"
            "This may take 1-2 minutes. QGIS may appear unresponsive.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
            
        self.install_btn.setEnabled(False)
        self.install_status.setText("⏳ Installing... (please wait)")
        self.install_status.setStyleSheet("color: orange;")
        
        # Force UI update
        from qgis.PyQt.QtWidgets import QApplication
        QApplication.processEvents()
        
        try:
            # Try multiple pip command approaches
            python_path = sys.executable
            
            # Approach 1: Use python -m pip
            result = subprocess.run(
                [python_path, '-m', 'pip', 'install', '--user', 'google-generativeai'],
                capture_output=True,
                text=True,
                timeout=180,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # Update UI
            QApplication.processEvents()
            
            if result.returncode == 0:
                self.install_status.setText("✅ Installed!")
                self.install_status.setStyleSheet("color: green;")
                QMessageBox.information(
                    self, 
                    "Success", 
                    "Package installed successfully!\n\n"
                    "Please restart QGIS to use Google Gemini."
                )
            else:
                # Try without --user flag
                result2 = subprocess.run(
                    [python_path, '-m', 'pip', 'install', 'google-generativeai'],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                
                if result2.returncode == 0:
                    self.install_status.setText("✅ Installed!")
                    self.install_status.setStyleSheet("color: green;")
                    QMessageBox.information(
                        self, 
                        "Success", 
                        "Package installed successfully!\n\n"
                        "Please restart QGIS to use Google Gemini."
                    )
                else:
                    self.install_status.setText("❌ Failed")
                    self.install_status.setStyleSheet("color: red;")
                    error_msg = result.stderr or result2.stderr or "Unknown error"
                    QMessageBox.warning(
                        self,
                        "Installation Failed",
                        f"Error:\n{error_msg[:500]}\n\n"
                        "Try running manually in Command Prompt:\n"
                        "pip install google-generativeai"
                    )
                
        except subprocess.TimeoutExpired:
            self.install_status.setText("❌ Timeout")
            self.install_status.setStyleSheet("color: red;")
            QMessageBox.warning(
                self, 
                "Timeout", 
                "Installation timed out after 3 minutes.\n\n"
                "Try running manually in Command Prompt:\n"
                "pip install google-generativeai"
            )
            
        except Exception as e:
            self.install_status.setText("❌ Error")
            self.install_status.setStyleSheet("color: red;")
            QMessageBox.warning(
                self, 
                "Error", 
                f"{str(e)}\n\n"
                "Try running manually in Command Prompt:\n"
                "pip install google-generativeai"
            )
            
        finally:
            self.install_btn.setEnabled(True)
            
    def test_gemini_connection(self):
        """Test Gemini API connection."""
        api_key = self.gemini_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter an API key first.")
            return
            
        self.gemini_test_result.setText("⏳ Testing...")
        self.gemini_test_result.setStyleSheet("color: orange;")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content("Say 'Hello ArcheoGlyph!' in one line.")
            
            if response and response.text:
                self.gemini_test_result.setText("✅ Connected!")
                self.gemini_test_result.setStyleSheet("color: green;")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Connection successful!\n\nResponse: {response.text[:100]}"
                )
            else:
                self.gemini_test_result.setText("⚠️ No response")
                self.gemini_test_result.setStyleSheet("color: orange;")
                
        except ImportError:
            self.gemini_test_result.setText("❌ Package not installed")
            self.gemini_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Package Not Found",
                "Please install google-generativeai first."
            )
            
        except Exception as e:
            self.gemini_test_result.setText("❌ Failed")
            self.gemini_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Connection Failed", str(e))
            
    def test_sd_connection(self):
        """Test Stable Diffusion server connection."""
        url = self.sd_url_input.text().strip()
        
        if not url:
            url = "http://127.0.0.1:7860"
            self.sd_url_input.setText(url)
            
        self.sd_test_result.setText("⏳ Testing...")
        self.sd_test_result.setStyleSheet("color: orange;")
        
        try:
            import requests
            response = requests.get(f"{url}/sdapi/v1/sd-models", timeout=5)
            
            if response.status_code == 200:
                models = response.json()
                self.sd_test_result.setText(f"✅ Connected! ({len(models)} models)")
                self.sd_test_result.setStyleSheet("color: green;")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Connected to Stable Diffusion server!\n\n"
                    f"Found {len(models)} model(s)."
                )
            else:
                self.sd_test_result.setText(f"⚠️ Status: {response.status_code}")
                self.sd_test_result.setStyleSheet("color: orange;")
                
        except ImportError:
            self.sd_test_result.setText("❌ requests package missing")
            self.sd_test_result.setStyleSheet("color: red;")
            
        except Exception as e:
            self.sd_test_result.setText("❌ Not connected")
            self.sd_test_result.setStyleSheet("color: red;")
            QMessageBox.warning(
                self,
                "Connection Failed",
                f"Cannot connect to {url}\n\n"
                f"Error: {str(e)}\n\n"
                "Make sure the server is running with --api flag."
            )
