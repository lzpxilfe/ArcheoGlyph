# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Main Dialog UI
"""

import os
import base64
from qgis.PyQt.QtCore import Qt, QSize, pyqtSignal, QThread
from qgis.PyQt.QtGui import QPixmap, QImage, QColor, QDragEnterEvent, QDropEvent
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QGroupBox, QRadioButton, QButtonGroup,
    QFileDialog, QColorDialog, QProgressBar, QMessageBox,
    QFrame, QSizePolicy, QWidget, QScrollArea
)
from qgis.core import QgsProject, QgsVectorLayer


class GenerationThread(QThread):
    """Thread for running generation tasks."""
    finished = pyqtSignal(object, str) # result (QPixmap), error_message

    def __init__(self, generator_func, **kwargs):
        super().__init__()
        self.generator_func = generator_func
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.generator_func(**self.kwargs)
            self.finished.emit(result, "")
        except Exception as e:
            self.finished.emit(None, str(e))


class ImageDropArea(QLabel):
    """A label that accepts image drops."""
    
    imageDropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setMaximumSize(300, 300)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #888;
                border-radius: 10px;
                background-color: #f5f5f5;
                color: #666;
                font-size: 14px;
            }
            QLabel:hover {
                border-color: #4a90d9;
                background-color: #e8f0fe;
            }
        """)
        self.setText("Drop Image Here\nor Click to Browse")
        self.image_path = None
        
    def mousePressEvent(self, event):
        """Handle mouse click to browse for image."""
        if event.button() == Qt.LeftButton:
            self.browse_image()
    
    def browse_image(self):
        """Open file dialog to select image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Artifact Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)"
        )
        if file_path:
            self.load_image(file_path)
            
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag if it contains image files."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    event.acceptProposedAction()
                    return
        event.ignore()
        
    def dropEvent(self, event: QDropEvent):
        """Handle dropped image files."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                self.load_image(file_path)
                break
                
    def load_image(self, file_path):
        """Load and display the image."""
        self.image_path = file_path
        pixmap = QPixmap(file_path)
        scaled = pixmap.scaled(
            self.size() - QSize(20, 20),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.imageDropped.emit(file_path)
        
    def clear_image(self):
        """Clear the loaded image."""
        self.image_path = None
        self.clear()
        self.setText("Drop Image Here\nor Click to Browse")


class PreviewLabel(QLabel):
    """Label for displaying generated symbol preview."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(150, 150)
        self.setMaximumSize(200, 200)
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 8px;
                background-color: white;
            }
        """)
        self.setText("Preview")
        self.generated_image = None
        
    def set_preview(self, pixmap_or_path):
        """Set the preview image."""
        if isinstance(pixmap_or_path, str):
            pixmap = QPixmap(pixmap_or_path)
        else:
            pixmap = pixmap_or_path
            
        self.generated_image = pixmap
        scaled = pixmap.scaled(
            self.size() - QSize(10, 10),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)


class ArcheoGlyphDialog(QDialog):
    """Main dialog for ArcheoGlyph plugin."""
    
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.plugin_dir = os.path.dirname(os.path.dirname(__file__))
        self.current_color = QColor("#8B4513")  # Default brown for artifacts
        self.generation_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("ArcheoGlyph - Symbol Generator")
        self.setMinimumSize(600, 500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel - Image input and preview
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        # Image drop area
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout(input_group)
        self.image_drop = ImageDropArea()
        self.image_drop.imageDropped.connect(self.on_image_loaded)
        input_layout.addWidget(self.image_drop, alignment=Qt.AlignCenter)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_input)
        input_layout.addWidget(clear_btn)
        left_panel.addWidget(input_group)
        
        # Preview area
        preview_group = QGroupBox("Generated Symbol")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = PreviewLabel()
        preview_layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        left_panel.addWidget(preview_group)
        
        left_panel.addStretch()
        main_layout.addLayout(left_panel)
        
        # Right panel - Settings
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        # Generation mode
        mode_group = QGroupBox("Generation Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_button_group = QButtonGroup(self)
        self.gemini_radio = QRadioButton("AI (Google Gemini)")
        self.local_radio = QRadioButton("AI (Local Stable Diffusion)")
        self.template_radio = QRadioButton("Use Template")
        
        self.gemini_radio.setChecked(True)
        self.mode_button_group.addButton(self.gemini_radio, 0)
        self.mode_button_group.addButton(self.local_radio, 1)
        self.mode_button_group.addButton(self.template_radio, 2)
        
        mode_layout.addWidget(self.gemini_radio)
        mode_layout.addWidget(self.local_radio)
        mode_layout.addWidget(self.template_radio)
        
        # Mode-specific connection
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
        
        right_panel.addWidget(mode_group)
        
        # Style selection
        style_group = QGroupBox("Style")
        style_layout = QVBoxLayout(style_group)
        
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "🎨 Cute / Kawaii",
            "📐 Minimal",
            "🏛️ Classic Archaeological"
        ])
        style_layout.addWidget(self.style_combo)
        right_panel.addWidget(style_group)
        
        # Template selection (initially hidden)
        self.template_group = QGroupBox("Template Type")
        template_layout = QVBoxLayout(self.template_group)
        
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Pottery (토기류)",
            "Stone Tools (석기류)",
            "Bronze Artifacts (청동기류)",
            "Iron Artifacts (철기류)",
            "Ornaments (장신구류)"
        ])
        template_layout.addWidget(self.template_combo)
        self.template_group.setVisible(False)
        right_panel.addWidget(self.template_group)
        
        # Color settings
        color_group = QGroupBox("Color")
        color_layout = QHBoxLayout(color_group)
        
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.update_color_preview()
        color_layout.addWidget(self.color_preview)
        
        color_btn = QPushButton("Pick Color")
        color_btn.clicked.connect(self.pick_color)
        color_layout.addWidget(color_btn)
        color_layout.addStretch()
        right_panel.addWidget(color_group)
        
        # Size settings
        size_group = QGroupBox("Size Scaling")
        size_layout = QVBoxLayout(size_group)
        
        size_mode_layout = QHBoxLayout()
        size_mode_layout.addWidget(QLabel("Mode:"))
        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItems([
            "Fixed Size",
            "By Data Count (Natural Breaks)",
            "By Data Count (Equal Interval)",
            "By Data Count (Quantile)"
        ])
        size_mode_layout.addWidget(self.size_mode_combo)
        size_layout.addLayout(size_mode_layout)
        
        minmax_layout = QHBoxLayout()
        minmax_layout.addWidget(QLabel("Min:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(8, 128)
        self.min_size_spin.setValue(16)
        minmax_layout.addWidget(self.min_size_spin)
        
        minmax_layout.addWidget(QLabel("Max:"))
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(8, 256)
        self.max_size_spin.setValue(64)
        minmax_layout.addWidget(self.max_size_spin)
        size_layout.addLayout(minmax_layout)
        right_panel.addWidget(size_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)
        
        right_panel.addStretch()
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("🎨 Generate")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_symbol)
        button_layout.addWidget(self.generate_btn)
        
        self.save_btn = QPushButton("💾 Save to Library")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_to_library)
        button_layout.addWidget(self.save_btn)
        
        self.apply_btn = QPushButton("📍 Apply to Layer")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_to_layer)
        button_layout.addWidget(self.apply_btn)
        
        # Settings button
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self.open_settings)
        button_layout.addWidget(settings_btn)
        
        right_panel.addLayout(button_layout)
        main_layout.addLayout(right_panel)
        
    def on_image_loaded(self, file_path):
        """Handle when an image is loaded."""
        self.generate_btn.setEnabled(True)
        
    def clear_input(self):
        """Clear the input image."""
        self.image_drop.clear_image()
        
    def on_mode_changed(self, button):
        """Handle generation mode change."""
        is_template = button == self.template_radio
        self.template_group.setVisible(is_template)
        self.style_combo.parentWidget().setVisible(not is_template)
        
    def update_color_preview(self):
        """Update the color preview label."""
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {self.current_color.name()};
                border: 1px solid #333;
                border-radius: 4px;
            }}
        """)
        
    def pick_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(self.current_color, self, "Select Symbol Color")
        if color.isValid():
            self.current_color = color
            self.update_color_preview()
            
    def generate_symbol(self):
        """Generate symbol based on current settings."""
        if not self.image_drop.image_path and not self.template_radio.isChecked():
            QMessageBox.warning(self, "No Image", "Please select an input image first.")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate mode since we can't track exact progress in thread
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        
        try:
            target_func = None
            kwargs = {}
            
            if self.gemini_radio.isChecked():
                from ..generators.gemini_generator import GeminiGenerator
                generator = GeminiGenerator()
                target_func = generator.generate
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'style': self.style_combo.currentText(),
                    'color': self.current_color.name()
                }
            elif self.local_radio.isChecked():
                from ..generators.local_generator import LocalGenerator
                generator = LocalGenerator()
                target_func = generator.generate
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'style': self.style_combo.currentText(),
                    'color': self.current_color.name()
                }
            else:
                from ..generators.template_generator import TemplateGenerator
                generator = TemplateGenerator(self.plugin_dir)
                target_func = generator.generate
                kwargs = {
                    'template_type': self.template_combo.currentText(),
                    'color': self.current_color.name()
                }
            
            if target_func:
                self.generation_thread = GenerationThread(target_func, **kwargs)
                self.generation_thread.finished.connect(self.on_generation_finished)
                self.generation_thread.start()
            
        except Exception as e:
            self.on_generation_finished(None, str(e))

    def on_generation_finished(self, result, error_message):
        """Handle generation results."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100) # Reset to normal
        self.generate_btn.setEnabled(True)
        
        if error_message:
            QMessageBox.critical(self, "Error", f"Generation failed: {error_message}")
            return
            
        if result:
            self.preview_label.set_preview(result)
            self.save_btn.setEnabled(True)
            self.apply_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Failed", "Generation returned no result.")
            
    def save_to_library(self):
        """Save generated symbol to QGIS symbol library."""
        if not self.preview_label.generated_image:
            QMessageBox.warning(self, "No Symbol", "Please generate a symbol first.")
            return
            
        from ..symbol_manager import SymbolManager
        
        manager = SymbolManager()
        success = manager.save_to_library(
            self.preview_label.generated_image,
            name="ArcheoGlyph Symbol"
        )
        
        if success:
            QMessageBox.information(self, "Saved", "Symbol saved to QGIS library!")
        else:
            QMessageBox.warning(self, "Error", "Failed to save symbol.")
            
    def apply_to_layer(self):
        """Apply generated symbol to current layer."""
        layer = self.iface.activeLayer()
        
        if not layer or not isinstance(layer, QgsVectorLayer):
            QMessageBox.warning(self, "No Layer", "Please select a vector layer first.")
            return
            
        if not self.preview_label.generated_image:
            QMessageBox.warning(self, "No Symbol", "Please generate a symbol first.")
            return
            
        from ..symbol_manager import SymbolManager
        
        manager = SymbolManager()
        
        # Get size scaling settings
        size_mode = self.size_mode_combo.currentIndex()
        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value()
        
        success = manager.apply_to_layer(
            layer=layer,
            symbol_image=self.preview_label.generated_image,
            size_mode=size_mode,
            min_size=min_size,
            max_size=max_size
        )
        
        if success:
            QMessageBox.information(self, "Applied", f"Symbol applied to layer: {layer.name()}")
            layer.triggerRepaint()
        else:
            QMessageBox.warning(self, "Error", "Failed to apply symbol to layer.")
            
    def open_settings(self):
        """Open the settings dialog."""
        from .settings_dialog import SettingsDialog
        
        dialog = SettingsDialog(self)
        dialog.exec_()
