# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Main Dialog UI
"""

import os
import base64
from qgis.PyQt.QtCore import Qt, QSize, pyqtSignal, QThread, QObject, QByteArray, QPointF, QRectF
from qgis.PyQt.QtGui import QPixmap, QImage, QColor, QDragEnterEvent, QDropEvent, QPainter, QPen, QBrush, QPainterPath, QCursor
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QGroupBox, QRadioButton, QButtonGroup,
    QFileDialog, QColorDialog, QProgressBar, QMessageBox,
    QFrame, QSizePolicy, QWidget, QScrollArea, QCheckBox,
    QLineEdit
)
from qgis.core import QgsProject, QgsVectorLayer

from ..generators.style_utils import STYLE_OPTIONS


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
    colorPicked = pyqtSignal(QColor)
    
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
        self.color_picking_mode = False
        
    def set_picking_mode(self, active):
        """Enable or disable color picking mode."""
        self.color_picking_mode = active
        if active:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
    def mousePressEvent(self, event):
        """Handle mouse click to browse or pick color."""
        if event.button() == Qt.LeftButton:
            if self.color_picking_mode and self.image_path:
                # Pick color from the clicked pixel
                pos = event.pos()
                pixmap = self.pixmap()
                if pixmap:
                    # Map widget coord to pixmap coord
                    img = pixmap.toImage()
                    
                    # Calculate scaling offset
                    scaled_w = pixmap.width()
                    scaled_h = pixmap.height()
                    widget_w = self.width()
                    widget_h = self.height()
                    
                    x_offset = (widget_w - scaled_w) / 2
                    y_offset = (widget_h - scaled_h) / 2
                    
                    img_x = int(pos.x() - x_offset)
                    img_y = int(pos.y() - y_offset)
                    
                    if 0 <= img_x < scaled_w and 0 <= img_y < scaled_h:
                        c = QColor(img.pixel(img_x, img_y))
                        self.colorPicked.emit(c)
            else:
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
        self.image_drop.colorPicked.connect(self.set_current_color)
        self.image_drop.setToolTip(
            "Use a representative photo of the artifact or archaeological feature.\n"
            "Clean backgrounds produce better silhouettes and internal detail lines."
        )
        input_layout.addWidget(self.image_drop, alignment=Qt.AlignCenter)
        
        # Photo tip label
        tip_label = QLabel(
            "💡 <i>Tip: Use a clear photo with a clean background for best results.</i>"
        )
        tip_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        tip_label.setWordWrap(True)
        input_layout.addWidget(tip_label)
        
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
        
        # Right panel - Settings (Main container)
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0) # Remove extra margins for the container
        
        # Scroll area for settings
        scan_scroll = QScrollArea()
        scan_scroll.setWidgetResizable(True)
        scan_scroll.setFrameShape(QFrame.NoFrame)
        
        # Widget to hold the scrollable content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        
        # --- Add groups to scroll_layout instead of right_panel ---
        
        # Generation mode
        mode_group = QGroupBox("Generation Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_button_group = QButtonGroup(self)
        self.autotrace_radio = QRadioButton("✂ Auto Trace")
        self.gemini_radio = QRadioButton("AI (Google Gemini)")
        self.hf_radio = QRadioButton("AI (Hugging Face)")
        self.local_radio = QRadioButton("AI (Local Stable Diffusion)")
        self.template_radio = QRadioButton("Use Template")
        
        self.autotrace_radio.setChecked(True)
        self.mode_button_group.addButton(self.autotrace_radio, 0)
        self.mode_button_group.addButton(self.gemini_radio, 1)
        self.mode_button_group.addButton(self.hf_radio, 2)
        self.mode_button_group.addButton(self.local_radio, 3)
        self.mode_button_group.addButton(self.template_radio, 4)
        
        mode_layout.addWidget(self.autotrace_radio)
        mode_layout.addWidget(self.hf_radio)
        mode_layout.addWidget(self.gemini_radio)
        mode_layout.addWidget(self.local_radio)
        mode_layout.addWidget(self.template_radio)
        
        # Mode description label
        self.mode_info_label = QLabel("✂ Extracts contour + internal feature lines from photo (fast, offline)")
        self.mode_info_label.setStyleSheet(
            "color: #555; font-size: 11px; background: #f0f8ff; "
            "padding: 4px; border-radius: 3px;"
        )
        self.mode_info_label.setWordWrap(True)
        mode_layout.addWidget(self.mode_info_label)
        
        # Mode-specific connection
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
        
        scroll_layout.addWidget(mode_group)
        
        # Style selection
        style_group = QGroupBox("Style")
        style_layout = QVBoxLayout(style_group)
        
        self.style_combo = QComboBox()
        self.style_combo.addItems(STYLE_OPTIONS)
        style_layout.addWidget(self.style_combo)
        
        # Symmetry checkbox
        self.symmetry_check = QCheckBox("Mirror symmetry")
        self.symmetry_check.setChecked(False)
        self.symmetry_check.setToolTip(
            "Produces a bilaterally symmetrical symbol by mirroring the contour."
        )
        style_layout.addWidget(self.symmetry_check)
        
        scroll_layout.addWidget(style_group)
        
        # Template selection (initially hidden)
        self.template_group = QGroupBox("Template Type")
        template_layout = QVBoxLayout(self.template_group)
        
        self.template_combo = QComboBox()
        try:
            from ..generators.template_generator import TemplateGenerator
            template_generator = TemplateGenerator(self.plugin_dir)
            self.template_combo.addItems(template_generator.get_available_templates())
        except Exception:
            self.template_combo.addItems([
                "Pottery",
                "Stone Tool",
                "Bronze Artifact",
                "Iron Artifact",
                "Weapon",
                "Excavation Area",
                "Survey Point",
                "Find Spot",
            ])
        template_layout.addWidget(self.template_combo)
        self.template_group.setVisible(False)
        scroll_layout.addWidget(self.template_group)
        
        # Color settings
        color_group = QGroupBox("Color")
        color_layout = QVBoxLayout(color_group) # Changed to QVBoxLayout for better density
        
        # Row 1: Checkbox
        self.override_color_check = QCheckBox("Override Color")
        self.override_color_check.setChecked(False) # Default: Use extracted/natural color
        self.override_color_check.setToolTip("If unchecked, the symbol will use the artifact's natural colors.")
        color_layout.addWidget(self.override_color_check)
        
        # Row 2: Picker controls
        picker_layout = QHBoxLayout()
        
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.update_color_preview()
        picker_layout.addWidget(self.color_preview)
        
        self.color_btn = QPushButton("Pick Color")
        self.color_btn.clicked.connect(self.pick_color)
        picker_layout.addWidget(self.color_btn)
        
        self.eyedrop_btn = QPushButton("🎨 Pick from Image")
        self.eyedrop_btn.setCheckable(True)
        self.eyedrop_btn.toggled.connect(self.toggle_picking_mode)
        picker_layout.addWidget(self.eyedrop_btn)
        
        picker_layout.addStretch()
        color_layout.addLayout(picker_layout)
        
        # Logic to enable/disable picker based on checkbox
        self.override_color_check.toggled.connect(lambda checked: self.color_preview.setEnabled(checked))
        self.override_color_check.toggled.connect(lambda checked: self.color_btn.setEnabled(checked))
        self.override_color_check.toggled.connect(lambda checked: self.eyedrop_btn.setEnabled(checked))
        
        # Initialize state
        self.color_preview.setEnabled(False)
        self.color_btn.setEnabled(False)
        self.eyedrop_btn.setEnabled(False)
        
        scroll_layout.addWidget(color_group)
        
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
        scroll_layout.addWidget(size_group)

        # Prompt input (for AI modes)
        self.prompt_group = QGroupBox("Text Prompt")
        prompt_layout = QVBoxLayout(self.prompt_group)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter a description for the icon (e.g., 'ancient pottery shard')")
        prompt_layout.addWidget(self.prompt_input)
        self.prompt_group.setVisible(False) # Hidden by default
        scroll_layout.addWidget(self.prompt_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        scroll_layout.addWidget(self.progress_bar)
        
        scroll_layout.addStretch() # Push everything up inside scroll area
        
        # Finish scroll area setup
        scan_scroll.setWidget(scroll_content)
        right_panel.addWidget(scan_scroll)
        
        # Action buttons (Fixed at bottom, outside scroll)
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
        
        # Update mode description label
        descriptions = {
            self.autotrace_radio: "✂ Extracts contour + internal feature lines from photo (fast, offline)",
            self.gemini_radio: "🤖 Google Gemini AI generates reference-constrained symbols (factual mode)",
            self.hf_radio: "🤗 Hugging Face AI refines symbols from the reference image (token required, HF Prompt Adaptive v3)",
            self.local_radio: "💻 Local Stable Diffusion generates symbols (GPU required)",
            self.template_radio: "📋 Uses built-in SVG templates by category",
        }
        self.mode_info_label.setText(descriptions.get(button, ""))

        # Show prompt input for HF mode (and maybe others in future)
        self.prompt_group.setVisible(
            button == self.hf_radio or 
            button == self.gemini_radio or 
            button == self.local_radio
        )
        
        # Update placeholder based on mode
        if button == self.hf_radio:
             self.prompt_input.setPlaceholderText("Optional: artifact/material note (e.g. 'green celadon vase')")
        elif button == self.gemini_radio:
             self.prompt_input.setPlaceholderText("Optional: factual notes (e.g. 'preserve chips and wear')")
        elif button == self.local_radio:
             self.prompt_input.setPlaceholderText("Enter generation prompt")

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
    
    def toggle_picking_mode(self, checked):
        """Toggle the cursor and mode for color picking."""
        self.image_drop.set_picking_mode(checked)
        if checked:
            self.eyedrop_btn.setText("Click Image to Pick")
        else:
            self.eyedrop_btn.setText("🎨 Pick from Image")

    def set_current_color(self, color):
        """Set color from picker."""
        if color.isValid():
            self.current_color = color
            self.update_color_preview()
            self.eyedrop_btn.setChecked(False)  # Turn off picking mode
            
    def generate_symbol(self):
        """Generate symbol based on current settings."""


        # Validate inputs
        if self.hf_radio.isChecked():
             # HF mode needs prompt (checked later), but doesn't strictly need an image
             pass
        elif not self.template_radio.isChecked() and not self.image_drop.image_path:
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
            selected_color = self.current_color.name() if self.override_color_check.isChecked() else None
            
            if self.autotrace_radio.isChecked():
                from ..generators.contour_generator import ContourGenerator
                self._current_generator = ContourGenerator()
                target_func = self._current_generator.generate
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'style': self.style_combo.currentText(),
                    'color': selected_color,
                    'symmetry': self.symmetry_check.isChecked()
                }
            elif self.gemini_radio.isChecked():
                from ..generators.gemini_generator import GeminiGenerator
                self._current_generator = GeminiGenerator()
                target_func = self._current_generator.generate
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'style': self.style_combo.currentText(),
                    'color': selected_color,
                    'symmetry': self.symmetry_check.isChecked()
                }
                
            elif self.hf_radio.isChecked():
                from ..generators.huggingface_generator import HuggingFaceGenerator
                self._current_generator = HuggingFaceGenerator()
                target_func = self._current_generator.generate
                
                # Use prompt input
                prompt = self.prompt_input.text().strip()

                if prompt:
                    self.mode_info_label.setText(
                        "HF Prompt Adaptive v3 active: custom text prompt will influence stylization."
                    )
                else:
                    self.mode_info_label.setText(
                        "HF Prompt Adaptive v3 active: no custom prompt detected; factual/default guidance is used."
                    )
                    prompt = "archaeological artifact from reference photo"

                kwargs = {
                    'prompt': prompt,
                    'style': self.style_combo.currentText(),
                    'color': selected_color,
                    'image_path': self.image_drop.image_path,
                    'symmetry': self.symmetry_check.isChecked()
                }

            elif self.local_radio.isChecked():
                from ..generators.local_generator import LocalGenerator
                self._current_generator = LocalGenerator()
                target_func = self._current_generator.generate
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'style': self.style_combo.currentText(),
                    'color': selected_color
                }
            else:
                from ..generators.template_generator import TemplateGenerator
                self._current_generator = TemplateGenerator(self.plugin_dir)
                target_func = self._current_generator.generate
                kwargs = {
                    'template_type': self.template_combo.currentText(),
                    'color': selected_color
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
        self._current_generator = None  # Release reference
        
        if error_message:
            QMessageBox.critical(self, "Error", f"Generation failed: {error_message}")
            return
            
        if result:
            # If result is a string (SVG code from Gemini), render to QPixmap on main thread
            if isinstance(result, str):
                pixmap = self._svg_to_pixmap(result)
                if pixmap:
                    self.preview_label.set_preview(pixmap)
                    self.save_btn.setEnabled(True)
                    self.apply_btn.setEnabled(True)
                else:
                    QMessageBox.warning(self, "Failed", "Generated SVG code was invalid.")
            else:
                # Result is already a QPixmap (from template or local SD)
                # OR it is a QImage (after thread safety fix)
                if isinstance(result, QImage):
                    pixmap = QPixmap.fromImage(result)
                    self.preview_label.set_preview(pixmap)
                else:
                    self.preview_label.set_preview(result)

                self.save_btn.setEnabled(True)
                self.apply_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Failed", "Generation returned no result.")

    def _svg_to_pixmap(self, svg_code):
        """Render SVG code to QPixmap (must be called on the main/GUI thread)."""
        from qgis.PyQt.QtCore import QByteArray
        from qgis.PyQt.QtSvg import QSvgRenderer
        from qgis.PyQt.QtGui import QPainter
        
        renderer = QSvgRenderer(QByteArray(svg_code.encode('utf-8')))
        
        if not renderer.isValid():
            return None

        renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        pixmap = QPixmap(256, 256)
        pixmap.fill(Qt.transparent)

        view_box = renderer.viewBoxF()
        if not view_box.isValid() or view_box.width() <= 0 or view_box.height() <= 0:
            default_size = renderer.defaultSize()
            if default_size.isValid() and default_size.width() > 0 and default_size.height() > 0:
                view_box = QRectF(0, 0, float(default_size.width()), float(default_size.height()))
            else:
                view_box = QRectF(0, 0, 256.0, 256.0)

        scale = min(256.0 / view_box.width(), 256.0 / view_box.height())
        target_w = view_box.width() * scale
        target_h = view_box.height() * scale
        target_x = (256.0 - target_w) * 0.5
        target_y = (256.0 - target_h) * 0.5
        target_rect = QRectF(target_x, target_y, target_w, target_h)

        painter = QPainter(pixmap)
        renderer.render(painter, target_rect)
        painter.end()

        return pixmap
            
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
