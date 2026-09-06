# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Main Dialog UI
"""

import inspect
import os
import threading
from qgis.PyQt.QtCore import Qt, QSize, pyqtSignal, QThread, QRectF, QSettings
from qgis.PyQt.QtGui import QPixmap, QColor, QDragEnterEvent, QDropEvent
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QGroupBox, QRadioButton, QButtonGroup,
    QFileDialog, QColorDialog, QProgressBar, QMessageBox,
    QFrame, QWidget, QScrollArea, QCheckBox, QSizePolicy,
    QLineEdit, QTabWidget, QSlider, QInputDialog
)
from qgis.core import QgsProject, QgsVectorLayer, QgsWkbTypes

from ..defaults import (
    DEFAULT_GRADUATED_CLASSES,
    DEFAULT_MAX_SYMBOL_SIZE_MM,
    DEFAULT_MIN_SYMBOL_SIZE_MM,
    PLUGIN_VERSION,
)
from ..i18n import apply_settings_language, tr
from ..generators.style_utils import STYLE_LEGEND, STYLE_OPTIONS
from ..generators.template_generator import template_display_name
from ..generators.symbol_result import SymbolResult
from ..generators.style_control_utils import (
    STYLE_CONTROL_DEFAULTS,
    STYLE_CONTROL_MAX,
    STYLE_CONTROL_MIN,
    STYLE_CONTROL_EXAGGERATION,
    STYLE_CONTROL_FACTUALITY,
    STYLE_CONTROL_SYMBOLIC_LOOSENESS,
    resolve_style_controls,
    save_style_controls,
    style_controls_short_text,
)


class GenerationThread(QThread):
    """Runs a generator off the GUI thread and emits a SymbolResult."""
    result_ready = pyqtSignal(object, str)  # SymbolResult or None, error_message

    def __init__(self, generator_func, source_label="", style_label="", **kwargs):
        super().__init__()
        self.generator_func = generator_func
        self.source_label = source_label
        self.style_label = style_label
        self.kwargs = kwargs
        self._cancel = threading.Event()

    def cancel(self):
        """Ask the generator to stop at its next checkpoint."""
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def run(self):
        try:
            kwargs = dict(self.kwargs)
            if "cancel_check" in inspect.signature(self.generator_func).parameters:
                kwargs["cancel_check"] = self._cancel.is_set
            raw = self.generator_func(**kwargs)
            if self._cancel.is_set():
                self.result_ready.emit(None, "")
                return
            result = SymbolResult.coerce(raw, source=self.source_label, style=self.style_label)
            self.result_ready.emit(result, "")
        except Exception as e:
            self.result_ready.emit(None, "" if self._cancel.is_set() else str(e))


class ImageDropArea(QLabel):
    """A label that accepts image drops."""
    
    imageDropped = pyqtSignal(str)
    colorPicked = pyqtSignal(QColor)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(200, 200)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
        self.setText(tr("Drop Image Here\nor Click to Browse"))
        self.image_path = None
        self._source_pixmap = None
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
            tr("Select Artifact Image"),
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
        if pixmap.isNull():
            return
        self._source_pixmap = pixmap
        self._render_image()
        self.imageDropped.emit(file_path)

    def _render_image(self):
        """Render source image to current label size safely."""
        if self._source_pixmap is None:
            return
        target = self.contentsRect().size() - QSize(14, 14)
        target_w = max(24, target.width())
        target_h = max(24, target.height())
        scaled = self._source_pixmap.scaled(
            QSize(target_w, target_h),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event):
        """Keep pixmap fitted when geometry changes."""
        super().resizeEvent(event)
        if self._source_pixmap is not None:
            self._render_image()
                
    def clear_image(self):
        """Clear the loaded image."""
        self.image_path = None
        self._source_pixmap = None
        self.clear()
        self.setText(tr("Drop Image Here\nor Click to Browse"))


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
        self.setText(tr("Preview"))
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

    MODE_DESCRIPTION = {
        "autotrace": "Extracts contour + internal feature lines from photo (fast, offline)",
        "gemini": "Google Gemini generates reference-constrained symbols (factual mode)",
        "hf": "Hugging Face generates factual symbols from the reference image (token required)",
        "local": "Local Stable Diffusion generates symbols (GPU required)",
        "template": "Uses built-in SVG templates by category",
    }

    TEMPLATE_CATEGORY_LABELS = (
        ("all", "All Categories"),
        ("artifacts", "Artifacts"),
        ("structures", "Structures"),
        ("remains", "Remains"),
        ("features", "Features"),
        ("survey", "Survey"),
    )
    
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        # Pick up the language before any widget text is built. Changing it
        # takes effect the next time the dialog is opened.
        apply_settings_language()
        self.iface = iface
        self.settings = QSettings()
        self.plugin_dir = os.path.dirname(os.path.dirname(__file__))
        self.plugin_version = PLUGIN_VERSION
        self.current_color = QColor("#8B4513")  # Default brown for artifacts
        self.generation_thread = None
        self.current_result = None
        
        self.setup_ui()

        project = QgsProject.instance()
        project.layersAdded.connect(lambda _layers: self.refresh_layer_list())
        project.layersRemoved.connect(lambda _ids: self.refresh_layer_list())
        
    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(
            tr("ArchaeoGlyph v{version} - Symbol Generator").format(
                version=self.plugin_version
            )
        )
        self.setMinimumSize(680, 560)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel - Image input and preview
        left_panel_container = QWidget()
        left_panel = QVBoxLayout(left_panel_container)
        left_panel.setSpacing(10)
        
        # Image drop area
        input_group = QGroupBox(tr("Input Image"))
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(6)
        self.image_drop = ImageDropArea()
        self.image_drop.imageDropped.connect(self.on_image_loaded)
        self.image_drop.colorPicked.connect(self.set_current_color)
        self.image_drop.setToolTip(
            tr("Use a representative photo of the artifact or archaeological feature.\n"
            "Clean backgrounds produce better silhouettes and internal detail lines.")
        )
        input_layout.addWidget(self.image_drop, alignment=Qt.AlignCenter)
        
        # Photo tip label
        tip_label = QLabel(
            tr("<i>Tip: Use a clear photo with a clean background for best results.</i>")
        )
        tip_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        tip_label.setWordWrap(True)
        input_layout.addWidget(tip_label)

        self.image_quality_hint_label = QLabel("")
        self.image_quality_hint_label.setStyleSheet(
            "color: #8a4b00; font-size: 10px; padding: 2px;"
        )
        self.image_quality_hint_label.setWordWrap(True)
        self.image_quality_hint_label.setMaximumHeight(58)
        self.image_quality_hint_label.setVisible(False)
        input_layout.addWidget(self.image_quality_hint_label)
        
        clear_btn = QPushButton(tr("Clear"))
        clear_btn.clicked.connect(self.clear_input)
        input_layout.addWidget(clear_btn)
        left_panel.addWidget(input_group)
        
        # Preview area
        preview_group = QGroupBox(tr("Generated Symbol"))
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = PreviewLabel()
        preview_layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        left_panel.addWidget(preview_group)
        
        left_panel.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel_container)
        left_scroll.setMinimumWidth(235)
        left_scroll.setMaximumWidth(280)
        main_layout.addWidget(left_scroll)
        
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
        mode_group = QGroupBox(tr("Generation Mode"))
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_button_group = QButtonGroup(self)
        self.autotrace_radio = QRadioButton(tr("Auto Trace"))
        self.gemini_radio = QRadioButton(tr("AI (Google Gemini)"))
        self.hf_radio = QRadioButton(tr("AI (Hugging Face)"))
        self.local_radio = QRadioButton(tr("AI (Local Stable Diffusion)"))
        self.template_radio = QRadioButton(tr("Use Template"))
        
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
        self.mode_info_label = QLabel(self.MODE_DESCRIPTION["autotrace"])
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
        self.style_group = QGroupBox(tr("Style"))
        style_layout = QVBoxLayout(self.style_group)
        style_tabs = QTabWidget()

        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)

        self.style_combo = QComboBox()
        for style_value in STYLE_OPTIONS:
            # The label may be translated; the data stays the canonical
            # English value the generators expect.
            self.style_combo.addItem(tr(style_value), style_value)
        self.style_combo.setToolTip(
            tr("Simple Symbol uses a two-tone fill with bold outlines for readable distribution maps.")
        )
        basic_layout.addWidget(self.style_combo)

        self.legend_quick_btn = QPushButton(tr("Simple Symbol Quick Setup"))
        self.legend_quick_btn.setToolTip(
            tr("Applies a stable preset to turn photos into simple map symbols.")
        )
        self.legend_quick_btn.clicked.connect(self._apply_legend_quick_setup)
        basic_layout.addWidget(self.legend_quick_btn)

        self.fast_quick_btn = QPushButton(tr("Fast Convert Setup"))
        self.fast_quick_btn.setToolTip(
            tr("Applies speed-priority settings for quick conversion.")
        )
        self.fast_quick_btn.clicked.connect(self._apply_fast_convert_setup)
        basic_layout.addWidget(self.fast_quick_btn)

        # Symmetry checkbox
        self.symmetry_check = QCheckBox(tr("Mirror symmetry"))
        self.symmetry_check.setChecked(False)
        self.symmetry_check.setToolTip(
            tr("Produces a bilaterally symmetrical symbol by mirroring the contour.")
        )
        basic_layout.addWidget(self.symmetry_check)

        upscale_default = self.settings.value(
            "ArcheoGlyph/autotrace_force_upscale",
            True,
            type=bool,
        )
        input_kind_default = str(
            self.settings.value("ArcheoGlyph/autotrace_input_kind", "auto")
        ).strip().lower()
        if input_kind_default not in ("auto", "photo", "drawing"):
            input_kind_default = "auto"
        input_kind_row = QHBoxLayout()
        input_kind_row.addWidget(QLabel(tr("Input type:")))
        self.input_kind_combo = QComboBox()
        self.input_kind_combo.addItem(tr("Auto detect"), "auto")
        self.input_kind_combo.addItem(tr("Photograph"), "photo")
        self.input_kind_combo.addItem(tr("Drawing / rubbing"), "drawing")
        self.input_kind_combo.setToolTip(
            tr("Drawings and rubbings are traced from their ink strokes; photographs\n"
            "go through background removal first.")
        )
        idx = self.input_kind_combo.findData(input_kind_default)
        self.input_kind_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.input_kind_combo.currentIndexChanged.connect(self._on_input_kind_changed)
        input_kind_row.addWidget(self.input_kind_combo, 1)
        basic_layout.addLayout(input_kind_row)

        self.synthetic_structure_check = QCheckBox(tr("Add schematic structure lines"))
        self.synthetic_structure_check.setChecked(
            self.settings.value("ArcheoGlyph/autotrace_synthetic_structure", False, type=bool)
        )
        self.synthetic_structure_check.setToolTip(
            tr("Off by default: only lines observed in the image are drawn.\n"
            "Enable to add conventional rim/shoulder, centre and terminal lines.")
        )
        self.synthetic_structure_check.toggled.connect(self._on_synthetic_structure_toggled)
        basic_layout.addWidget(self.synthetic_structure_check)

        self.autotrace_upscale_check = QCheckBox(tr("Low-res detail boost (upscale)"))
        self.autotrace_upscale_check.setChecked(bool(upscale_default))
        self.autotrace_upscale_check.setToolTip(
            tr("Auto Trace only. Aggressively upscales low-resolution images before contour analysis.")
        )
        self.autotrace_upscale_check.toggled.connect(self._on_autotrace_upscale_toggled)
        basic_layout.addWidget(self.autotrace_upscale_check)

        detail_mode_default = str(
            self.settings.value("ArcheoGlyph/autotrace_detail_mode", "fast")
        ).strip().lower()
        if detail_mode_default not in ("fast", "precise"):
            detail_mode_default = "fast"

        detail_mode_row = QHBoxLayout()
        detail_mode_row.addWidget(QLabel(tr("Auto Trace quality:")))
        self.autotrace_detail_mode_combo = QComboBox()
        self.autotrace_detail_mode_combo.addItem(tr("Fast (speed priority)"), "fast")
        self.autotrace_detail_mode_combo.addItem(tr("Precise (detail priority)"), "precise")
        detail_idx = self.autotrace_detail_mode_combo.findData(detail_mode_default)
        if detail_idx < 0:
            detail_idx = 1
        self.autotrace_detail_mode_combo.setCurrentIndex(detail_idx)
        self.autotrace_detail_mode_combo.currentIndexChanged.connect(
            self._on_autotrace_detail_mode_changed
        )
        detail_mode_row.addWidget(self.autotrace_detail_mode_combo, 1)
        basic_layout.addLayout(detail_mode_row)

        round_strategy_default = str(
            self.settings.value("ArcheoGlyph/round_strategy", "image_first")
        ).strip().lower()
        if round_strategy_default not in ("image_first", "hybrid", "structure_first"):
            round_strategy_default = "image_first"
        round_strategy_row = QHBoxLayout()
        round_strategy_row.addWidget(QLabel(tr("Round artifact mode:")))
        self.round_strategy_combo = QComboBox()
        self.round_strategy_combo.addItem(tr("Image-first (recommended)"), "image_first")
        self.round_strategy_combo.addItem(tr("Hybrid (rescue on failure)"), "hybrid")
        self.round_strategy_combo.addItem(tr("Structure-first (stable)"), "structure_first")
        round_strategy_idx = self.round_strategy_combo.findData(round_strategy_default)
        if round_strategy_idx < 0:
            round_strategy_idx = 0
        self.round_strategy_combo.setCurrentIndex(round_strategy_idx)
        self.round_strategy_combo.currentIndexChanged.connect(
            self._on_round_strategy_changed
        )
        round_strategy_row.addWidget(self.round_strategy_combo, 1)
        basic_layout.addLayout(round_strategy_row)
        basic_layout.addStretch()
        style_tabs.addTab(basic_tab, tr("Basic"))

        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)

        params_hint = QLabel(
            tr("Adjust expression balance for symbol output.")
        )
        params_hint.setWordWrap(True)
        params_hint.setStyleSheet("color: #666; font-size: 11px;")
        params_layout.addWidget(params_hint)
        controls = resolve_style_controls(self.settings)

        factual_layout = QHBoxLayout()
        factual_layout.addWidget(QLabel(tr("Factuality:")))
        self.factuality_slider = QSlider(Qt.Horizontal)
        self.factuality_slider.setRange(STYLE_CONTROL_MIN, STYLE_CONTROL_MAX)
        self.factuality_slider.setValue(int(controls[STYLE_CONTROL_FACTUALITY]))
        self.factuality_slider.setToolTip(tr("0 = expressive symbol, 100 = measured/documentary."))
        self.factuality_slider.valueChanged.connect(self._on_style_params_changed)
        factual_layout.addWidget(self.factuality_slider)
        self.factuality_value_label = QLabel(str(STYLE_CONTROL_DEFAULTS[STYLE_CONTROL_FACTUALITY]))
        self.factuality_value_label.setMinimumWidth(34)
        factual_layout.addWidget(self.factuality_value_label)
        params_layout.addLayout(factual_layout)

        symbolic_layout = QHBoxLayout()
        symbolic_layout.addWidget(QLabel(tr("Symbol Looseness:")))
        self.symbolic_looseness_slider = QSlider(Qt.Horizontal)
        self.symbolic_looseness_slider.setRange(STYLE_CONTROL_MIN, STYLE_CONTROL_MAX)
        self.symbolic_looseness_slider.setValue(int(controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS]))
        self.symbolic_looseness_slider.setToolTip(tr("0 = tight measured shape, 100 = loose symbolic simplification."))
        self.symbolic_looseness_slider.valueChanged.connect(self._on_style_params_changed)
        symbolic_layout.addWidget(self.symbolic_looseness_slider)
        self.symbolic_looseness_value_label = QLabel(str(STYLE_CONTROL_DEFAULTS[STYLE_CONTROL_SYMBOLIC_LOOSENESS]))
        self.symbolic_looseness_value_label.setMinimumWidth(34)
        symbolic_layout.addWidget(self.symbolic_looseness_value_label)
        params_layout.addLayout(symbolic_layout)

        exaggeration_layout = QHBoxLayout()
        exaggeration_layout.addWidget(QLabel(tr("Exaggeration:")))
        self.exaggeration_slider = QSlider(Qt.Horizontal)
        self.exaggeration_slider.setRange(STYLE_CONTROL_MIN, STYLE_CONTROL_MAX)
        self.exaggeration_slider.setValue(int(controls[STYLE_CONTROL_EXAGGERATION]))
        self.exaggeration_slider.setToolTip(tr("0 = none, 100 = strong stylization and simplified emphasis."))
        self.exaggeration_slider.valueChanged.connect(self._on_style_params_changed)
        exaggeration_layout.addWidget(self.exaggeration_slider)
        self.exaggeration_value_label = QLabel(str(STYLE_CONTROL_DEFAULTS[STYLE_CONTROL_EXAGGERATION]))
        self.exaggeration_value_label.setMinimumWidth(34)
        exaggeration_layout.addWidget(self.exaggeration_value_label)
        params_layout.addLayout(exaggeration_layout)
        params_layout.addStretch()
        style_tabs.addTab(params_tab, tr("Parameters"))

        style_layout.addWidget(style_tabs)
        scroll_layout.addWidget(self.style_group)
        self._update_style_param_labels()
        
        # Template selection (initially hidden)
        self.template_group = QGroupBox(tr("Template Type"))
        template_layout = QVBoxLayout(self.template_group)

        category_row = QHBoxLayout()
        category_row.addWidget(QLabel(tr("Category:")))
        self.template_category_combo = QComboBox()
        for value, label in self.TEMPLATE_CATEGORY_LABELS:
            self.template_category_combo.addItem(label, value)
        self.template_category_combo.currentIndexChanged.connect(self._refresh_template_list)
        category_row.addWidget(self.template_category_combo, 1)
        template_layout.addLayout(category_row)

        self.template_search_input = QLineEdit()
        self.template_search_input.setPlaceholderText(tr("Filter templates (e.g., dagger, tomb, survey)"))
        self.template_search_input.textChanged.connect(self._refresh_template_list)
        template_layout.addWidget(self.template_search_input)

        self.template_combo = QComboBox()
        try:
            from ..generators.template_generator import TemplateGenerator
            self._template_generator = TemplateGenerator(self.plugin_dir)
            self._all_templates = sorted(self._template_generator.get_available_templates())
            self._template_categories = {
                key: sorted(values)
                for key, values in self._template_generator.get_categories().items()
            }
        except Exception:
            self._template_generator = None
            self._all_templates = [
                "Pottery",
                "Stone Tool",
                "Bronze Artifact",
                "Iron Artifact",
                "Weapon",
                "Excavation Area",
                "Survey Point",
                "Find Spot",
            ]
            self._template_categories = {}
        self._refresh_template_list()
        template_layout.addWidget(self.template_combo)
        self.template_group.setVisible(False)
        scroll_layout.addWidget(self.template_group)
        
        # Color settings
        color_group = QGroupBox(tr("Color"))
        color_layout = QVBoxLayout(color_group) # Changed to QVBoxLayout for better density
        
        # Row 1: Checkbox
        self.override_color_check = QCheckBox(tr("Override Color"))
        self.override_color_check.setChecked(False) # Default: Use extracted/natural color
        self.override_color_check.setToolTip(tr("If unchecked, the symbol will use the artifact's natural colors."))
        color_layout.addWidget(self.override_color_check)
        
        # Row 2: Picker controls
        picker_layout = QHBoxLayout()
        
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(30, 30)
        self.update_color_preview()
        picker_layout.addWidget(self.color_preview)
        
        self.color_btn = QPushButton(tr("Pick Color"))
        self.color_btn.clicked.connect(self.pick_color)
        picker_layout.addWidget(self.color_btn)
        
        self.eyedrop_btn = QPushButton(tr("Pick from Image"))
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
        size_group = QGroupBox(tr("Size Scaling"))
        size_layout = QVBoxLayout(size_group)
        
        size_mode_layout = QHBoxLayout()
        size_mode_layout.addWidget(QLabel(tr("Mode:")))
        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItems([
            tr("Fixed Size"),
            tr("By Data Count (Natural Breaks)"),
            tr("By Data Count (Equal Interval)"),
            tr("By Data Count (Quantile)")
        ])
        self.size_mode_combo.currentIndexChanged.connect(self._on_size_mode_changed)
        size_mode_layout.addWidget(self.size_mode_combo)
        size_layout.addLayout(size_mode_layout)

        minmax_layout = QHBoxLayout()
        minmax_layout.addWidget(QLabel(tr("Min:")))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(2, 128)
        self.min_size_spin.setValue(int(DEFAULT_MIN_SYMBOL_SIZE_MM))
        minmax_layout.addWidget(self.min_size_spin)
        
        minmax_layout.addWidget(QLabel(tr("Max:")))
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(2, 256)
        self.max_size_spin.setValue(int(DEFAULT_MAX_SYMBOL_SIZE_MM))
        minmax_layout.addWidget(self.max_size_spin)
        size_layout.addLayout(minmax_layout)

        size_field_layout = QHBoxLayout()
        size_field_layout.addWidget(QLabel(tr("Size Field:")))
        self.size_field_combo = QComboBox()
        self.size_field_combo.setToolTip(
            tr("Choose a numeric attribute for graduated size. "
            "Use Auto to pick the first numeric field.")
        )
        self.size_field_combo.addItem(tr("Auto (first numeric field)"), "")
        size_field_layout.addWidget(self.size_field_combo, 1)
        size_layout.addLayout(size_field_layout)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel(tr("Classes:")))
        self.class_count_spin = QSpinBox()
        self.class_count_spin.setRange(2, 12)
        self.class_count_spin.setValue(int(DEFAULT_GRADUATED_CLASSES))
        self.class_count_spin.setToolTip(tr("Number of size classes for graduated rendering."))
        class_layout.addWidget(self.class_count_spin)
        class_layout.addStretch()
        size_layout.addLayout(class_layout)
        scroll_layout.addWidget(size_group)

        # Target layer selection
        layer_group = QGroupBox(tr("Target Layer"))
        layer_layout = QHBoxLayout(layer_group)
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip(tr("Choose the point layer that will receive the generated symbol."))
        self.layer_combo.currentIndexChanged.connect(self._refresh_size_field_list)
        layer_layout.addWidget(self.layer_combo, 1)
        refresh_layers_btn = QPushButton(tr("Refresh"))
        refresh_layers_btn.clicked.connect(self.refresh_layer_list)
        layer_layout.addWidget(refresh_layers_btn)
        scroll_layout.addWidget(layer_group)

        # Prompt input (for AI modes)
        self.prompt_group = QGroupBox(tr("Text Prompt"))
        prompt_layout = QVBoxLayout(self.prompt_group)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText(tr("Enter a description for the icon (e.g., 'ancient pottery shard')"))
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
        
        self.generate_btn = QPushButton(tr("Generate"))
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

        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip(tr("Stop the running generation."))
        self.cancel_btn.clicked.connect(self.cancel_generation)
        button_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton(tr("Save to Library"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_to_library)
        button_layout.addWidget(self.save_btn)
        
        self.apply_btn = QPushButton(tr("Apply to Layer"))
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_to_layer)
        button_layout.addWidget(self.apply_btn)
        
        # Settings button
        settings_btn = QPushButton(tr("Settings"))
        settings_btn.clicked.connect(self.open_settings)
        button_layout.addWidget(settings_btn)
        
        right_panel.addLayout(button_layout)
        main_layout.addLayout(right_panel)
        self._set_mode_info_with_controls(show_controls=False)
        self.refresh_layer_list()
        self._on_size_mode_changed(self.size_mode_combo.currentIndex())

    def showEvent(self, event):
        """Refresh layer choices whenever dialog is shown."""
        super().showEvent(event)
        self.refresh_layer_list()
        
    def on_image_loaded(self, file_path):
        """Handle when an image is loaded."""
        self.generate_btn.setEnabled(True)
        self._update_input_quality_notice(file_path)
        
    def clear_input(self):
        """Clear the input image."""
        self.image_drop.clear_image()
        self._update_input_quality_notice(None)
        
    def on_mode_changed(self, button):
        """Handle generation mode change."""
        is_template = button == self.template_radio
        is_autotrace = button == self.autotrace_radio
        self.template_group.setVisible(is_template)
        self.style_group.setVisible(not is_template)
        self._set_mode_info_with_controls(show_controls=False)
        self.autotrace_upscale_check.setEnabled(is_autotrace)
        self.autotrace_detail_mode_combo.setEnabled(is_autotrace)
        self.round_strategy_combo.setEnabled(is_autotrace)
        self.input_kind_combo.setEnabled(is_autotrace)
        self.synthetic_structure_check.setEnabled(is_autotrace)

        # Show prompt input for HF mode (and maybe others in future)
        self.prompt_group.setVisible(
            button == self.hf_radio or 
            button == self.gemini_radio or 
            button == self.local_radio
        )
        
        # Update placeholder based on mode
        if button == self.hf_radio:
             self.prompt_input.setPlaceholderText(
                 tr("Optional: style note (e.g., 'typology plate icon with clear shoulder line')")
             )
        elif button == self.gemini_radio:
             self.prompt_input.setPlaceholderText(
                 tr("Optional: factual note (e.g., 'preserve chips and asymmetry, no decorative background')")
             )
        elif button == self.local_radio:
             self.prompt_input.setPlaceholderText(
                 tr("Optional: local SD prompt hint (e.g., 'flat archaeological icon, muted tones')")
             )

    def update_color_preview(self):
        """Update the color preview label."""
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {self.current_color.name()};
                border: 1px solid #333;
                border-radius: 4px;
            }}
        """)

    def _on_input_kind_changed(self, _index):
        """Persist the input-type choice."""
        kind = str(self.input_kind_combo.currentData() or "auto").strip().lower()
        if kind not in ("auto", "photo", "drawing"):
            kind = "auto"
        self.settings.setValue("ArcheoGlyph/autotrace_input_kind", kind)

    def _on_synthetic_structure_toggled(self, checked):
        """Persist the schematic-structure preference."""
        self.settings.setValue("ArcheoGlyph/autotrace_synthetic_structure", bool(checked))

    def _on_autotrace_upscale_toggled(self, checked):
        """Persist Auto Trace low-resolution upscale preference."""
        self.settings.setValue("ArcheoGlyph/autotrace_force_upscale", bool(checked))

    def _on_autotrace_detail_mode_changed(self, _index):
        """Persist Auto Trace detail mode preference."""
        mode = str(self.autotrace_detail_mode_combo.currentData() or "fast").strip().lower()
        if mode not in ("fast", "precise"):
            mode = "fast"
        self.settings.setValue("ArcheoGlyph/autotrace_detail_mode", mode)

    def _on_round_strategy_changed(self, _index):
        """Persist strategy for round artifact extraction."""
        strategy = str(self.round_strategy_combo.currentData() or "image_first").strip().lower()
        if strategy not in ("image_first", "hybrid", "structure_first"):
            strategy = "image_first"
        self.settings.setValue("ArcheoGlyph/round_strategy", strategy)

    def selected_style(self):
        """
        The canonical English style value behind the combo's label.

        The label is translated; the generators are not, so everything that
        passes a style onwards must go through here.
        """
        return self.style_combo.currentData() or STYLE_LEGEND

    def _apply_legend_quick_setup(self):
        """Apply a practical legend-like preset without requiring special terminology."""
        style_idx = self.style_combo.findData(STYLE_LEGEND)
        if style_idx >= 0:
            self.style_combo.setCurrentIndex(style_idx)

        # Stable, readable simple-symbol defaults.
        self.factuality_slider.setValue(84)
        self.symbolic_looseness_slider.setValue(22)
        self.exaggeration_slider.setValue(16)

        detail_idx = self.autotrace_detail_mode_combo.findData("precise")
        if detail_idx >= 0:
            self.autotrace_detail_mode_combo.setCurrentIndex(detail_idx)

        round_idx = self.round_strategy_combo.findData("structure_first")
        if round_idx >= 0:
            self.round_strategy_combo.setCurrentIndex(round_idx)

        self._persist_style_parameters()
        self._set_mode_info_with_controls(
            show_controls=True,
            base_text=(
                "Simple symbol preset applied: stable silhouette, bold outline, minimal structure lines."
            ),
        )

    def _apply_fast_convert_setup(self):
        """Apply a speed-priority preset for faster image-to-symbol conversion."""
        detail_idx = self.autotrace_detail_mode_combo.findData("fast")
        if detail_idx >= 0:
            self.autotrace_detail_mode_combo.setCurrentIndex(detail_idx)

        round_idx = self.round_strategy_combo.findData("image_first")
        if round_idx >= 0:
            self.round_strategy_combo.setCurrentIndex(round_idx)

        # Disable expensive low-res recovery by default for speed.
        self.autotrace_upscale_check.setChecked(False)

        # Keep style readable while reducing heavy internal-detail extraction pressure.
        self.factuality_slider.setValue(70)
        self.symbolic_looseness_slider.setValue(30)
        self.exaggeration_slider.setValue(14)

        self._persist_style_parameters()
        self._set_mode_info_with_controls(
            show_controls=True,
            base_text=(
                "Fast preset applied: speed priority (Fast mode, image-first, upscale off)."
            ),
        )

    def _image_sharpness(self, file_path):
        """
        Variance of the Laplacian: a blur measure. Returns None when OpenCV is
        unavailable or the file cannot be read.
        """
        try:
            import cv2
            import numpy as np

            with open(file_path, "rb") as stream:
                buffer = np.frombuffer(stream.read(), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            side = max(image.shape[:2])
            if side > 800:
                scale = 800.0 / side
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            return float(cv2.Laplacian(image, cv2.CV_64F).var())
        except Exception:
            return None

    def _update_input_quality_notice(self, file_path):
        """Warn when the source image is too small or too soft to trace well."""
        if not file_path or not os.path.exists(file_path):
            self.image_quality_hint_label.setVisible(False)
            self.image_quality_hint_label.setText("")
            return

        def _s_int(key, default):
            try:
                return int(self.settings.value(key, default))
            except (TypeError, ValueError):
                return int(default)

        weak_short_px = max(128, _s_int("ArcheoGlyph/image_warn_min_short_px", 700))
        recommended_short_px = max(
            weak_short_px, _s_int("ArcheoGlyph/image_warn_recommended_short_px", 900)
        )
        min_sharpness = float(_s_int("ArcheoGlyph/image_warn_min_sharpness", 60))

        px = QPixmap(file_path)
        if px.isNull():
            self.image_quality_hint_label.setVisible(False)
            self.image_quality_hint_label.setText("")
            return

        width = int(px.width())
        height = int(px.height())
        short_side = min(width, height)
        sharpness = self._image_sharpness(file_path)

        problems = []
        if short_side < weak_short_px:
            problems.append(f"the short side is {short_side}px (recommended {recommended_short_px}px)")
        elif short_side < recommended_short_px:
            problems.append(f"the short side is only {short_side}px")
        if sharpness is not None and sharpness < min_sharpness:
            problems.append("the image looks blurred or heavily compressed")

        if not problems:
            self.image_quality_hint_label.setVisible(False)
            self.image_quality_hint_label.setText("")
            return

        self.image_quality_hint_label.setText(
            f"<b>Input may trace poorly</b> ({width}x{height}): " + "; ".join(problems)
            + ". A tighter crop of a sharper photo gives cleaner symbols."
        )
        self.image_quality_hint_label.setVisible(True)

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
            self.eyedrop_btn.setText(tr("Click Image to Pick"))
        else:
            self.eyedrop_btn.setText(tr("Pick from Image"))

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
            QMessageBox.warning(self, tr("No Image"), tr("Please select an input image first."))
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate mode since we can't track exact progress in thread
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        # Ensure slider values are persisted before any generator reads QSettings.
        self._persist_style_parameters()
        
        try:
            target_func = None
            source_label = "unknown"
            kwargs = {}
            prompt = self.prompt_input.text().strip()
            selected_color = self.current_color.name() if self.override_color_check.isChecked() else None
            controls = self._current_style_controls()
            
            if self.autotrace_radio.isChecked():
                from ..generators.contour_generator import ContourGenerator
                self._current_generator = ContourGenerator()
                target_func = self._current_generator.generate_result
                source_label = "autotrace"
                detail_mode = str(
                    self.autotrace_detail_mode_combo.currentData() or "fast"
                ).strip().lower()
                if detail_mode not in ("fast", "precise"):
                    detail_mode = "fast"
                round_strategy = str(
                    self.round_strategy_combo.currentData() or "image_first"
                ).strip().lower()
                if round_strategy not in ("image_first", "hybrid", "structure_first"):
                    round_strategy = "image_first"
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'style': self.selected_style(),
                    'color': selected_color,
                    'symmetry': self.symmetry_check.isChecked(),
                    'force_lowres_upscale': self.autotrace_upscale_check.isChecked(),
                    'detail_mode': detail_mode,
                    'round_strategy': round_strategy,
                    'input_kind': str(self.input_kind_combo.currentData() or "auto"),
                    'synthetic_structure': self.synthetic_structure_check.isChecked(),
                    STYLE_CONTROL_FACTUALITY: controls[STYLE_CONTROL_FACTUALITY],
                    STYLE_CONTROL_SYMBOLIC_LOOSENESS: controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS],
                    STYLE_CONTROL_EXAGGERATION: controls[STYLE_CONTROL_EXAGGERATION],
                }
            elif self.gemini_radio.isChecked():
                from ..generators.gemini_generator import GeminiGenerator
                self._current_generator = GeminiGenerator()
                target_func = self._current_generator.generate
                source_label = "gemini"
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'prompt': prompt,
                    'style': self.selected_style(),
                    'color': selected_color,
                    'symmetry': self.symmetry_check.isChecked(),
                    STYLE_CONTROL_FACTUALITY: controls[STYLE_CONTROL_FACTUALITY],
                    STYLE_CONTROL_SYMBOLIC_LOOSENESS: controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS],
                    STYLE_CONTROL_EXAGGERATION: controls[STYLE_CONTROL_EXAGGERATION],
                }
                
            elif self.hf_radio.isChecked():
                from ..generators.huggingface_generator import HuggingFaceGenerator
                self._current_generator = HuggingFaceGenerator()
                target_func = self._current_generator.generate
                source_label = "huggingface"

                if prompt:
                    self.mode_info_label.setText(
                        tr("HF custom prompt active: text guidance will influence stylization.")
                    )
                else:
                    self.mode_info_label.setText(
                        tr("HF default factual guidance active.")
                    )
                    prompt = "archaeological artifact from reference photo"

                kwargs = {
                    'prompt': prompt,
                    'style': self.selected_style(),
                    'color': selected_color,
                    'image_path': self.image_drop.image_path,
                    'symmetry': self.symmetry_check.isChecked(),
                    STYLE_CONTROL_FACTUALITY: controls[STYLE_CONTROL_FACTUALITY],
                    STYLE_CONTROL_SYMBOLIC_LOOSENESS: controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS],
                    STYLE_CONTROL_EXAGGERATION: controls[STYLE_CONTROL_EXAGGERATION],
                }

            elif self.local_radio.isChecked():
                from ..generators.local_generator import LocalGenerator
                self._current_generator = LocalGenerator()
                target_func = self._current_generator.generate
                source_label = "local-sd"
                kwargs = {
                    'image_path': self.image_drop.image_path,
                    'prompt': prompt,
                    'style': self.selected_style(),
                    'color': selected_color,
                    STYLE_CONTROL_FACTUALITY: controls[STYLE_CONTROL_FACTUALITY],
                    STYLE_CONTROL_SYMBOLIC_LOOSENESS: controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS],
                    STYLE_CONTROL_EXAGGERATION: controls[STYLE_CONTROL_EXAGGERATION],
                }
            else:
                from ..generators.template_generator import TemplateGenerator
                self._current_generator = TemplateGenerator(self.plugin_dir)
                target_func = self._current_generator.generate
                source_label = "template"
                template_name = self.template_combo.currentData()
                if not template_name:
                    QMessageBox.warning(self, tr("No Template"), tr("Adjust template filters and select a valid template."))
                    self.progress_bar.setVisible(False)
                    self.progress_bar.setRange(0, 100)
                    self.generate_btn.setEnabled(True)
                    return
                kwargs = {
                    'template_type': template_name,
                    'color': selected_color
                }
            
            if target_func:
                self.generation_thread = GenerationThread(
                    target_func, source_label, self.selected_style(), **kwargs
                )
                self.generation_thread.result_ready.connect(self.on_generation_finished)
                self.generation_thread.start()
            
        except Exception as e:
            self.on_generation_finished(None, str(e))

    def on_generation_finished(self, result, error_message):
        """Handle generation results."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100) # Reset to normal
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        cancelled = self.generation_thread is not None and self.generation_thread.cancelled
        self._current_generator = None  # Release reference
        self._set_mode_info_with_controls(show_controls=True)
        
        if cancelled:
            self._set_mode_info_with_controls(show_controls=False, base_text="Generation cancelled.")
            return

        if error_message:
            message = str(error_message or "")
            lower = message.lower()
            if "quota exceeded" in lower or "resourceexhausted" in lower:
                QMessageBox.critical(
                    self,
                    tr("Quota Exceeded"),
                    tr("Google Gemini quota is currently exhausted for this API key/project.\n\n"
                    "Actions:\n"
                    "1. Wait for quota reset and retry.\n"
                    "2. Use Auto Trace or Hugging Face in the meantime.\n"
                    "3. Check quota/billing in Google AI Studio.")
                )
                return
            QMessageBox.critical(
                    self, tr("Error"),
                    tr("Generation failed: {message}").format(message=message),
                )
            return
            
        if result is None or result.is_empty:
            QMessageBox.warning(self, tr("Failed"), tr("Generation returned no result."))
            return

        if not result.is_vector and result.raster_png:
            # AI backends return raster images; trace them so the symbol still
            # reaches QGIS as a scalable, recolourable SVG marker.
            try:
                from ..generators.raster_vectorize import vectorize_result

                vectorize_result(result, style=result.style)
            except Exception as e:
                result.add_warning(f"Vectorisation failed: {e}")

        pixmap = self._result_to_pixmap(result)
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, tr("Failed"), tr("Generated symbol could not be rendered."))
            return

        self.current_result = result
        self.preview_label.set_preview(pixmap)
        self.save_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)

        kind = "vector SVG" if result.is_vector else "raster PNG"
        info = f"Result: {kind} from {result.source}"
        if result.warnings:
            info += " | " + "; ".join(result.warnings[:3])
        self._set_mode_info_with_controls(show_controls=False, base_text=info)

    def cancel_generation(self):
        """Stop a running generation."""
        if self.generation_thread is not None and self.generation_thread.isRunning():
            self.generation_thread.cancel()
            self.cancel_btn.setEnabled(False)
            self._set_mode_info_with_controls(show_controls=False, base_text="Cancelling...")

    def closeEvent(self, event):
        """Never let the dialog die while its worker thread is still running."""
        if self.generation_thread is not None and self.generation_thread.isRunning():
            self.generation_thread.cancel()
            self.generation_thread.wait(3000)
        super().closeEvent(event)

    def _result_to_pixmap(self, result):
        """Render a SymbolResult for the preview (GUI thread only)."""
        if result.is_vector:
            pixmap = self._svg_to_pixmap(result.svg)
            if pixmap is not None:
                return pixmap
        if result.raster_png:
            pixmap = QPixmap()
            if pixmap.loadFromData(result.raster_png, "PNG"):
                return pixmap
        return None

    def _svg_to_pixmap(self, svg_code, size=512):
        """Render SVG code to a square QPixmap (must be called on the main/GUI thread)."""
        from qgis.PyQt.QtCore import QByteArray
        from qgis.PyQt.QtSvg import QSvgRenderer
        from qgis.PyQt.QtGui import QPainter
        
        renderer = QSvgRenderer(QByteArray(svg_code.encode('utf-8')))
        
        if not renderer.isValid():
            return None

        renderer.setAspectRatioMode(Qt.KeepAspectRatio)
        pixmap = QPixmap(int(size), int(size))
        pixmap.fill(Qt.transparent)

        view_box = renderer.viewBoxF()
        if not view_box.isValid() or view_box.width() <= 0 or view_box.height() <= 0:
            default_size = renderer.defaultSize()
            if default_size.isValid() and default_size.width() > 0 and default_size.height() > 0:
                view_box = QRectF(0, 0, float(default_size.width()), float(default_size.height()))
            else:
                view_box = QRectF(0, 0, float(size), float(size))

        side = float(size)
        scale = min(side / view_box.width(), side / view_box.height())
        target_w = view_box.width() * scale
        target_h = view_box.height() * scale
        target_x = (side - target_w) * 0.5
        target_y = (side - target_h) * 0.5
        target_rect = QRectF(target_x, target_y, target_w, target_h)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter, target_rect)
        painter.end()

        return pixmap
            
    def save_to_library(self):
        """Save generated symbol to the QGIS symbol library."""
        if self.current_result is None:
            QMessageBox.warning(self, tr("No Symbol"), tr("Please generate a symbol first."))
            return

        name, ok = QInputDialog.getText(
            self, tr("Save to Library"), tr("Symbol name:"), text="ArchaeoGlyph Symbol"
        )
        if not ok:
            return

        from ..symbol_manager import SymbolManager

        final_name = SymbolManager().save_to_library(self.current_result, name=name)
        if final_name:
            QMessageBox.information(
                    self, tr("Saved"),
                    tr("Symbol saved to QGIS library as '{name}'.").format(name=final_name),
                )
        else:
            QMessageBox.warning(self, tr("Error"), tr("Failed to save symbol. See the ArchaeoGlyph message log."))

    def apply_to_layer(self):
        """Apply generated symbol to current layer."""
        layer = self._get_selected_layer()
        
        if not layer:
            QMessageBox.warning(self, tr("No Layer"), tr("Please choose a point layer in Target Layer."))
            return
            
        if self.current_result is None:
            QMessageBox.warning(self, tr("No Symbol"), tr("Please generate a symbol first."))
            return
            
        from ..symbol_manager import SymbolManager
        
        manager = SymbolManager()
        
        # Get size scaling settings
        size_mode = self.size_mode_combo.currentIndex()
        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value()
        size_field = self.size_field_combo.currentData() if hasattr(self, "size_field_combo") else ""
        class_count = self.class_count_spin.value() if hasattr(self, "class_count_spin") else DEFAULT_GRADUATED_CLASSES

        success = manager.apply_to_layer(
            layer=layer,
            result=self.current_result,
            size_mode=size_mode,
            min_size=min_size,
            max_size=max_size,
            size_field=size_field or None,
            num_classes=class_count,
        )
        
        if success:
            QMessageBox.information(
                    self, tr("Applied"),
                    tr("Symbol applied to layer: {layer}").format(layer=layer.name()),
                )
            layer.triggerRepaint()
        else:
            QMessageBox.warning(self, tr("Error"), tr("Failed to apply symbol to layer. See the ArchaeoGlyph message log."))
            
    def open_settings(self):
        """Open the settings dialog."""
        from .settings_dialog import SettingsDialog
        
        dialog = SettingsDialog(self)
        dialog.exec_()
        detail_mode = str(
            self.settings.value("ArcheoGlyph/autotrace_detail_mode", "fast")
        ).strip().lower()
        if detail_mode not in ("fast", "precise"):
            detail_mode = "fast"
        detail_idx = self.autotrace_detail_mode_combo.findData(detail_mode)
        if detail_idx >= 0:
            self.autotrace_detail_mode_combo.setCurrentIndex(detail_idx)
        self.autotrace_upscale_check.setChecked(
            self.settings.value("ArcheoGlyph/autotrace_force_upscale", True, type=bool)
        )
        round_strategy = str(
            self.settings.value("ArcheoGlyph/round_strategy", "image_first")
        ).strip().lower()
        if round_strategy not in ("image_first", "hybrid", "structure_first"):
            round_strategy = "image_first"
        round_strategy_idx = self.round_strategy_combo.findData(round_strategy)
        if round_strategy_idx >= 0:
            self.round_strategy_combo.setCurrentIndex(round_strategy_idx)
        self._update_input_quality_notice(self.image_drop.image_path)

    def _update_style_param_labels(self):
        """Update labels for style parameter sliders."""
        controls = self._current_style_controls()
        self.factuality_value_label.setText(str(controls[STYLE_CONTROL_FACTUALITY]))
        self.symbolic_looseness_value_label.setText(str(controls[STYLE_CONTROL_SYMBOLIC_LOOSENESS]))
        self.exaggeration_value_label.setText(str(controls[STYLE_CONTROL_EXAGGERATION]))

    def _on_style_params_changed(self, _value):
        """Handle slider changes."""
        self._update_style_param_labels()
        self._persist_style_parameters()

    def _current_style_controls(self):
        """Return normalized style controls from current slider values."""
        return resolve_style_controls(
            settings=None,
            factuality=self.factuality_slider.value(),
            symbolic_looseness=self.symbolic_looseness_slider.value(),
            exaggeration=self.exaggeration_slider.value(),
        )

    def _active_mode_description(self):
        """Return current mode description text."""
        if self.gemini_radio.isChecked():
            return self.MODE_DESCRIPTION["gemini"]
        if self.hf_radio.isChecked():
            return self.MODE_DESCRIPTION["hf"]
        if self.local_radio.isChecked():
            return self.MODE_DESCRIPTION["local"]
        if self.template_radio.isChecked():
            return self.MODE_DESCRIPTION["template"]
        return self.MODE_DESCRIPTION["autotrace"]

    def _set_mode_info_with_controls(self, show_controls=False, base_text=None):
        """Update mode info label, optionally appending style-control values."""
        text = str(base_text or self._active_mode_description())
        if show_controls:
            text += f" | Controls: {style_controls_short_text(self._current_style_controls())}"
        self.mode_info_label.setText(text)

    def _persist_style_parameters(self):
        """Persist style parameter controls for generators."""
        save_style_controls(self.settings, self._current_style_controls())

    def refresh_layer_list(self):
        """Refresh selectable point layers."""
        if not hasattr(self, "layer_combo"):
            return

        previous_layer_id = self.layer_combo.currentData()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()

        point_layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            if layer.geometryType() != QgsWkbTypes.PointGeometry:
                continue
            point_layers.append(layer)

        point_layers.sort(key=lambda layer: layer.name().lower())

        if not point_layers:
            self.layer_combo.addItem(tr("No point layers available"), "")
            self.layer_combo.setEnabled(False)
            self.layer_combo.blockSignals(False)
            self._refresh_size_field_list()
            return

        self.layer_combo.setEnabled(True)
        for layer in point_layers:
            self.layer_combo.addItem(layer.name(), layer.id())

        target_layer_id = previous_layer_id
        if not target_layer_id:
            active = self.iface.activeLayer()
            if isinstance(active, QgsVectorLayer) and active.geometryType() == QgsWkbTypes.PointGeometry:
                target_layer_id = active.id()

        selected_index = 0
        for idx in range(self.layer_combo.count()):
            if self.layer_combo.itemData(idx) == target_layer_id:
                selected_index = idx
                break
        self.layer_combo.setCurrentIndex(selected_index)
        self.layer_combo.blockSignals(False)
        self._refresh_size_field_list()

    def _get_selected_layer(self):
        """Return currently selected point layer."""
        layer_id = self.layer_combo.currentData()
        if not layer_id:
            return None

        layer = QgsProject.instance().mapLayer(layer_id)
        if not isinstance(layer, QgsVectorLayer):
            return None
        if layer.geometryType() != QgsWkbTypes.PointGeometry:
            return None
        return layer

    def _refresh_template_list(self):
        """Filter template list by category and search text."""
        if not hasattr(self, "template_combo"):
            return

        selected = self.template_combo.currentData() if self.template_combo.count() else None
        category = self.template_category_combo.currentData() if hasattr(self, "template_category_combo") else "all"
        query = self.template_search_input.text().strip().lower() if hasattr(self, "template_search_input") else ""

        if category == "all":
            filtered = list(self._all_templates or [])
        else:
            filtered = sorted(list((self._template_categories or {}).get(category, [])))

        if query:
            filtered = [name for name in filtered if query in name.lower()]

        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        if filtered:
            for name in filtered:
                # Canonical English name as data: it keys TEMPLATE_INFO and is
                # what gets stored, so translating the label is safe.
                self.template_combo.addItem(template_display_name(name), name)
            if selected and selected in filtered:
                self.template_combo.setCurrentIndex(self.template_combo.findData(selected))
        else:
            # No match: an item with no data, so callers see an empty selection
            # instead of comparing against a sentence.
            self.template_combo.addItem(tr("No templates match current filter"), None)
        self.template_combo.blockSignals(False)

    def _on_size_mode_changed(self, index):
        """Toggle graduated-size options based on selected mode."""
        graduated = int(index) != 0
        self.max_size_spin.setEnabled(graduated)
        self.size_field_combo.setEnabled(graduated)
        self.class_count_spin.setEnabled(graduated)

    def _refresh_size_field_list(self):
        """Refresh numeric field choices for selected target layer."""
        if not hasattr(self, "size_field_combo"):
            return

        previous = self.size_field_combo.currentData()
        self.size_field_combo.blockSignals(True)
        self.size_field_combo.clear()
        self.size_field_combo.addItem(tr("Auto (first numeric field)"), "")

        layer = self._get_selected_layer()
        if layer:
            for field in layer.fields():
                if field.isNumeric():
                    self.size_field_combo.addItem(field.name(), field.name())

        for idx in range(self.size_field_combo.count()):
            if self.size_field_combo.itemData(idx) == previous:
                self.size_field_combo.setCurrentIndex(idx)
                break

        self.size_field_combo.blockSignals(False)
