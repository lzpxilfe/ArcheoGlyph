# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Main Plugin Class
"""

import os
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .i18n import apply_settings_language, tr
from .ui.main_dialog import ArcheoGlyphDialog
from .defaults import HF_DEFAULT_MODEL_ID, HF_LEGACY_MODEL_ALIASES, PLUGIN_VERSION
from .log import log_exception


class ArcheoGlyph:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.
        
        :param iface: An interface instance that provides the hook to QGIS.
        :type iface: QgsInterface
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        
        # Initialize locale
        locale = (QSettings().value('locale/userLocale') or 'en')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            f'ArcheoGlyph_{locale}.qm'
        )

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # The plugin ships no compiled .qm catalogues; the Python catalogue in
        # i18n.py is what actually translates the UI. Menu and toolbar text is
        # built once, so a language change shows up after a QGIS restart.
        apply_settings_language()

        self.actions = []
        self._register_symbol_search_path()
        self.menu = self.tr('&ArchaeoGlyph')
        self.toolbar = self.iface.addToolBar('ArchaeoGlyph')
        self.toolbar.setObjectName('ArchaeoGlyph')
        
        self.dialog = None
        self._migrate_settings()

    def _register_symbol_search_path(self):
        """
        Let QGIS resolve generated symbols by name, so saved projects keep
        working when they are opened on another machine.
        """
        try:
            from .symbol_manager import register_svg_search_path

            register_svg_search_path()
        except Exception as e:
            log_exception("Could not register the symbol folder with QGIS", e)

    def _migrate_settings(self):
        """
        One-time settings migration for this plugin code version.
        Prevents stale old-model settings from lingering across updates.
        """
        settings = QSettings()
        if settings.value('ArcheoGlyph/auto_update_models', None) is None:
            settings.setValue('ArcheoGlyph/auto_update_models', 'true')
        if not str(settings.value('ArcheoGlyph/sam_model_type', '')).strip():
            settings.setValue('ArcheoGlyph/sam_model_type', 'hf:facebook/sam2.1-hiera-large')
        saved_version = str(settings.value('ArcheoGlyph/code_version', '')).strip()
        if saved_version == PLUGIN_VERSION:
            return

        hf_model = str(settings.value('ArcheoGlyph/hf_model_id', '')).strip()
        if not hf_model:
            settings.setValue('ArcheoGlyph/hf_model_id', HF_DEFAULT_MODEL_ID)
        elif hf_model in HF_LEGACY_MODEL_ALIASES:
            settings.setValue('ArcheoGlyph/hf_model_id', HF_LEGACY_MODEL_ALIASES.get(hf_model, HF_DEFAULT_MODEL_ID))

        # Safety migration: invalid SAM setup should not block Auto Trace.
        mask_backend = str(settings.value('ArcheoGlyph/mask_backend', 'auto')).strip().lower()
        if mask_backend not in ("auto", "opencv", "onnx", "sam"):
            mask_backend = "auto"
            settings.setValue('ArcheoGlyph/mask_backend', mask_backend)
        sam_model_type = str(settings.value('ArcheoGlyph/sam_model_type', 'hf:facebook/sam2.1-hiera-large')).strip().lower()
        if not sam_model_type:
            sam_model_type = "hf:facebook/sam2.1-hiera-large"
            settings.setValue('ArcheoGlyph/sam_model_type', sam_model_type)
        elif sam_model_type == "hf:facebook/sam3-hiera-large":
            # SAM3 public model availability can fluctuate; keep a stable default.
            sam_model_type = "hf:facebook/sam2.1-hiera-large"
            settings.setValue('ArcheoGlyph/sam_model_type', sam_model_type)
        uses_hf_sam = sam_model_type.startswith("hf:")
        sam_checkpoint = str(settings.value('ArcheoGlyph/sam_checkpoint_path', '')).strip()
        if mask_backend == "sam" and (not uses_hf_sam) and (not sam_checkpoint or not os.path.exists(sam_checkpoint)):
            settings.setValue('ArcheoGlyph/mask_backend', 'auto')

        # Persist plugin code version marker.
        settings.setValue('ArcheoGlyph/code_version', PLUGIN_VERSION)

    def tr(self, message):
        """
        Translate a string.

        The plugin catalogue is tried first; Qt's own translation is kept as a
        fallback so a compiled .qm catalogue would still be honoured.
        """
        translated = tr(message)
        if translated != message:
            return translated
        return QCoreApplication.translate('ArcheoGlyph', message)

    def add_action(
            self,
            icon_path,
            text,
            callback,
            enabled_flag=True,
            add_to_menu=True,
            add_to_toolbar=True,
            status_tip=None,
            whats_this=None,
            parent=None):
        """Add a toolbar icon to the toolbar."""

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = os.path.join(self.plugin_dir, 'resources', 'icon.svg')
        
        self.add_action(
            icon_path,
            text=self.tr('ArchaeoGlyph Symbol Generator'),
            callback=self.run,
            parent=self.iface.mainWindow()
        )

    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr('&ArchaeoGlyph'), action)
            self.iface.removeToolBarIcon(action)
        del self.toolbar

    def run(self):
        """Run method that opens the plugin dialog."""
        if self.dialog is None:
            self.dialog = ArcheoGlyphDialog(self.iface, parent=self.iface.mainWindow())
        
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
