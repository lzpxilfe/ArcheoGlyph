# -*- coding: utf-8 -*-
"""
ArchaeoGlyph - Symbol Manager
Stores generated symbols in the QGIS user profile and applies them to layers
or the symbol library as SVG (preferred) or raster marker layers.
"""

import math
import os
import re

from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsApplication,
    QgsGraduatedSymbolRenderer,
    QgsMarkerSymbol,
    QgsRasterMarkerSymbolLayer,
    QgsRendererRange,
    QgsSingleSymbolRenderer,
    QgsStyle,
    QgsSvgMarkerSymbolLayer,
    QgsUnitTypes,
)

from .defaults import (
    DEFAULT_GRADUATED_CLASSES,
    DEFAULT_LIBRARY_SYMBOL_SIZE_MM,
    DEFAULT_MAX_SYMBOL_SIZE_MM,
    DEFAULT_MIN_SYMBOL_SIZE_MM,
)
from .generators.symbol_result import SymbolResult
from .log import log, log_exception
from .symbol_breaks import compute_breaks

STORE_SUBDIR = os.path.join("archeoglyph", "symbols")
SVG_SEARCH_PATH_SETTING = "svg/searchPathsForSVG"


def merge_search_paths(existing, new_path):
    """
    Add ``new_path`` to a list of QGIS SVG search paths, without duplicates.

    Returns the new list, or None when nothing needs to change. Kept separate
    from QSettings so it can be tested directly.
    """
    target = os.path.normpath(str(new_path))
    paths = []
    if isinstance(existing, str):
        candidates = existing.split("|") if "|" in existing else [existing]
    else:
        candidates = list(existing or [])
    for candidate in candidates:
        text = str(candidate).strip()
        if text:
            paths.append(text)
    if any(os.path.normpath(p) == target for p in paths):
        return None
    return paths + [target]


def register_svg_search_path(directory=None):
    """
    Make the symbol store one of QGIS's SVG search paths.

    QGIS writes SVG marker paths into a project relative to its search paths,
    so a project saved here still finds its symbols on another machine that
    has the plugin installed. Without this, projects carry absolute paths.
    """
    from qgis.PyQt.QtCore import QSettings

    directory = directory or symbol_store_dir()
    settings = QSettings()
    updated = merge_search_paths(settings.value(SVG_SEARCH_PATH_SETTING, []), directory)
    if updated is None:
        return False
    settings.setValue(SVG_SEARCH_PATH_SETTING, updated)
    log(f"Registered the ArchaeoGlyph symbol folder as a QGIS SVG search path: {directory}")
    return True


def symbol_store_dir():
    """
    Directory that holds generated symbol files.

    Lives inside the active QGIS user profile, so it survives plugin updates,
    is writable on system-wide installs, and is never cleaned up behind a
    project that references it.
    """
    base = QgsApplication.qgisSettingsDirPath() or os.path.expanduser("~")
    path = os.path.join(base, STORE_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


class SymbolManager:
    """Manager for symbol operations in QGIS."""

    def __init__(self):
        self.symbol_dir = symbol_store_dir()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, result):
        """
        Write ``result`` (SymbolResult or legacy QPixmap/QImage/str) to the
        store and return ``(path, result)``. Files are content-addressed, so
        the same symbol is never written twice and old references stay valid.
        """
        result = SymbolResult.coerce(result)
        if result is None or result.is_empty:
            raise ValueError("No symbol content to store.")
        path = os.path.join(self.symbol_dir, f"{result.content_hash()}.{result.extension}")
        if not os.path.exists(path):
            with open(path, "wb") as stream:
                stream.write(result.payload_bytes())
        return path, result

    def _make_symbol_layer(self, path, result, size_mm):
        """Build an SVG or raster marker layer for a stored symbol file."""
        size_mm = float(size_mm)
        if path.lower().endswith(".svg"):
            layer = QgsSvgMarkerSymbolLayer(path)
            meta = result.meta if result is not None else {}
            fill = meta.get("fill")
            outline = meta.get("outline")
            if fill:
                layer.setFillColor(QColor(str(fill)))
            if outline:
                layer.setStrokeColor(QColor(str(outline)))
            width_units = meta.get("outline_width")
            viewbox = meta.get("viewbox")
            if width_units and viewbox and len(viewbox) == 4 and float(viewbox[2]) > 0:
                # Convert the SVG's own stroke width into millimetres at this size.
                width_mm = float(width_units) * size_mm / float(viewbox[2])
                layer.setStrokeWidth(max(0.05, width_mm))
                layer.setStrokeWidthUnit(QgsUnitTypes.RenderMillimeters)
        else:
            layer = QgsRasterMarkerSymbolLayer(path)
        layer.setSize(size_mm)
        layer.setSizeUnit(QgsUnitTypes.RenderMillimeters)
        return layer

    def _make_symbol(self, path, result, size_mm):
        symbol = QgsMarkerSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        symbol.appendSymbolLayer(self._make_symbol_layer(path, result, size_mm))
        return symbol

    # ------------------------------------------------------------------
    # Library
    # ------------------------------------------------------------------

    def save_to_library(self, result, name="ArchaeoGlyph Symbol"):
        """
        Save a symbol to the QGIS default style database.

        :return: the final (unique) symbol name, or None on failure
        """
        try:
            path, result = self.store(result)
            symbol = self._make_symbol(path, result, DEFAULT_LIBRARY_SYMBOL_SIZE_MM)

            style = QgsStyle.defaultStyle()
            base_name = re.sub(r"\s+", " ", str(name or "ArchaeoGlyph Symbol")).strip() or "ArchaeoGlyph Symbol"
            final_name = base_name
            counter = 1
            existing = set(style.symbolNames())
            while final_name in existing:
                final_name = f"{base_name} {counter}"
                counter += 1

            if not style.addSymbol(final_name, symbol, True):
                log(f"QgsStyle refused symbol '{final_name}'", level="warning")
                return None
            try:
                style.tagSymbol(QgsStyle.SymbolEntity, final_name, ["ArchaeoGlyph"])
            except Exception as e:  # tagging is cosmetic
                log_exception("Could not tag symbol", e)
            return final_name
        except Exception as e:
            log_exception("Error saving symbol", e)
            return None

    # ------------------------------------------------------------------
    # Layer rendering
    # ------------------------------------------------------------------

    def apply_to_layer(
        self,
        layer,
        result,
        size_mode=0,
        min_size=DEFAULT_MIN_SYMBOL_SIZE_MM,
        max_size=DEFAULT_MAX_SYMBOL_SIZE_MM,
        size_field=None,
        num_classes=DEFAULT_GRADUATED_CLASSES,
    ):
        """
        Apply a symbol to a point layer.

        :param size_mode: 0=fixed, 1=natural breaks, 2=equal interval, 3=quantile
        :return: True if successful
        """
        try:
            path, result = self.store(result)
            if int(size_mode) == 0:
                return self._apply_single_symbol(layer, path, result, min_size)
            return self._apply_graduated_symbol(
                layer, path, result, size_mode, min_size, max_size, size_field, num_classes
            )
        except Exception as e:
            log_exception("Error applying symbol", e)
            return False

    def _apply_single_symbol(self, layer, path, result, size):
        symbol = self._make_symbol(path, result, size)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        return True

    def _apply_graduated_symbol(
        self, layer, path, result, size_mode, min_size, max_size, size_field=None,
        num_classes=DEFAULT_GRADUATED_CLASSES,
    ):
        """Graduated-size renderer driven by a numeric attribute."""
        fallback_size = (float(min_size) + float(max_size)) / 2.0

        if not size_field:
            for field in layer.fields():
                if field.isNumeric():
                    size_field = field.name()
                    break
        if not size_field:
            return self._apply_single_symbol(layer, path, result, fallback_size)

        idx = layer.fields().indexOf(size_field)
        values = self._extract_numeric_values(layer, idx)
        if not values or min(values) == max(values):
            return self._apply_single_symbol(layer, path, result, fallback_size)

        class_count = int(num_classes) if num_classes is not None else DEFAULT_GRADUATED_CLASSES
        class_count = max(2, min(class_count, len(values)))
        breaks = compute_breaks(values, class_count, size_mode)
        if len(breaks) < 2:
            return self._apply_single_symbol(layer, path, result, fallback_size)

        ranges = []
        break_count = len(breaks) - 1
        for i in range(break_count):
            lower = float(breaks[i])
            upper = float(breaks[i + 1])
            if upper <= lower:
                continue
            size = float(min_size) + (float(max_size) - float(min_size)) * ((i + 0.5) / float(break_count))
            range_symbol = self._make_symbol(path, result, size)
            ranges.append(QgsRendererRange(lower, upper, range_symbol, f"{lower:.2f} - {upper:.2f}"))

        if not ranges:
            return self._apply_single_symbol(layer, path, result, fallback_size)

        layer.setRenderer(QgsGraduatedSymbolRenderer(size_field, ranges))
        return True

    def _extract_numeric_values(self, layer, field_index):
        """Extract finite numeric values from a layer field index."""
        values = []
        for feature in layer.getFeatures():
            try:
                value = feature[field_index]
                if value is None:
                    continue
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values.append(numeric_value)
            except (TypeError, ValueError):
                continue
        return values

    def get_saved_symbols(self):
        """List symbol files in the store."""
        symbols = []
        if os.path.exists(self.symbol_dir):
            for name in sorted(os.listdir(self.symbol_dir)):
                if name.lower().endswith((".svg", ".png")):
                    symbols.append({
                        "name": os.path.splitext(name)[0],
                        "path": os.path.join(self.symbol_dir, name),
                    })
        return symbols
