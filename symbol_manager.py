# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Symbol Manager
Manages saving, loading, and applying symbols to QGIS layers.
"""

import os
import math
import re
import tempfile

from .log import log_exception
from .symbol_breaks import compute_breaks
from qgis.core import (
    QgsMarkerSymbol,
    QgsRasterMarkerSymbolLayer, QgsSingleSymbolRenderer,
    QgsGraduatedSymbolRenderer, QgsRendererRange,
    QgsStyle, QgsUnitTypes
)
import time as import_time

from .defaults import (
    DEFAULT_GRADUATED_CLASSES,
    DEFAULT_LIBRARY_SYMBOL_SIZE_MM,
    DEFAULT_MAX_SYMBOL_SIZE_MM,
    DEFAULT_MIN_SYMBOL_SIZE_MM,
)


class SymbolManager:
    """Manager for symbol operations in QGIS."""
    
    def __init__(self):
        """Initialize the symbol manager."""
        self.symbol_dir = self._get_symbol_directory()
        
    def _get_symbol_directory(self):
        """Get or create the symbol storage directory."""
        base_dir = os.path.join(
            os.path.dirname(__file__),
            'symbols'
        )
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return base_dir

    def _get_runtime_cache_directory(self):
        """Get or create a deterministic runtime cache directory for layer-applied symbols."""
        cache_dir = os.path.join(tempfile.gettempdir(), "archeoglyph_symbols")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return cache_dir

    def _cleanup_runtime_cache(self, cache_dir, max_age_days=30):
        """Remove very old cache files to avoid unbounded temp growth."""
        try:
            now_ts = import_time.time()
            max_age_seconds = float(max_age_days) * 24.0 * 60.0 * 60.0
            for name in os.listdir(cache_dir):
                if not name.lower().startswith("archeoglyph_symbol_"):
                    continue
                if not name.lower().endswith(".png"):
                    continue
                path = os.path.join(cache_dir, name)
                if not os.path.isfile(path):
                    continue
                age_seconds = now_ts - os.path.getmtime(path)
                if age_seconds > max_age_seconds:
                    try:
                        os.remove(path)
                    except Exception:
                        continue
        except Exception:
            # Cache cleanup should never break symbol application.
            return

    def _layer_symbol_cache_path(self, layer):
        """Return deterministic cache path per layer id."""
        cache_dir = self._get_runtime_cache_directory()
        self._cleanup_runtime_cache(cache_dir)

        layer_id = ""
        try:
            layer_id = str(layer.id() or "")
        except Exception:
            layer_id = ""
        safe_layer_id = re.sub(r"[^A-Za-z0-9_\-\.]", "_", layer_id).strip("._")
        if not safe_layer_id:
            safe_layer_id = f"tmp_{int(import_time.time() * 1000)}"
        return os.path.join(cache_dir, f"archeoglyph_symbol_{safe_layer_id}.png")
        
    def save_to_library(self, pixmap, name="ArcheoGlyph Symbol"):
        """
        Save a symbol to QGIS style library.
        
        :param pixmap: QPixmap of the symbol
        :param name: Name for the symbol
        :return: True if successful
        """
        try:
            # Save the pixmap to a file
            file_path = os.path.join(self.symbol_dir, f"{name}.png")
            pixmap.save(file_path, "PNG")
            
            # Create a marker symbol with the image
            symbol = QgsMarkerSymbol.createSimple({})
            symbol.deleteSymbolLayer(0)
            
            raster_layer = QgsRasterMarkerSymbolLayer(file_path)
            raster_layer.setSize(DEFAULT_LIBRARY_SYMBOL_SIZE_MM)
            symbol.appendSymbolLayer(raster_layer)
            
            # Add to QGIS default style
            style = QgsStyle.defaultStyle()
            
            # Generate unique name if exists
            final_name = name
            counter = 1
            while style.symbolNames().count(final_name) > 0:
                final_name = f"{name}_{counter}"
                counter += 1
                
            style.addSymbol(final_name, symbol)
            style.saveSymbol(final_name, symbol, True, [])
            
            return True
            
        except Exception as e:
            log_exception("Error saving symbol", e)
            return False
            
    def apply_to_layer(
        self,
        layer,
        symbol_image,
        size_mode=0,
        min_size=DEFAULT_MIN_SYMBOL_SIZE_MM,
        max_size=DEFAULT_MAX_SYMBOL_SIZE_MM,
        size_field=None,
        num_classes=DEFAULT_GRADUATED_CLASSES,
    ):
        """
        Apply a symbol to a vector layer.
        
        :param layer: QgsVectorLayer to apply symbol to
        :param symbol_image: QPixmap of the symbol
        :param size_mode: 0=fixed, 1=natural breaks, 2=equal interval, 3=quantile
        :param min_size: Minimum symbol size
        :param max_size: Maximum symbol size
        :param size_field: Field name for graduated sizing (optional)
        :param num_classes: Number of graduated classes (2..9 recommended)
        :return: True if successful
        """
        try:
            # Save to a deterministic per-layer cache path.
            # This avoids unbounded temp-file accumulation while preserving QGIS renderer references.
            temp_path = self._layer_symbol_cache_path(layer)
            symbol_image.save(temp_path, "PNG")
            
            if size_mode == 0:
                # Fixed size - single symbol renderer
                return self._apply_single_symbol(layer, temp_path, min_size)
            else:
                # Graduated size
                return self._apply_graduated_symbol(
                    layer, temp_path, size_mode, min_size, max_size, size_field, num_classes
                )
                
        except Exception as e:
            log_exception("Error applying symbol", e)
            return False
            
    def _apply_single_symbol(self, layer, image_path, size):
        """Apply a single symbol renderer."""
        symbol = QgsMarkerSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        
        raster_layer = QgsRasterMarkerSymbolLayer(image_path)
        raster_layer.setSize(float(size))
        raster_layer.setSizeUnit(QgsUnitTypes.RenderMillimeters)
        symbol.appendSymbolLayer(raster_layer)
        
        renderer = QgsSingleSymbolRenderer(symbol)
        layer.setRenderer(renderer)
        
        return True
        
    def _apply_graduated_symbol(
        self,
        layer,
        image_path,
        size_mode,
        min_size,
        max_size,
        size_field=None,
        num_classes=DEFAULT_GRADUATED_CLASSES,
    ):
        """Apply a graduated symbol renderer based on data count."""
        # If no field specified, try to use feature count or first numeric field
        if not size_field:
            # Find first numeric field
            fields = layer.fields()
            for field in fields:
                if field.isNumeric():
                    size_field = field.name()
                    break
                    
        if not size_field:
            # Fallback to single symbol if no numeric field
            return self._apply_single_symbol(layer, image_path, (min_size + max_size) / 2)
            
        # Create base symbol
        base_symbol = QgsMarkerSymbol.createSimple({})
        base_symbol.deleteSymbolLayer(0)
        
        raster_layer = QgsRasterMarkerSymbolLayer(image_path)
        raster_layer.setSizeUnit(QgsUnitTypes.RenderMillimeters)
        base_symbol.appendSymbolLayer(raster_layer)
        
        # Get field statistics
        idx = layer.fields().indexOf(size_field)
        values = self._extract_numeric_values(layer, idx)
        if not values:
            return self._apply_single_symbol(layer, image_path, (min_size + max_size) / 2)

        min_val = float(min(values))
        max_val = float(max(values))
        if min_val == max_val:
            return self._apply_single_symbol(layer, image_path, (min_size + max_size) / 2)

        # Create ranges based on selected size mode.
        class_count = int(num_classes) if num_classes is not None else DEFAULT_GRADUATED_CLASSES
        class_count = max(2, min(class_count, len(values)))
        breaks = self._compute_breaks(values, class_count, size_mode)
        if len(breaks) < 2:
            return self._apply_single_symbol(layer, image_path, (min_size + max_size) / 2)

        ranges = []
        break_count = len(breaks) - 1
        for i in range(break_count):
            lower = float(breaks[i])
            upper = float(breaks[i + 1])
            if upper <= lower:
                continue

            size = min_size + (max_size - min_size) * ((i + 0.5) / max(1.0, float(break_count)))

            range_symbol = base_symbol.clone()
            range_layer = range_symbol.symbolLayer(0)
            range_layer.setSize(float(size))
            range_layer.setSizeUnit(QgsUnitTypes.RenderMillimeters)

            label = f"{lower:.2f} - {upper:.2f}"
            ranges.append(QgsRendererRange(lower, upper, range_symbol, label))

        if not ranges:
            return self._apply_single_symbol(layer, image_path, (min_size + max_size) / 2)
            
        renderer = QgsGraduatedSymbolRenderer(size_field, ranges)
        layer.setRenderer(renderer)
        
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
                if not math.isfinite(numeric_value):
                    continue
                values.append(numeric_value)
            except Exception:
                continue
        return values

    def _compute_breaks(self, values, num_classes, size_mode):
        """Delegate to the QGIS-free implementation in symbol_breaks."""
        return compute_breaks(values, num_classes, size_mode)

    def get_saved_symbols(self):
        """Get list of saved symbols in the library."""
        symbols = []
        if os.path.exists(self.symbol_dir):
            for f in os.listdir(self.symbol_dir):
                if f.lower().endswith('.png'):
                    symbols.append({
                        'name': os.path.splitext(f)[0],
                        'path': os.path.join(self.symbol_dir, f)
                    })
        return symbols
