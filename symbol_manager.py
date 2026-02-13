# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Symbol Manager
Manages saving, loading, and applying symbols to QGIS layers.
"""

import os
import tempfile
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
            os.path.dirname(os.path.dirname(__file__)),
            'symbols'
        )
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return base_dir
        
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
            print(f"Error saving symbol: {e}")
            return False
            
    def apply_to_layer(self, layer, symbol_image, size_mode=0, 
                       min_size=DEFAULT_MIN_SYMBOL_SIZE_MM, max_size=DEFAULT_MAX_SYMBOL_SIZE_MM, size_field=None):
        """
        Apply a symbol to a vector layer.
        
        :param layer: QgsVectorLayer to apply symbol to
        :param symbol_image: QPixmap of the symbol
        :param size_mode: 0=fixed, 1=natural breaks, 2=equal interval, 3=quantile
        :param min_size: Minimum symbol size
        :param max_size: Maximum symbol size
        :param size_field: Field name for graduated sizing (optional)
        :return: True if successful
        """
        try:
            # Save to a unique temp file to prevent overwriting previous symbols on other layers
            # We use delete=False so QGIS can access the file. 
            # Note: These files will persist in the temp dir until OS cleanup.
            timestamp = int(import_time.time() * 1000)
            temp_path = os.path.join(tempfile.gettempdir(), f'archeoglyph_symbol_{timestamp}_{os.urandom(4).hex()}.png')
            symbol_image.save(temp_path, "PNG")
            
            if size_mode == 0:
                # Fixed size - single symbol renderer
                return self._apply_single_symbol(layer, temp_path, min_size)
            else:
                # Graduated size
                return self._apply_graduated_symbol(
                    layer, temp_path, size_mode, min_size, max_size, size_field
                )
                
        except Exception as e:
            print(f"Error applying symbol: {e}")
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
        
    def _apply_graduated_symbol(self, layer, image_path, size_mode, 
                                min_size, max_size, size_field=None):
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
        min_val = layer.minimumValue(idx)
        max_val = layer.maximumValue(idx)
        
        if min_val == max_val:
            return self._apply_single_symbol(layer, image_path, (min_size + max_size) / 2)
            
        # Create ranges
        num_classes = DEFAULT_GRADUATED_CLASSES
        ranges = []
        
        for i in range(num_classes):
            lower = min_val + (max_val - min_val) * i / num_classes
            upper = min_val + (max_val - min_val) * (i + 1) / num_classes
            
            # Calculate size for this range
            size = min_size + (max_size - min_size) * (i + 0.5) / num_classes
            
            # Clone and modify symbol
            range_symbol = base_symbol.clone()
            range_layer = range_symbol.symbolLayer(0)
            range_layer.setSize(float(size))
            range_layer.setSizeUnit(QgsUnitTypes.RenderMillimeters)
            
            label = f"{lower:.1f} - {upper:.1f}"
            ranges.append(QgsRendererRange(lower, upper, range_symbol, label))
            
        renderer = QgsGraduatedSymbolRenderer(size_field, ranges)
        layer.setRenderer(renderer)
        
        return True
        
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
