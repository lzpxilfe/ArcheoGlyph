# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Symbol Manager
Manages saving, loading, and applying symbols to QGIS layers.
"""

import os
import tempfile
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtCore import Qt, QSettings
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsSymbol, QgsMarkerSymbol,
    QgsRasterMarkerSymbolLayer, QgsSingleSymbolRenderer,
    QgsGraduatedSymbolRenderer, QgsRendererRange,
    QgsClassificationMethod, QgsApplication,
    QgsStyle
)


class SymbolManager:
    """Manager for symbol operations in QGIS."""
    
    def __init__(self):
        """Initialize the symbol manager."""
        self.settings = QSettings()
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
            raster_layer.setSize(10)  # Default size in mm
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
                       min_size=16, max_size=64, size_field=None):
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
            # Save temp image
            temp_path = os.path.join(tempfile.gettempdir(), 'archeoglyph_temp.png')
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
        raster_layer.setSize(size / 10.0)  # Convert to mm
        raster_layer.setSizeUnit(0)  # Millimeters
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
        base_symbol.appendSymbolLayer(raster_layer)
        
        # Get field statistics
        idx = layer.fields().indexOf(size_field)
        min_val = layer.minimumValue(idx)
        max_val = layer.maximumValue(idx)
        
        if min_val == max_val:
            return self._apply_single_symbol(layer, image_path, (min_size + max_size) / 2)
            
        # Create ranges
        num_classes = 5
        ranges = []
        
        for i in range(num_classes):
            lower = min_val + (max_val - min_val) * i / num_classes
            upper = min_val + (max_val - min_val) * (i + 1) / num_classes
            
            # Calculate size for this range
            size = min_size + (max_size - min_size) * (i + 0.5) / num_classes
            
            # Clone and modify symbol
            range_symbol = base_symbol.clone()
            range_symbol.symbolLayer(0).setSize(size / 10.0)
            
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
