# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Size Scaler
Provides proportional sizing based on data values.
"""

import math
from qgis.core import QgsVectorLayer


class SizeScaler:
    """Size scaling utilities for symbols."""
    
    # Classification methods
    METHOD_FIXED = 0
    METHOD_NATURAL_BREAKS = 1
    METHOD_EQUAL_INTERVAL = 2
    METHOD_QUANTILE = 3
    
    @staticmethod
    def calculate_size(value, min_val, max_val, min_size, max_size, method=0):
        """
        Calculate symbol size based on value and method.
        
        :param value: The data value
        :param min_val: Minimum value in dataset
        :param max_val: Maximum value in dataset
        :param min_size: Minimum symbol size
        :param max_size: Maximum symbol size
        :param method: Classification method
        :return: Calculated size
        """
        if min_val == max_val:
            return (min_size + max_size) / 2
            
        # Normalize value to 0-1 range
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))
        
        # Apply scaling
        if method == SizeScaler.METHOD_NATURAL_BREAKS:
            # Square root scaling for more natural perception
            normalized = math.sqrt(normalized)
        elif method == SizeScaler.METHOD_EQUAL_INTERVAL:
            # Linear scaling
            pass
        elif method == SizeScaler.METHOD_QUANTILE:
            # Logarithmic scaling
            if normalized > 0:
                normalized = math.log(normalized * 9 + 1) / math.log(10)
                
        # Map to size range
        return min_size + (max_size - min_size) * normalized
        
    @staticmethod
    def get_layer_value_range(layer, field_name):
        """
        Get min and max values from a layer field.
        
        :param layer: QgsVectorLayer
        :param field_name: Name of the numeric field
        :return: (min_value, max_value) tuple
        """
        if not isinstance(layer, QgsVectorLayer):
            return (0, 1)
            
        idx = layer.fields().indexOf(field_name)
        if idx < 0:
            return (0, 1)
            
        min_val = layer.minimumValue(idx)
        max_val = layer.maximumValue(idx)
        
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = 1
            
        return (min_val, max_val)
        
    @staticmethod
    def get_numeric_fields(layer):
        """
        Get list of numeric field names from a layer.
        
        :param layer: QgsVectorLayer
        :return: List of field names
        """
        if not isinstance(layer, QgsVectorLayer):
            return []
            
        numeric_fields = []
        for field in layer.fields():
            if field.isNumeric():
                numeric_fields.append(field.name())
                
        return numeric_fields
        
    @staticmethod
    def create_size_expression(field_name, min_val, max_val, min_size, max_size):
        """
        Create a QGIS expression for data-driven sizing.
        
        :param field_name: Name of the field to use
        :param min_val: Minimum value
        :param max_val: Maximum value
        :param min_size: Minimum size
        :param max_size: Maximum size
        :return: QGIS expression string
        """
        if min_val == max_val:
            return str((min_size + max_size) / 2)
            
        # Linear interpolation expression
        expression = (
            f"scale_linear(\"{field_name}\", "
            f"{min_val}, {max_val}, {min_size}, {max_size})"
        )
        
        return expression
