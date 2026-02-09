# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Color Adjuster
Provides HSL-based color adjustment for symbols.
"""

from qgis.PyQt.QtGui import QPixmap, QImage, QColor, QPainter
from qgis.PyQt.QtCore import Qt


class ColorAdjuster:
    """Color adjustment utilities for symbols."""
    
    # Preset color palettes for archaeological artifacts
    PALETTES = {
        'earth': ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#D2691E'],
        'stone': ['#708090', '#778899', '#B0C4DE', '#A9A9A9', '#696969'],
        'metal': ['#CD7F32', '#FFD700', '#C0C0C0', '#B87333', '#4A4A4A'],
        'clay': ['#D2691E', '#8B4513', '#A52A2A', '#BC8F8F', '#F4A460'],
        'jade': ['#00A86B', '#50C878', '#3EB489', '#29AB87', '#4CBB17']
    }
    
    @staticmethod
    def adjust_hsl(pixmap, hue_shift=0, saturation_factor=1.0, lightness_factor=1.0):
        """
        Adjust the hue, saturation, and lightness of a pixmap.
        
        :param pixmap: Input QPixmap
        :param hue_shift: Hue shift in degrees (-180 to 180)
        :param saturation_factor: Saturation multiplier (0 to 2)
        :param lightness_factor: Lightness multiplier (0 to 2)
        :return: Adjusted QPixmap
        """
        image = pixmap.toImage()
        
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor(image.pixel(x, y))
                
                if color.alpha() > 0:
                    h, s, l, a = color.getHsl()
                    
                    # Adjust hue
                    h = (h + hue_shift) % 360
                    
                    # Adjust saturation
                    s = max(0, min(255, int(s * saturation_factor)))
                    
                    # Adjust lightness
                    l = max(0, min(255, int(l * lightness_factor)))
                    
                    color.setHsl(h, s, l, a)
                    image.setPixel(x, y, color.rgba())
                    
        return QPixmap.fromImage(image)
        
    @staticmethod
    def apply_color(pixmap, target_color):
        """
        Apply a target color to a pixmap, preserving luminance.
        
        :param pixmap: Input QPixmap
        :param target_color: Target QColor or hex string
        :return: Colorized QPixmap
        """
        if isinstance(target_color, str):
            target_color = QColor(target_color)
            
        image = pixmap.toImage()
        target_h, target_s, target_l, _ = target_color.getHsl()
        
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor(image.pixel(x, y))
                
                if color.alpha() > 0:
                    _, _, l, a = color.getHsl()
                    
                    # Apply target hue and saturation, keep original lightness
                    color.setHsl(target_h, target_s, l, a)
                    image.setPixel(x, y, color.rgba())
                    
        return QPixmap.fromImage(image)
        
    @staticmethod
    def set_transparency(pixmap, alpha):
        """
        Set overall transparency of a pixmap.
        
        :param pixmap: Input QPixmap
        :param alpha: Alpha value (0-255)
        :return: Adjusted QPixmap
        """
        image = pixmap.toImage()
        
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor(image.pixel(x, y))
                
                if color.alpha() > 0:
                    # Scale alpha relative to original
                    new_alpha = int(color.alpha() * alpha / 255)
                    color.setAlpha(new_alpha)
                    image.setPixel(x, y, color.rgba())
                    
        return QPixmap.fromImage(image)
        
    @classmethod
    def get_palette(cls, name):
        """Get a preset color palette."""
        return cls.PALETTES.get(name, cls.PALETTES['earth'])
