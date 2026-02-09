# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Template Generator
Generates symbols from built-in SVG templates.
"""

import os
import re
from qgis.PyQt.QtGui import QPixmap, QColor, QPainter
from qgis.PyQt.QtCore import Qt, QByteArray
from qgis.PyQt.QtSvg import QSvgRenderer


class TemplateGenerator:
    """Generator using built-in SVG templates."""
    
    # Template categories with default colors
    TEMPLATE_INFO = {
        "Pottery (토기류)": {
            "file": "pottery.svg",
            "default_color": "#8B4513"  # Saddle brown
        },
        "Stone Tools (석기류)": {
            "file": "stone_tool.svg",
            "default_color": "#708090"  # Slate gray
        },
        "Bronze Artifacts (청동기류)": {
            "file": "bronze.svg",
            "default_color": "#CD7F32"  # Bronze
        },
        "Iron Artifacts (철기류)": {
            "file": "iron.svg",
            "default_color": "#434343"  # Dark gray
        },
        "Ornaments (장신구류)": {
            "file": "ornament.svg",
            "default_color": "#FFD700"  # Gold
        }
    }
    
    def __init__(self, plugin_dir):
        """Initialize the template generator."""
        self.plugin_dir = plugin_dir
        self.template_dir = os.path.join(plugin_dir, 'resources', 'templates')
        
    def generate(self, template_type, color=None):
        """
        Generate a symbol from a template.
        
        :param template_type: Template type name
        :param color: Optional hex color for the symbol
        :return: QPixmap of generated symbol or None on failure
        """
        template_info = self.TEMPLATE_INFO.get(template_type)
        if not template_info:
            return None
            
        template_path = os.path.join(self.template_dir, template_info['file'])
        
        # If template file doesn't exist, create a placeholder
        if not os.path.exists(template_path):
            return self._create_placeholder(template_type, color or template_info['default_color'])
            
        # Load and colorize the SVG
        svg_data = self._load_and_colorize_svg(template_path, color or template_info['default_color'])
        
        if svg_data:
            return self._svg_to_pixmap(svg_data)
            
        return None
        
    def _load_and_colorize_svg(self, svg_path, color):
        """Load SVG file and replace colors."""
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
                
            # Replace fill colors with the new color
            # This is a simple approach - works for basic SVGs
            svg_content = re.sub(
                r'fill="[^"]*"',
                f'fill="{color}"',
                svg_content
            )
            
            return svg_content
        except Exception:
            return None
            
    def _svg_to_pixmap(self, svg_data, size=256):
        """Convert SVG data to QPixmap."""
        renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
        
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        
        return pixmap
        
    def _create_placeholder(self, template_type, color):
        """Create a placeholder symbol when template is not available."""
        # Create a simple placeholder based on template type
        size = 256
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set color
        q_color = QColor(color)
        painter.setBrush(q_color)
        painter.setPen(q_color.darker(120))
        
        # Draw different shapes based on type
        margin = 20
        
        if "Pottery" in template_type:
            # Draw a jar shape
            self._draw_pottery(painter, size, margin)
        elif "Stone" in template_type:
            # Draw a stone tool shape
            self._draw_stone_tool(painter, size, margin)
        elif "Bronze" in template_type:
            # Draw a bronze artifact shape (dagger/sword)
            self._draw_bronze(painter, size, margin)
        elif "Iron" in template_type:
            # Draw an iron artifact shape
            self._draw_iron(painter, size, margin)
        elif "Ornament" in template_type:
            # Draw an ornament shape
            self._draw_ornament(painter, size, margin)
        else:
            # Default circle
            painter.drawEllipse(margin, margin, size - 2*margin, size - 2*margin)
            
        painter.end()
        return pixmap
        
    def _draw_pottery(self, painter, size, margin):
        """Draw a simple pottery jar shape."""
        from qgis.PyQt.QtGui import QPainterPath
        
        path = QPainterPath()
        cx = size / 2
        
        # Jar outline
        path.moveTo(cx - 30, margin + 20)  # Left rim
        path.lineTo(cx + 30, margin + 20)  # Right rim
        path.lineTo(cx + 25, margin + 40)  # Right neck
        path.quadTo(cx + 60, size/2, cx + 50, size - margin)  # Right body
        path.lineTo(cx - 50, size - margin)  # Bottom
        path.quadTo(cx - 60, size/2, cx - 25, margin + 40)  # Left body
        path.lineTo(cx - 30, margin + 20)  # Back to rim
        
        painter.drawPath(path)
        
    def _draw_stone_tool(self, painter, size, margin):
        """Draw a simple stone tool shape."""
        from qgis.PyQt.QtGui import QPainterPath, QPolygonF
        from qgis.PyQt.QtCore import QPointF
        
        # Arrowhead/point shape
        points = [
            QPointF(size/2, margin),        # Top point
            QPointF(size - margin, size - margin - 40),  # Right
            QPointF(size/2, size - margin),  # Bottom middle
            QPointF(margin, size - margin - 40),  # Left
        ]
        
        painter.drawPolygon(QPolygonF(points))
        
    def _draw_bronze(self, painter, size, margin):
        """Draw a simple bronze dagger shape."""
        from qgis.PyQt.QtGui import QPainterPath
        
        path = QPainterPath()
        cx = size / 2
        
        # Dagger blade
        path.moveTo(cx, margin)  # Tip
        path.lineTo(cx + 25, size/2 - 30)  # Right blade
        path.lineTo(cx + 15, size/2)  # Right guard
        path.lineTo(cx + 10, size - margin)  # Right handle
        path.lineTo(cx - 10, size - margin)  # Left handle
        path.lineTo(cx - 15, size/2)  # Left guard
        path.lineTo(cx - 25, size/2 - 30)  # Left blade
        path.lineTo(cx, margin)  # Back to tip
        
        painter.drawPath(path)
        
    def _draw_iron(self, painter, size, margin):
        """Draw a simple iron tool shape (axe head)."""
        from qgis.PyQt.QtGui import QPainterPath
        
        path = QPainterPath()
        
        # Axe head shape
        path.moveTo(margin + 20, size/2 - 60)
        path.lineTo(size - margin, size/2 - 30)
        path.quadTo(size - margin + 10, size/2, size - margin, size/2 + 30)
        path.lineTo(margin + 20, size/2 + 60)
        path.lineTo(margin, size/2 + 40)
        path.lineTo(margin + 40, size/2)
        path.lineTo(margin, size/2 - 40)
        path.lineTo(margin + 20, size/2 - 60)
        
        painter.drawPath(path)
        
    def _draw_ornament(self, painter, size, margin):
        """Draw a simple ornament shape (bead/pendant)."""
        # Draw a circular pendant with hole
        outer_margin = margin + 20
        painter.drawEllipse(outer_margin, outer_margin, 
                          size - 2*outer_margin, size - 2*outer_margin)
        
        # Inner hole
        hole_size = 30
        painter.setBrush(Qt.white)
        painter.drawEllipse(
            int(size/2 - hole_size/2), 
            int(margin + 40), 
            hole_size, hole_size
        )
        
    def get_available_templates(self):
        """Return list of available template types."""
        return list(self.TEMPLATE_INFO.keys())
