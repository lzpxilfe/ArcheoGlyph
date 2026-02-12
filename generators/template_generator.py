# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Template Generator
Generates symbols from built-in SVG templates with comprehensive archaeological categories.
"""

import os
import re
from qgis.PyQt.QtGui import QImage, QColor, QPainter, QPainterPath, QPolygonF, QPen
from qgis.PyQt.QtCore import Qt, QByteArray, QPointF, QRectF
from qgis.PyQt.QtSvg import QSvgRenderer


class TemplateGenerator:
    """Generator using built-in SVG templates."""
    
    # ── Comprehensive archaeological template categories ──
    TEMPLATE_INFO = {
        # ─── Artifacts (유물) ───
        "Pottery (토기류)": {
            "file": "pottery.svg",
            "default_color": "#8B4513",
            "category": "artifacts"
        },
        "Stone Tools (석기류)": {
            "file": "stone_tool.svg",
            "default_color": "#708090",
            "category": "artifacts"
        },
        "Bronze Artifacts (청동기류)": {
            "file": "bronze.svg",
            "default_color": "#CD7F32",
            "category": "artifacts"
        },
        "Iron Artifacts (철기류)": {
            "file": "iron.svg",
            "default_color": "#434343",
            "category": "artifacts"
        },
        "Ornaments (장신구류)": {
            "file": "ornament.svg",
            "default_color": "#FFD700",
            "category": "artifacts"
        },
        "Coins (화폐/주화)": {
            "file": "coin.svg",
            "default_color": "#DAA520",
            "category": "artifacts"
        },
        "Bone/Antler Tools (골각기류)": {
            "file": "bone_tool.svg",
            "default_color": "#F5DEB3",
            "category": "artifacts"
        },
        "Weapons (무기류)": {
            "file": "weapon.svg",
            "default_color": "#696969",
            "category": "artifacts"
        },

        # ─── Structures (유구/건축) ───
        "Fortress/Castle (성곽)": {
            "file": "fortress.svg",
            "default_color": "#8B7355",
            "category": "structures"
        },
        "Dwelling/House (주거지)": {
            "file": "dwelling.svg",
            "default_color": "#A0522D",
            "category": "structures"
        },
        "Tomb/Burial (고분/무덤)": {
            "file": "tomb.svg",
            "default_color": "#556B2F",
            "category": "structures"
        },
        "Temple/Shrine (사찰/신전)": {
            "file": "temple.svg",
            "default_color": "#B22222",
            "category": "structures"
        },
        "Kiln/Furnace (가마/요지)": {
            "file": "kiln.svg",
            "default_color": "#D2691E",
            "category": "structures"
        },
        "Well (우물)": {
            "file": "well.svg",
            "default_color": "#4682B4",
            "category": "structures"
        },
        "Wall/Rampart (담장/성벽)": {
            "file": "wall.svg",
            "default_color": "#808080",
            "category": "structures"
        },
        "Pit (수혈/구덩이)": {
            "file": "pit.svg",
            "default_color": "#6B4226",
            "category": "structures"
        },

        # ─── Human Remains (인골) ───
        "Human Remains (인골)": {
            "file": "skull.svg",
            "default_color": "#DEB887",
            "category": "remains"
        },
        "Burial (매장)": {
            "file": "burial.svg",
            "default_color": "#8B8378",
            "category": "remains"
        },

        # ─── Features (현상) ───
        "Hearth/Fire Pit (노지/화덕)": {
            "file": "hearth.svg",
            "default_color": "#FF4500",
            "category": "features"
        },
        "Midden/Shell Mound (패총)": {
            "file": "midden.svg",
            "default_color": "#BDB76B",
            "category": "features"
        },
        "Ditch/Moat (환호/도랑)": {
            "file": "ditch.svg",
            "default_color": "#2E8B57",
            "category": "features"
        },
        "Stone Alignment (열석/선돌)": {
            "file": "stone_align.svg",
            "default_color": "#778899",
            "category": "features"
        },
        "Dolmen (고인돌)": {
            "file": "dolmen.svg",
            "default_color": "#A9A9A9",
            "category": "features"
        },
        "Rock Art (암각화)": {
            "file": "rock_art.svg",
            "default_color": "#CD853F",
            "category": "features"
        },

        # ─── Survey / General (조사/일반) ───
        "Excavation Area (발굴구역)": {
            "file": "excavation.svg",
            "default_color": "#FF8C00",
            "category": "survey"
        },
        "Survey Point (조사지점)": {
            "file": "survey_point.svg",
            "default_color": "#4169E1",
            "category": "survey"
        },
        "Find Spot (유물산포지)": {
            "file": "find_spot.svg",
            "default_color": "#DC143C",
            "category": "survey"
        },
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
        :return: QImage of generated symbol or None on failure
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
            return self._svg_to_image(svg_data)
            
        return None
        
    def _load_and_colorize_svg(self, svg_path, color):
        """Load SVG file and replace colors using XML parsing."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Namespace handling (SVG usually has one)
            # We need to handle tags with and without namespaces generally
            # But specific fill/stroke attributes are usually direct
            
            def update_element_color(element, new_color):
                # Update fill
                if 'fill' in element.attrib and element.attrib['fill'] != 'none':
                    element.attrib['fill'] = new_color
                
                # Update stroke
                if 'stroke' in element.attrib and element.attrib['stroke'] != 'none':
                    element.attrib['stroke'] = new_color
                    
                # Handle style attribute (css-like)
                if 'style' in element.attrib:
                    style = element.attrib['style']
                    new_style = []
                    for part in style.split(';'):
                        if not part.strip(): continue
                        key, _, val = part.partition(':')
                        key = key.strip().lower()
                        if key == 'fill':
                             if val.strip() != 'none':
                                new_style.append(f"fill:{new_color}")
                             else:
                                new_style.append(part)
                        elif key == 'stroke':
                             if val.strip() != 'none':
                                new_style.append(f"stroke:{new_color}")
                             else:
                                new_style.append(part)
                        else:
                            new_style.append(part)
                    element.attrib['style'] = ';'.join(new_style)

            # Recursive update
            for elem in root.iter():
                update_element_color(elem, color)
                
            # Convert back to string
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            # Fallback to regex if XML parsing fails (for malformed SVGs)
            try:
                with open(svg_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return re.sub(r'fill="[^"]*"', f'fill="{color}"', content)
            except:
                return None
            
    def _svg_to_image(self, svg_data, size=256):
        """Convert SVG data to QImage."""
        renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
        
        image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        
        return image
        
    def _create_placeholder(self, template_type, color):
        """Create a placeholder symbol when template SVG is not available."""
        size = 256
        image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        q_color = QColor(color)
        painter.setBrush(q_color)
        painter.setPen(QPen(q_color.darker(130), 2.0))
        
        m = 25  # margin
        cx, cy = size / 2, size / 2

        # ── Dispatch drawing by keyword ──
        key = template_type.split("(")[0].strip().lower()
        
        if "pottery" in key:
            self._draw_pottery(painter, size, m)
        elif "stone tool" in key:
            self._draw_stone_tool(painter, size, m)
        elif "bronze" in key:
            self._draw_bronze(painter, size, m)
        elif "iron" in key:
            self._draw_iron(painter, size, m)
        elif "ornament" in key:
            self._draw_ornament(painter, size, m)
        elif "coin" in key:
            self._draw_coin(painter, size, m, q_color)
        elif "bone" in key:
            self._draw_bone_tool(painter, size, m)
        elif "weapon" in key:
            self._draw_weapon(painter, size, m)
        elif "fortress" in key or "castle" in key:
            self._draw_fortress(painter, size, m)
        elif "dwelling" in key or "house" in key:
            self._draw_dwelling(painter, size, m)
        elif "tomb" in key:
            self._draw_tomb(painter, size, m)
        elif "temple" in key or "shrine" in key:
            self._draw_temple(painter, size, m, q_color)
        elif "kiln" in key or "furnace" in key:
            self._draw_kiln(painter, size, m)
        elif "well" in key:
            self._draw_well(painter, size, m, q_color)
        elif "wall" in key or "rampart" in key:
            self._draw_wall(painter, size, m)
        elif "pit" in key:
            self._draw_pit(painter, size, m, q_color)
        elif "human" in key or "skull" in key:
            self._draw_skull(painter, size, m, q_color)
        elif "burial" in key:
            self._draw_burial(painter, size, m, q_color)
        elif "hearth" in key or "fire" in key:
            self._draw_hearth(painter, size, m, q_color)
        elif "midden" in key or "shell" in key:
            self._draw_midden(painter, size, m)
        elif "ditch" in key or "moat" in key:
            self._draw_ditch(painter, size, m, q_color)
        elif "stone align" in key:
            self._draw_stone_alignment(painter, size, m)
        elif "dolmen" in key:
            self._draw_dolmen(painter, size, m)
        elif "rock art" in key:
            self._draw_rock_art(painter, size, m, q_color)
        elif "excavation" in key:
            self._draw_excavation(painter, size, m, q_color)
        elif "survey" in key:
            self._draw_survey_point(painter, size, m, q_color)
        elif "find" in key:
            self._draw_find_spot(painter, size, m, q_color)
        else:
            painter.drawEllipse(m, m, size - 2*m, size - 2*m)
            
        painter.end()
        return image

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Artifacts
    # ═══════════════════════════════════════════════════════

    def _draw_pottery(self, painter, s, m):
        """Jar/vessel shape."""
        p = QPainterPath()
        cx = s / 2
        p.moveTo(cx - 30, m + 20)
        p.lineTo(cx + 30, m + 20)
        p.lineTo(cx + 25, m + 40)
        p.quadTo(cx + 65, s * 0.55, cx + 50, s - m)
        p.lineTo(cx - 50, s - m)
        p.quadTo(cx - 65, s * 0.55, cx - 25, m + 40)
        p.closeSubpath()
        painter.drawPath(p)
        
    def _draw_stone_tool(self, painter, s, m):
        """Arrowhead/point shape."""
        pts = [
            QPointF(s/2, m),
            QPointF(s - m, s - m - 40),
            QPointF(s/2, s - m),
            QPointF(m, s - m - 40),
        ]
        painter.drawPolygon(QPolygonF(pts))
        
    def _draw_bronze(self, painter, s, m):
        """Bronze dagger shape."""
        p = QPainterPath()
        cx = s / 2
        p.moveTo(cx, m)
        p.lineTo(cx + 25, s/2 - 30)
        p.lineTo(cx + 15, s/2)
        p.lineTo(cx + 10, s - m)
        p.lineTo(cx - 10, s - m)
        p.lineTo(cx - 15, s/2)
        p.lineTo(cx - 25, s/2 - 30)
        p.closeSubpath()
        painter.drawPath(p)
        
    def _draw_iron(self, painter, s, m):
        """Axe head shape."""
        p = QPainterPath()
        p.moveTo(m + 20, s/2 - 60)
        p.lineTo(s - m, s/2 - 30)
        p.quadTo(s - m + 10, s/2, s - m, s/2 + 30)
        p.lineTo(m + 20, s/2 + 60)
        p.lineTo(m, s/2 + 40)
        p.lineTo(m + 40, s/2)
        p.lineTo(m, s/2 - 40)
        p.closeSubpath()
        painter.drawPath(p)
        
    def _draw_ornament(self, painter, s, m):
        """Circular pendant with hole."""
        om = m + 20
        painter.drawEllipse(om, om, s - 2*om, s - 2*om)
        painter.setBrush(Qt.white)
        hs = 30
        painter.drawEllipse(int(s/2 - hs/2), int(m + 40), hs, hs)

    def _draw_coin(self, painter, s, m, color):
        """Coin — double circle with cross."""
        painter.drawEllipse(m + 10, m + 10, s - 2*m - 20, s - 2*m - 20)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(color.darker(150), 2.5))
        inner = 35
        painter.drawEllipse(m + inner, m + inner, s - 2*m - 2*inner, s - 2*m - 2*inner)
        cx, cy = s/2, s/2
        r = s/2 - m - inner
        painter.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))
        painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))

    def _draw_bone_tool(self, painter, s, m):
        """Bone/awl shape — elongated with rounded ends."""
        p = QPainterPath()
        cx = s / 2
        p.moveTo(cx, m)
        p.quadTo(cx + 12, m + 40, cx + 8, s * 0.4)
        p.quadTo(cx + 15, s * 0.7, cx + 6, s - m - 10)
        p.quadTo(cx, s - m + 5, cx - 6, s - m - 10)
        p.quadTo(cx - 15, s * 0.7, cx - 8, s * 0.4)
        p.quadTo(cx - 12, m + 40, cx, m)
        p.closeSubpath()
        painter.drawPath(p)

    def _draw_weapon(self, painter, s, m):
        """Spearhead shape."""
        p = QPainterPath()
        cx = s / 2
        p.moveTo(cx, m)
        p.quadTo(cx + 35, s * 0.35, cx + 20, s * 0.55)
        p.lineTo(cx + 8, s * 0.55)
        p.lineTo(cx + 8, s - m)
        p.lineTo(cx - 8, s - m)
        p.lineTo(cx - 8, s * 0.55)
        p.lineTo(cx - 20, s * 0.55)
        p.quadTo(cx - 35, s * 0.35, cx, m)
        p.closeSubpath()
        painter.drawPath(p)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Structures
    # ═══════════════════════════════════════════════════════

    def _draw_fortress(self, painter, s, m):
        """Castle/fortress — crenellated rectangle."""
        p = QPainterPath()
        bw = s - 2 * m  # base width
        bh = s - 2 * m  # base height
        cw = bw / 5     # crenel width
        ch = 25          # crenel height
        
        # Bottom-left, go clockwise
        p.moveTo(m, s - m)
        p.lineTo(m, m + ch)
        # Crenellations across the top
        for i in range(5):
            x = m + i * cw
            if i % 2 == 0:
                p.lineTo(x, m)
                p.lineTo(x + cw, m)
                p.lineTo(x + cw, m + ch)
            else:
                p.lineTo(x, m + ch)
                p.lineTo(x + cw, m + ch)
        p.lineTo(s - m, s - m)
        p.closeSubpath()
        painter.drawPath(p)
        # Gate
        painter.setBrush(Qt.white)
        gw, gh = 30, 45
        painter.drawRect(int(s/2 - gw/2), int(s - m - gh), gw, gh)

    def _draw_dwelling(self, painter, s, m):
        """House/dwelling — house shape with roof."""
        p = QPainterPath()
        cx = s / 2
        # Roof
        p.moveTo(cx, m)
        p.lineTo(s - m, s * 0.45)
        # Right wall
        p.lineTo(s - m - 15, s - m)
        # Bottom
        p.lineTo(m + 15, s - m)
        # Left wall
        p.lineTo(m, s * 0.45)
        p.closeSubpath()
        painter.drawPath(p)
        # Door
        painter.setBrush(Qt.white)
        dw, dh = 28, 40
        painter.drawRect(int(cx - dw/2), int(s - m - dh), dw, dh)

    def _draw_tomb(self, painter, s, m):
        """Burial mound — dome/tumulus shape."""
        p = QPainterPath()
        p.moveTo(m, s - m)
        p.quadTo(m, s * 0.3, s / 2, m + 10)
        p.quadTo(s - m, s * 0.3, s - m, s - m)
        p.closeSubpath()
        painter.drawPath(p)

    def _draw_temple(self, painter, s, m, color):
        """Temple — pagoda/traditional roof shape."""
        p = QPainterPath()
        cx = s / 2
        # Roof
        p.moveTo(cx, m)
        p.lineTo(s - m - 10, m + 60)
        p.lineTo(s - m - 30, m + 55)
        p.lineTo(s - m, m + 110)
        p.lineTo(s - m - 20, m + 105)
        # Right pillar
        p.lineTo(s - m - 30, s - m)
        # Base
        p.lineTo(m + 30, s - m)
        # Left pillar
        p.lineTo(m + 20, m + 105)
        p.lineTo(m, m + 110)
        p.lineTo(m + 30, m + 55)
        p.lineTo(m + 10, m + 60)
        p.closeSubpath()
        painter.drawPath(p)

    def _draw_kiln(self, painter, s, m):
        """Kiln — dome with opening."""
        p = QPainterPath()
        p.moveTo(m + 20, s - m)
        p.quadTo(m, s * 0.4, s / 2, m + 15)
        p.quadTo(s - m, s * 0.4, s - m - 20, s - m)
        p.closeSubpath()
        painter.drawPath(p)
        # Opening
        painter.setBrush(Qt.white)
        ow, oh = 35, 30
        painter.drawEllipse(int(s/2 - ow/2), int(s - m - oh - 5), ow, oh)

    def _draw_well(self, painter, s, m, color):
        """Well — circle with inner circle."""
        painter.drawEllipse(m + 15, m + 15, s - 2*m - 30, s - 2*m - 30)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
        inner = 50
        painter.drawEllipse(m + inner, m + inner, s - 2*m - 2*inner, s - 2*m - 2*inner)

    def _draw_wall(self, painter, s, m):
        """Wall segment — thick horizontal bar with stone texture hint."""
        wall_h = 60
        cy = s / 2
        painter.drawRect(m, int(cy - wall_h/2), s - 2*m, wall_h)
        # Stone lines
        painter.setBrush(Qt.NoBrush)
        pen = painter.pen()
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(m, int(cy), s - m, int(cy))
        step = (s - 2*m) // 4
        for i in range(1, 4):
            x = m + i * step
            painter.drawLine(x, int(cy - wall_h/2), x, int(cy))
            painter.drawLine(x + step//2, int(cy), x + step//2, int(cy + wall_h/2))

    def _draw_pit(self, painter, s, m, color):
        """Pit — dashed circle."""
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
        pen = QPen(color.darker(120), 2.5, Qt.DashLine)
        painter.setPen(pen)
        painter.drawEllipse(m + 20, m + 20, s - 2*m - 40, s - 2*m - 40)
        # Cross inside
        cx, cy = s/2, s/2
        r = 30
        painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
        painter.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Human Remains
    # ═══════════════════════════════════════════════════════

    def _draw_skull(self, painter, s, m, color):
        """Skull — cranium + jaw."""
        p = QPainterPath()
        cx = s / 2
        # Cranium
        p.addEllipse(QRectF(m + 30, m + 10, s - 2*m - 60, s * 0.55))
        painter.drawPath(p)
        # Jaw
        p2 = QPainterPath()
        p2.moveTo(cx - 35, s * 0.5)
        p2.quadTo(cx - 30, s * 0.75, cx, s - m - 20)
        p2.quadTo(cx + 30, s * 0.75, cx + 35, s * 0.5)
        painter.drawPath(p2)
        # Eyes
        painter.setBrush(Qt.white)
        ew, eh = 22, 20
        painter.drawEllipse(int(cx - 28), int(s * 0.32), ew, eh)
        painter.drawEllipse(int(cx + 6), int(s * 0.32), ew, eh)

    def _draw_burial(self, painter, s, m, color):
        """Burial — body outline (flexed position)."""
        painter.setBrush(Qt.NoBrush)
        pen = QPen(color, 3.0)
        painter.setPen(pen)
        # Head
        painter.drawEllipse(int(s * 0.35), m + 10, 35, 35)
        # Spine curve
        p = QPainterPath()
        p.moveTo(s * 0.52, m + 45)
        p.quadTo(s * 0.6, s * 0.4, s * 0.55, s * 0.6)
        p.quadTo(s * 0.45, s * 0.8, s * 0.35, s - m - 10)
        painter.drawPath(p)
        # Legs (flexed)
        p2 = QPainterPath()
        p2.moveTo(s * 0.35, s - m - 10)
        p2.quadTo(s * 0.55, s - m + 5, s * 0.65, s * 0.7)
        painter.drawPath(p2)
        # Arms
        p3 = QPainterPath()
        p3.moveTo(s * 0.55, s * 0.35)
        p3.quadTo(s * 0.35, s * 0.45, s * 0.38, s * 0.55)
        painter.drawPath(p3)
        painter.setBrush(color)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Features
    # ═══════════════════════════════════════════════════════

    def _draw_hearth(self, painter, s, m, color):
        """Hearth — flame inside circle."""
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 50))
        painter.drawEllipse(m + 20, m + 20, s - 2*m - 40, s - 2*m - 40)
        # Flame
        painter.setBrush(color)
        p = QPainterPath()
        cx = s / 2
        p.moveTo(cx, m + 40)
        p.quadTo(cx + 30, s * 0.4, cx + 15, s * 0.55)
        p.quadTo(cx + 25, s * 0.65, cx, s - m - 30)
        p.quadTo(cx - 25, s * 0.65, cx - 15, s * 0.55)
        p.quadTo(cx - 30, s * 0.4, cx, m + 40)
        painter.drawPath(p)

    def _draw_midden(self, painter, s, m):
        """Shell mound — layered mound."""
        # Bottom layer
        p1 = QPainterPath()
        p1.moveTo(m, s - m)
        p1.quadTo(s/2, s * 0.5, s - m, s - m)
        p1.closeSubpath()
        painter.drawPath(p1)
        # Top layer (lighter)
        old_brush = painter.brush()
        painter.setBrush(old_brush.color().lighter(130))
        p2 = QPainterPath()
        p2.moveTo(m + 30, s - m - 30)
        p2.quadTo(s/2, s * 0.35, s - m - 30, s - m - 30)
        p2.closeSubpath()
        painter.drawPath(p2)
        painter.setBrush(old_brush)

    def _draw_ditch(self, painter, s, m, color):
        """Ditch/moat — concentric dashed arcs."""
        painter.setBrush(Qt.NoBrush)
        pen = QPen(color, 3.0, Qt.DashLine)
        painter.setPen(pen)
        painter.drawArc(m + 20, m + 20, s - 2*m - 40, s - 2*m - 40, 30 * 16, 300 * 16)
        pen.setWidth(2)
        painter.setPen(pen)
        inner = 50
        painter.drawArc(m + inner, m + inner, s - 2*m - 2*inner, s - 2*m - 2*inner, 30 * 16, 300 * 16)
        painter.setBrush(color)

    def _draw_stone_alignment(self, painter, s, m):
        """Standing stones — row of vertical rectangles."""
        stones = 5
        gap = (s - 2 * m) / (stones * 2 - 1)
        sw = gap * 0.8
        for i in range(stones):
            x = m + i * gap * 2
            h = 50 + (i % 3) * 25
            y = s - m - h
            painter.drawRect(int(x), int(y), int(sw), int(h))

    def _draw_dolmen(self, painter, s, m):
        """Dolmen — capstone on two uprights."""
        # Two uprights
        uw, uh = 30, 90
        painter.drawRect(m + 30, int(s - m - uh), uw, uh)
        painter.drawRect(int(s - m - 30 - uw), int(s - m - uh), uw, uh)
        # Capstone
        p = QPainterPath()
        top_y = s - m - uh - 25
        p.moveTo(m + 10, s - m - uh + 5)
        p.lineTo(m + 40, top_y)
        p.lineTo(s - m - 40, top_y)
        p.lineTo(s - m - 10, s - m - uh + 5)
        p.closeSubpath()
        painter.drawPath(p)

    def _draw_rock_art(self, painter, s, m, color):
        """Rock art — spiral petroglyph."""
        painter.setBrush(Qt.NoBrush)
        pen = QPen(color, 3.0)
        painter.setPen(pen)
        cx, cy = s / 2, s / 2
        import math
        turns = 3.5
        points = 80
        for i in range(1, points):
            t0 = (i - 1) / points * turns * 2 * math.pi
            t1 = i / points * turns * 2 * math.pi
            r0 = 8 + (i - 1) / points * (s/2 - m - 15)
            r1 = 8 + i / points * (s/2 - m - 15)
            x0 = cx + r0 * math.cos(t0)
            y0 = cy + r0 * math.sin(t0)
            x1 = cx + r1 * math.cos(t1)
            y1 = cy + r1 * math.sin(t1)
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
        painter.setBrush(color)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Survey / General
    # ═══════════════════════════════════════════════════════

    def _draw_excavation(self, painter, s, m, color):
        """Excavation area — square with grid lines."""
        painter.drawRect(m + 15, m + 15, s - 2*m - 30, s - 2*m - 30)
        painter.setBrush(Qt.NoBrush)
        pen = QPen(color.darker(130), 1.5, Qt.DotLine)
        painter.setPen(pen)
        sz = s - 2*m - 30
        step = sz / 3
        for i in range(1, 3):
            y = m + 15 + i * step
            painter.drawLine(m + 15, int(y), s - m - 15, int(y))
            x = m + 15 + i * step
            painter.drawLine(int(x), m + 15, int(x), s - m - 15)
        painter.setBrush(color)

    def _draw_survey_point(self, painter, s, m, color):
        """Survey point — crosshair with circle."""
        cx, cy = s/2, s/2
        r = s/2 - m - 20
        painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(color.darker(130), 2.0))
        ext = 15
        painter.drawLine(int(cx), int(cy - r - ext), int(cx), int(cy + r + ext))
        painter.drawLine(int(cx - r - ext), int(cy), int(cx + r + ext), int(cy))
        # Center dot
        painter.setBrush(color)
        painter.drawEllipse(int(cx - 5), int(cy - 5), 10, 10)

    def _draw_find_spot(self, painter, s, m, color):
        """Find spot — location pin / drop marker."""
        p = QPainterPath()
        cx = s / 2
        p.moveTo(cx, s - m - 10)
        p.quadTo(cx - 55, s * 0.5, cx - 50, s * 0.35)
        p.quadTo(cx - 50, m + 10, cx, m + 5)
        p.quadTo(cx + 50, m + 10, cx + 50, s * 0.35)
        p.quadTo(cx + 55, s * 0.5, cx, s - m - 10)
        p.closeSubpath()
        painter.drawPath(p)
        # Inner circle
        painter.setBrush(Qt.white)
        ir = 20
        painter.drawEllipse(int(cx - ir), int(s * 0.28), ir * 2, ir * 2)

    def get_available_templates(self):
        """Return list of available template types."""
        return list(self.TEMPLATE_INFO.keys())
    
    def get_categories(self):
        """Return templates grouped by category."""
        categories = {}
        for name, info in self.TEMPLATE_INFO.items():
            cat = info.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories
