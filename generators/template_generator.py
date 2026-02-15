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
    
    # Comprehensive archaeological template catalog (English-first)
    TEMPLATE_INFO = {
        # Artifacts
        "Pottery": {
            "file": "pottery.svg",
            "default_color": "#8B4513",
            "category": "artifacts"
        },
        "Stone Tool": {
            "file": "stone_tool.svg",
            "default_color": "#708090",
            "category": "artifacts"
        },
        "Bronze Artifact": {
            "file": "bronze.svg",
            "default_color": "#CD7F32",
            "category": "artifacts"
        },
        "Iron Artifact": {
            "file": "iron.svg",
            "default_color": "#434343",
            "category": "artifacts"
        },
        "Ornament": {
            "file": "ornament.svg",
            "default_color": "#FFD700",
            "category": "artifacts"
        },
        "Coin": {
            "file": "coin.svg",
            "default_color": "#DAA520",
            "category": "artifacts"
        },
        "Bone Tool": {
            "file": "bone_tool.svg",
            "default_color": "#F5DEB3",
            "category": "artifacts"
        },
        "Weapon": {
            "file": "weapon.svg",
            "default_color": "#696969",
            "category": "artifacts"
        },
        "Arrowhead": {
            "file": "arrowhead.svg",
            "default_color": "#5F6A72",
            "category": "artifacts"
        },
        "Blade": {
            "file": "blade.svg",
            "default_color": "#50565D",
            "category": "artifacts"
        },
        "Scraper": {
            "file": "scraper.svg",
            "default_color": "#7A828B",
            "category": "artifacts"
        },
        "Needle / Pin": {
            "file": "needle_pin.svg",
            "default_color": "#8A7F73",
            "category": "artifacts"
        },
        "Bead": {
            "file": "bead.svg",
            "default_color": "#C68E3A",
            "category": "artifacts"
        },
        "Bracelet / Ring": {
            "file": "bracelet_ring.svg",
            "default_color": "#C9A227",
            "category": "artifacts"
        },
        "Seal / Stamp": {
            "file": "seal_stamp.svg",
            "default_color": "#8B5A2B",
            "category": "artifacts"
        },
        "Spindle Whorl": {
            "file": "spindle_whorl.svg",
            "default_color": "#7C5C46",
            "category": "artifacts"
        },
        "Chisel": {
            "file": "chisel.svg",
            "default_color": "#5B6168",
            "category": "artifacts"
        },
        "Bronze Dagger (Liaoning-style)": {
            "file": "bronze_dagger_liaoning.svg",
            "default_color": "#B66A62",
            "category": "artifacts"
        },
        "Bronze Dagger (Ordos-style)": {
            "file": "bronze_dagger_ordos.svg",
            "default_color": "#5E79B4",
            "category": "artifacts"
        },
        "Bronze Dagger (Antenna-style)": {
            "file": "bronze_dagger_antenna.svg",
            "default_color": "#4EA7A6",
            "category": "artifacts"
        },
        "Bronze Dagger (Slender)": {
            "file": "bronze_dagger_slender.svg",
            "default_color": "#B39A58",
            "category": "artifacts"
        },
        "Bronze Dagger (Tao type)": {
            "file": "bronze_dagger_tao.svg",
            "default_color": "#58A05A",
            "category": "artifacts"
        },
        "Bronze Dagger (Medium-fine)": {
            "file": "bronze_dagger_medium_fine.svg",
            "default_color": "#5E79B4",
            "category": "artifacts"
        },
        "Bronze Dagger (Flat bladed)": {
            "file": "bronze_dagger_flat_bladed.svg",
            "default_color": "#A06AC2",
            "category": "artifacts"
        },
        "Bronze Dagger (Type IA)": {
            "file": "bronze_dagger_type_ia.svg",
            "default_color": "#A3645C",
            "category": "artifacts"
        },
        "Bronze Dagger (Type IB)": {
            "file": "bronze_dagger_type_ib.svg",
            "default_color": "#8F7E5B",
            "category": "artifacts"
        },
        "Bronze Dagger (Other)": {
            "file": "bronze_dagger_other.svg",
            "default_color": "#8E8E8E",
            "category": "artifacts"
        },
        "Projectile Point (Leaf-shaped)": {
            "file": "projectile_point_leaf.svg",
            "default_color": "#6E8FA3",
            "category": "artifacts"
        },
        "Projectile Point (Side-notched)": {
            "file": "projectile_point_side_notched.svg",
            "default_color": "#5B7FA2",
            "category": "artifacts"
        },
        "Projectile Point (Corner-notched)": {
            "file": "projectile_point_corner_notched.svg",
            "default_color": "#5B76A2",
            "category": "artifacts"
        },
        "Projectile Point (Stemmed)": {
            "file": "projectile_point_stemmed.svg",
            "default_color": "#667F95",
            "category": "artifacts"
        },
        "Projectile Point (Triangular)": {
            "file": "projectile_point_triangular.svg",
            "default_color": "#7F8FA1",
            "category": "artifacts"
        },

        # Structures
        "Fortress / Castle": {
            "file": "fortress.svg",
            "default_color": "#8B7355",
            "category": "structures"
        },
        "Dwelling / House": {
            "file": "dwelling.svg",
            "default_color": "#A0522D",
            "category": "structures"
        },
        "Tomb": {
            "file": "tomb.svg",
            "default_color": "#556B2F",
            "category": "structures"
        },
        "Keyhole Tomb (Normal)": {
            "file": "keyhole_tomb_normal.svg",
            "default_color": "#A88A5F",
            "category": "structures"
        },
        "Keyhole Tomb (With Moat)": {
            "file": "keyhole_tomb_with_moat.svg",
            "default_color": "#A88A5F",
            "category": "structures"
        },
        "Keyhole Tomb (Stepped)": {
            "file": "keyhole_tomb_stepped.svg",
            "default_color": "#A88A5F",
            "category": "structures"
        },
        "Keyhole Tomb (With Fukiishi)": {
            "file": "keyhole_tomb_with_fukiishi.svg",
            "default_color": "#A88A5F",
            "category": "structures"
        },
        "Keyhole Tomb (Tsumishizuka)": {
            "file": "keyhole_tomb_tsumishizuka.svg",
            "default_color": "#A88A5F",
            "category": "structures"
        },
        "Keyhole Tomb (Makinokuchi)": {
            "file": "keyhole_tomb_makinokuchi.svg",
            "default_color": "#A88A5F",
            "category": "structures"
        },
        "Temple / Shrine": {
            "file": "temple.svg",
            "default_color": "#B22222",
            "category": "structures"
        },
        "Kiln / Furnace": {
            "file": "kiln.svg",
            "default_color": "#D2691E",
            "category": "structures"
        },
        "Well": {
            "file": "well.svg",
            "default_color": "#4682B4",
            "category": "structures"
        },
        "Wall / Rampart": {
            "file": "wall.svg",
            "default_color": "#808080",
            "category": "structures"
        },
        "Pit": {
            "file": "pit.svg",
            "default_color": "#6B4226",
            "category": "structures"
        },
        "Gate": {
            "file": "gate.svg",
            "default_color": "#8C6E4B",
            "category": "structures"
        },
        "Road / Pavement": {
            "file": "road_pavement.svg",
            "default_color": "#8A8A8A",
            "category": "structures"
        },
        "Bridge": {
            "file": "bridge.svg",
            "default_color": "#7A6C5D",
            "category": "structures"
        },
        "Storage Pit": {
            "file": "storage_pit.svg",
            "default_color": "#6F4F37",
            "category": "structures"
        },
        "Posthole": {
            "file": "posthole.svg",
            "default_color": "#5A4A3A",
            "category": "structures"
        },
        "Workshop": {
            "file": "workshop.svg",
            "default_color": "#8C5A3C",
            "category": "structures"
        },
        "Tower": {
            "file": "tower.svg",
            "default_color": "#707070",
            "category": "structures"
        },

        # Remains
        "Human Remains": {
            "file": "skull.svg",
            "default_color": "#DEB887",
            "category": "remains"
        },
        "Burial": {
            "file": "burial.svg",
            "default_color": "#8B8378",
            "category": "remains"
        },
        "Skeleton": {
            "file": "skeleton.svg",
            "default_color": "#C4A484",
            "category": "remains"
        },
        "Cremation Burial": {
            "file": "cremation_burial.svg",
            "default_color": "#A89F91",
            "category": "remains"
        },
        "Animal Remains": {
            "file": "animal_remains.svg",
            "default_color": "#BFA88D",
            "category": "remains"
        },

        # Features
        "Hearth / Fire Pit": {
            "file": "hearth.svg",
            "default_color": "#FF4500",
            "category": "features"
        },
        "Midden / Shell Mound": {
            "file": "midden.svg",
            "default_color": "#BDB76B",
            "category": "features"
        },
        "Ditch / Moat": {
            "file": "ditch.svg",
            "default_color": "#2E8B57",
            "category": "features"
        },
        "Stone Alignment": {
            "file": "stone_align.svg",
            "default_color": "#778899",
            "category": "features"
        },
        "Dolmen": {
            "file": "dolmen.svg",
            "default_color": "#A9A9A9",
            "category": "features"
        },
        "Rock Art": {
            "file": "rock_art.svg",
            "default_color": "#CD853F",
            "category": "features"
        },
        "Canal / Water Channel": {
            "file": "canal_water_channel.svg",
            "default_color": "#3B7EA1",
            "category": "features"
        },
        "Terrace": {
            "file": "terrace.svg",
            "default_color": "#8A7760",
            "category": "features"
        },
        "Ash Layer": {
            "file": "ash_layer.svg",
            "default_color": "#7D7D7D",
            "category": "features"
        },
        "Burnt Area": {
            "file": "burnt_area.svg",
            "default_color": "#6A4E42",
            "category": "features"
        },
        "Mound / Barrow": {
            "file": "mound_barrow.svg",
            "default_color": "#7A6A50",
            "category": "features"
        },
        "Standing Stone": {
            "file": "standing_stone.svg",
            "default_color": "#8A9096",
            "category": "features"
        },

        # Survey
        "Excavation Area": {
            "file": "excavation.svg",
            "default_color": "#FF8C00",
            "category": "survey"
        },
        "Survey Point": {
            "file": "survey_point.svg",
            "default_color": "#4169E1",
            "category": "survey"
        },
        "Find Spot": {
            "file": "find_spot.svg",
            "default_color": "#DC143C",
            "category": "survey"
        },
        "Trench": {
            "file": "trench.svg",
            "default_color": "#D97706",
            "category": "survey"
        },
        "Datum Point": {
            "file": "datum_point.svg",
            "default_color": "#1D4ED8",
            "category": "survey"
        },
        "Sample Location": {
            "file": "sample_location.svg",
            "default_color": "#BE123C",
            "category": "survey"
        },
        "Photo Point": {
            "file": "photo_point.svg",
            "default_color": "#7C3AED",
            "category": "survey"
        },
        "Grid Corner": {
            "file": "grid_corner.svg",
            "default_color": "#0F766E",
            "category": "survey"
        },
        "Test Pit": {
            "file": "test_pit.svg",
            "default_color": "#92400E",
            "category": "survey"
        },
    }

    # Backward compatibility for older naming variants
    LEGACY_TEMPLATE_ALIASES = {
        "Stone Tools": "Stone Tool",
        "Bronze Artifacts": "Bronze Artifact",
        "Iron Artifacts": "Iron Artifact",
        "Ornaments": "Ornament",
        "Coins": "Coin",
        "Bone/Antler Tools": "Bone Tool",
        "Weapons": "Weapon",
        "Fortress/Castle": "Fortress / Castle",
        "Dwelling/House": "Dwelling / House",
        "Tomb/Burial": "Tomb",
        "Temple/Shrine": "Temple / Shrine",
        "Kiln/Furnace": "Kiln / Furnace",
        "Wall/Rampart": "Wall / Rampart",
        "Hearth/Fire Pit": "Hearth / Fire Pit",
        "Midden/Shell Mound": "Midden / Shell Mound",
        "Ditch/Moat": "Ditch / Moat",
        "Liaoning-style bronze dagger": "Bronze Dagger (Liaoning-style)",
        "Ordos-style bronze dagger": "Bronze Dagger (Ordos-style)",
        "Antenna-style bronze dagger": "Bronze Dagger (Antenna-style)",
        "Slender bronze dagger": "Bronze Dagger (Slender)",
        "Tao Shi Jian sword": "Bronze Dagger (Tao type)",
        "Medium-fine bronze sword": "Bronze Dagger (Medium-fine)",
        "Flat bladed bronze sword": "Bronze Dagger (Flat bladed)",
        "Type IA bronze dagger": "Bronze Dagger (Type IA)",
        "Type IB bronze dagger": "Bronze Dagger (Type IB)",
        "Other bronze sword": "Bronze Dagger (Other)",
        "Leppy Hills point": "Projectile Point (Leaf-shaped)",
        "Pequop side-notched point": "Projectile Point (Side-notched)",
        "Dead Cedar point": "Projectile Point (Corner-notched)",
        "Elko-eared point": "Projectile Point (Stemmed)",
        "with Shugo": "Keyhole Tomb (With Moat)",
        "with moat": "Keyhole Tomb (With Moat)",
        "with Fukiishi": "Keyhole Tomb (With Fukiishi)",
        "Tsumishizuka": "Keyhole Tomb (Tsumishizuka)",
        "Makinokuchi": "Keyhole Tomb (Makinokuchi)",
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
        template_type = self._normalize_template_type(template_type)
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

    def _normalize_template_type(self, template_type):
        """Normalize template names for backward compatibility."""
        key = str(template_type or "").strip()
        if not key:
            return ""
        if key in self.TEMPLATE_INFO:
            return key
        return self.LEGACY_TEMPLATE_ALIASES.get(key, key)
        
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
        except Exception:
            # Fallback to regex if XML parsing fails (for malformed SVGs)
            try:
                with open(svg_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return re.sub(r'fill="[^"]*"', f'fill="{color}"', content)
            except Exception:
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

        # Dispatch drawing by keyword
        key = str(template_type or "").strip().lower().replace("/", " ")
        
        if "keyhole tomb" in key or "shugo" in key or "fukiishi" in key or "tsumishizuka" in key:
            if "moat" in key or "shugo" in key:
                self._draw_keyhole_tomb(painter, size, m, "moat", q_color)
            elif "fukiishi" in key:
                self._draw_keyhole_tomb(painter, size, m, "fukiishi", q_color)
            elif "tsumishizuka" in key:
                self._draw_keyhole_tomb(painter, size, m, "tsumishizuka", q_color)
            elif "makinokuchi" in key:
                self._draw_keyhole_tomb(painter, size, m, "makinokuchi", q_color)
            elif "stepped" in key:
                self._draw_keyhole_tomb(painter, size, m, "stepped", q_color)
            else:
                self._draw_keyhole_tomb(painter, size, m, "normal", q_color)
        elif "bronze dagger" in key or "bronze sword" in key:
            if "liaoning" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "liaoning", q_color)
            elif "ordos" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "ordos", q_color)
            elif "antenna" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "antenna", q_color)
            elif "slender" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "slender", q_color)
            elif "tao" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "tao", q_color)
            elif "type ia" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "type_ia", q_color)
            elif "type ib" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "type_ib", q_color)
            elif "medium" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "medium", q_color)
            elif "flat" in key:
                self._draw_bronze_dagger_typology(painter, size, m, "flat", q_color)
            else:
                self._draw_bronze_dagger_typology(painter, size, m, "other", q_color)
        elif "projectile point" in key or "side-notched" in key or "corner-notched" in key:
            if "leaf" in key:
                self._draw_projectile_point_typology(painter, size, m, "leaf")
            elif "side" in key:
                self._draw_projectile_point_typology(painter, size, m, "side_notched")
            elif "corner" in key or "dead cedar" in key:
                self._draw_projectile_point_typology(painter, size, m, "corner_notched")
            elif "stemmed" in key or "elko" in key:
                self._draw_projectile_point_typology(painter, size, m, "stemmed")
            elif "triangular" in key:
                self._draw_projectile_point_typology(painter, size, m, "triangular")
            else:
                self._draw_projectile_point_typology(painter, size, m, "leaf")
        elif "pottery" in key:
            self._draw_pottery(painter, size, m)
        elif "stone tool" in key or "arrowhead" in key or "scraper" in key:
            self._draw_stone_tool(painter, size, m)
        elif "bronze" in key:
            self._draw_bronze(painter, size, m)
        elif "iron" in key or "chisel" in key:
            self._draw_iron(painter, size, m)
        elif "ornament" in key or "bead" in key or "bracelet" in key or "ring" in key:
            self._draw_ornament(painter, size, m)
        elif "coin" in key or "seal" in key or "stamp" in key or "spindle" in key:
            self._draw_coin(painter, size, m, q_color)
        elif "bone" in key or "needle" in key or "pin" in key or "animal remains" in key:
            self._draw_bone_tool(painter, size, m)
        elif "weapon" in key or "blade" in key or "arrow shaft" in key:
            self._draw_weapon(painter, size, m)
        elif "fortress" in key or "castle" in key or "gate" in key or "tower" in key:
            if "gate" in key:
                self._draw_gate(painter, size, m)
            elif "tower" in key:
                self._draw_tower(painter, size, m)
            else:
                self._draw_fortress(painter, size, m)
        elif "dwelling" in key or "house" in key or "workshop" in key:
            if "workshop" in key:
                self._draw_workshop(painter, size, m)
            else:
                self._draw_dwelling(painter, size, m)
        elif "road" in key or "pavement" in key:
            self._draw_road(painter, size, m, q_color)
        elif "bridge" in key:
            self._draw_bridge(painter, size, m, q_color)
        elif "terrace" in key:
            self._draw_terrace(painter, size, m, q_color)
        elif "wall" in key or "rampart" in key:
            self._draw_wall(painter, size, m)
        elif "posthole" in key:
            self._draw_posthole(painter, size, m, q_color)
        elif "test pit" in key:
            self._draw_test_pit(painter, size, m, q_color)
        elif "pit" in key:
            self._draw_pit(painter, size, m, q_color)
        elif "ash layer" in key:
            self._draw_ash_layer(painter, size, m, q_color)
        elif "burnt" in key:
            self._draw_burnt_area(painter, size, m, q_color)
        elif "canal" in key or "water channel" in key:
            self._draw_canal(painter, size, m, q_color)
        elif "ditch" in key or "moat" in key:
            self._draw_ditch(painter, size, m, q_color)
        elif "standing stone" in key:
            self._draw_standing_stone(painter, size, m, q_color)
        elif "stone align" in key:
            self._draw_stone_alignment(painter, size, m)
        elif "trench" in key:
            self._draw_trench(painter, size, m, q_color)
        elif "grid corner" in key:
            self._draw_grid_corner(painter, size, m, q_color)
        elif "excavation" in key:
            self._draw_excavation(painter, size, m, q_color)
        elif "datum" in key:
            self._draw_datum_point(painter, size, m, q_color)
        elif "photo point" in key:
            self._draw_photo_point(painter, size, m, q_color)
        elif "survey" in key:
            self._draw_survey_point(painter, size, m, q_color)
        elif "sample location" in key:
            self._draw_sample_location(painter, size, m, q_color)
        elif "find" in key:
            self._draw_find_spot(painter, size, m, q_color)
        elif "tomb" in key or "barrow" in key or ("mound" in key and "shell" not in key and "midden" not in key):
            self._draw_tomb(painter, size, m)
        elif "temple" in key or "shrine" in key:
            self._draw_temple(painter, size, m, q_color)
        elif "kiln" in key or "furnace" in key:
            self._draw_kiln(painter, size, m)
        elif "well" in key:
            self._draw_well(painter, size, m, q_color)
        elif "human" in key or "skull" in key or "skeleton" in key:
            self._draw_skull(painter, size, m, q_color)
        elif "burial" in key or "cremation" in key:
            self._draw_burial(painter, size, m, q_color)
        elif "hearth" in key or "fire" in key:
            self._draw_hearth(painter, size, m, q_color)
        elif "midden" in key or "shell" in key:
            self._draw_midden(painter, size, m)
        elif "dolmen" in key:
            self._draw_dolmen(painter, size, m)
        elif "rock art" in key:
            self._draw_rock_art(painter, size, m, q_color)
        else:
            painter.drawEllipse(m, m, size - 2*m, size - 2*m)
            
        painter.end()
        return image

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Artifacts
    # ═══════════════════════════════════════════════════════

    def _draw_pottery(self, painter, s, m):
        """Vessel profile with section-style interior cues."""
        p = QPainterPath()
        cx = s / 2
        p.moveTo(cx - 24, m + 22)
        p.lineTo(cx + 24, m + 22)
        p.quadTo(cx + 28, m + 36, cx + 24, m + 44)
        p.quadTo(cx + 68, s * 0.56, cx + 52, s - m)
        p.lineTo(cx - 52, s - m)
        p.quadTo(cx - 68, s * 0.56, cx - 24, m + 44)
        p.quadTo(cx - 28, m + 36, cx - 24, m + 22)
        p.closeSubpath()
        painter.drawPath(p)

        old_pen = painter.pen()
        old_brush = painter.brush()
        line_pen = QPen(old_pen.color().darker(140), 1.1)
        painter.setPen(line_pen)
        painter.setBrush(Qt.NoBrush)

        # Split-profile convention used in ceramic illustration.
        painter.drawLine(int(cx), int(m + 24), int(cx), int(s - m - 2))
        painter.drawLine(int(cx - 24), int(m + 30), int(cx + 24), int(m + 30))
        painter.drawLine(int(cx - 44), int(s - m - 8), int(cx + 44), int(s - m - 8))
        for i in range(6):
            y = int(m + 50 + (i * 24))
            painter.drawLine(int(cx - 46 + (i % 2) * 4), y, int(cx - 12), y + 8)

        painter.setPen(old_pen)
        painter.setBrush(old_brush)
        
    def _draw_stone_tool(self, painter, s, m):
        """Arrowhead/point with flake-scar style internal lines."""
        pts = [
            QPointF(s/2, m),
            QPointF(s - m, s - m - 40),
            QPointF(s/2, s - m),
            QPointF(m, s - m - 40),
        ]
        painter.drawPolygon(QPolygonF(pts))

        cx = s / 2.0
        old_pen = painter.pen()
        scar_pen = QPen(old_pen.color().darker(145), 1.0)
        painter.setPen(scar_pen)
        painter.drawLine(int(cx), int(m + 14), int(cx), int(s - m - 12))
        for i in range(4):
            y = int(m + 38 + i * 34)
            offset = 12 + i * 2
            painter.drawLine(int(cx - offset), y, int(cx - 4), y + 10)
            painter.drawLine(int(cx + offset), y, int(cx + 4), y + 10)
        painter.setPen(old_pen)
        
    def _draw_bronze(self, painter, s, m):
        """Default bronze icon: medium typological dagger silhouette."""
        color = painter.brush().color() if painter.brush().style() != Qt.NoBrush else QColor("#8C8C8C")
        self._draw_bronze_dagger_typology(painter, s, m, "medium", color)
        
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
        old_pen = painter.pen()
        ridge_pen = QPen(old_pen.color().darker(130), 1.25)
        painter.setPen(ridge_pen)
        painter.drawLine(int(cx), int(m + 14), int(cx), int(s - m - 8))
        painter.setPen(old_pen)

    def _draw_bronze_dagger_typology(self, painter, s, m, variant, color):
        """Typological bronze dagger variants inspired by catalog symbol conventions."""
        cx = s / 2.0
        top = float(m + 8)
        bottom = float(s - m - 10)
        height = max(40.0, bottom - top)

        profiles = {
            "liaoning": [0, 10, 24, 17, 23, 11, 2],
            "ordos": [0, 8, 18, 12, 16, 9, 2],
            "antenna": [0, 8, 18, 11, 15, 9, 2],
            "slender": [0, 6, 11, 9, 10, 6, 1],
            "tao": [0, 7, 12, 10, 8, 5, 1],
            "medium": [0, 8, 16, 10, 12, 7, 2],
            "flat": [0, 12, 18, 18, 16, 8, 2],
            "type_ia": [0, 9, 21, 16, 20, 9, 2],
            "type_ib": [0, 8, 18, 14, 20, 11, 2],
            "other": [0, 7, 14, 10, 11, 6, 1],
        }
        t_values = [0.00, 0.12, 0.30, 0.54, 0.74, 0.90, 1.00]
        widths = profiles.get(variant, profiles["other"])

        right = []
        left = []
        for t, w in zip(t_values, widths):
            y = top + (height * float(t))
            right.append(QPointF(cx + float(w), y))
            left.append(QPointF(cx - float(w), y))

        polygon_points = right + list(reversed(left))
        painter.drawPolygon(QPolygonF(polygon_points))

        old_pen = painter.pen()
        ridge_pen = QPen(old_pen.color().darker(135), 1.20)
        painter.setPen(ridge_pen)
        painter.drawLine(int(cx), int(top + (height * 0.08)), int(cx), int(bottom + 8))

        if variant == "flat":
            shoulder_y = int(top + (height * 0.30))
            painter.drawLine(int(cx - 18), shoulder_y, int(cx + 18), shoulder_y)
        elif variant == "antenna":
            antenna_y = int(top + (height * 0.78))
            painter.drawLine(int(cx - 24), antenna_y, int(cx - 10), antenna_y)
            painter.drawLine(int(cx + 10), antenna_y, int(cx + 24), antenna_y)
            painter.setBrush(color)
            painter.drawEllipse(int(cx - 27), antenna_y - 3, 6, 6)
            painter.drawEllipse(int(cx + 21), antenna_y - 3, 6, 6)
        elif variant == "liaoning":
            ring_y = int(top + (height * 0.67))
            painter.drawLine(int(cx - 14), ring_y, int(cx + 14), ring_y)
        elif variant == "type_ia":
            ring_y = int(top + (height * 0.62))
            painter.drawLine(int(cx - 16), ring_y, int(cx + 16), ring_y)
        elif variant == "type_ib":
            band_y = int(top + (height * 0.58))
            painter.drawLine(int(cx - 13), band_y, int(cx + 13), band_y)
            painter.drawLine(int(cx - 15), band_y + 8, int(cx + 15), band_y + 8)

        painter.setPen(old_pen)

    def _draw_projectile_point_typology(self, painter, s, m, variant):
        """Projectile point variants inspired by typology catalog symbols."""
        cx = s / 2.0
        top = float(m + 12)
        bottom = float(s - m - 8)
        mid = (top + bottom) / 2.0

        shapes = {
            "leaf": [(-2, top), (22, mid - 30), (28, mid), (14, bottom - 14), (4, bottom), (0, bottom + 2)],
            "side_notched": [(-2, top), (20, mid - 34), (26, mid - 8), (17, mid + 6), (11, bottom - 18), (6, bottom - 6), (0, bottom + 2)],
            "corner_notched": [(-2, top), (18, mid - 34), (24, mid - 10), (18, mid + 8), (8, bottom - 26), (8, bottom - 8), (0, bottom + 2)],
            "stemmed": [(-2, top), (20, mid - 28), (22, mid + 6), (13, bottom - 24), (7, bottom - 18), (7, bottom - 6), (0, bottom + 2)],
            "triangular": [(-2, top), (24, mid - 20), (20, bottom - 18), (10, bottom - 10), (6, bottom - 4), (0, bottom + 2)],
        }
        right = shapes.get(variant, shapes["leaf"])
        points = []
        for x_off, y in right:
            points.append(QPointF(cx + float(x_off), float(y)))
        for x_off, y in reversed(right):
            points.append(QPointF(cx - float(x_off), float(y)))
        painter.drawPolygon(QPolygonF(points))

        # Midrib line for legibility in typology-like symbols.
        old_pen = painter.pen()
        painter.setPen(QPen(old_pen.color().darker(135), 1.1))
        painter.drawLine(int(cx), int(top + 6), int(cx), int(bottom - 6))
        painter.setPen(old_pen)

    def _draw_keyhole_tomb(self, painter, s, m, variant, color):
        """Keyhole-shaped tomb variants (normal / moat / stepped)."""
        cx = s / 2.0
        circle_y = float(m + 54)
        circle_r = 34.0
        join_y = circle_y + circle_r - 4.0
        tail_bottom = float(s - m - 8)

        tail_top_half = 20.0
        tail_bottom_half = 38.0
        if variant in ("stepped", "fukiishi", "tsumishizuka"):
            tail_top_half = 18.0
            tail_bottom_half = 32.0

        mound_path = QPainterPath()
        mound_path.addEllipse(QRectF(cx - circle_r, circle_y - circle_r, circle_r * 2.0, circle_r * 2.0))

        tail_path = QPainterPath()
        tail_path.moveTo(cx - tail_top_half, join_y)
        tail_path.lineTo(cx - tail_bottom_half, tail_bottom)
        tail_path.lineTo(cx + tail_bottom_half, tail_bottom)
        tail_path.lineTo(cx + tail_top_half, join_y)
        tail_path.closeSubpath()

        composite = QPainterPath(mound_path)
        composite.addPath(tail_path)

        if variant in ("moat", "makinokuchi"):
            old_brush = painter.brush()
            old_pen = painter.pen()
            moat_width = 8.0 if variant == "moat" else 5.0
            moat_pen = QPen(QColor(106, 143, 168), moat_width)
            painter.setPen(moat_pen)
            painter.setBrush(Qt.NoBrush)

            moat_path = QPainterPath()
            moat_path.addEllipse(QRectF(cx - (circle_r + 11.0), circle_y - (circle_r + 11.0), (circle_r + 11.0) * 2.0, (circle_r + 11.0) * 2.0))
            moat_tail = QPainterPath()
            moat_tail.moveTo(cx - (tail_top_half + 9.0), join_y + 1.0)
            moat_tail.lineTo(cx - (tail_bottom_half + 11.0), tail_bottom + 8.0)
            moat_tail.lineTo(cx + (tail_bottom_half + 11.0), tail_bottom + 8.0)
            moat_tail.lineTo(cx + (tail_top_half + 9.0), join_y + 1.0)
            moat_tail.closeSubpath()
            moat_path.addPath(moat_tail)
            painter.drawPath(moat_path)

            if variant == "makinokuchi":
                painter.setPen(QPen(old_pen.color().darker(125), 1.0))
                painter.drawLine(int(cx - 26), int(join_y + 12), int(cx + 26), int(join_y + 12))
                painter.drawLine(int(cx - 30), int(join_y + 24), int(cx + 30), int(join_y + 24))

            painter.setBrush(old_brush)
            painter.setPen(old_pen)

        painter.drawPath(composite)

        if variant in ("stepped", "fukiishi", "tsumishizuka"):
            old_pen = painter.pen()
            step_pen = QPen(old_pen.color().darker(130), 1.1)
            painter.setPen(step_pen)
            for i in range(3):
                y = int(join_y + 14 + (i * 16))
                width = int((tail_top_half + 6) + (i * 7))
                painter.drawLine(int(cx - width), y, int(cx + width), y)
            if variant == "tsumishizuka":
                for i in range(10):
                    y = int(join_y + 18 + (i * 7))
                    x1 = int(cx - 20 - (i % 3))
                    x2 = int(cx + 20 + (i % 3))
                    painter.drawLine(x1, y, x1 + 6, y + 3)
                    painter.drawLine(x2, y, x2 - 6, y + 3)
            painter.setPen(old_pen)

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
        old_pen = painter.pen()
        hatch_pen = QPen(old_pen.color().darker(140), 1.0)
        painter.setPen(hatch_pen)
        span = float(s - (2 * m) - 36)
        for i in range(8):
            x = int(m + 18 + ((span / 7.0) * i))
            y = int((s - m - 26) - (18 - abs(3.5 - i) * 3.5))
            painter.drawLine(x, y, x - 7, y + 11)
        painter.setPen(old_pen)

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

    def _draw_gate(self, painter, s, m):
        """Gate icon with twin posts and lintel."""
        old_brush = painter.brush()
        old_pen = painter.pen()
        post_w = 30
        top_y = m + 40
        bottom_y = s - m
        painter.drawRect(m + 24, top_y, post_w, bottom_y - top_y)
        painter.drawRect(s - m - 24 - post_w, top_y, post_w, bottom_y - top_y)
        painter.drawRect(m + 16, m + 16, s - 2 * m - 32, 24)
        painter.setBrush(Qt.NoBrush)
        arch_pen = QPen(old_pen.color().darker(130), 1.4)
        painter.setPen(arch_pen)
        painter.drawArc(m + 40, top_y + 10, s - 2 * m - 80, 70, 0, 180 * 16)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_tower(self, painter, s, m):
        """Tower icon with crenellation and slit windows."""
        old_brush = painter.brush()
        old_pen = painter.pen()
        x = int(s / 2 - 40)
        y = m + 20
        w = 80
        h = s - 2 * m - 20
        painter.drawRect(x, y, w, h)
        crenel_w = 16
        for i in range(5):
            if i % 2 == 0:
                painter.drawRect(x + i * crenel_w, y - 14, crenel_w, 14)
        painter.setBrush(Qt.white)
        painter.drawRect(x + 32, y + 32, 16, 18)
        painter.drawRect(x + 32, y + 66, 16, 18)
        painter.drawRect(x + 30, y + h - 42, 20, 28)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_workshop(self, painter, s, m):
        """Workshop icon: dwelling body + crossed tool cue."""
        self._draw_dwelling(painter, s, m)
        old_pen = painter.pen()
        tool_pen = QPen(old_pen.color().darker(145), 1.8)
        painter.setPen(tool_pen)
        cx = s / 2
        y = int(s * 0.6)
        painter.drawLine(int(cx - 34), y - 8, int(cx + 18), y + 20)
        painter.drawLine(int(cx + 34), y - 8, int(cx - 18), y + 20)
        painter.drawRect(int(cx + 14), y + 16, 12, 6)
        painter.setPen(old_pen)

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
        old_pen = painter.pen()
        stipple_pen = QPen(old_pen.color().darker(135), 1.0)
        painter.setPen(stipple_pen)
        for i in range(14):
            x = int(m + 20 + (i * 14))
            y = int(s - m - 16 - ((i % 3) * 9))
            painter.drawEllipse(x, y, 4, 3)
        painter.setPen(old_pen)

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

    def _draw_canal(self, painter, s, m, color):
        """Canal/water-channel with paired lines and flow arrows."""
        old_brush = painter.brush()
        old_pen = painter.pen()
        painter.setBrush(Qt.NoBrush)
        flow_pen = QPen(color.darker(120), 2.4)
        painter.setPen(flow_pen)
        painter.drawArc(m + 18, m + 30, s - 2 * m - 36, s - 2 * m - 60, 40 * 16, 270 * 16)
        painter.drawArc(m + 36, m + 48, s - 2 * m - 72, s - 2 * m - 96, 40 * 16, 270 * 16)
        for i in range(3):
            x = int(m + 76 + i * 44)
            y = int(s / 2 + (i % 2) * 8)
            painter.drawLine(x, y, x + 12, y)
            painter.drawLine(x + 12, y, x + 7, y - 4)
            painter.drawLine(x + 12, y, x + 7, y + 4)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

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

    def _draw_standing_stone(self, painter, s, m, color):
        """Single monolith with pecked face marks."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        p = QPainterPath()
        p.moveTo(s * 0.42, s - m)
        p.quadTo(s * 0.32, s * 0.62, s * 0.36, s * 0.34)
        p.quadTo(s * 0.41, m + 6, s * 0.50, m + 14)
        p.quadTo(s * 0.62, m + 22, s * 0.64, s * 0.42)
        p.quadTo(s * 0.66, s * 0.66, s * 0.58, s - m)
        p.closeSubpath()
        painter.drawPath(p)
        painter.setBrush(Qt.NoBrush)
        peck_pen = QPen(color.darker(145), 1.0)
        painter.setPen(peck_pen)
        for i in range(5):
            y = int(m + 42 + i * 28)
            painter.drawLine(int(s * 0.46), y, int(s * 0.54), y + 5)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

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

    def _draw_ash_layer(self, painter, s, m, color):
        """Ash layer as horizontal banding with dense stipple."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        band_h = int((s - 2 * m) * 0.55)
        top = int(s / 2 - band_h / 2)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 85))
        painter.drawRect(m + 14, top, s - 2 * m - 28, band_h)
        painter.setBrush(Qt.NoBrush)
        stipple_pen = QPen(color.darker(150), 1.0)
        painter.setPen(stipple_pen)
        for i in range(9):
            y = top + 12 + i * 12
            painter.drawLine(m + 20, y, s - m - 20, y)
        for i in range(28):
            x = int(m + 24 + (i * 7))
            y = int(top + 8 + ((i * 11) % max(12, band_h - 12)))
            painter.drawPoint(x, y)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_burnt_area(self, painter, s, m, color):
        """Burnt feature with charred irregular boundary."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        p = QPainterPath()
        p.moveTo(m + 30, s * 0.72)
        p.quadTo(s * 0.28, s * 0.36, s * 0.46, m + 24)
        p.quadTo(s * 0.70, m + 34, s - m - 20, s * 0.54)
        p.quadTo(s * 0.72, s * 0.78, s * 0.52, s - m - 12)
        p.quadTo(s * 0.34, s - m - 4, m + 30, s * 0.72)
        p.closeSubpath()
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 95))
        painter.drawPath(p)
        painter.setBrush(Qt.NoBrush)
        char_pen = QPen(color.darker(160), 1.2)
        painter.setPen(char_pen)
        for i in range(12):
            x = int(m + 38 + i * 13)
            y = int(s * 0.42 + (i % 4) * 18)
            painter.drawLine(x - 4, y - 3, x + 4, y + 3)
            painter.drawLine(x - 3, y + 4, x + 3, y - 4)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

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
        old_pen = painter.pen()
        n_pen = QPen(color.darker(150), 1.6)
        painter.setPen(n_pen)
        nx = s - m - 34
        ny = m + 24
        painter.drawLine(nx, ny + 16, nx, ny - 10)
        painter.drawLine(nx, ny - 10, nx - 5, ny - 3)
        painter.drawLine(nx, ny - 10, nx + 5, ny - 3)
        painter.setPen(old_pen)

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

    def _draw_trench(self, painter, s, m, color):
        """Trench as elongated rectangle with cut hatch."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        x = m + 20
        y = int(s * 0.36)
        w = s - 2 * m - 40
        h = int(s * 0.28)
        painter.drawRect(x, y, w, h)
        painter.setBrush(Qt.NoBrush)
        hatch_pen = QPen(color.darker(145), 1.0, Qt.DashLine)
        painter.setPen(hatch_pen)
        for i in range(8):
            dx = int(x + 8 + i * (w - 16) / 7.0)
            painter.drawLine(dx, y + 4, dx - 8, y + h - 4)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_datum_point(self, painter, s, m, color):
        """Datum point: control-point triangle with center marker."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        cx = s / 2.0
        top = m + 24
        left = m + 34
        right = s - m - 34
        base = s - m - 24
        tri = QPolygonF([
            QPointF(cx, top),
            QPointF(right, base),
            QPointF(left, base),
        ])
        painter.drawPolygon(tri)
        painter.setBrush(Qt.white)
        painter.drawEllipse(int(cx - 8), int(s / 2 - 8), 16, 16)
        painter.setBrush(Qt.NoBrush)
        x_pen = QPen(color.darker(145), 1.4)
        painter.setPen(x_pen)
        painter.drawLine(int(cx), int(s / 2 - 18), int(cx), int(s / 2 + 18))
        painter.drawLine(int(cx - 18), int(s / 2), int(cx + 18), int(s / 2))
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_photo_point(self, painter, s, m, color):
        """Photo point: camera body + viewing cone."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        body_x = int(s * 0.34)
        body_y = int(s * 0.42)
        body_w = int(s * 0.32)
        body_h = int(s * 0.22)
        painter.drawRect(body_x, body_y, body_w, body_h)
        painter.setBrush(Qt.white)
        painter.drawEllipse(int(s / 2 - 16), int(body_y + 12), 32, 32)
        painter.drawRect(int(body_x + 8), int(body_y - 10), 18, 10)
        cone_pen = QPen(color.darker(150), 1.2, Qt.DotLine)
        painter.setPen(cone_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(int(s / 2), int(body_y + body_h / 2), s - m - 6, int(s * 0.28))
        painter.drawLine(int(s / 2), int(body_y + body_h / 2), s - m - 6, int(s * 0.72))
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_grid_corner(self, painter, s, m, color):
        """Grid corner: L marker with tied coordinate ticks."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setBrush(Qt.NoBrush)
        grid_pen = QPen(color.darker(145), 3.0)
        painter.setPen(grid_pen)
        x0 = m + 26
        y0 = s - m - 26
        painter.drawLine(x0, y0, x0 + 120, y0)
        painter.drawLine(x0, y0, x0, y0 - 120)
        tick_pen = QPen(color.darker(150), 1.3)
        painter.setPen(tick_pen)
        for i in range(1, 4):
            painter.drawLine(x0 + i * 30, y0 - 6, x0 + i * 30, y0 + 6)
            painter.drawLine(x0 - 6, y0 - i * 30, x0 + 6, y0 - i * 30)
        painter.setBrush(color)
        painter.drawEllipse(x0 - 5, y0 - 5, 10, 10)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_sample_location(self, painter, s, m, color):
        """Sample location: core tube marker inside ring."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        cx = s / 2.0
        cy = s / 2.0
        r = s / 2.0 - m - 24
        painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        tube = QPainterPath()
        tube.moveTo(cx - 12, m + 36)
        tube.lineTo(cx + 12, m + 36)
        tube.lineTo(cx + 8, s - m - 34)
        tube.lineTo(cx - 8, s - m - 34)
        tube.closeSubpath()
        painter.drawPath(tube)
        painter.setBrush(Qt.white)
        painter.drawEllipse(int(cx - 9), int(m + 28), 18, 18)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_road(self, painter, s, m, color):
        """Road/pavement with carriageway edges and center line."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setBrush(Qt.NoBrush)
        edge_pen = QPen(color.darker(130), 2.2)
        painter.setPen(edge_pen)
        painter.drawArc(m + 12, m + 34, s - 2 * m - 24, s - 2 * m - 68, 25 * 16, 310 * 16)
        painter.drawArc(m + 40, m + 56, s - 2 * m - 80, s - 2 * m - 112, 25 * 16, 310 * 16)
        center_pen = QPen(color.darker(150), 1.3, Qt.DashLine)
        painter.setPen(center_pen)
        painter.drawArc(m + 26, m + 45, s - 2 * m - 52, s - 2 * m - 90, 25 * 16, 310 * 16)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_bridge(self, painter, s, m, color):
        """Bridge with deck and two arch openings."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        deck_y = int(s * 0.42)
        painter.drawRect(m + 16, deck_y, s - 2 * m - 32, 22)
        painter.setBrush(Qt.white)
        painter.drawArc(m + 28, deck_y + 8, int((s - 2 * m - 56) / 2), 80, 0, 180 * 16)
        painter.drawArc(int(s / 2), deck_y + 8, int((s - 2 * m - 56) / 2), 80, 0, 180 * 16)
        painter.setBrush(Qt.NoBrush)
        water_pen = QPen(color.darker(145), 1.1, Qt.DotLine)
        painter.setPen(water_pen)
        painter.drawLine(m + 24, int(s * 0.78), s - m - 24, int(s * 0.78))
        painter.drawLine(m + 30, int(s * 0.84), s - m - 30, int(s * 0.84))
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_terrace(self, painter, s, m, color):
        """Terrace with stepped contour bands."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setBrush(Qt.NoBrush)
        contour_pen = QPen(color.darker(140), 2.0)
        painter.setPen(contour_pen)
        for i in range(4):
            y = int(m + 36 + i * 40)
            inset = 18 + i * 10
            painter.drawLine(m + inset, y, s - m - inset, y)
            painter.drawLine(m + inset, y, m + inset + 12, y - 8)
            painter.drawLine(s - m - inset, y, s - m - inset - 12, y - 8)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_posthole(self, painter, s, m, color):
        """Posthole with center post and packing stones."""
        import math
        old_pen = painter.pen()
        old_brush = painter.brush()
        cx = s / 2.0
        cy = s / 2.0
        r = s / 2.0 - m - 24
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 65))
        ring_pen = QPen(color.darker(125), 2.2, Qt.DashLine)
        painter.setPen(ring_pen)
        painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        painter.setBrush(color.darker(140))
        painter.setPen(QPen(color.darker(150), 1.0))
        painter.drawEllipse(int(cx - 8), int(cy - 8), 16, 16)
        for i in range(6):
            rad = (math.pi / 3.0) * i
            px = cx + (r - 10) * math.cos(rad)
            py = cy + (r - 10) * math.sin(rad)
            painter.drawEllipse(int(px - 4), int(py - 3), 8, 6)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_test_pit(self, painter, s, m, color):
        """Test pit as square cut with section cross."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        x = m + 28
        y = m + 28
        w = s - 2 * m - 56
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 55))
        pit_pen = QPen(color.darker(130), 2.0, Qt.DashLine)
        painter.setPen(pit_pen)
        painter.drawRect(x, y, w, w)
        painter.setBrush(Qt.NoBrush)
        cross_pen = QPen(color.darker(145), 1.4)
        painter.setPen(cross_pen)
        painter.drawLine(x + 8, y + 8, x + w - 8, y + w - 8)
        painter.drawLine(x + w - 8, y + 8, x + 8, y + w - 8)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def get_available_templates(self):
        """Return list of available template types."""
        return sorted(list(self.TEMPLATE_INFO.keys()))

    def get_templates_by_category(self, category):
        """Return sorted template names for a given category key."""
        cat = str(category or "").strip().lower()
        if not cat or cat == "all":
            return self.get_available_templates()
        return sorted(
            [name for name, info in self.TEMPLATE_INFO.items() if str(info.get("category", "")).lower() == cat]
        )
    
    def get_categories(self):
        """Return templates grouped by category."""
        categories = {}
        for name, info in self.TEMPLATE_INFO.items():
            cat = info.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        for cat in categories:
            categories[cat] = sorted(categories[cat])
        return categories
