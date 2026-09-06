# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Template Generator
Generates symbols from built-in SVG templates with comprehensive archaeological categories.
"""

import os
import re
from qgis.PyQt.QtGui import QImage, QColor, QPainter, QPainterPath, QPolygonF, QPen
from qgis.PyQt.QtCore import Qt, QBuffer, QByteArray, QIODevice, QPointF, QRect, QRectF, QSize
from qgis.PyQt.QtSvg import QSvgGenerator, QSvgRenderer

from ..i18n import tr
from ..log import log_exception
from . import template_catalog


# -- House style -----------------------------------------------------------
#
# These symbols are markers on a map, read at 5-10 mm. At that size a hairline
# disappears and a mitred corner turns into a speck, which is what makes a
# drawing look scratchy rather than drawn. Every stroke in this module goes
# through _pen() so the whole catalogue is drawn in one hand.

DETAIL_WIDTH = 3.0      # internal lines: section marks, hatching, decoration
OUTLINE_WIDTH = 4.8     # the silhouette and anything that carries the shape


def _weight(width):
    """Lift a requested stroke width onto the house steps."""
    width = float(width)
    if width <= 2.0:
        return DETAIL_WIDTH
    if width <= 3.4:
        return OUTLINE_WIDTH
    return width * 1.3      # a deliberately heavy stroke stays heavy


def _clip_detail(painter, *paths):
    """
    Confine internal detail to the silhouette it belongs to.

    Hatching, burnish marks and section lines are laid out from a bounding
    box rather than from the curve, so without this they run past the edge of
    the shape - which is what makes a symbol look unfinished. Pair every call
    with painter.restore().
    """
    outline = QPainterPath()
    for path in paths:
        outline.addPath(path)
    painter.save()
    painter.setClipPath(outline)


def _pen(color, width=1.0, style=None):
    """
    A stroke in the house style: round, weighted, and darker than its fill.

    Callers pass the colour they mean and the relative weight they mean; the
    deepening and the rounding are applied here so they cannot drift between
    the 59 drawing methods.
    """
    pen = QPen(QColor(color).darker(140), _weight(width))
    if style is not None:
        pen.setStyle(style)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def template_display_name(name):
    """
    The label to show for a template.

    Template names are English identifiers: they key TEMPLATE_INFO, get stored
    in settings and travel with saved projects. Only the label is translated.
    """
    return tr(name)


class TemplateGenerator:
    """Generator using built-in SVG templates."""
    
    # The catalogue is data, so it lives in template_catalog where it can be
    # read without QGIS. Re-exported here because saved code and tests reach
    # for TemplateGenerator.TEMPLATE_INFO.
    TEMPLATE_INFO = template_catalog.TEMPLATE_INFO
    LEGACY_TEMPLATE_ALIASES = template_catalog.LEGACY_TEMPLATE_ALIASES

    def __init__(self, plugin_dir):
        """Initialize the template generator."""
        self.plugin_dir = plugin_dir
        self.template_dir = os.path.join(plugin_dir, 'resources', 'templates')
        
    def generate(self, template_type, color=None):
        """
        Generate a symbol from a built-in template.

        :return: SymbolResult carrying parametrised SVG (plus a raster preview)
        """
        from .symbol_result import SymbolResult
        from .autotrace.svg_builder import add_provenance, finalize_svg
        from ..defaults import PLUGIN_VERSION

        template_type = self._normalize_template_type(template_type)
        template_info = self.TEMPLATE_INFO.get(template_type)
        if not template_info:
            return None

        color = color or template_info['default_color']
        result = SymbolResult(source="template", style=str(template_type))

        template_path = self._template_file(template_info)
        svg_data = None
        if template_path:
            svg_data = self._load_and_colorize_svg(template_path, color)
        if not svg_data:
            svg_data = self._create_placeholder_svg(template_type, color)

        if svg_data:
            svg, info = finalize_svg(svg_data)
            result.meta.update(info)
            result.record_provenance(title=str(template_type), plugin_version=PLUGIN_VERSION)
            result.svg = add_provenance(svg, result.meta)
        image = self._create_placeholder(template_type, color)
        if image is not None and not image.isNull():
            png = SymbolResult.coerce(image).raster_png
            result.raster_png = png
        if result.is_empty:
            return None
        return result

    def _template_file(self, template_info):
        """
        Path to an SVG file shipped for this template, or "" when there is none.

        Most templates are drawn in code and carry no ``file`` key at all, and
        an entry that has one may still have no file on disk, so this must
        never assume either.
        """
        filename = str((template_info or {}).get("file") or "").strip()
        if not filename:
            return ""
        path = os.path.join(self.template_dir, filename)
        return path if os.path.exists(path) else ""

    def _normalize_template_type(self, template_type):
        """Normalize template names for backward compatibility."""
        key = str(template_type or "").strip()
        if not key:
            return ""
        key = re.sub(r"\s*\([A-Z]{2,6}\)\s*$", "", key).strip()
        if key in self.TEMPLATE_INFO:
            return key
        if key in self.LEGACY_TEMPLATE_ALIASES:
            return self.LEGACY_TEMPLATE_ALIASES[key]

        key_fold = key.casefold()
        for template_name in self.TEMPLATE_INFO:
            if template_name.casefold() == key_fold:
                return template_name
        for alias_name, canonical_name in self.LEGACY_TEMPLATE_ALIASES.items():
            if alias_name.casefold() == key_fold:
                return canonical_name
        return key
        
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
            except Exception as e:
                log_exception(f"Could not read the template SVG {svg_path}", e)
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
        
    # Sentinel for "pass the resolved QColor here".
    COLOR = object()

    def _resolve_draw(self, template_type):
        """
        Map a template name to (draw method name, extra args).

        An explicit ``draw`` entry in TEMPLATE_INFO wins; otherwise the name is
        matched by keyword. Returns (None, ()) for the generic fallback shape.
        """
        info = self.TEMPLATE_INFO.get(template_type) or {}
        explicit = info.get("draw")
        if explicit:
            name, *extra = explicit if isinstance(explicit, (list, tuple)) else (explicit,)
            return name, tuple(self.COLOR if a == "COLOR" else a for a in extra)

        key = str(template_type or "").strip().lower().replace("/", " ")
        COLOR = self.COLOR
        
        if (
            "kofun" in key
            or "enpun" in key
            or "zenpokouen" in key
            or "zenpokoen" in key
            or "makimuku-en" in key
            or "hotategai" in key
            or "sohochuen" in key
            or "hofun" in key
            or "zenpokoho" in key
            or "makimuku-ho" in key
            or "yosumi" in key
            or "daijobo" in key
        ):
            if "with shugo" in key:
                return ("_draw_keyhole_tomb", ("moat", COLOR))
            elif "with fukiishi" in key:
                return ("_draw_keyhole_tomb", ("fukiishi", COLOR))
            elif "tsumiishizuka" in key or "tsumishizuka" in key:
                return ("_draw_keyhole_tomb", ("tsumishizuka", COLOR))
            elif "normal" in key:
                return ("_draw_keyhole_tomb", ("normal", COLOR))
            elif "enpun" in key:
                return ("_draw_kofun_shape", ("enpun", COLOR))
            elif "zenpokouen" in key or "zenpokoen" in key:
                return ("_draw_kofun_shape", ("zenpokouen", COLOR))
            elif "makimuku-en" in key:
                return ("_draw_kofun_shape", ("makimuku_en", COLOR))
            elif "hotategai" in key:
                return ("_draw_kofun_shape", ("hotategai", COLOR))
            elif "sohochuen" in key:
                return ("_draw_kofun_shape", ("sohochuen", COLOR))
            elif "zenpokoho" in key:
                return ("_draw_kofun_shape", ("zenpokoho", COLOR))
            elif "makimuku-ho" in key:
                return ("_draw_kofun_shape", ("makimuku_ho", COLOR))
            elif "yosumi" in key:
                return ("_draw_kofun_shape", ("yosumi", COLOR))
            elif "daijobo" in key:
                return ("_draw_kofun_shape", ("daijobo", COLOR))
            elif "hofun" in key:
                return ("_draw_kofun_shape", ("hofun", COLOR))
            else:
                return ("_draw_keyhole_tomb", ("normal", COLOR))
        elif "keyhole tomb" in key or "shugo" in key or "fukiishi" in key or "tsumishizuka" in key:
            if "moat" in key or "shugo" in key:
                return ("_draw_keyhole_tomb", ("moat", COLOR))
            elif "fukiishi" in key:
                return ("_draw_keyhole_tomb", ("fukiishi", COLOR))
            elif "tsumishizuka" in key:
                return ("_draw_keyhole_tomb", ("tsumishizuka", COLOR))
            elif "makinokuchi" in key:
                return ("_draw_keyhole_tomb", ("makinokuchi", COLOR))
            elif "stepped" in key:
                return ("_draw_keyhole_tomb", ("stepped", COLOR))
            else:
                return ("_draw_keyhole_tomb", ("normal", COLOR))
        elif "bronze dagger-axe" in key:
            return ("_draw_bronze_weapon_symbol", ("dagger_axe", COLOR))
        elif "bronze spear" in key:
            return ("_draw_bronze_weapon_symbol", ("spear", COLOR))
        elif "bronze sword" in key:
            return ("_draw_bronze_weapon_symbol", ("sword", COLOR))
        elif "bronze dagger" in key or "bronze sword" in key:
            if "liaoning" in key:
                return ("_draw_bronze_dagger_typology", ("liaoning", COLOR))
            elif "ordos" in key:
                return ("_draw_bronze_dagger_typology", ("ordos", COLOR))
            elif "antenna" in key:
                return ("_draw_bronze_dagger_typology", ("antenna", COLOR))
            elif "slender" in key:
                return ("_draw_bronze_dagger_typology", ("slender", COLOR))
            elif "tao" in key:
                return ("_draw_bronze_dagger_typology", ("tao", COLOR))
            elif "type ia" in key:
                return ("_draw_bronze_dagger_typology", ("type_ia", COLOR))
            elif "type ib" in key:
                return ("_draw_bronze_dagger_typology", ("type_ib", COLOR))
            elif "medium" in key:
                return ("_draw_bronze_dagger_typology", ("medium", COLOR))
            elif "flat" in key:
                return ("_draw_bronze_dagger_typology", ("flat", COLOR))
            else:
                return ("_draw_bronze_dagger_typology", ("other", COLOR))
        elif "projectile point" in key or "side-notched" in key or "corner-notched" in key:
            if "leaf" in key:
                return ("_draw_projectile_point_typology", ("leaf",))
            elif "side" in key:
                return ("_draw_projectile_point_typology", ("side_notched",))
            elif "corner" in key or "dead cedar" in key:
                return ("_draw_projectile_point_typology", ("corner_notched",))
            elif "stemmed" in key or "elko" in key:
                return ("_draw_projectile_point_typology", ("stemmed",))
            elif "triangular" in key:
                return ("_draw_projectile_point_typology", ("triangular",))
            else:
                return ("_draw_projectile_point_typology", ("leaf",))
        elif "rim sherd" in key:
            return ("_draw_pottery_sherd_section", ("rim", COLOR))
        elif "base sherd" in key:
            return ("_draw_pottery_sherd_section", ("base", COLOR))
        elif "body sherd" in key:
            return ("_draw_pottery_sherd_section", ("body", COLOR))
        elif "pottery" in key:
            return ("_draw_pottery", ())
        elif "stone tool" in key or "arrowhead" in key or "scraper" in key:
            return ("_draw_stone_tool", ())
        elif "bronze" in key:
            return ("_draw_bronze", ())
        elif "iron" in key or "chisel" in key:
            return ("_draw_iron", ())
        elif "ornament" in key or "bead" in key or "bracelet" in key or "ring" in key:
            return ("_draw_ornament", ())
        elif "coin" in key or "seal" in key or "stamp" in key or "spindle" in key:
            return ("_draw_coin", (COLOR,))
        elif "bone" in key or "needle" in key or "pin" in key or "animal remains" in key:
            return ("_draw_bone_tool", ())
        elif "weapon" in key or "blade" in key or "arrow shaft" in key:
            return ("_draw_weapon", ())
        elif "fortress" in key or "castle" in key or "gate" in key or "tower" in key:
            if "gate" in key:
                return ("_draw_gate", ())
            elif "tower" in key:
                return ("_draw_tower", ())
            else:
                return ("_draw_fortress", ())
        elif "dwelling" in key or "house" in key or "workshop" in key:
            if "workshop" in key:
                return ("_draw_workshop", ())
            else:
                return ("_draw_dwelling", ())
        elif "road" in key or "pavement" in key:
            return ("_draw_road", (COLOR,))
        elif "bridge" in key:
            return ("_draw_bridge", (COLOR,))
        elif "terrace" in key:
            return ("_draw_terrace", (COLOR,))
        elif "wall" in key or "rampart" in key:
            return ("_draw_wall", ())
        elif "posthole" in key:
            return ("_draw_posthole", (COLOR,))
        elif "test pit" in key:
            return ("_draw_test_pit", (COLOR,))
        elif "pit" in key:
            return ("_draw_pit", (COLOR,))
        elif "ash layer" in key:
            return ("_draw_ash_layer", (COLOR,))
        elif "burnt" in key:
            return ("_draw_burnt_area", (COLOR,))
        elif "canal" in key or "water channel" in key:
            return ("_draw_canal", (COLOR,))
        elif "ditch" in key or "moat" in key:
            return ("_draw_ditch", (COLOR,))
        elif "standing stone" in key:
            return ("_draw_standing_stone", (COLOR,))
        elif "stone align" in key:
            return ("_draw_stone_alignment", ())
        elif "trench" in key:
            return ("_draw_trench", (COLOR,))
        elif "grid corner" in key:
            return ("_draw_grid_corner", (COLOR,))
        elif "excavation" in key:
            return ("_draw_excavation", (COLOR,))
        elif "north arrow" in key:
            return ("_draw_north_arrow", (COLOR,))
        elif "scale bar" in key:
            return ("_draw_scale_bar", (COLOR,))
        elif "harris matrix" in key or "harris context" in key:
            return ("_draw_harris_matrix_context", (COLOR,))
        elif "stratigraphic unit" in key:
            return ("_draw_stratigraphic_unit", (COLOR,))
        elif "datum" in key:
            return ("_draw_datum_point", (COLOR,))
        elif "photo point" in key:
            return ("_draw_photo_point", (COLOR,))
        elif "survey" in key:
            return ("_draw_survey_point", (COLOR,))
        elif "sample location" in key:
            return ("_draw_sample_location", (COLOR,))
        elif "find" in key:
            return ("_draw_find_spot", (COLOR,))
        elif "tomb" in key or "barrow" in key or ("mound" in key and "shell" not in key and "midden" not in key):
            return ("_draw_tomb", ())
        elif "temple" in key or "shrine" in key:
            return ("_draw_temple", (COLOR,))
        elif "kiln" in key or "furnace" in key:
            return ("_draw_kiln", ())
        elif "well" in key:
            return ("_draw_well", (COLOR,))
        elif "human" in key or "skull" in key or "skeleton" in key:
            return ("_draw_skull", (COLOR,))
        elif "burial" in key or "cremation" in key:
            return ("_draw_burial", (COLOR,))
        elif "hearth" in key or "fire" in key:
            return ("_draw_hearth", (COLOR,))
        elif "midden" in key or "shell" in key:
            return ("_draw_midden", ())
        elif "dolmen" in key:
            return ("_draw_dolmen", ())
        elif "rock art" in key:
            return ("_draw_rock_art", (COLOR,))
        else:
            return (None, ())

    def _create_placeholder_svg(self, template_type, color, size=256):
        """Draw a built-in template into an SVG document and return its text."""
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        generator = QSvgGenerator()
        generator.setOutputDevice(buffer)
        generator.setSize(QSize(size, size))
        generator.setViewBox(QRect(0, 0, size, size))
        generator.setTitle(str(template_type))

        painter = QPainter(generator)
        try:
            self._paint_template(painter, template_type, color, size)
        finally:
            painter.end()
        buffer.close()
        return bytes(buffer.data()).decode("utf-8", "replace")

    def _create_placeholder(self, template_type, color, size=256):
        """Raster fallback of the same drawing (preview and legacy callers)."""
        image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            self._paint_template(painter, template_type, color, size)
        finally:
            painter.end()
        return image

    def _paint_template(self, painter, template_type, color, size=256):
        """
        Shared painting used by both the SVG and raster placeholder paths.

        Two rules bind every ``_draw_*`` method, because the SVG is
        parametrised for QGIS afterwards (see svg_builder.parametrize):

        * Fill only with the symbol colour. QGIS gives every ``param(fill)``
          the same value, so a lighter or darker *colour* collapses into a
          flat tone the moment the user recolours the symbol. Vary the alpha
          instead — it survives as per-element ``fill-opacity``.
        * Use ``Qt.NoBrush`` for an unfilled shape, never a transparent
          colour: a transparent colour still emits a solid ``fill`` attribute,
          which the parametriser can take as the symbol's fallback colour.

        Both rules are enforced by tests/test_template_drawing.py.
        """
        q_color = QColor(color)
        painter.setBrush(q_color)
        painter.setPen(_pen(q_color, 2.6))
        m = 25  # margin

        name, extra = self._resolve_draw(template_type)
        if not name:
            painter.drawEllipse(m, m, size - 2 * m, size - 2 * m)
            return
        method = getattr(self, name, None)
        if method is None:
            painter.drawEllipse(m, m, size - 2 * m, size - 2 * m)
            return
        args = tuple(q_color if a is self.COLOR else a for a in extra)
        method(painter, size, m, *args)


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
        line_pen = _pen(old_pen.color().darker(140), 1.1)
        _clip_detail(painter, p)
        painter.setPen(line_pen)
        painter.setBrush(Qt.NoBrush)

        # Split-profile convention used in ceramic illustration: the centre
        # line, the rim and the base. The section hatching that used to fill
        # the left half reads as scribble once the symbol is map-sized.
        painter.drawLine(int(cx), int(m + 24), int(cx), int(s - m - 2))
        painter.drawLine(int(cx - 24), int(m + 30), int(cx + 24), int(m + 30))
        painter.drawLine(int(cx - 44), int(s - m - 8), int(cx + 44), int(s - m - 8))
        painter.restore()

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_pottery_sherd_section(self, painter, s, m, variant, color):
        """Section-style ceramic sherd snippets used in typology figures."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        edge_pen = _pen(color.darker(145), 2.0)
        hatch_pen = _pen(color.darker(165), 1.0)
        painter.setPen(edge_pen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 95))

        path = QPainterPath()
        if variant == "rim":
            path.moveTo(m + 22, m + 52)
            path.quadTo(s * 0.42, m + 24, s * 0.70, m + 34)
            path.quadTo(s - m - 18, m + 50, s - m - 42, m + 70)
            path.lineTo(m + 52, s - m - 28)
            path.quadTo(m + 30, s * 0.62, m + 22, m + 52)
        elif variant == "base":
            path.moveTo(m + 32, s - m - 68)
            path.quadTo(s * 0.35, s - m - 24, s * 0.52, s - m - 16)
            path.quadTo(s * 0.70, s - m - 24, s - m - 28, s - m - 66)
            path.lineTo(s - m - 60, m + 42)
            path.quadTo(s * 0.56, m + 30, m + 54, m + 44)
            path.closeSubpath()
        else:
            path.moveTo(m + 28, m + 46)
            path.quadTo(s * 0.36, m + 20, s * 0.62, m + 32)
            path.quadTo(s - m - 20, m + 54, s - m - 30, s * 0.62)
            path.quadTo(s * 0.70, s - m - 18, s * 0.45, s - m - 16)
            path.quadTo(m + 34, s - m - 20, m + 24, s * 0.60)
            path.closeSubpath()
        painter.drawPath(path)

        _clip_detail(painter, path)
        painter.setPen(hatch_pen)
        painter.setBrush(Qt.NoBrush)
        for i in range(5):
            x = int(m + 48 + i * 34)
            y1 = int(m + 60 + (i % 3) * 14)
            y2 = int(s - m - 28 - (i % 2) * 10)
            painter.drawLine(x - 9, y1, x + 8, y2)
        painter.restore()

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
        scar_pen = _pen(old_pen.color().darker(145), 1.0)
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
        painter.setBrush(Qt.NoBrush)
        hs = 30
        painter.drawEllipse(int(s/2 - hs/2), int(m + 40), hs, hs)

    def _draw_coin(self, painter, s, m, color):
        """Coin — double circle with cross."""
        painter.drawEllipse(m + 10, m + 10, s - 2*m - 20, s - 2*m - 20)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color.darker(150), 2.5))
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
        ridge_pen = _pen(old_pen.color().darker(130), 1.25)
        painter.setPen(ridge_pen)
        painter.drawLine(int(cx), int(m + 14), int(cx), int(s - m - 8))
        painter.setPen(old_pen)

    def _draw_bronze_weapon_symbol(self, painter, s, m, variant, color):
        """Bronze weapon symbol variants (sword, dagger-axe, spear)."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        cx = s / 2.0
        painter.setPen(_pen(color.darker(170), 2.2))
        painter.setBrush(color)

        path = QPainterPath()
        if variant == "dagger_axe":
            path.moveTo(cx, m + 8)
            path.quadTo(cx + 28, s * 0.26, cx + 22, s * 0.48)
            path.lineTo(cx + 30, s * 0.66)
            path.quadTo(cx + 12, s - m - 26, cx + 4, s - m - 14)
            path.lineTo(cx - 4, s - m - 14)
            path.quadTo(cx - 12, s - m - 26, cx - 30, s * 0.66)
            path.lineTo(cx - 22, s * 0.48)
            path.quadTo(cx - 28, s * 0.26, cx, m + 8)
            path.closeSubpath()
        elif variant == "spear":
            path.moveTo(cx, m + 6)
            path.quadTo(cx + 14, s * 0.30, cx + 12, s * 0.66)
            path.lineTo(cx + 9, s - m - 30)
            path.lineTo(cx + 9, s - m - 18)
            path.lineTo(cx - 9, s - m - 18)
            path.lineTo(cx - 9, s - m - 30)
            path.lineTo(cx - 12, s * 0.66)
            path.quadTo(cx - 14, s * 0.30, cx, m + 6)
            path.closeSubpath()
        else:
            path.moveTo(cx, m + 6)
            path.quadTo(cx + 20, s * 0.26, cx + 18, s * 0.60)
            path.lineTo(cx + 12, s * 0.75)
            path.lineTo(cx + 12, s - m - 28)
            path.lineTo(cx + 22, s - m - 28)
            path.lineTo(cx + 22, s - m - 14)
            path.lineTo(cx - 22, s - m - 14)
            path.lineTo(cx - 22, s - m - 28)
            path.lineTo(cx - 12, s - m - 28)
            path.lineTo(cx - 12, s * 0.75)
            path.lineTo(cx - 18, s * 0.60)
            path.quadTo(cx - 20, s * 0.26, cx, m + 6)
            path.closeSubpath()

        painter.drawPath(path)
        painter.setPen(_pen(color.darker(185), 1.3))
        painter.setBrush(Qt.NoBrush)
        ridge_bottom = int(s - m - 22 if variant == "sword" else s - m - 20)
        painter.drawLine(int(cx), int(m + 14), int(cx), ridge_bottom)
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

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
        ridge_pen = _pen(old_pen.color().darker(135), 1.20)
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
        painter.setPen(_pen(old_pen.color().darker(135), 1.1))
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
            moat_pen = _pen(color.lighter(135), moat_width)
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
                painter.setPen(_pen(old_pen.color().darker(125), 1.0))
                painter.drawLine(int(cx - 26), int(join_y + 12), int(cx + 26), int(join_y + 12))
                painter.drawLine(int(cx - 30), int(join_y + 24), int(cx + 30), int(join_y + 24))

            painter.setBrush(old_brush)
            painter.setPen(old_pen)

        painter.drawPath(composite)

        if variant in ("stepped", "fukiishi", "tsumishizuka"):
            old_pen = painter.pen()
            step_pen = _pen(old_pen.color().darker(130), 1.1)
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

    def _draw_kofun_shape(self, painter, s, m, variant, color):
        """Kofun plan-view variants for regional map symbols."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setPen(_pen(color.darker(160), 2.0))
        painter.setBrush(color)

        cx = s / 2.0
        if variant == "zenpokouen":
            self._draw_keyhole_tomb(painter, s, m, "normal", color)
        elif variant == "makimuku_en":
            self._draw_keyhole_tomb(painter, s, m, "normal", color)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(175), 1.2))
            painter.drawLine(int(cx - 24), int(s * 0.55), int(cx + 24), int(s * 0.55))
            painter.drawLine(int(cx - 30), int(s * 0.63), int(cx + 30), int(s * 0.63))
        elif variant == "enpun":
            r = s / 2.0 - m - 14
            painter.drawEllipse(int(cx - r), int(cx - r), int(2 * r), int(2 * r))
        elif variant == "hotategai":
            top_r = 46.0
            circle_y = m + 62
            p = QPainterPath()
            p.addEllipse(QRectF(cx - top_r, circle_y - top_r, top_r * 2, top_r * 2))
            p.moveTo(cx - 30, circle_y + top_r - 6)
            p.lineTo(cx - 44, s - m - 26)
            p.lineTo(cx + 44, s - m - 26)
            p.lineTo(cx + 30, circle_y + top_r - 6)
            p.closeSubpath()
            painter.drawPath(p)
        elif variant == "sohochuen":
            p = QPainterPath()
            p.addEllipse(QRectF(cx - 38, m + 26, 76, 76))
            p.addEllipse(QRectF(cx - 28, s * 0.49, 56, 92))
            painter.drawPath(p)
        elif variant == "hofun":
            side = s - (2 * m) - 36
            painter.drawRect(int(cx - side / 2), int(cx - side / 2), int(side), int(side))
        elif variant == "zenpokoho":
            p = QPainterPath()
            p.addEllipse(QRectF(cx - 34, m + 24, 68, 68))
            p.moveTo(cx - 32, s * 0.50)
            p.lineTo(cx - 42, s - m - 20)
            p.lineTo(cx + 42, s - m - 20)
            p.lineTo(cx + 32, s * 0.50)
            p.closeSubpath()
            painter.drawPath(p)
        elif variant == "makimuku_ho":
            p = QPainterPath()
            p.addEllipse(QRectF(cx - 34, m + 24, 68, 68))
            p.moveTo(cx - 24, s * 0.50)
            p.lineTo(cx - 44, s - m - 26)
            p.lineTo(cx + 44, s - m - 26)
            p.lineTo(cx + 24, s * 0.50)
            p.closeSubpath()
            painter.drawPath(p)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(178), 1.2))
            painter.drawLine(int(cx - 30), int(s * 0.62), int(cx + 30), int(s * 0.62))
        elif variant == "yosumi":
            p = QPainterPath()
            x1 = m + 34
            y1 = m + 34
            x2 = s - m - 34
            y2 = s - m - 34
            protrusion = 16
            p.moveTo(x1, y1 - protrusion)
            p.lineTo((x1 + x2) / 2, y1)
            p.lineTo(x2, y1 - protrusion)
            p.lineTo(x2 + protrusion, y1)
            p.lineTo(x2, (y1 + y2) / 2)
            p.lineTo(x2 + protrusion, y2)
            p.lineTo(x2, y2 + protrusion)
            p.lineTo((x1 + x2) / 2, y2)
            p.lineTo(x1, y2 + protrusion)
            p.lineTo(x1 - protrusion, y2)
            p.lineTo(x1, (y1 + y2) / 2)
            p.lineTo(x1 - protrusion, y1)
            p.closeSubpath()
            painter.drawPath(p)
        elif variant == "daijobo":
            side = s - (2 * m) - 30
            x = int(cx - side / 2)
            y = int(cx - side / 2)
            painter.drawRect(x, y, int(side), int(side))
            inner = int(side * 0.46)
            inner_x = int(cx - inner / 2)
            inner_y = int(cx - inner / 2)
            # Lighter through opacity, so QGIS recolouring keeps the contrast.
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 120))
            painter.drawRect(inner_x, inner_y, inner, inner)
        else:
            self._draw_keyhole_tomb(painter, s, m, "normal", color)

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_fortress(self, painter, s, m):
        """Castle/fortress — crenellated rectangle."""
        p = QPainterPath()
        bw = s - 2 * m  # base width
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
        painter.setBrush(Qt.NoBrush)
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
        painter.setBrush(Qt.NoBrush)
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
        hatch_pen = _pen(old_pen.color().darker(140), 1.0)
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
        painter.setBrush(Qt.NoBrush)
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
        arch_pen = _pen(old_pen.color().darker(130), 1.4)
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
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(x + 32, y + 32, 16, 18)
        painter.drawRect(x + 32, y + 66, 16, 18)
        painter.drawRect(x + 30, y + h - 42, 20, 28)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_workshop(self, painter, s, m):
        """Workshop icon: dwelling body + crossed tool cue."""
        self._draw_dwelling(painter, s, m)
        old_pen = painter.pen()
        tool_pen = _pen(old_pen.color().darker(145), 1.8)
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
        pen = _pen(color.darker(120), 2.5, Qt.DashLine)
        painter.setPen(pen)
        painter.drawEllipse(m + 20, m + 20, s - 2*m - 40, s - 2*m - 40)
        # Cross inside
        cx, cy = s/2, s/2
        r = 30
        painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
        painter.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean tomb types (한국 무덤)
    # ═══════════════════════════════════════════════════════

    def _draw_korean_tomb(self, painter, s, m, variant, color):
        """
        Korean burial types as the schematic each one is recognised by.

        Section view for the dolmens and pit graves, plan view for the chamber
        tombs — the same convention as an excavation report figure. Detail is
        deliberately coarse: these have to stay readable at a 5-10 mm marker.
        """
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        fill = QColor(color.red(), color.green(), color.blue(), 110)
        faint = QColor(color.red(), color.green(), color.blue(), 55)
        edge = _pen(color.darker(150), 2.6)
        thin = _pen(color.darker(165), 1.4)
        dashed = _pen(color.darker(140), 2.0, Qt.DashLine)
        cx = s / 2.0
        ground = s - m - 34

        painter.setPen(edge)
        painter.setBrush(solid)

        if variant == "table":
            # 탁자식: a capstone carried clear of the ground on tall slabs.
            cap = QPainterPath()
            cap.moveTo(m, m + 78)
            cap.lineTo(m + 26, m + 44)
            cap.lineTo(s - m - 26, m + 44)
            cap.lineTo(s - m, m + 78)
            cap.closeSubpath()
            painter.drawPath(cap)
            painter.setBrush(fill)
            painter.drawRect(QRectF(m + 44, m + 78, 26, ground - (m + 78)))
            painter.drawRect(QRectF(s - m - 70, m + 78, 26, ground - (m + 78)))
            painter.setPen(thin)
            painter.drawLine(int(m - 4), int(ground), int(s - m + 4), int(ground))

        elif variant == "go_board":
            # 기반식: a thick capstone resting on short supports over a low mound.
            painter.setBrush(faint)
            painter.setPen(thin)
            mound = QPainterPath()
            mound.moveTo(m - 4, ground)
            mound.quadTo(cx, ground - 46, s - m + 4, ground)
            mound.closeSubpath()
            painter.drawPath(mound)
            painter.setPen(edge)
            painter.setBrush(fill)
            for x in (m + 40, cx - 13, s - m - 66):
                painter.drawRect(QRectF(x, ground - 40, 26, 40))
            painter.setBrush(solid)
            cap = QPainterPath()
            cap.moveTo(m - 2, ground - 46)
            cap.quadTo(cx, ground - 92, s - m + 2, ground - 46)
            cap.quadTo(cx, ground - 30, m - 2, ground - 46)
            cap.closeSubpath()
            painter.drawPath(cap)

        elif variant == "capstone":
            # 개석식: the capstone lies on the ground over a buried cist.
            cap = QPainterPath()
            cap.moveTo(m - 2, ground - 20)
            cap.quadTo(cx, ground - 74, s - m + 2, ground - 20)
            cap.quadTo(cx, ground - 2, m - 2, ground - 20)
            cap.closeSubpath()
            painter.drawPath(cap)
            painter.setPen(thin)
            painter.drawLine(int(m - 6), int(ground - 8), int(s - m + 6), int(ground - 8))
            painter.setPen(dashed)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(cx - 44, ground + 6, 88, 40))

        elif variant == "stone_cist":
            # 석관묘: four slabs set on edge, drawn in plan with open corners.
            painter.setBrush(Qt.NoBrush)
            left, right = m + 30, s - m - 30
            top, bottom = m + 16, s - m - 16
            painter.setBrush(fill)
            painter.drawRect(QRectF(left + 12, top, right - left - 24, 16))
            painter.drawRect(QRectF(left + 12, bottom - 16, right - left - 24, 16))
            painter.drawRect(QRectF(left, top + 12, 16, bottom - top - 24))
            painter.drawRect(QRectF(right - 16, top + 12, 16, bottom - top - 24))

        elif variant == "stone_lined":
            # 석곽묘: a chamber walled with piled stones, drawn in plan.
            left, right = m + 22, s - m - 22
            top, bottom = m + 10, s - m - 10
            painter.setBrush(fill)
            painter.drawRect(QRectF(left, top, right - left, bottom - top))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            painter.drawRect(QRectF(left + 22, top + 22, right - left - 44, bottom - top - 44))
            painter.setBrush(solid)
            step = (bottom - top - 24) / 5.0
            for i in range(5):
                y = top + 12 + step * i
                painter.drawEllipse(QRectF(left + 5, y, 12, 12))
                painter.drawEllipse(QRectF(right - 17, y, 12, 12))
            step = (right - left - 24) / 5.0
            for i in range(5):
                x = left + 12 + step * i
                painter.drawEllipse(QRectF(x, top + 5, 12, 12))
                painter.drawEllipse(QRectF(x, bottom - 17, 12, 12))

        elif variant in ("wooden_coffin", "wooden_chamber"):
            # 목관묘 / 목곽묘: the grave pit dashed, the timber solid inside it.
            painter.setBrush(faint)
            painter.setPen(dashed)
            painter.drawRect(QRectF(m + 6, m + 2, s - 2 * m - 12, s - 2 * m - 4))
            painter.setPen(edge)
            if variant == "wooden_chamber":
                painter.setBrush(fill)
                painter.drawRect(QRectF(m + 26, m + 20, s - 2 * m - 52, s - 2 * m - 40))
                painter.setBrush(solid)
                painter.drawRect(QRectF(m + 48, m + 42, s - 2 * m - 96, s - 2 * m - 84))
            else:
                painter.setBrush(solid)
                painter.drawRect(QRectF(m + 36, m + 24, s - 2 * m - 72, s - 2 * m - 48))
            painter.setPen(thin)
            for i in range(3):
                y = int(m + 60 + i * 34)
                painter.drawLine(int(m + 58), y, int(s - m - 58), y)

        elif variant == "jar_coffin":
            # 옹관묘: two jars set mouth to mouth.
            painter.setBrush(fill)
            for direction in (1, -1):
                jar = QPainterPath()
                mouth = cx + direction * 8
                tip = cx + direction * (s / 2.0 - m)
                jar.moveTo(mouth, s / 2.0 - 46)
                jar.quadTo(mouth + direction * 46, s / 2.0 - 60, tip, s / 2.0 - 20)
                jar.quadTo(tip + direction * 8, s / 2.0, tip, s / 2.0 + 20)
                jar.quadTo(mouth + direction * 46, s / 2.0 + 60, mouth, s / 2.0 + 46)
                jar.closeSubpath()
                painter.drawPath(jar)
            painter.setPen(thin)
            painter.drawLine(int(cx), int(s / 2.0 - 48), int(cx), int(s / 2.0 + 48))

        elif variant == "stone_mound_chamber":
            # 적석목곽분: a stone pile heaped over a timber chamber.
            painter.setBrush(fill)
            mound = QPainterPath()
            mound.moveTo(m - 2, ground)
            mound.quadTo(cx, m + 4, s - m + 2, ground)
            mound.closeSubpath()
            painter.drawPath(mound)
            painter.setBrush(solid)
            painter.setPen(thin)
            for row, count in ((ground - 96, 3), (ground - 64, 5), (ground - 32, 7)):
                span = 22.0 * (count - 1)
                for i in range(count):
                    painter.drawEllipse(QRectF(cx - span / 2 + 22 * i - 8, row - 8, 16, 16))
            painter.setPen(edge)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(cx - 46, ground - 26, 92, 26))
            painter.setPen(thin)
            painter.drawLine(int(m - 6), int(ground), int(s - m + 6), int(ground))

        elif variant == "corridor_chamber":
            # 횡혈식석실분: a chamber reached by a corridor, drawn in plan
            # inside the mound.
            painter.setBrush(faint)
            painter.setPen(dashed)
            painter.drawEllipse(QRectF(m - 4, m - 4, s - 2 * m + 8, s - 2 * m + 8))
            painter.setPen(edge)
            painter.setBrush(fill)
            painter.drawRect(QRectF(cx - 52, m + 26, 104, 88))
            painter.drawRect(QRectF(cx - 20, m + 114, 40, s - m - 20 - (m + 114)))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(cx - 34, m + 40, 68, 60))

        elif variant == "earthen_mound":
            # 봉토분: an earthen mound with its build-up layers.
            painter.setBrush(fill)
            mound = QPainterPath()
            mound.moveTo(m - 4, ground)
            mound.quadTo(cx, m - 4, s - m + 4, ground)
            mound.closeSubpath()
            painter.drawPath(mound)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for shrink in (26, 52):
                layer = QPainterPath()
                layer.moveTo(m - 4 + shrink, ground)
                layer.quadTo(cx, m - 4 + shrink * 1.6, s - m + 4 - shrink, ground)
                painter.drawPath(layer)
            painter.setPen(edge)
            painter.drawLine(int(m - 8), int(ground), int(s - m + 8), int(ground))

        elif variant == "ditch_encircled":
            # 주구묘: an open ditch ring around a central grave.
            painter.setBrush(fill)
            painter.setPen(thin)
            outer = QPainterPath()
            outer.addRect(QRectF(m - 2, m - 2, s - 2 * m + 4, s - 2 * m + 4))
            inner = QPainterPath()
            inner.addRect(QRectF(m + 24, m + 24, s - 2 * m - 48, s - 2 * m - 48))
            ring = outer.subtracted(inner)
            gap = QPainterPath()
            gap.addRect(QRectF(cx - 22, m - 6, 44, 40))
            gap.addRect(QRectF(cx - 22, s - m - 34, 44, 40))
            painter.drawPath(ring.subtracted(gap))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawRect(QRectF(cx - 26, s / 2.0 - 44, 52, 88))

        elif variant == "pit_grave":
            # 토광묘: a plain earth-cut pit, in section.
            painter.setBrush(fill)
            pit = QPainterPath()
            pit.moveTo(m + 6, ground - 96)
            pit.lineTo(s - m - 6, ground - 96)
            pit.lineTo(s - m - 26, ground)
            pit.lineTo(m + 26, ground)
            pit.closeSubpath()
            painter.drawPath(pit)
            painter.setPen(thin)
            painter.drawLine(int(m - 8), int(ground - 96), int(m + 6), int(ground - 96))
            painter.drawLine(int(s - m - 6), int(ground - 96), int(s - m + 8), int(ground - 96))
            painter.setBrush(solid)
            painter.setPen(edge)
            painter.drawRect(QRectF(cx - 48, ground - 44, 96, 30))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean settlement, production and defence features
    # ═══════════════════════════════════════════════════════

    def _draw_korean_feature(self, painter, s, m, variant, color):
        """
        Settlement, production and defence features as excavated plans.

        Plan view for dwellings, kilns and fields — that is how they appear on
        a site drawing — and section view for the ramparts and the basin,
        where the profile is what identifies them.
        """
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        fill = QColor(color.red(), color.green(), color.blue(), 110)
        faint = QColor(color.red(), color.green(), color.blue(), 55)
        edge = _pen(color.darker(150), 2.6)
        thin = _pen(color.darker(165), 1.4)
        dashed = _pen(color.darker(140), 2.0, Qt.DashLine)
        cx, cy = s / 2.0, s / 2.0
        ground = s - m - 34

        painter.setPen(edge)
        painter.setBrush(fill)

        def postholes(points, radius=7):
            painter.setBrush(solid)
            painter.setPen(thin)
            for px, py in points:
                painter.drawEllipse(QRectF(px - radius, py - radius, radius * 2, radius * 2))
            painter.setPen(edge)
            painter.setBrush(fill)

        def hearth(hx, hy, radius=14):
            painter.setPen(thin)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 170))
            painter.drawEllipse(QRectF(hx - radius, hy - radius, radius * 2, radius * 2))
            painter.setPen(edge)
            painter.setBrush(fill)

        if variant == "pit_house_round":
            # 원형 수혈주거지: a round cut with postholes and a central hearth.
            painter.drawEllipse(QRectF(m + 4, m + 4, s - 2 * m - 8, s - 2 * m - 8))
            r = (s - 2 * m - 8) / 2.0 - 30
            postholes([
                (cx - r * 0.71, cy - r * 0.71), (cx + r * 0.71, cy - r * 0.71),
                (cx - r * 0.71, cy + r * 0.71), (cx + r * 0.71, cy + r * 0.71),
            ])
            hearth(cx, cy)

        elif variant == "pit_house_square":
            # 방형 수혈주거지.
            painter.drawRect(QRectF(m + 6, m + 6, s - 2 * m - 12, s - 2 * m - 12))
            inset = 44
            postholes([
                (m + inset, m + inset), (s - m - inset, m + inset),
                (m + inset, s - m - inset), (s - m - inset, s - m - inset),
            ])
            hearth(cx, cy)

        elif variant == "pit_house_convex":
            # 凸자형: a square room with a short entrance passage.
            body = QPainterPath()
            body.moveTo(m + 6, m + 34)
            body.lineTo(s - m - 6, m + 34)
            body.lineTo(s - m - 6, s - m - 40)
            body.lineTo(cx + 26, s - m - 40)
            body.lineTo(cx + 26, s - m - 6)
            body.lineTo(cx - 26, s - m - 6)
            body.lineTo(cx - 26, s - m - 40)
            body.lineTo(m + 6, s - m - 40)
            body.closeSubpath()
            painter.drawPath(body)
            postholes([
                (m + 40, m + 66), (s - m - 40, m + 66),
                (m + 40, s - m - 68), (s - m - 40, s - m - 68),
            ])
            hearth(cx, cy - 10)

        elif variant == "pit_house_twin":
            # 呂자형: a main room and a smaller front room, joined.
            # Drawn as one outline: two rooms joined by a neck, which is what
            # the 呂 shape is. Two separate rectangles read as two dwellings.
            twin = QPainterPath()
            twin.moveTo(m + 12, m + 4)
            twin.lineTo(s - m - 12, m + 4)
            twin.lineTo(s - m - 12, m + 96)
            twin.lineTo(cx + 26, m + 96)
            twin.lineTo(cx + 26, s - m - 96)
            twin.lineTo(s - m - 34, s - m - 96)
            twin.lineTo(s - m - 34, s - m - 4)
            twin.lineTo(m + 34, s - m - 4)
            twin.lineTo(m + 34, s - m - 96)
            twin.lineTo(cx - 26, s - m - 96)
            twin.lineTo(cx - 26, m + 96)
            twin.lineTo(m + 12, m + 96)
            twin.closeSubpath()
            painter.drawPath(twin)
            hearth(cx, m + 50)

        elif variant == "raised_floor":
            # 굴립주건물: known only from its posthole grid.
            painter.setBrush(faint)
            painter.setPen(dashed)
            painter.drawRect(QRectF(m + 4, m + 26, s - 2 * m - 8, s - 2 * m - 52))
            grid = []
            for row in range(3):
                for col in range(4):
                    grid.append((m + 30 + col * (s - 2 * m - 60) / 3.0,
                                 m + 52 + row * (s - 2 * m - 104) / 2.0))
            postholes(grid, radius=9)

        elif variant == "kamado":
            # 부뚜막: the clay body seen in plan, with the pot seat cut into
            # it and the stoke opening at the front. Drawn as one outline -
            # a rectangle with a circle on top read as neither.
            painter.setBrush(fill)
            stove = QPainterPath()
            stove.moveTo(m + 12, cy + 54)
            stove.lineTo(m + 12, cy - 30)
            stove.quadTo(cx, cy - 78, s - m - 12, cy - 30)
            stove.lineTo(s - m - 12, cy + 54)
            stove.lineTo(cx + 26, cy + 54)
            stove.lineTo(cx + 26, cy + 18)
            stove.lineTo(cx - 26, cy + 18)
            stove.lineTo(cx - 26, cy + 54)
            stove.closeSubpath()
            painter.drawPath(stove)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(150), 2.6))
            painter.drawEllipse(QRectF(cx - 40, cy - 44, 80, 62))

        elif variant == "ondol":
            # 온돌: the heated floor as a long flue run - firebox at one end,
            # chimney rising at the other. Drawn with notched flues it read as
            # a plumbing fitting.
            painter.setBrush(fill)
            flue = QPainterPath()
            flue.moveTo(m + 8, cy + 34)
            flue.lineTo(m + 8, cy - 34)
            flue.lineTo(s - m - 62, cy - 34)
            flue.lineTo(s - m - 62, m + 10)
            flue.lineTo(s - m - 8, m + 10)
            flue.lineTo(s - m - 8, cy + 34)
            flue.closeSubpath()
            painter.drawPath(flue)
            _clip_detail(painter, flue)
            painter.setPen(_pen(color.darker(150), 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(m + 30), int(cy), int(s - m - 40), int(cy))
            painter.restore()
            painter.setBrush(solid)
            painter.setPen(edge)
            painter.drawEllipse(QRectF(m + 2, cy - 26, 52, 52))

        elif variant in ("pottery_kiln", "tile_kiln"):
            # 토기가마 / 기와가마: the sloping tunnel kiln in section - the
            # firebox low at one end, the chamber climbing to the flue at the
            # other. The two differ by how tall the chamber is, and by colour.
            # A tile kiln has the broader chamber; that width is what tells
            # the two apart, since they share a colour and a profile.
            width = 62 if variant == "tile_kiln" else 38
            rise = 60
            painter.setBrush(fill)
            kiln = QPainterPath()
            kiln.moveTo(m + 6, s - m - 22)
            kiln.lineTo(m + 6, s - m - 22 - width)
            kiln.quadTo(cx - 10, s - m - 72 - rise, s - m - 54, m + 30)
            kiln.lineTo(s - m - 12, m + 30)
            kiln.lineTo(s - m - 12, m + 30 + width)
            kiln.quadTo(cx, s - m - 30 - rise * 0.5, m + 58, s - m - 22)
            kiln.closeSubpath()
            painter.drawPath(kiln)
            painter.setBrush(solid)
            painter.setPen(edge)
            painter.drawRect(QRectF(m + 2, s - m - 44, 44, 34))

        elif variant == "iron_smelting":
            # 제철유구: the shaft furnace in section, waisted, on its wider
            # base. Adding the slag run turned the silhouette into a boot.
            painter.setBrush(fill)
            furnace = QPainterPath()
            furnace.moveTo(cx - 40, m + 10)
            furnace.lineTo(cx + 40, m + 10)
            furnace.quadTo(cx + 28, cy, cx + 46, s - m - 46)
            furnace.lineTo(cx + 74, s - m - 10)
            furnace.lineTo(cx - 74, s - m - 10)
            furnace.lineTo(cx - 46, s - m - 46)
            furnace.quadTo(cx - 28, cy, cx - 40, m + 10)
            furnace.closeSubpath()
            painter.drawPath(furnace)
            _clip_detail(painter, furnace)
            painter.setBrush(solid)
            painter.setPen(edge)
            # The tap hole at the base of the shaft.
            painter.drawEllipse(QRectF(cx - 22, s - m - 62, 44, 40))
            painter.restore()

        elif variant == "charcoal_kiln":
            # 숯가마: an oval chamber, its stoke hole, and charcoal inside.
            painter.setBrush(fill)
            painter.drawEllipse(QRectF(m + 10, m + 40, s - 2 * m - 20, s - 2 * m - 80))
            painter.setBrush(solid)
            painter.drawRect(QRectF(cx - 22, s - m - 56, 44, 44))
            painter.setPen(thin)
            for row, count in ((cy - 26, 4), (cy + 6, 5), (cy + 36, 3)):
                span = 30.0 * (count - 1)
                for i in range(count):
                    painter.drawEllipse(QRectF(cx - span / 2 + 30 * i - 9, row - 9, 18, 18))
            painter.setPen(edge)

        elif variant == "paddy_field":
            # 논: level plots divided by levees, with the water inlet.
            painter.setBrush(fill)
            painter.drawRect(QRectF(m, m + 14, s - 2 * m, s - 2 * m - 28))
            painter.setPen(_pen(color.darker(150), 3.4))
            painter.setBrush(Qt.NoBrush)
            for i in range(1, 3):
                y = int(m + 14 + i * (s - 2 * m - 28) / 3.0)
                painter.drawLine(int(m), y, int(s - m), y)
            painter.drawLine(int(cx), int(m + 14), int(cx), int(s - m - 14))
            painter.setPen(thin)
            for i in range(3):
                y = int(m + 34 + i * (s - 2 * m - 28) / 3.0)
                painter.drawLine(int(m + 14), y, int(m + 74), y)
                painter.drawLine(int(cx + 14), y, int(cx + 74), y)
            painter.setPen(edge)

        elif variant == "dry_field":
            # 밭: ridge and furrow.
            painter.setBrush(fill)
            painter.drawRect(QRectF(m, m + 20, s - 2 * m, s - 2 * m - 40))
            painter.setPen(_pen(color.darker(155), 3.0))
            for i in range(6):
                x = int(m + 16 + i * (s - 2 * m - 32) / 5.0)
                painter.drawLine(x, int(m + 26), x, int(s - m - 26))
            painter.setPen(thin)
            for i in range(5):
                x = int(m + 32 + i * (s - 2 * m - 32) / 5.0)
                painter.drawLine(x, int(m + 34), x, int(s - m - 34))
            painter.setPen(edge)

        elif variant == "earthen_rampart":
            # 토성: a rammed-earth bank in section, with its outer ditch.
            bank = QPainterPath()
            bank.moveTo(m + 4, ground)
            bank.lineTo(m + 54, m + 34)
            bank.lineTo(s - m - 74, m + 34)
            bank.lineTo(s - m - 30, ground)
            bank.closeSubpath()
            painter.drawPath(bank)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(1, 4):
                y = m + 34 + i * (ground - m - 34) / 4.0
                shrink = 14 * (4 - i)
                painter.drawLine(int(m + 10 + shrink), int(y), int(s - m - 36 - shrink), int(y))
            painter.setPen(edge)
            painter.drawLine(int(m - 6), int(ground), int(s - m + 6), int(ground))
            painter.setPen(dashed)
            ditch = QPainterPath()
            ditch.moveTo(s - m - 26, ground)
            ditch.lineTo(s - m - 14, ground + 30)
            ditch.lineTo(s - m, ground + 30)
            painter.drawPath(ditch)

        elif variant == "stone_rampart":
            # 석성: a stone-faced wall in section, coursed.
            wall = QPainterPath()
            wall.moveTo(m + 10, ground)
            wall.lineTo(m + 40, m + 30)
            wall.lineTo(s - m - 40, m + 30)
            wall.lineTo(s - m - 10, ground)
            wall.closeSubpath()
            painter.drawPath(wall)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            courses = 5
            for i in range(1, courses):
                y = m + 30 + i * (ground - m - 30) / courses
                inset = 30 - i * 4
                painter.drawLine(int(m + inset), int(y), int(s - m - inset), int(y))
            painter.setBrush(solid)
            for i in range(4):
                painter.drawRect(QRectF(m + 46 + i * 42, m + 12, 30, 18))
            painter.setPen(edge)
            painter.drawLine(int(m - 6), int(ground), int(s - m + 6), int(ground))

        elif variant == "mountain_fortress":
            # 산성: a wall line following the ridge, over contour lines.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            for i in range(3):
                inset = 16 + i * 26
                contour = QPainterPath()
                contour.moveTo(m + inset, s - m - 10)
                contour.quadTo(cx, m + 10 + i * 34, s - m - inset, s - m - 10)
                painter.drawPath(contour)
            painter.setPen(_pen(color.darker(150), 4.0))
            wall = QPainterPath()
            wall.moveTo(m + 6, s - m - 10)
            wall.quadTo(cx, m - 12, s - m - 6, s - m - 10)
            painter.drawPath(wall)
            painter.setPen(thin)
            painter.setBrush(solid)
            painter.drawRect(QRectF(cx - 16, m + 24, 32, 26))
            painter.setPen(edge)

        elif variant == "palisade":
            # 목책: a line of pointed timbers with its tie beam.
            painter.setBrush(solid)
            painter.setPen(edge)
            count = 6
            step = (s - 2 * m - 24) / (count - 1.0)
            for i in range(count):
                x = m + 12 + step * i
                post = QPainterPath()
                post.moveTo(x, m + 26)
                post.lineTo(x + 13, m + 46)
                post.lineTo(x + 13, ground + 6)
                post.lineTo(x - 13, ground + 6)
                post.lineTo(x - 13, m + 46)
                post.closeSubpath()
                painter.drawPath(post)
            painter.setPen(_pen(color.darker(160), 3.4))
            painter.drawLine(int(m + 4), int(m + 80), int(s - m - 4), int(m + 80))
            painter.setPen(thin)
            painter.drawLine(int(m - 6), int(ground + 6), int(s - m + 6), int(ground + 6))
            painter.setPen(edge)

        elif variant == "encircling_ditch":
            # 환호: a broad cut ditch ringing a settlement, with its entrance.
            # Two thin circles and a few dots read as a shirt button, so the
            # ditch is drawn as a filled band and the houses as buildings.
            outer = QPainterPath()
            outer.addEllipse(QRectF(m - 2, m - 2, s - 2 * m + 4, s - 2 * m + 4))
            inner = QPainterPath()
            inner.addEllipse(QRectF(m + 26, m + 26, s - 2 * m - 52, s - 2 * m - 52))
            entrance = QPainterPath()
            entrance.addRect(QRectF(cx - 26, m - 8, 52, 42))
            painter.setBrush(fill)
            painter.setPen(edge)
            painter.drawPath(outer.subtracted(inner).subtracted(entrance))
            painter.setBrush(solid)
            painter.setPen(thin)
            for dx, dy in ((-30, -14), (26, -20), (-16, 30), (30, 24)):
                painter.drawRect(QRectF(cx + dx - 15, cy + dy - 13, 30, 26))
            painter.setPen(edge)

        elif variant == "beacon":
            # 봉수: the stone platform and its smoke.
            painter.setBrush(fill)
            base = QPainterPath()
            base.moveTo(m + 10, ground + 8)
            base.lineTo(m + 46, cy + 6)
            base.lineTo(s - m - 46, cy + 6)
            base.lineTo(s - m - 10, ground + 8)
            base.closeSubpath()
            painter.drawPath(base)
            painter.setBrush(solid)
            painter.drawRect(QRectF(cx - 32, cy - 26, 64, 32))
            painter.setPen(_pen(color.darker(150), 3.4))
            painter.setBrush(Qt.NoBrush)
            smoke = QPainterPath()
            smoke.moveTo(cx, cy - 30)
            smoke.quadTo(cx - 34, cy - 62, cx, m + 46)
            smoke.quadTo(cx + 34, m + 24, cx - 6, m + 6)
            painter.drawPath(smoke)
            painter.setPen(edge)

        elif variant == "water_basin":
            # 집수정: a timber-lined basin in section, with the water level.
            painter.setBrush(fill)
            basin = QPainterPath()
            basin.moveTo(m + 6, m + 44)
            basin.lineTo(s - m - 6, m + 44)
            basin.lineTo(s - m - 40, ground + 10)
            basin.lineTo(m + 40, ground + 10)
            basin.closeSubpath()
            painter.drawPath(basin)
            painter.setPen(_pen(color.darker(150), 3.0))
            painter.setBrush(Qt.NoBrush)
            for i in range(3):
                y = int(m + 74 + i * 26)
                inset = 16 + i * 10
                painter.drawLine(int(m + inset), y, int(s - m - inset), y)
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawRect(QRectF(m + 6, m + 30, s - 2 * m - 12, 18))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean pottery and ceramics (토기·도자기)
    # ═══════════════════════════════════════════════════════

    def _draw_korean_pottery(self, painter, s, m, variant, color):
        """
        Korean ceramic types by profile.

        Each is the silhouette the type is identified by, with just enough
        surface treatment to tell neighbours apart — comb impressions, paddle
        marks, burnish, slip — and nothing finer, since a map marker is a few
        millimetres across.
        """
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        edge = _pen(color.darker(150), 2.6)
        thin = _pen(color.darker(170), 1.3)
        cx = s / 2.0
        top, bottom = m + 6, s - m - 6

        painter.setPen(edge)
        painter.setBrush(solid)
        body = QPainterPath()

        if variant == "comb_pattern":
            # 빗살무늬토기: a deep conical vessel with a pointed base.
            body.moveTo(cx - 74, top)
            body.lineTo(cx + 74, top)
            body.lineTo(cx + 12, bottom)
            body.quadTo(cx, bottom + 6, cx - 12, bottom)
            body.closeSubpath()
            painter.drawPath(body)
            _clip_detail(painter, body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            # Four rows read as comb impressions; the twenty-five the shape
            # could hold turn into mud once the symbol is map-sized.
            for row in range(4):
                y = top + 30 + row * 34
                half = 64 - row * 13
                for i in range(4):
                    x = cx - half + (2.0 * half / 3.0) * i
                    painter.drawLine(int(x), int(y), int(x + 9), int(y + 15))
            painter.restore()

        elif variant == "plain_coarse":
            # 민무늬토기: a deep flat-based vessel, undecorated.
            body.moveTo(cx - 68, top)
            body.quadTo(cx - 62, s * 0.5, cx - 44, bottom)
            body.lineTo(cx + 44, bottom)
            body.quadTo(cx + 62, s * 0.5, cx + 68, top)
            body.closeSubpath()
            painter.drawPath(body)
            _clip_detail(painter, body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 66), int(top + 16), int(cx + 66), int(top + 16))
            painter.restore()

        elif variant == "red_burnished":
            # 붉은간토기: a globular jar with a short everted neck.
            body.moveTo(cx - 26, top)
            body.quadTo(cx - 34, top + 18, cx - 24, top + 34)
            body.quadTo(cx - 84, s * 0.52, cx - 40, bottom)
            body.lineTo(cx + 40, bottom)
            body.quadTo(cx + 84, s * 0.52, cx + 24, top + 34)
            body.quadTo(cx + 34, top + 18, cx + 26, top)
            body.closeSubpath()
            painter.drawPath(body)
            _clip_detail(painter, body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(4):
                x = cx - 46 + i * 22
                painter.drawLine(int(x), int(s * 0.42), int(x + 16), int(s * 0.62))
            painter.restore()

        elif variant == "black_burnished":
            # 검은간토기: the tall-necked burnished jar.
            body.moveTo(cx - 22, top)
            body.lineTo(cx - 14, top + 52)
            body.quadTo(cx - 80, s * 0.58, cx - 36, bottom)
            body.lineTo(cx + 36, bottom)
            body.quadTo(cx + 80, s * 0.58, cx + 14, top + 52)
            body.lineTo(cx + 22, top)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 20), int(top + 12), int(cx + 20), int(top + 12))
            for i in range(3):
                x = cx - 34 + i * 26
                painter.drawLine(int(x), int(s * 0.56), int(x + 14), int(s * 0.72))

        elif variant == "wajil":
            # 와질토기: a round-bottomed short-necked jar, paddle-marked.
            body.moveTo(cx - 34, top + 10)
            body.quadTo(cx - 40, top + 28, cx - 30, top + 42)
            body.quadTo(cx - 88, s * 0.56, cx, bottom)
            body.quadTo(cx + 88, s * 0.56, cx + 30, top + 42)
            body.quadTo(cx + 40, top + 28, cx + 34, top + 10)
            body.closeSubpath()
            painter.drawPath(body)
            _clip_detail(painter, body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(4):
                x = cx - 48 + i * 30
                painter.drawLine(int(x), int(s * 0.46), int(x + 18), int(s * 0.66))
            painter.restore()

        elif variant == "gyeongjil":
            # 경질토기: a hard-fired jar on a ring foot, paddled.
            body.moveTo(cx - 40, top + 6)
            body.quadTo(cx - 46, top + 24, cx - 34, top + 40)
            body.quadTo(cx - 90, s * 0.56, cx - 30, bottom - 22)
            body.lineTo(cx + 30, bottom - 22)
            body.quadTo(cx + 90, s * 0.56, cx + 34, top + 40)
            body.quadTo(cx + 46, top + 24, cx + 40, top + 6)
            body.closeSubpath()
            painter.drawPath(body)
            painter.drawRect(QRectF(cx - 36, bottom - 24, 72, 22))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(5):
                y = s * 0.40 + i * 16
                painter.drawLine(int(cx - 64), int(y), int(cx + 64), int(y))

        elif variant == "gobae":
            # 굽다리접시: a shallow dish on a pierced pedestal.
            body.moveTo(cx - 74, top + 22)
            body.lineTo(cx + 74, top + 22)
            body.quadTo(cx + 58, top + 66, cx + 22, top + 74)
            body.lineTo(cx - 22, top + 74)
            body.quadTo(cx - 58, top + 66, cx - 74, top + 22)
            body.closeSubpath()
            painter.drawPath(body)
            stand = QPainterPath()
            stand.moveTo(cx - 22, top + 74)
            stand.lineTo(cx + 22, top + 74)
            stand.lineTo(cx + 58, bottom)
            stand.lineTo(cx - 58, bottom)
            stand.closeSubpath()
            painter.drawPath(stand)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for row, half in ((top + 98, 16), (top + 130, 26)):
                painter.drawRect(QRectF(cx - half - 12, row, 18, 20))
                painter.drawRect(QRectF(cx - 6 + half, row, 18, 20))

        elif variant == "storage_jar":
            # 항아리: a wide-shouldered jar with lugs.
            body.moveTo(cx - 34, top + 8)
            body.quadTo(cx - 42, top + 26, cx - 32, top + 40)
            body.quadTo(cx - 92, s * 0.50, cx - 42, bottom)
            body.lineTo(cx + 42, bottom)
            body.quadTo(cx + 92, s * 0.50, cx + 32, top + 40)
            body.quadTo(cx + 42, top + 26, cx + 34, top + 8)
            body.closeSubpath()
            painter.drawPath(body)
            painter.drawEllipse(QRectF(cx - 92, s * 0.42, 26, 26))
            painter.drawEllipse(QRectF(cx + 66, s * 0.42, 26, 26))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 32), int(top + 20), int(cx + 32), int(top + 20))

        elif variant == "siru":
            # 시루: a steaming vessel — handles and a perforated base.
            body.moveTo(cx - 72, top + 14)
            body.lineTo(cx + 72, top + 14)
            body.lineTo(cx + 40, bottom - 16)
            body.lineTo(cx - 40, bottom - 16)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(int(cx - 106), int(top + 20), 44, 52, -80 * 16, 160 * 16)
            painter.drawArc(int(cx + 62), int(top + 20), 44, 52, 100 * 16, 160 * 16)
            painter.setPen(thin)
            painter.setBrush(solid)
            for i in range(5):
                x = cx - 30 + i * 15
                painter.drawEllipse(QRectF(x - 6, bottom - 22, 12, 12))

        elif variant == "celadon":
            # 청자: the maebyeong profile — high shoulder, narrow foot.
            body.moveTo(cx - 20, top)
            body.lineTo(cx - 24, top + 20)
            body.quadTo(cx - 76, top + 34, cx - 72, s * 0.46)
            body.quadTo(cx - 62, bottom - 20, cx - 36, bottom)
            body.lineTo(cx + 36, bottom)
            body.quadTo(cx + 62, bottom - 20, cx + 72, s * 0.46)
            body.quadTo(cx + 76, top + 34, cx + 24, top + 20)
            body.lineTo(cx + 20, top)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(3):
                painter.drawEllipse(QRectF(cx - 34 + i * 26, s * 0.46, 22, 30))

        elif variant == "buncheong":
            # 분청사기: a bottle with brushed white slip.
            body.moveTo(cx - 16, top)
            body.lineTo(cx - 16, top + 48)
            body.quadTo(cx - 78, s * 0.52, cx - 44, bottom)
            body.lineTo(cx + 44, bottom)
            body.quadTo(cx + 78, s * 0.52, cx + 16, top + 48)
            body.lineTo(cx + 16, top)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(_pen(color.lighter(155), 4.0))
            painter.setBrush(Qt.NoBrush)
            for i in range(3):
                y = s * 0.56 + i * 22
                brush_stroke = QPainterPath()
                brush_stroke.moveTo(cx - 50, y)
                brush_stroke.quadTo(cx, y + 16, cx + 50, y)
                painter.drawPath(brush_stroke)

        elif variant == "white_porcelain":
            # 백자 달항아리: nearly spherical, but the mouth and foot have to
            # show or it reads as a plain circle.
            body.moveTo(cx - 34, top + 16)
            body.lineTo(cx - 30, top + 30)
            body.quadTo(cx - 96, top + 62, cx - 88, s * 0.58)
            body.quadTo(cx - 78, bottom - 22, cx - 36, bottom - 10)
            body.lineTo(cx - 32, bottom)
            body.lineTo(cx + 32, bottom)
            body.lineTo(cx + 36, bottom - 10)
            body.quadTo(cx + 78, bottom - 22, cx + 88, s * 0.58)
            body.quadTo(cx + 96, top + 62, cx + 30, top + 30)
            body.lineTo(cx + 34, top + 16)
            body.closeSubpath()
            painter.drawPath(body)
            _clip_detail(painter, body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            # The seam where the two thrown halves were joined.
            painter.drawLine(int(cx - 88), int(s * 0.56), int(cx + 88), int(s * 0.56))
            painter.restore()

        elif variant == "onggi":
            # 옹기: a large storage jar under its lid.
            body.moveTo(cx - 60, top + 34)
            body.quadTo(cx - 94, s * 0.50, cx - 46, bottom)
            body.lineTo(cx + 46, bottom)
            body.quadTo(cx + 94, s * 0.50, cx + 60, top + 34)
            body.closeSubpath()
            painter.drawPath(body)
            lid = QPainterPath()
            lid.moveTo(cx - 70, top + 34)
            lid.quadTo(cx, top - 6, cx + 70, top + 34)
            lid.closeSubpath()
            painter.drawPath(lid)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 68), int(top + 34), int(cx + 68), int(top + 34))
            for i in range(3):
                y = s * 0.54 + i * 20
                painter.drawLine(int(cx - 64), int(y), int(cx + 64), int(y))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean stone, bronze and iron tools
    # ═══════════════════════════════════════════════════════

    def _draw_korean_tool(self, painter, s, m, variant, color):
        """
        Stone, bronze and iron implements as typology silhouettes.

        Long objects are drawn upright so they fill the tile; the surface
        detail is limited to what distinguishes the type — flake scars, a
        midrib, mirror bands, armour plates.
        """
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        edge = _pen(color.darker(150), 2.4)
        thin = _pen(color.darker(170), 1.3)
        cx, cy = s / 2.0, s / 2.0
        top, bottom = m + 4, s - m - 4

        painter.setPen(edge)
        painter.setBrush(solid)
        body = QPainterPath()

        if variant == "handaxe":
            # 주먹도끼: a pointed biface, flaked all over.
            body.moveTo(cx, top)
            body.quadTo(cx + 62, s * 0.42, cx + 46, bottom - 26)
            body.quadTo(cx, bottom + 6, cx - 46, bottom - 26)
            body.quadTo(cx - 62, s * 0.42, cx, top)
            body.closeSubpath()
            painter.drawPath(body)
            _clip_detail(painter, body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(2):
                y = top + 62 + i * 54
                half = 30 + i * 12
                painter.drawLine(int(cx - half), int(y), int(cx - 6), int(y - 16))
                painter.drawLine(int(cx + half), int(y), int(cx + 6), int(y - 16))
            painter.restore()

        elif variant == "chopper":
            # 찍개: a cobble with one flaked working edge.
            body.moveTo(cx - 20, top + 10)
            body.quadTo(cx + 66, top + 22, cx + 62, cy + 10)
            body.quadTo(cx + 50, bottom, cx - 10, bottom - 6)
            body.quadTo(cx - 64, bottom - 30, cx - 60, cy - 20)
            body.quadTo(cx - 58, top + 20, cx - 20, top + 10)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(_pen(color.darker(175), 2.6))
            painter.setBrush(Qt.NoBrush)
            zigzag = QPainterPath()
            zigzag.moveTo(cx - 58, cy - 20)
            for i in range(4):
                zigzag.lineTo(cx - 34 + i * 8, cy + 6 + i * 22)
                zigzag.lineTo(cx - 56 + i * 10, cy + 18 + i * 22)
            painter.drawPath(zigzag)

        elif variant == "tanged_point":
            # 슴베찌르개: a blade with a tang for hafting.
            body.moveTo(cx, top)
            body.quadTo(cx + 32, s * 0.36, cx + 22, s * 0.62)
            body.lineTo(cx + 12, s * 0.68)
            body.lineTo(cx + 12, bottom)
            body.lineTo(cx - 12, bottom)
            body.lineTo(cx - 12, s * 0.68)
            body.lineTo(cx - 22, s * 0.62)
            body.quadTo(cx - 32, s * 0.36, cx, top)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx), int(top + 14), int(cx), int(s * 0.62))

        elif variant == "microblade_core":
            # 좀돌날: a wedge-shaped core with its blade scars.
            body.moveTo(cx - 54, top + 20)
            body.lineTo(cx + 54, top + 32)
            body.lineTo(cx + 34, bottom - 10)
            body.lineTo(cx - 30, bottom - 20)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(6):
                x = cx - 44 + i * 18
                painter.drawLine(int(x), int(top + 24), int(x - 6), int(bottom - 18))

        elif variant == "polished_dagger":
            # 간돌검: a polished blade with a midrib and a stepped hilt.
            body.moveTo(cx, top)
            body.lineTo(cx + 22, s * 0.30)
            body.lineTo(cx + 16, s * 0.56)
            body.lineTo(cx - 16, s * 0.56)
            body.lineTo(cx - 22, s * 0.30)
            body.closeSubpath()
            painter.drawPath(body)
            painter.drawRect(QRectF(cx - 38, s * 0.56, 76, 14))
            painter.drawRect(QRectF(cx - 16, s * 0.56 + 14, 32, 46))
            painter.drawRect(QRectF(cx - 30, bottom - 22, 60, 20))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx), int(top + 12), int(cx), int(s * 0.54))

        elif variant == "semilunar_knife":
            # 반달돌칼: a half-moon harvesting knife, two-holed.
            body.moveTo(cx - 84, cy - 22)
            body.quadTo(cx, cy - 76, cx + 84, cy - 22)
            body.quadTo(cx, cy + 56, cx - 84, cy - 22)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cx - 40, cy - 34, 18, 18))
            painter.drawEllipse(QRectF(cx + 22, cy - 34, 18, 18))

        elif variant == "stone_hoe":
            # 돌괭이: a broad blade notched for hafting.
            body.moveTo(cx - 18, top + 6)
            body.lineTo(cx + 18, top + 6)
            body.lineTo(cx + 26, s * 0.34)
            body.lineTo(cx + 62, bottom - 26)
            body.quadTo(cx, bottom + 8, cx - 62, bottom - 26)
            body.lineTo(cx - 26, s * 0.34)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 26), int(s * 0.34), int(cx + 26), int(s * 0.34))

        elif variant == "grinding_slab":
            # 갈판갈돌: the saddle quern with its muller resting on it.
            slab = QPainterPath()
            slab.moveTo(cx - 92, cy + 22)
            slab.quadTo(cx, cy - 14, cx + 92, cy + 22)
            slab.lineTo(cx + 78, bottom - 6)
            slab.quadTo(cx, bottom + 12, cx - 78, bottom - 6)
            slab.closeSubpath()
            painter.drawPath(slab)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 190))
            muller = QPainterPath()
            muller.moveTo(cx - 54, cy - 16)
            muller.quadTo(cx, cy - 60, cx + 54, cy - 16)
            muller.quadTo(cx, cy + 16, cx - 54, cy - 16)
            muller.closeSubpath()
            painter.drawPath(muller)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 70), int(cy + 34), int(cx + 70), int(cy + 34))

        elif variant == "stone_arrowhead":
            # 돌화살촉: a triangular point on a single stem.
            body.moveTo(cx, top)
            body.lineTo(cx + 34, s * 0.52)
            body.lineTo(cx + 10, s * 0.52)
            body.lineTo(cx + 10, bottom)
            body.lineTo(cx - 10, bottom)
            body.lineTo(cx - 10, s * 0.52)
            body.lineTo(cx - 34, s * 0.52)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx), int(top + 16), int(cx), int(s * 0.50))

        elif variant == "net_sinker":
            # 어망추: a pebble notched at both ends for the net line.
            body.moveTo(cx - 46, cy - 62)
            body.quadTo(cx + 52, cy - 46, cx + 46, cy + 8)
            body.quadTo(cx + 40, cy + 66, cx - 8, cy + 62)
            body.quadTo(cx - 56, cy + 52, cx - 46, cy - 62)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(_pen(color.darker(180), 3.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(int(cx - 56), int(cy - 60), 30, 34, 90 * 16, 180 * 16)
            painter.drawArc(int(cx + 28), int(cy + 22), 30, 34, -90 * 16, 180 * 16)

        elif variant in ("coarse_mirror", "fine_mirror"):
            # 다뉴조문경 / 다뉴세문경: a decorated mirror back. Concentric
            # rings with two dots in the middle read as a shirt button, so the
            # decoration is a saw-tooth band - coarse or fine - and the two
            # loops sit off-centre where they really are.
            import math

            painter.drawEllipse(QRectF(m + 4, m + 4, s - 2 * m - 8, s - 2 * m - 8))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(155), 2.2))
            teeth = 10 if variant == "coarse_mirror" else 18
            depth = 26.0 if variant == "coarse_mirror" else 16.0
            outer = (s - 2 * m) / 2.0 - 16
            band = QPainterPath()
            for i in range(teeth * 2 + 1):
                angle = math.pi * i / teeth
                radius = outer if i % 2 == 0 else outer - depth
                px, py = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
                if i == 0:
                    band.moveTo(px, py)
                else:
                    band.lineTo(px, py)
            band.closeSubpath()
            painter.drawPath(band)
            painter.setBrush(solid)
            painter.setPen(edge)
            for dx in (-20, 14):
                painter.drawEllipse(QRectF(cx + dx, cy - 12, 22, 22))

        elif variant == "bronze_rattle":
            # 청동방울 (팔주령): eight bells on one disc, so it is drawn as a
            # single eight-lobed outline. Eight separate circles read as a
            # loading spinner.
            import math

            painter.setBrush(solid)
            star = QPainterPath()
            for i in range(8):
                angle = 2.0 * math.pi * i / 8.0
                nxt = 2.0 * math.pi * (i + 1) / 8.0
                mid = (angle + nxt) / 2.0
                lobe = 86.0
                waist = 44.0
                px, py = cx + lobe * math.cos(angle), cy + lobe * math.sin(angle)
                if i == 0:
                    star.moveTo(px, py)
                star.quadTo(cx + waist * math.cos(mid), cy + waist * math.sin(mid),
                            cx + lobe * math.cos(nxt), cy + lobe * math.sin(nxt))
            star.closeSubpath()
            painter.drawPath(star)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(150), 2.6))
            painter.drawEllipse(QRectF(cx - 30, cy - 30, 60, 60))

        elif variant == "bronze_bell":
            # 동탁: a bell with its suspension loop and clapper.
            body.moveTo(cx - 26, top + 40)
            body.lineTo(cx + 26, top + 40)
            body.lineTo(cx + 52, bottom - 30)
            body.quadTo(cx, bottom - 12, cx - 52, bottom - 30)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(int(cx - 22), int(top), 44, 52, 0, 180 * 16)
            painter.setBrush(solid)
            painter.setPen(thin)
            painter.drawEllipse(QRectF(cx - 9, bottom - 26, 18, 18))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 44), int(bottom - 34), int(cx + 44), int(bottom - 34))

        elif variant == "iron_sword":
            # 철검: a long straight blade with guard and grip.
            body.moveTo(cx, top)
            body.lineTo(cx + 15, top + 30)
            body.lineTo(cx + 15, s * 0.66)
            body.lineTo(cx - 15, s * 0.66)
            body.lineTo(cx - 15, top + 30)
            body.closeSubpath()
            painter.drawPath(body)
            painter.drawRect(QRectF(cx - 42, s * 0.66, 84, 14))
            painter.drawRect(QRectF(cx - 13, s * 0.66 + 14, 26, 48))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cx - 22, bottom - 44, 44, 44))
            painter.setPen(thin)
            painter.drawLine(int(cx), int(top + 12), int(cx), int(s * 0.64))

        elif variant == "iron_spearhead":
            # 철모: a leaf blade over a socket.
            body.moveTo(cx, top)
            body.quadTo(cx + 38, s * 0.34, cx + 20, s * 0.60)
            body.lineTo(cx - 20, s * 0.60)
            body.quadTo(cx - 38, s * 0.34, cx, top)
            body.closeSubpath()
            painter.drawPath(body)
            painter.drawRect(QRectF(cx - 20, s * 0.60, 40, bottom - s * 0.60))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx), int(top + 14), int(cx), int(s * 0.58))
            painter.drawLine(int(cx - 18), int(bottom - 22), int(cx + 18), int(bottom - 22))

        elif variant == "iron_arrowhead":
            # 철촉: a narrow head on a long tang.
            body.moveTo(cx, top)
            body.lineTo(cx + 22, s * 0.40)
            body.lineTo(cx + 6, s * 0.46)
            body.lineTo(cx + 6, bottom)
            body.lineTo(cx - 6, bottom)
            body.lineTo(cx - 6, s * 0.46)
            body.lineTo(cx - 22, s * 0.40)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx), int(top + 12), int(cx), int(s * 0.44))

        elif variant == "iron_axe":
            # 철부: a socketed axe with a flaring edge.
            body.moveTo(cx - 30, top + 16)
            body.lineTo(cx + 30, top + 16)
            body.lineTo(cx + 44, bottom - 34)
            body.quadTo(cx, bottom + 4, cx - 44, bottom - 34)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            painter.drawRect(QRectF(cx - 24, top + 22, 48, 34))
            painter.setPen(_pen(color.darker(175), 2.6))
            painter.drawLine(int(cx - 40), int(bottom - 30), int(cx + 40), int(bottom - 30))

        elif variant == "iron_ard":
            # 따비: a forked digging blade on its shaft.
            painter.drawRect(QRectF(cx - 12, top, 24, s * 0.46))
            fork = QPainterPath()
            fork.moveTo(cx - 12, s * 0.46)
            fork.lineTo(cx + 12, s * 0.46)
            fork.lineTo(cx + 46, bottom - 6)
            fork.lineTo(cx + 24, bottom - 6)
            fork.lineTo(cx, s * 0.68)
            fork.lineTo(cx - 24, bottom - 6)
            fork.lineTo(cx - 46, bottom - 6)
            fork.closeSubpath()
            painter.drawPath(fork)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 34), int(top + 26), int(cx + 34), int(top + 26))

        elif variant == "iron_sickle":
            # 낫: a curved blade with its tang.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(150), 9.0))
            blade = QPainterPath()
            blade.moveTo(cx + 62, top + 26)
            blade.quadTo(cx - 4, top + 6, cx - 62, cy + 6)
            blade.quadTo(cx - 20, bottom - 16, cx + 34, bottom - 34)
            painter.drawPath(blade)
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawRect(QRectF(cx + 34, bottom - 46, 46, 20))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            inner = QPainterPath()
            inner.moveTo(cx + 56, top + 36)
            inner.quadTo(cx - 4, top + 20, cx - 50, cy + 6)
            painter.drawPath(inner)

        elif variant == "plate_armour":
            # 판갑: a riveted cuirass, seen from the front.
            body.moveTo(cx - 48, top + 10)
            body.lineTo(cx + 48, top + 10)
            body.quadTo(cx + 72, cy, cx + 56, bottom - 8)
            body.lineTo(cx - 56, bottom - 8)
            body.quadTo(cx - 72, cy, cx - 48, top + 10)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(1, 5):
                y = top + 10 + i * (bottom - top - 18) / 5.0
                painter.drawLine(int(cx - 62), int(y), int(cx + 62), int(y))
            painter.setBrush(solid)
            for i in range(4):
                y = top + 24 + i * (bottom - top - 18) / 5.0
                painter.drawEllipse(QRectF(cx - 56, y, 10, 10))
                painter.drawEllipse(QRectF(cx + 46, y, 10, 10))

        elif variant == "lamellar_armour":
            # 찰갑: small scales laced into a sheet.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(edge)
            painter.drawRect(QRectF(m + 8, m + 8, s - 2 * m - 16, s - 2 * m - 16))
            painter.setPen(thin)
            painter.setBrush(solid)
            cols, rows_n = 5, 5
            w = (s - 2 * m - 36) / cols
            h = (s - 2 * m - 36) / rows_n
            for r in range(rows_n):
                offset = (w / 2.0) if r % 2 else 0.0
                for c in range(cols):
                    x = m + 18 + c * w + offset - (w if offset and c == cols - 1 else 0)
                    scale = QPainterPath()
                    scale.moveTo(x + 2, m + 18 + r * h)
                    scale.lineTo(x + w - 4, m + 18 + r * h)
                    scale.lineTo(x + w - 4, m + 18 + r * h + h * 0.6)
                    scale.quadTo(x + w / 2.0, m + 18 + r * h + h,
                                 x + 2, m + 18 + r * h + h * 0.6)
                    scale.closeSubpath()
                    painter.drawPath(scale)

        elif variant == "horse_bit":
            # 재갈: two cheek rings and the jointed mouthpiece.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(150), 7.0))
            painter.drawEllipse(QRectF(m + 2, cy - 46, 92, 92))
            painter.drawEllipse(QRectF(s - m - 94, cy - 46, 92, 92))
            painter.setPen(_pen(color.darker(150), 8.0))
            painter.drawLine(int(m + 90), int(cy), int(cx + 2), int(cy - 14))
            painter.drawLine(int(cx - 2), int(cy - 14), int(s - m - 90), int(cy))
            painter.setBrush(solid)
            painter.setPen(thin)
            painter.drawEllipse(QRectF(cx - 12, cy - 26, 24, 24))

        elif variant == "stirrup":
            # 등자: the suspension loop over a flat footplate.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(150), 8.0))
            loop = QPainterPath()
            loop.moveTo(cx - 8, top + 6)
            loop.lineTo(cx - 8, top + 34)
            loop.quadTo(cx - 76, cy + 10, cx - 46, bottom - 30)
            loop.lineTo(cx + 46, bottom - 30)
            loop.quadTo(cx + 76, cy + 10, cx + 8, top + 34)
            loop.lineTo(cx + 8, top + 6)
            painter.drawPath(loop)
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawRect(QRectF(cx - 56, bottom - 34, 112, 20))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            painter.drawRect(QRectF(cx - 16, top + 6, 32, 26))

        elif variant == "iron_ingot":
            # 철정: the spade-shaped bar ingot.
            body.moveTo(cx - 52, top + 8)
            body.lineTo(cx + 52, top + 8)
            body.lineTo(cx + 18, top + 52)
            body.lineTo(cx + 18, bottom - 52)
            body.lineTo(cx + 52, bottom - 8)
            body.lineTo(cx - 52, bottom - 8)
            body.lineTo(cx - 18, bottom - 52)
            body.lineTo(cx - 18, top + 52)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx), int(top + 52), int(cx), int(bottom - 52))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean ornaments, tiles and other finds
    # ═══════════════════════════════════════════════════════

    def _draw_korean_ornament(self, painter, s, m, variant, color):
        """
        Ornaments, roof tiles and other finds that identify a Korean site.

        Ornaments are drawn at the scale of the object itself rather than in
        proportion to each other, so a bead and a crown both fill the tile.
        """
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        edge = _pen(color.darker(150), 2.4)
        thin = _pen(color.darker(170), 1.3)
        hollow = Qt.NoBrush   # fill="none": never picked up as the fallback colour
        cx, cy = s / 2.0, s / 2.0
        top, bottom = m + 4, s - m - 4

        painter.setPen(edge)
        painter.setBrush(solid)
        body = QPainterPath()

        if variant == "gogok":
            # 곡옥: a fat perforated head with a tail that hooks back under
            # it. Drawn as a crescent - a head circle alone reads as a bean.
            hx, hy, r = cx - 4, top + 48, 42
            body.moveTo(hx - r, hy)
            body.quadTo(hx - r, hy - r * 1.35, hx + 6, hy - r)
            body.quadTo(hx + r * 1.5, hy - r * 0.5, hx + r * 1.25, hy + r * 0.9)
            body.quadTo(hx + r * 0.95, bottom - 18, hx - r * 0.9, bottom - 10)
            body.quadTo(hx - r * 0.2, bottom - 46, hx + r * 0.35, hy + r * 0.75)
            body.quadTo(hx + r * 0.5, hy + r * 0.1, hx - r, hy)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(hollow)
            painter.drawEllipse(QRectF(hx - 34, hy - 34, 30, 30))

        elif variant == "gwanok":
            # 관옥: tubular beads threaded on a cord.
            painter.setPen(_pen(color.darker(170), 2.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(m + 2), int(cy), int(s - m - 2), int(cy))
            painter.setPen(edge)
            painter.setBrush(solid)
            for i in range(3):
                x = m + 18 + i * 62
                painter.drawRect(QRectF(x, cy - 26, 54, 52))
            # The bore lines inside each bead were six marks nobody could see
            # at map size; the cord through them already says "threaded".

        elif variant == "glass_bead":
            # 유리구슬: a strung line of small round beads.
            painter.setPen(_pen(color.darker(170), 2.2))
            painter.setBrush(Qt.NoBrush)
            cord = QPainterPath()
            cord.moveTo(m, cy - 30)
            cord.quadTo(cx, cy + 56, s - m, cy - 30)
            painter.drawPath(cord)
            painter.setPen(edge)
            painter.setBrush(solid)
            for i in range(6):
                t = i / 5.0
                x = (1 - t) ** 2 * m + 2 * (1 - t) * t * cx + t ** 2 * (s - m)
                y = (1 - t) ** 2 * (cy - 30) + 2 * (1 - t) * t * (cy + 56) + t ** 2 * (cy - 30)
                painter.drawEllipse(QRectF(x - 21, y - 21, 42, 42))

        elif variant == "gold_earring":
            # 금귀걸이: the thick main ring, its link and the drop.
            painter.setBrush(hollow)
            painter.setPen(_pen(color.darker(150), 11.0))
            painter.drawEllipse(QRectF(cx - 54, top + 6, 108, 96))
            painter.setPen(_pen(color.darker(150), 5.0))
            painter.drawEllipse(QRectF(cx - 20, top + 96, 40, 38))
            painter.setPen(edge)
            painter.setBrush(solid)
            drop = QPainterPath()
            drop.moveTo(cx - 30, top + 136)
            drop.lineTo(cx + 30, top + 136)
            drop.quadTo(cx + 24, bottom - 10, cx, bottom)
            drop.quadTo(cx - 24, bottom - 10, cx - 30, top + 136)
            drop.closeSubpath()
            painter.drawPath(drop)

        elif variant == "gold_crown":
            # 금관: the headband with its 出-shaped uprights. The arms have to
            # turn upwards at their ends - drawn straight they read as
            # scaffolding rather than a crown.
            painter.setBrush(solid)
            painter.drawRect(QRectF(m + 2, bottom - 44, s - 2 * m - 4, 32))
            painter.setPen(_pen(color.darker(150), 7.0))
            painter.setBrush(Qt.NoBrush)
            for offset, height in ((-64, 118), (0, 146), (64, 118)):
                stem = cx + offset
                foot = bottom - 44
                upright = QPainterPath()
                upright.moveTo(stem, foot)
                upright.lineTo(stem, foot - height)
                painter.drawPath(upright)
                for step, reach in enumerate((26, 20)):
                    arm_y = foot - 44 - step * 40
                    if arm_y < foot - height:
                        continue
                    for side in (-1, 1):
                        arm = QPainterPath()
                        arm.moveTo(stem, arm_y)
                        arm.lineTo(stem + side * reach, arm_y)
                        arm.lineTo(stem + side * reach, arm_y - 24)
                        painter.drawPath(arm)
            painter.setPen(thin)
            painter.setBrush(solid)
            for offset in (-86, -18, 50):
                painter.drawEllipse(QRectF(cx + offset, bottom - 6, 18, 18))

        elif variant == "belt_fitting":
            # 대금구: the buckle, the strap plates and a pendant.
            painter.setBrush(solid)
            painter.drawRect(QRectF(m + 2, cy - 34, 68, 68))
            painter.setBrush(hollow)
            painter.setPen(thin)
            painter.drawRect(QRectF(m + 16, cy - 20, 40, 40))
            painter.setPen(edge)
            painter.setBrush(solid)
            for i in range(2):
                painter.drawRect(QRectF(m + 82 + i * 60, cy - 30, 52, 60))
            painter.setPen(_pen(color.darker(150), 4.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(m + 108), int(cy + 30), int(m + 108), int(cy + 60))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawEllipse(QRectF(m + 90, cy + 58, 36, 36))

        elif variant == "mokgan":
            # 목간: an inked wooden slip, notched for binding.
            body.moveTo(cx - 34, top)
            body.lineTo(cx + 34, top)
            body.lineTo(cx + 34, cy - 26)
            body.lineTo(cx + 24, cy - 14)
            body.lineTo(cx + 34, cy - 2)
            body.lineTo(cx + 34, bottom - 20)
            body.lineTo(cx, bottom)
            body.lineTo(cx - 34, bottom - 20)
            body.lineTo(cx - 34, cy - 2)
            body.lineTo(cx - 24, cy - 14)
            body.lineTo(cx - 34, cy - 26)
            body.closeSubpath()
            painter.drawPath(body)
            # A single column of ink down the slip. The five crossed marks
            # that used to stand in for writing read as plus signs.
            _clip_detail(painter, body)
            painter.setPen(_pen(color.darker(190), 4.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx), int(top + 26), int(cx), int(bottom - 34))
            painter.restore()

        elif variant == "round_roof_tile":
            # 수막새: the round tile face, lotus-petalled.
            import math
            painter.drawEllipse(QRectF(m + 2, m + 2, s - 2 * m - 4, s - 2 * m - 4))
            painter.setPen(thin)
            painter.setBrush(hollow)
            radius = (s - 2 * m) / 2.0 - 12
            for i in range(8):
                angle = 2.0 * math.pi * i / 8.0
                px = cx + radius * 0.62 * math.cos(angle)
                py = cy + radius * 0.62 * math.sin(angle)
                painter.drawEllipse(QRectF(px - 26, py - 20, 52, 40))
            painter.setBrush(solid)
            painter.setPen(edge)
            painter.drawEllipse(QRectF(cx - 22, cy - 22, 44, 44))

        elif variant == "eaves_roof_tile":
            # 암막새: the decorated eaves face over the curved tile.
            painter.setBrush(solid)
            face = QPainterPath()
            face.moveTo(m + 2, cy - 6)
            face.lineTo(s - m - 2, cy - 6)
            face.lineTo(s - m - 2, cy + 46)
            face.quadTo(cx, cy + 74, m + 2, cy + 46)
            face.closeSubpath()
            painter.drawPath(face)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 130))
            tile = QPainterPath()
            tile.moveTo(m + 22, cy - 6)
            tile.quadTo(cx, top - 8, s - m - 22, cy - 6)
            tile.quadTo(cx, cy - 42, m + 22, cy - 6)
            tile.closeSubpath()
            painter.drawPath(tile)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(3):
                x = cx - 58 + i * 58
                scroll = QPainterPath()
                scroll.moveTo(x - 20, cy + 34)
                scroll.quadTo(x, cy + 2, x + 20, cy + 34)
                painter.drawPath(scroll)

        elif variant == "floor_brick":
            # 전돌: a square brick with its stamped panel.
            body.moveTo(m + 20, m + 8)
            body.lineTo(s - m - 2, m + 26)
            body.lineTo(s - m - 20, s - m - 8)
            body.lineTo(m + 2, s - m - 26)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(hollow)
            inner = QPainterPath()
            inner.moveTo(m + 42, m + 34)
            inner.lineTo(s - m - 26, m + 48)
            inner.lineTo(s - m - 42, s - m - 34)
            inner.lineTo(m + 26, s - m - 48)
            inner.closeSubpath()
            painter.drawPath(inner)
            painter.drawLine(int(m + 34), int(cy - 4), int(s - m - 34), int(cy + 8))
            painter.drawLine(int(cx - 8), int(m + 22), int(cx + 8), int(s - m - 22))

        elif variant == "inkstone":
            # 벼루: the grinding surface, its water well and the foot.
            painter.drawRect(QRectF(m + 2, cy - 54, s - 2 * m - 4, 88))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 150))
            painter.drawRect(QRectF(m + 22, bottom - 40, s - 2 * m - 44, 26))
            painter.setPen(thin)
            painter.setBrush(hollow)
            painter.drawRect(QRectF(m + 18, cy - 40, s - 2 * m - 36, 60))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 235))
            painter.drawEllipse(QRectF(cx + 26, cy - 32, 56, 44))

        elif variant == "clay_figurine":
            # 토우: a simple modelled figure.
            painter.drawEllipse(QRectF(cx - 26, top + 2, 52, 52))
            trunk = QPainterPath()
            trunk.moveTo(cx - 30, top + 58)
            trunk.lineTo(cx + 30, top + 58)
            trunk.quadTo(cx + 44, cy + 40, cx + 26, bottom)
            trunk.lineTo(cx - 26, bottom)
            trunk.quadTo(cx - 44, cy + 40, cx - 30, top + 58)
            trunk.closeSubpath()
            painter.drawPath(trunk)
            painter.setPen(_pen(color.darker(150), 9.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(int(cx - 30), int(top + 74), int(cx - 72), int(cy + 26))
            painter.drawLine(int(cx + 30), int(top + 74), int(cx + 72), int(cy + 26))
            painter.setPen(thin)
            painter.drawLine(int(cx - 12), int(top + 26), int(cx - 4), int(top + 26))
            painter.drawLine(int(cx + 4), int(top + 26), int(cx + 12), int(top + 26))

        elif variant == "chimi":
            # 치미: the ridge-end ornament, ribbed like a tail.
            body.moveTo(cx - 34, bottom)
            body.lineTo(cx + 34, bottom)
            body.quadTo(cx + 52, cy, cx + 30, top + 30)
            body.quadTo(cx + 14, top - 2, cx - 30, top + 16)
            body.quadTo(cx - 62, cy - 20, cx - 34, bottom)
            body.closeSubpath()
            painter.drawPath(body)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for i in range(4):
                rib = QPainterPath()
                rib.moveTo(cx - 26 + i * 6, bottom - 20)
                rib.quadTo(cx - 34 + i * 16, cy - 10, cx - 16 + i * 14, top + 26)
                painter.drawPath(rib)

        elif variant == "foundation_stone":
            # 초석: the base stone with its column seat, in plan.
            painter.drawRect(QRectF(m + 2, m + 2, s - 2 * m - 4, s - 2 * m - 4))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 150))
            painter.setPen(thin)
            painter.drawEllipse(QRectF(m + 32, m + 32, s - 2 * m - 64, s - 2 * m - 64))
            painter.setBrush(solid)
            painter.setPen(edge)
            painter.drawEllipse(QRectF(cx - 34, cy - 34, 68, 68))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cx - 20, cy - 20, 40, 40))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

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
        painter.setBrush(Qt.NoBrush)
        ew, eh = 22, 20
        painter.drawEllipse(int(cx - 28), int(s * 0.32), ew, eh)
        painter.drawEllipse(int(cx + 6), int(s * 0.32), ew, eh)

    def _draw_burial(self, painter, s, m, color):
        """Burial — body outline (flexed position)."""
        painter.setBrush(Qt.NoBrush)
        pen = _pen(color, 3.0)
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
        # Top layer, lightened through opacity rather than a second colour:
        # QGIS gives every param(fill) the same value, so a lighter colour
        # would vanish the moment the symbol is recoloured.
        old_brush = painter.brush()
        base = old_brush.color()
        painter.setBrush(QColor(base.red(), base.green(), base.blue(), 110))
        p2 = QPainterPath()
        p2.moveTo(m + 30, s - m - 30)
        p2.quadTo(s/2, s * 0.35, s - m - 30, s - m - 30)
        p2.closeSubpath()
        painter.drawPath(p2)
        painter.setBrush(old_brush)
        old_pen = painter.pen()
        stipple_pen = _pen(old_pen.color().darker(135), 1.0)
        painter.setPen(stipple_pen)
        for i in range(14):
            x = int(m + 20 + (i * 14))
            y = int(s - m - 16 - ((i % 3) * 9))
            painter.drawEllipse(x, y, 4, 3)
        painter.setPen(old_pen)

    def _draw_ditch(self, painter, s, m, color):
        """Ditch/moat — concentric dashed arcs."""
        painter.setBrush(Qt.NoBrush)
        pen = _pen(color, 3.0, Qt.DashLine)
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
        cx = s / 2.0
        # A channel is two banks and the direction of flow. Three small
        # arrowheads down the middle just filled it with clutter.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color.darker(120), 2.6))
        painter.drawArc(m + 18, m + 30, s - 2 * m - 36, s - 2 * m - 60, 40 * 16, 270 * 16)
        painter.drawArc(m + 36, m + 48, s - 2 * m - 72, s - 2 * m - 96, 40 * 16, 270 * 16)
        painter.setPen(_pen(color.darker(150), 3.0))
        arrow = QPainterPath()
        arrow.moveTo(cx - 24, s / 2.0)
        arrow.lineTo(cx + 22, s / 2.0)
        arrow.moveTo(cx + 8, s / 2.0 - 13)
        arrow.lineTo(cx + 22, s / 2.0)
        arrow.lineTo(cx + 8, s / 2.0 + 13)
        painter.drawPath(arrow)
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
        peck_pen = _pen(color.darker(145), 1.0)
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
        import math

        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color, 4.0))
        cx, cy = s / 2.0, s / 2.0
        # One path, four quarter-turns per revolution. Eighty line segments
        # drew the same curve but as eighty marks, which is what made it
        # scratchy - and the joins showed at map size.
        spiral = QPainterPath()
        outer = s / 2.0 - m - 8
        steps = 11
        spiral.moveTo(cx, cy)
        for i in range(1, steps + 1):
            a0 = (i - 1) * math.pi / 2.0
            a1 = i * math.pi / 2.0
            r0 = 6 + (i - 1) / steps * outer
            r1 = 6 + i / steps * outer
            rm = (r0 + r1) / 2.0 * 1.22
            am = (a0 + a1) / 2.0
            spiral.quadTo(cx + rm * math.cos(am), cy + rm * math.sin(am),
                          cx + r1 * math.cos(a1), cy + r1 * math.sin(a1))
        painter.drawPath(spiral)

    def _draw_ash_layer(self, painter, s, m, color):
        """Ash layer as horizontal banding with dense stipple."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        band_h = int((s - 2 * m) * 0.55)
        top = int(s / 2 - band_h / 2)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 85))
        painter.drawRect(m + 14, top, s - 2 * m - 28, band_h)
        # Two partings read as bedding; nine rules and a stipple field read as
        # a barcode once the symbol is map-sized.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color.darker(150), 1.4))
        for fraction in (0.36, 0.68):
            y = int(top + band_h * fraction)
            painter.drawLine(m + 22, y, s - m - 22, y)
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
        # A darker core inside the scorched outline says "burnt" more
        # plainly than a field of char marks.
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 205))
        painter.setPen(_pen(color.darker(160), 1.2))
        core = QPainterPath()
        core.moveTo(m + 62, s * 0.66)
        core.quadTo(s * 0.40, s * 0.44, s * 0.54, m + 62)
        core.quadTo(s * 0.68, s * 0.52, s * 0.56, s * 0.74)
        core.quadTo(s * 0.44, s * 0.80, m + 62, s * 0.66)
        core.closeSubpath()
        painter.drawPath(core)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Survey / General
    # ═══════════════════════════════════════════════════════

    def _draw_excavation(self, painter, s, m, color):
        """Excavation area — square with grid lines."""
        painter.drawRect(m + 15, m + 15, s - 2*m - 30, s - 2*m - 30)
        painter.setBrush(Qt.NoBrush)
        pen = _pen(color.darker(130), 1.5, Qt.DotLine)
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
        n_pen = _pen(color.darker(150), 1.6)
        painter.setPen(n_pen)
        nx = s - m - 34
        ny = m + 24
        painter.drawLine(nx, ny + 16, nx, ny - 10)
        painter.drawLine(nx, ny - 10, nx - 5, ny - 3)
        painter.drawLine(nx, ny - 10, nx + 5, ny - 3)
        painter.setPen(old_pen)

    def _draw_north_arrow(self, painter, s, m, color):
        """Map-style north arrow used in archaeological figures."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setPen(_pen(color.darker(165), 2.0))
        painter.setBrush(color)
        cx = s / 2.0
        arrow = QPainterPath()
        arrow.moveTo(cx, m + 12)
        arrow.lineTo(cx + 34, s - m - 56)
        arrow.lineTo(cx + 10, s - m - 56)
        arrow.lineTo(cx + 10, s - m - 16)
        arrow.lineTo(cx - 10, s - m - 16)
        arrow.lineTo(cx - 10, s - m - 56)
        arrow.lineTo(cx - 34, s - m - 56)
        arrow.closeSubpath()
        painter.drawPath(arrow)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color.darker(185), 1.6))
        painter.drawText(int(cx - 10), m + 26, "N")
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_scale_bar(self, painter, s, m, color):
        """Segmented scale bar convention for map figures."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setPen(_pen(color.darker(160), 1.8))
        seg_w = 34
        bar_h = 18
        x0 = int(s / 2 - (seg_w * 2))
        y0 = int(s * 0.54)
        for i in range(4):
            if i % 2 == 0:
                painter.setBrush(color)
            else:
                painter.setBrush(Qt.NoBrush)
            painter.drawRect(x0 + i * seg_w, y0, seg_w, bar_h)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(x0, y0 + bar_h + 2, x0 + seg_w * 4, y0 + bar_h + 2)
        for i in range(5):
            tx = x0 + i * seg_w
            painter.drawLine(tx, y0 + bar_h + 2, tx, y0 + bar_h + 9)
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_harris_matrix_context(self, painter, s, m, color):
        """Simplified Harris matrix context box + relation connectors."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setPen(_pen(color.darker(160), 1.8))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 70))

        top = QRectF(m + 40, m + 28, s - 2 * m - 80, 36)
        mid = QRectF(m + 26, s * 0.44, s - 2 * m - 52, 40)
        bot_l = QRectF(m + 22, s - m - 56, 72, 32)
        bot_r = QRectF(s - m - 94, s - m - 56, 72, 32)
        painter.drawRect(top)
        painter.drawRect(mid)
        painter.drawRect(bot_l)
        painter.drawRect(bot_r)

        painter.setBrush(Qt.NoBrush)
        painter.drawLine(int(top.center().x()), int(top.bottom()), int(mid.center().x()), int(mid.top()))
        painter.drawLine(int(mid.left() + 24), int(mid.bottom()), int(bot_l.center().x()), int(bot_l.top()))
        painter.drawLine(int(mid.right() - 24), int(mid.bottom()), int(bot_r.center().x()), int(bot_r.top()))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_stratigraphic_unit(self, painter, s, m, color):
        """Layered context symbol inspired by section stratigraphy notation."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setPen(_pen(color.darker(160), 1.5))
        x = m + 16
        y = m + 24
        w = s - 2 * m - 32
        h = s - 2 * m - 48
        layers = 4
        for i in range(layers):
            top = int(y + i * (h / layers))
            lh = int(h / layers)
            shade = 70 + (i * 35)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), min(190, shade)))
            painter.drawRect(x, top, w, lh)
            painter.setPen(_pen(color.darker(170), 1.0))
            painter.drawLine(x + 8, top + lh - 4, x + w - 8, top + lh - 10)
            painter.setPen(_pen(color.darker(160), 1.5))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_survey_point(self, painter, s, m, color):
        """Survey point — crosshair with circle."""
        cx, cy = s/2, s/2
        r = s/2 - m - 20
        painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color.darker(130), 2.0))
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
        painter.setBrush(Qt.NoBrush)
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
        hatch_pen = _pen(color.darker(145), 1.0, Qt.DashLine)
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
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(int(cx - 8), int(s / 2 - 8), 16, 16)
        painter.setBrush(Qt.NoBrush)
        x_pen = _pen(color.darker(145), 1.4)
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
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(int(s / 2 - 16), int(body_y + 12), 32, 32)
        painter.drawRect(int(body_x + 8), int(body_y - 10), 18, 10)
        cone_pen = _pen(color.darker(150), 1.2, Qt.DotLine)
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
        grid_pen = _pen(color.darker(145), 3.0)
        painter.setPen(grid_pen)
        x0 = m + 26
        y0 = s - m - 26
        painter.drawLine(x0, y0, x0 + 120, y0)
        painter.drawLine(x0, y0, x0, y0 - 120)
        tick_pen = _pen(color.darker(150), 1.3)
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
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(int(cx - 9), int(m + 28), 18, 18)
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_road(self, painter, s, m, color):
        """Road/pavement with carriageway edges and center line."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        painter.setBrush(Qt.NoBrush)
        edge_pen = _pen(color.darker(130), 2.2)
        painter.setPen(edge_pen)
        painter.drawArc(m + 12, m + 34, s - 2 * m - 24, s - 2 * m - 68, 25 * 16, 310 * 16)
        painter.drawArc(m + 40, m + 56, s - 2 * m - 80, s - 2 * m - 112, 25 * 16, 310 * 16)
        center_pen = _pen(color.darker(150), 1.3, Qt.DashLine)
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
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(m + 28, deck_y + 8, int((s - 2 * m - 56) / 2), 80, 0, 180 * 16)
        painter.drawArc(int(s / 2), deck_y + 8, int((s - 2 * m - 56) / 2), 80, 0, 180 * 16)
        painter.setBrush(Qt.NoBrush)
        water_pen = _pen(color.darker(145), 1.1, Qt.DotLine)
        painter.setPen(water_pen)
        painter.drawLine(m + 24, int(s * 0.78), s - m - 24, int(s * 0.78))
        painter.drawLine(m + 30, int(s * 0.84), s - m - 30, int(s * 0.84))
        painter.setBrush(old_brush)
        painter.setPen(old_pen)

    def _draw_terrace(self, painter, s, m, color):
        """Terrace with stepped contour bands."""
        old_pen = painter.pen()
        old_brush = painter.brush()
        # The stepped profile is the type; a stack of rules with ticks is not.
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 120))
        painter.setPen(_pen(color.darker(140), 2.6))
        steps = QPainterPath()
        steps.moveTo(m + 4, s - m - 8)
        for i in range(3):
            y = s - m - 8 - i * 46
            steps.lineTo(m + 30 + i * 54, y)
            steps.lineTo(m + 30 + i * 54, y - 46)
        steps.lineTo(s - m - 4, m + 12)
        steps.lineTo(s - m - 4, s - m - 8)
        steps.closeSubpath()
        painter.drawPath(steps)
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
        ring_pen = _pen(color.darker(125), 2.2, Qt.DashLine)
        painter.setPen(ring_pen)
        painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 245))
        painter.setPen(_pen(color.darker(150), 1.0))
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
        pit_pen = _pen(color.darker(130), 2.0, Qt.DashLine)
        painter.setPen(pit_pen)
        painter.drawRect(x, y, w, w)
        painter.setBrush(Qt.NoBrush)
        cross_pen = _pen(color.darker(145), 1.4)
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
