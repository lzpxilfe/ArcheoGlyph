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
from . import icon_grid, template_catalog


# -- House style -----------------------------------------------------------
#
# These symbols are markers on a map, read at 5-10 mm. At that size a hairline
# disappears and a mitred corner turns into a speck, which is what makes a
# drawing look scratchy rather than drawn. Every stroke in this module goes
# through _pen() so the whole catalogue is drawn in one hand.

#: One grid unit at the 256px canvas the templates are painted on.
_UNIT = 256.0 / icon_grid.UNITS

DETAIL_WIDTH = icon_grid.DETAIL * _UNIT       # internal lines: 2 units
OUTLINE_WIDTH = icon_grid.OUTLINE * _UNIT     # the silhouette: 3 units


def _weight(width):
    """Lift a requested stroke width onto the grid's two steps."""
    width = float(width)
    if width <= 2.0:
        return DETAIL_WIDTH
    if width <= 3.4:
        return OUTLINE_WIDTH
    return width * 1.3      # a deliberately heavy stroke stays heavy


class _SoftPainter:
    """
    Draws every rectangle with the house corner radius.

    Eighty-six of the shapes in this file are rectangles, and a sharp-cornered
    rectangle is what makes an icon look like a diagram rather than a drawn
    object. Rounding them here softens the whole set at once, instead of 59
    methods each picking a radius - and the radius is clamped to a quarter of
    the shorter side, so a scale-bar segment stays a segment while a tomb
    chamber gets a proper corner.

    Everything else is forwarded untouched.
    """

    def __init__(self, painter, grid):
        self._painter = painter
        self._grid = grid

    def __getattr__(self, name):
        return getattr(self._painter, name)

    def drawRect(self, *args):
        if len(args) == 1:
            rect = args[0]
            x, y = rect.left(), rect.top()
            w, h = rect.width(), rect.height()
        else:
            x, y, w, h = (float(value) for value in args[:4])
        unit = self._grid.unit
        radius = min(icon_grid.RADIUS * unit, abs(w) / 4.0, abs(h) / 4.0)
        self._painter.drawPath(_rounded_rect_path(x, y, w, h, radius))


def _rounded_rect_path(x, y, w, h, r):
    """A rounded rectangle as one path, in canvas pixels."""
    x1, y1 = x + w, y + h
    path = QPainterPath()
    path.moveTo(x + r, y)
    path.lineTo(x1 - r, y)
    path.quadTo(x1, y, x1, y + r)
    path.lineTo(x1, y1 - r)
    path.quadTo(x1, y1, x1 - r, y1)
    path.lineTo(x + r, y1)
    path.quadTo(x, y1, x, y1 - r)
    path.lineTo(x, y + r)
    path.quadTo(x, y, x + r, y)
    path.closeSubpath()
    return path


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
        method(_SoftPainter(painter, icon_grid.Grid(size)), size, m, *args)


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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))

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
        """
        The bronze dagger series, as one profile with ten sets of numbers.

        Every blade here is the same ``symmetric`` call - half-widths read down
        the blade in grid units - so the ten types share a tip angle, a tang
        and a margin without anyone matching them by hand. That shared
        skeleton is what makes a typology series look like a series.
        """
        g = icon_grid.Grid(s)
        # y positions down the blade, in grid units
        stations = (6, 12, 20, 33, 44, 52, 58)
        profiles = {
            "liaoning": (0, 2.5, 8, 4, 7.5, 3, 2),     # 비파형: the lute waist,
                                                        # pinched hard so the type reads
            "ordos":    (0, 2.5, 6, 4, 5, 3, 2),
            "antenna":  (0, 2.5, 6, 3.5, 5, 3, 2),
            "slender":  (0, 2, 4, 3, 3.5, 2.5, 1.5),    # 세형: narrow throughout
            "tao":      (0, 2, 4.5, 3.5, 3, 2, 1.5),
            "medium":   (0, 2.5, 5, 3.5, 4, 2.5, 2),
            "flat":     (0, 4, 6, 6, 5, 3, 2),          # 평인: no waist at all
            "type_ia":  (0, 3, 6.5, 5, 6.5, 3, 2),
            "type_ib":  (0, 2.5, 6, 4.5, 6.5, 3.5, 2),
            "other":    (0, 2, 4.5, 3, 3.5, 2.5, 1.5),
        }
        widths = profiles.get(variant, profiles["other"])
        blade = g.symmetric(list(zip(widths, stations)))
        painter.drawPath(blade)

        old_pen = painter.pen()
        ridge_pen = _pen(old_pen.color().darker(135), 1.20)
        _clip_detail(painter, blade)
        painter.setPen(ridge_pen)
        painter.drawPath(g.line(32, 8, 32, 58))
        painter.restore()
        painter.setPen(ridge_pen)

        # What separates the types beyond the profile: a guard, a hilt band,
        # or the antenna finials.
        if variant == "flat":
            painter.drawPath(g.line(27, 22, 37, 22))
        elif variant == "antenna":
            painter.drawPath(g.line(24, 48, 29, 48))
            painter.drawPath(g.line(35, 48, 40, 48))
            painter.setBrush(color)
            painter.drawPath(g.circle(23, 48, 2))
            painter.drawPath(g.circle(41, 48, 2))
        elif variant == "liaoning":
            painter.drawPath(g.line(28, 43, 36, 43))
        elif variant == "type_ia":
            painter.drawPath(g.line(27, 41, 37, 41))
        elif variant == "type_ib":
            painter.drawPath(g.line(28, 39, 36, 39))
            painter.drawPath(g.line(27, 43, 37, 43))

        painter.setPen(old_pen)

    def _draw_projectile_point_typology(self, painter, s, m, variant):
        """
        The projectile point series, built the same way as the daggers.

        The types differ only in how the base is worked - notched, stemmed or
        left straight - so everything above the base is deliberately identical
        across them.
        """
        g = icon_grid.Grid(s)
        shapes = {
            #        (half width, y) down the point
            "leaf":           ((0, 7), (6, 20), (7, 32), (4, 48), (1.5, 56), (0, 58)),
            "side_notched":   ((0, 7), (5.5, 19), (6.5, 30), (4, 35), (5, 44),
                               (2, 48), (2, 57)),
            "corner_notched": ((0, 7), (5, 19), (6, 30), (4.5, 36), (1.5, 42),
                               (2, 50), (2, 57)),
            "stemmed":        ((0, 7), (5.5, 20), (6, 34), (3, 42), (2, 44), (2, 57)),
            "triangular":     ((0, 7), (6.5, 26), (5.5, 46), (2.5, 50), (1.5, 57)),
        }
        head = g.symmetric(list(shapes.get(variant, shapes["leaf"])))
        painter.drawPath(head)

        # The midrib belongs inside the head, not running out through its tip.
        old_pen = painter.pen()
        _clip_detail(painter, head)
        painter.setPen(_pen(old_pen.color().darker(135), 1.1))
        painter.drawPath(g.line(32, 9, 32, 56))
        painter.restore()
        painter.setPen(old_pen)

    def _draw_keyhole_tomb(self, painter, s, m, variant, color):
        """
        The keyhole tomb, as one outline shared by every variant.

        The reference plates make the point: the variants are the same
        silhouette in different colours, with at most a mark added. Giving
        each one its own tail width, as this used to, only blurred the family.
        """
        g = icon_grid.Grid(s)
        mound = g.keyhole(head_cy=22, head_r=13, join_y=30, foot_half=14, foot_y=57)

        if variant in ("moat", "makinokuchi"):
            # 주호: the ditch ringing the mound, drawn as the same outline
            # standing off it.
            old_pen, old_brush = painter.pen(), painter.brush()
            painter.setPen(_pen(color.lighter(135), 3.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(
                g.keyhole(head_cy=22, head_r=17, join_y=32, foot_half=18, foot_y=59)
            )
            painter.setBrush(old_brush)
            painter.setPen(old_pen)

        painter.drawPath(mound)

        if variant == "fukiishi":
            # 즙석: the stone facing, as one band across the mound.
            _clip_detail(painter, mound)
            painter.setPen(_pen(color.darker(150), 1.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(14, 40, 50, 40))
            painter.restore()
        elif variant in ("tsumishizuka", "makinokuchi"):
            _clip_detail(painter, mound)
            painter.setPen(_pen(color.darker(150), 1.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(16, 38, 48, 38))
            painter.drawPath(g.line(14, 46, 50, 46))
            painter.restore()
        elif variant == "stepped":
            # 단축: the mound built in tiers. A second, concentric outline
            # says that better than more bands, and keeps it apart from the
            # 즙석 and 적석총 marks, which are bands.
            _clip_detail(painter, mound)
            painter.setPen(_pen(color.darker(150), 1.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(
                g.keyhole(head_cy=23, head_r=8.5, join_y=29,
                          foot_half=9, foot_y=51)
            )
            painter.restore()

    def _draw_kofun_shape(self, painter, s, m, variant, color):
        """
        The kofun plan series: ten silhouettes on one grid.

        These are pure plan outlines - a circle, a square, a keyhole, a
        scallop - which is exactly the case where the reference adds no
        internal detail at all. What separates them is the shape.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(color)

        if variant in ("zenpokouen", "makimuku_en"):
            painter.drawPath(g.keyhole(head_cy=22, head_r=13, join_y=30,
                                       foot_half=14, foot_y=57))
        elif variant == "enpun":                        # 원분
            painter.drawPath(g.circle(32, 32, 25))
        elif variant == "hofun":                        # 방분
            painter.drawPath(g.rect(9, 9, 46, 46))
        elif variant == "hotategai":                    # 가리비형: short front
            painter.drawPath(g.keyhole(head_cy=26, head_r=17, join_y=40,
                                       foot_half=13, foot_y=56))
        elif variant in ("zenpokoho", "makimuku_ho"):
            # 전방후방분: the same mound as 전방후원분 with a square rear, so
            # the two read as a pair. The numbers are the keyhole's, which is
            # what puts the shoulder in the same place in both.
            painter.drawPath(g.poly([(19, 9), (45, 9), (45, 28), (42, 30),
                                     (46, 57), (18, 57), (22, 30), (19, 28)]))
        elif variant == "sohochuen":                    # 쌍방중원분
            painter.drawPath(g.spindle(r=16, waist_half=10, foot_half=7,
                                       top_y=7, bottom_y=57))
        elif variant == "yosumi":                       # 사우돌출형
            # The corners are the whole point of the type, so the sides have
            # to fall in between them. Sampling a circle, as this used to,
            # only produced an octagon with nothing protruding.
            painter.drawPath(g.poly([
                (7, 7), (32, 14), (57, 7), (50, 32),
                (57, 57), (32, 50), (7, 57), (14, 32),
            ]))
        elif variant == "daijobo":                      # 대상묘: a low platform
            painter.drawPath(g.rect(7, 20, 50, 24))
        else:
            painter.drawPath(g.circle(32, 32, 25))

        # 마키무쿠형 is the one that carries a mark: the terraces on the front.
        if variant in ("makimuku_en", "makimuku_ho"):
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color.darker(175), 1.2))
            painter.drawPath(g.line(20, 44, 44, 44))
            painter.drawPath(g.line(18, 50, 46, 50))

        painter.setBrush(old_brush)
        painter.setPen(old_pen)

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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
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
        Korean burial types, as the schematic each one is recognised by.

        Section view for the dolmens and the mounds, plan view for the
        chambers - the same split an excavation report uses. The section
        types all share one ground line at y=50 and the plan types all sit in
        the same box, so the family lines up instead of each schematic
        picking its own horizon.

        Detail is held to what separates a type from its neighbour. The
        earlier drawings answered "piled stones" with twenty small circles
        and "timber" with a stack of rules; at a 5-10 mm marker both turn
        into a smudge, which is the whole reason this catalogue is being
        rebuilt on a grid.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        ground_tone = QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT)
        edge = _pen(color, 2.6)
        thin = _pen(color, 1.4)
        dashed = _pen(color, 1.8, Qt.DashLine)
        GROUND = 50

        painter.setPen(edge)
        painter.setBrush(solid)

        if variant == "table":
            # 탁자식: the chamber stands clear of the ground, so the legs are
            # tall and the capstone is a thin slab.
            painter.drawPath(g.rect(6, 13, 52, 8))
            painter.setBrush(body)
            painter.drawPath(g.rect(17, 21, 8, GROUND - 21))
            painter.drawPath(g.rect(39, 21, 8, GROUND - 21))
            painter.setPen(thin)
            painter.drawPath(g.line(5, GROUND, 59, GROUND))

        elif variant == "go_board":
            # 기반식: the same capstone, thick and domed, on short supports.
            # Against 탁자식 the difference is leg height, which is the
            # difference in the field.
            painter.drawPath(g.symmetric([(5, 17), (20, 24), (27, 33)], curved=True))
            painter.setBrush(body)
            for x in (16, 28, 40):
                painter.drawPath(g.rect(x, 33, 8, GROUND - 33))
            painter.setPen(thin)
            painter.drawPath(g.line(5, GROUND, 59, GROUND))

        elif variant == "capstone":
            # 개석식: the capstone lies on the ground and the cist is buried,
            # so the buried half is the dashed one.
            painter.drawPath(g.symmetric([(6, 22), (22, 28), (28, 34)], curved=True))
            painter.setPen(thin)
            painter.drawPath(g.line(5, 34, 59, 34))
            painter.setPen(dashed)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.rect(21, 39, 22, 13))

        elif variant == "stone_cist":
            # 석관묘: four slabs set on edge. The open corners are what says
            # slabs rather than a built wall.
            painter.setBrush(body)
            painter.drawPath(g.rect(18, 11, 28, 7))
            painter.drawPath(g.rect(18, 46, 28, 7))
            painter.drawPath(g.rect(11, 18, 7, 28))
            painter.drawPath(g.rect(46, 18, 7, 28))

        elif variant == "stone_lined":
            # 석곽묘: a wall built of piled stone, so it is a continuous band
            # rather than four slabs - drawn as one thick outline with its
            # courses ticked, not as a ring of twenty pebbles.
            painter.setBrush(body)
            painter.drawPath(g.rect(9, 12, 46, 40))
            painter.setBrush(solid)
            painter.drawPath(g.rect(17, 20, 30, 24))

        elif variant == "wooden_coffin":
            # 목관묘: the grave pit dashed, one timber coffin inside it.
            painter.setBrush(ground_tone)
            painter.setPen(dashed)
            painter.drawPath(g.rect(9, 10, 46, 44))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.rect(21, 20, 22, 24))

        elif variant == "wooden_chamber":
            # 목곽묘: the same pit with a chamber around the coffin. One ring
            # more than 목관묘 - which is exactly the distinction.
            painter.setBrush(ground_tone)
            painter.setPen(dashed)
            painter.drawPath(g.rect(9, 10, 46, 44))
            painter.setPen(edge)
            painter.setBrush(body)
            painter.drawPath(g.rect(16, 17, 32, 30))
            painter.setBrush(solid)
            painter.drawPath(g.rect(25, 26, 14, 12))

        elif variant == "jar_coffin":
            # 옹관묘: two jars set mouth to mouth. Drawn upright rather than
            # laid down, because at 64 units a horizontal pair reads as one
            # bean and an upright pair reads as two pots.
            painter.setBrush(body)
            # Two jars of the same size read as one peanut. A small lid jar
            # over a large body jar is both what 합구식 옹관 actually is and
            # what makes the pair legible at marker size.
            painter.drawPath(g.symmetric(
                [(5, 10), (10, 15), (11, 23), (11, 30)], curved=True))
            painter.drawPath(g.symmetric(
                [(11, 32), (15, 39), (14, 49), (6, 55)], curved=True))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(20, 31, 44, 31))

        elif variant == "stone_mound_chamber":
            # 적석목곽분: a stone pile heaped over a timber chamber. Three
            # stones say pile; fifteen said noise.
            painter.setBrush(ground_tone)
            painter.drawPath(g.symmetric([(4, 16), (17, 28), (28, GROUND)],
                                         curved=True))
            painter.setBrush(solid)
            painter.setPen(thin)
            for cx, cy in ((22, 33), (32, 29), (42, 33)):
                painter.drawPath(g.circle(cx, cy, 4.5))
            painter.setPen(edge)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.rect(23, 39, 18, 11))

        elif variant == "corridor_chamber":
            # 횡혈식석실분: the chamber and the passage that reaches it, in
            # plan, inside the mound.
            painter.setBrush(ground_tone)
            painter.setPen(dashed)
            painter.drawPath(g.circle(32, 32, 26))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.rect(22, 15, 20, 19))
            painter.setBrush(body)
            painter.drawPath(g.rect(28, 34, 8, 17))

        elif variant == "earthen_mound":
            # 봉토분: a plain earthen mound, built up in tiers. No chamber
            # and no stones, which is what tells it from 적석목곽분.
            mound = g.symmetric([(4, 16), (17, 28), (28, GROUND)], curved=True)
            painter.setBrush(body)
            painter.drawPath(mound)
            _clip_detail(painter, mound)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(12, 38, 52, 38))
            painter.drawPath(g.line(18, 29, 46, 29))
            painter.restore()

        elif variant == "ditch_encircled":
            # 주구묘: a grave inside its ring ditch. The ditch is the stroke,
            # so it stays a ditch instead of turning into a frame.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 3.0))
            painter.drawPath(g.rect(9, 9, 46, 46))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.rect(25, 22, 14, 20))

        elif variant == "pit_grave":
            # 토광묘: a plain earth-cut pit in section, the body laid in it.
            painter.setBrush(ground_tone)
            painter.drawPath(g.poly([(9, 20), (55, 20), (48, GROUND),
                                     (16, GROUND)]))
            painter.setBrush(solid)
            painter.drawPath(g.rect(21, 34, 22, 10))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(5, 20, 59, 20))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean settlement, production and defence features
    # ═══════════════════════════════════════════════════════

    def _draw_korean_feature(self, painter, s, m, variant, color):
        """
        Settlement, production and defence features as excavated plans.

        Plan view for the dwellings, the fields and the kiln floors - that is
        how a site drawing shows them - and section view for the ramparts,
        the ovens and the climbing kilns, where the profile is what names the
        feature. Section types share one ground line at y=50, so the group
        sits on a single horizon.

        Repetition is the trap in this family. Sixteen postholes for a
        raised-floor building, thirteen billets in a charcoal kiln and eight
        furrows in a dry field all say "many" at drawing size and "grey" at
        marker size, so each is cut to the smallest count that still reads as
        a series.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        ground_tone = QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT)
        edge = _pen(color, 2.6)
        thin = _pen(color, 1.4)
        GROUND = 50

        def floor(outline, posts, hearth=(32, 32)):
            """A pit dwelling: the cut, its postholes and its hearth."""
            painter.setPen(edge)
            painter.setBrush(body)
            painter.drawPath(outline)
            painter.setBrush(solid)
            painter.setPen(thin)
            for px, py in posts:
                painter.drawPath(g.circle(px, py, 3))
            painter.setPen(edge)
            painter.drawPath(g.circle(hearth[0], hearth[1], 5))

        def ground_line():
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(5, GROUND, 59, GROUND))

        def climbing_kiln(marks):
            """
            A kiln in section: firebox at one end, domed chamber, flue at the
            other. Both kilns are this one silhouette - the reference plates
            make variants differ by their mark, not their shape - so ``marks``
            is what says pottery or roof tile.

            Drawn as a sloping tube, which is what this used to be, the whole
            thing reads as a diagonal bar: nothing in it says which end is
            the fire. The asymmetry is the information.
            """
            painter.setPen(edge)
            painter.setBrush(body)
            dome = g.symmetric([(5, 19), (15, 27), (17, GROUND)],
                               curved=True, cx=32)
            painter.drawPath(dome)
            painter.setBrush(solid)
            painter.drawPath(g.circle(12, 42, 7))          # firebox
            painter.drawPath(g.rect(48, 13, 8, 37))        # flue
            _clip_detail(painter, dome)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush if marks == "tile" else solid)
            if marks == "tile":
                painter.drawPath(g.line(26, 40, 34, 33))
                painter.drawPath(g.line(30, 45, 38, 38))
            else:
                painter.drawPath(g.circle(28, 38, 4))
                painter.drawPath(g.circle(38, 42, 4))
            painter.restore()
            ground_line()

        painter.setPen(edge)
        painter.setBrush(body)

        if variant == "pit_house_round":
            # 원형 수혈주거지
            floor(g.circle(32, 32, 24),
                  [(20, 20), (44, 20), (20, 44), (44, 44)])

        elif variant == "pit_house_square":
            # 방형 수혈주거지
            floor(g.rect(9, 9, 46, 46),
                  [(19, 19), (45, 19), (19, 45), (45, 45)])

        elif variant == "pit_house_convex":
            # 철(凸)자형: the same floor with an entrance passage added.
            floor(g.poly([(9, 11), (55, 11), (55, 43), (39, 43), (39, 53),
                          (25, 53), (25, 43), (9, 43)]),
                  [(19, 19), (45, 19), (19, 36), (45, 36)], hearth=(32, 27))

        elif variant == "pit_house_twin":
            # 여(呂)자형: two rooms joined by a short passage.
            floor(g.poly([(11, 8), (53, 8), (53, 30), (37, 30), (37, 36),
                          (46, 36), (46, 56), (18, 56), (18, 36), (27, 36),
                          (27, 30), (11, 30)]),
                  [(20, 15), (44, 15), (24, 49), (40, 49)], hearth=(32, 20))

        elif variant == "raised_floor":
            # 굴립주건물: the floor stands on earth-fast posts. Drawn in
            # section, because raised is the whole point and a plan of it is
            # a field of dots - which is what this used to be.
            painter.setBrush(body)
            painter.drawPath(g.poly([(12, 21), (32, 9), (52, 21)]))
            painter.setBrush(solid)
            painter.drawPath(g.rect(8, 21, 48, 8))
            painter.setBrush(body)
            for x in (16, 29, 42):
                painter.drawPath(g.rect(x, 29, 6, GROUND - 29))
            ground_line()

        elif variant == "kamado":
            # 부뚜막: a clay stove block with an arched fire mouth and a flue.
            painter.setBrush(body)
            painter.drawPath(g.poly([(9, GROUND), (14, 22), (48, 22),
                                     (53, GROUND)]))
            painter.setBrush(solid)
            painter.drawPath(g.rect(43, 12, 8, 10))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.2))
            painter.drawPath(g.rect(22, 33, 16, 17, r=8))
            ground_line()

        elif variant == "ondol":
            # 온돌: firebox, the flues that run under the floor, and the
            # chimney at the far end - in plan, left to right.
            # A stubby block at each end reads as a barbell, so the firebox
            # is a wide arch and the chimney a narrow stack: the shape says
            # which way the smoke runs.
            painter.setBrush(solid)
            painter.drawPath(g.rect(6, 22, 14, 20, r=7))
            painter.setBrush(body)
            for y in (24, 31, 38):
                painter.drawPath(g.rect(20, y, 27, 5, r=1))
            painter.setBrush(solid)
            painter.drawPath(g.rect(47, 12, 8, 40))

        elif variant == "pottery_kiln":
            climbing_kiln("pot")

        elif variant == "tile_kiln":
            climbing_kiln("tile")

        elif variant == "iron_smelting":
            # 제철유구: a shaft furnace with its tap hole, not the cooling
            # tower the old hourglass profile read as.
            painter.setBrush(body)
            painter.drawPath(g.symmetric(
                [(8, 12), (13, 24), (11, 38), (16, GROUND)], curved=True))
            painter.setBrush(solid)
            painter.setPen(edge)
            painter.drawPath(g.circle(32, 43, 5))
            ground_line()

        elif variant == "charcoal_kiln":
            # 숯가마: an elongated chamber in plan, fired from one end and
            # vented at the other.
            painter.setBrush(body)
            painter.drawPath(g.symmetric(
                [(7, 15), (19, 26), (19, 41), (9, 49)], curved=True))
            painter.setBrush(solid)
            painter.drawPath(g.rect(27, 47, 10, 11))       # stoking mouth
            painter.drawPath(g.circle(32, 10, 5))          # vent

        elif variant == "paddy_field":
            # 논: basins held by bunds, so the cells are what is drawn.
            painter.setBrush(ground_tone)
            painter.drawPath(g.rect(7, 13, 50, 38))
            painter.setBrush(body)
            painter.setPen(thin)
            for row in (17, 33):
                for col in (11, 27, 43):
                    painter.drawPath(g.rect(col, row, 10, 13, r=1))

        elif variant == "dry_field":
            # 밭: ridge and furrow. Four ridges read as ploughing; eight read
            # as a barcode.
            painter.setBrush(ground_tone)
            painter.drawPath(g.rect(8, 13, 48, 38))
            painter.setBrush(body)
            painter.setPen(thin)
            for i in range(4):
                painter.drawPath(g.rect(12 + i * 11, 17, 6, 30, r=1))

        elif variant == "earthen_rampart":
            # 토성: an earth bank, wide and sloping, raised in layers.
            bank = g.poly([(6, GROUND), (19, 17), (45, 17), (58, GROUND)])
            painter.setBrush(body)
            painter.drawPath(bank)
            _clip_detail(painter, bank)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(11, 39, 53, 39))
            painter.drawPath(g.line(16, 28, 48, 28))
            painter.restore()
            ground_line()

        elif variant == "stone_rampart":
            # 석성: the same section built in stone - steeper, and topped
            # with its crenellation.
            wall = g.poly([(13, GROUND), (19, 18), (45, 18), (51, GROUND)])
            painter.setBrush(body)
            painter.drawPath(wall)
            painter.setBrush(solid)
            for x in (18, 29, 40):
                painter.drawPath(g.rect(x, 11, 7, 7))
            _clip_detail(painter, wall)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(14, 38, 50, 38))
            painter.drawPath(g.line(16, 28, 48, 28))
            painter.restore()
            ground_line()

        elif variant == "mountain_fortress":
            # 산성: a wall carried along a ridge. The ridge is what makes it
            # a mountain fortress rather than a town wall.
            painter.setBrush(ground_tone)
            painter.setPen(thin)
            painter.drawPath(g.poly([(5, GROUND), (20, 16), (32, 35),
                                     (44, 23), (57, GROUND)]))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 3.0))
            painter.drawPath(g.poly([(9, 45), (20, 20), (32, 39), (44, 27),
                                     (53, 45)], close=False))
            painter.setBrush(solid)
            painter.setPen(edge)
            painter.drawPath(g.rect(28, 35, 8, 8, r=1))

        elif variant == "palisade":
            # 목책: a line of sharpened stakes behind a rail.
            painter.setBrush(body)
            for i in range(5):
                x = 8 + i * 10
                painter.drawPath(g.poly([(x, 20), (x + 4, 13), (x + 8, 20),
                                         (x + 8, GROUND), (x, GROUND)]))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            painter.drawPath(g.line(6, 32, 58, 32))

        elif variant == "encircling_ditch":
            # 환호: a ditch ringing a settlement. Drawn as the stroke it is,
            # with the houses it encloses.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 3.0))
            painter.drawPath(g.circle(32, 32, 25))
            painter.setBrush(solid)
            painter.setPen(edge)
            for cx, cy in ((25, 26), (41, 29), (31, 41)):
                painter.drawPath(g.circle(cx, cy, 5))

        elif variant == "beacon":
            # 봉수: the fire platform and its smoke.
            painter.setBrush(body)
            painter.drawPath(g.poly([(10, GROUND), (20, 31), (44, 31),
                                     (54, GROUND)]))
            painter.setBrush(solid)
            painter.drawPath(g.rect(25, 23, 14, 8))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.0))
            painter.drawPath(g.poly([(32, 22), (27, 17), (36, 13), (30, 7)],
                                    close=False))
            ground_line()

        elif variant == "water_basin":
            # 집수정: a tapered basin holding water.
            basin = g.poly([(11, 15), (53, 15), (45, GROUND), (19, GROUND)])
            painter.setBrush(ground_tone)
            painter.drawPath(basin)
            _clip_detail(painter, basin)
            painter.setBrush(body)
            painter.setPen(Qt.NoPen)
            painter.drawPath(g.rect(12, 28, 40, 24, r=0))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(16, 28, 48, 28))
            painter.restore()

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean pottery and ceramics (토기·도자기)
    # ═══════════════════════════════════════════════════════

    def _draw_korean_pottery(self, painter, s, m, variant, color):
        """
        The ceramic series as one curved profile with thirteen sets of numbers.

        Every vessel is a ``symmetric`` call over half-widths read down the
        wall, so rim heights, shoulder positions and foot widths line up
        across the series instead of each pot having its own curve. Surface
        treatment is what separates the wares, and it is clipped to the body.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        thin = _pen(color.darker(170), 1.3)
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(solid)

        #                  (half width, y) down the wall, in grid units
        profiles = {
            "comb_pattern":    ((18, 6), (18, 11), (16, 13), (5, 54), (1, 58)),
            "plain_coarse":    ((17, 6), (16, 22), (14, 42), (12, 57)),
            "red_burnished":   ((6, 6), (8, 11), (20, 26), (17, 44), (10, 57)),
            "black_burnished": ((5, 5), (5, 19), (19, 33), (16, 48), (9, 57)),
            "wajil":           ((8, 7), (9, 12), (21, 30), (14, 50), (4, 57)),
            "gyeongjil":       ((10, 6), (11, 11), (22, 30), (16, 47), (11, 57)),
            "storage_jar":     ((8, 6), (10, 11), (22, 26), (19, 46), (11, 57)),
            "siru":            ((19, 8), (16, 28), (12, 50), (11, 56)),
            "celadon":         ((5, 5), (6, 10), (20, 21), (14, 44), (9, 57)),
            "buncheong":       ((4, 5), (4, 19), (18, 37), (12, 57)),
            "white_porcelain": ((8, 7), (20, 20), (21, 33), (18, 46), (9, 57)),
            "onggi":           ((17, 14), (21, 30), (18, 46), (11, 57)),
            "gobae":           ((19, 8), (16, 14), (7, 22)),   # the dish only
        }
        body = g.symmetric(list(profiles.get(variant, profiles["plain_coarse"])),
                           curved=True)
        painter.drawPath(body)

        # A ring foot or a lid is a second shape, not part of the wall.
        if variant == "gyeongjil":
            painter.drawPath(g.rect(23, 52, 18, 6))
        elif variant == "onggi":
            painter.drawPath(g.symmetric([(20, 8), (21, 12), (17, 14)], curved=True))
        elif variant == "storage_jar":
            painter.drawPath(g.circle(9, 30, 4))
            painter.drawPath(g.circle(55, 30, 4))
        elif variant == "gobae":
            # The pedestal is its own shape; merged into the dish profile the
            # whole thing read as an hourglass.
            painter.drawPath(g.symmetric([(7, 22), (8, 44), (17, 52), (17, 57)]))

        _clip_detail(painter, body)
        painter.setPen(thin)
        painter.setBrush(Qt.NoBrush)
        if variant == "comb_pattern":
            painter.drawPath(g.line(46, 13, 18, 13))
            for half, y in ((14, 20), (11, 30), (8, 40)):
                for step in range(3):
                    x = 32 - half + (half * step)
                    painter.drawPath(g.line(x, y, x + 3, y + 5))
        elif variant in ("red_burnished", "wajil"):
            # Burnish strokes: swept, and few.
            for step in range(3):
                x = 24 + step * 8
                painter.drawPath(g.line(x, 26, x + 5, 42))
        elif variant in ("black_burnished", "celadon", "buncheong"):
            painter.drawPath(g.line(20, 34, 44, 34))
        elif variant == "gyeongjil":
            # 타날문: the paddled bands that name the ware.
            for y in (26, 33, 40):
                painter.drawPath(g.line(12, y, 52, y))
        elif variant == "white_porcelain":
            # The seam where the two thrown halves meet.
            painter.drawPath(g.line(12, 33, 52, 33))
        elif variant == "onggi":
            for y in (34, 42):
                painter.drawPath(g.line(14, y, 50, y))
        elif variant == "siru":
            painter.setBrush(solid)
            for step in range(4):
                painter.drawPath(g.circle(25 + step * 5, 53, 1.5))
        elif variant == "gobae":
            # The pierced pedestal is what makes it a 굽다리접시.
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.rect(28, 26, 3, 6, r=1))
            painter.drawPath(g.rect(33, 26, 3, 6, r=1))
        painter.restore()

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

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
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.MID))
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
            # The tail has to stay fat enough to read at marker size; drawn
            # thinner it turns into a figure 9 once the outline is on it.
            body.quadTo(hx - r * 0.05, bottom - 40, hx + r * 0.15, hy + r * 0.8)
            body.quadTo(hx + r * 0.28, hy + r * 0.05, hx - r, hy)
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
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
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
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.MID))
            painter.drawRect(QRectF(m + 22, bottom - 40, s - 2 * m - 44, 26))
            painter.setPen(thin)
            painter.setBrush(hollow)
            painter.drawRect(QRectF(m + 18, cy - 40, s - 2 * m - 36, 60))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOLID))
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
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.MID))
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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
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
        painter.setBrush(QColor(base.red(), base.green(), base.blue(), icon_grid.MID))
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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
        painter.drawPath(p)
        # A darker core inside the scorched outline says "burnt" more
        # plainly than a field of char marks.
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.MID))
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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))

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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
        ring_pen = _pen(color.darker(125), 2.2, Qt.DashLine)
        painter.setPen(ring_pen)
        painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOLID))
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
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), icon_grid.SOFT))
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
