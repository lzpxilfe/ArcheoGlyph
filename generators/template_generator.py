# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Template Generator
Generates symbols from built-in SVG templates with comprehensive archaeological categories.
"""

import math
import os
import re
from qgis.PyQt.QtGui import QImage, QColor, QPainter, QPainterPath, QPen
from qgis.PyQt.QtCore import Qt, QBuffer, QByteArray, QIODevice, QRect, QSize
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


#: Family-wide size corrections, applied to the half-width tables below.
#:
#: Every symbol was drawn to fill the same safe area, which left the ink each
#: one lays on its tile ranging from 9 percent to 66 - a seven-fold spread.
#: On a legend that reads as some symbols shouting while others go missing.
#: The blade series were the thinnest things in the catalogue and the vessels
#: the heaviest, so each family is corrected once against its own table, which
#: leaves the differences *within* a family untouched.
#:
#: Named per family on purpose: a bare `scale` local in three methods is one
#: careless search-and-replace away from silently rescaling the wrong one.
BLADE_SCALE = 1.7        # 동검 10종, 첨두기 5종
VESSEL_SCALE = 0.8       # 토기 13종


def _weight(width):
    """Lift a requested stroke width onto the grid's two steps."""
    width = float(width)
    if width <= 2.0:
        return DETAIL_WIDTH
    if width <= 3.4:
        return OUTLINE_WIDTH
    # A deliberately heavy stroke - a bracelet, an earring hoop, a ring ditch
    # - is a shape drawn as a line, so it is given in grid units like every
    # other measurement here. Scaling it by a bare 1.3 made the heaviest
    # strokes in the catalogue *thinner* than an ordinary detail line.
    return width * _UNIT


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
            return ("_draw_general_find", ("pottery", COLOR))
        elif "scraper" in key:
            return ("_draw_general_find", ("scraper", COLOR))
        elif "arrowhead" in key:
            return ("_draw_general_find", ("arrowhead", COLOR))
        elif "stone tool" in key:
            return ("_draw_general_find", ("stone_tool", COLOR))
        elif "bronze" in key:
            return ("_draw_general_find", ("bronze", COLOR))
        elif "chisel" in key:
            return ("_draw_general_find", ("chisel", COLOR))
        elif "iron" in key:
            return ("_draw_general_find", ("iron", COLOR))
        elif "bracelet" in key or "ring" in key:
            return ("_draw_general_find", ("bracelet", COLOR))
        elif "bead" in key:
            return ("_draw_general_find", ("bead", COLOR))
        elif "ornament" in key:
            return ("_draw_general_find", ("ornament", COLOR))
        elif "spindle" in key:
            return ("_draw_general_find", ("whorl", COLOR))
        elif "seal" in key or "stamp" in key:
            return ("_draw_general_find", ("seal", COLOR))
        elif "coin" in key:
            return ("_draw_general_find", ("coin", COLOR))
        elif "needle" in key or "pin" in key:
            return ("_draw_general_find", ("needle", COLOR))
        elif "animal remains" in key:
            return ("_draw_general_find", ("animal_bone", COLOR))
        elif "bone" in key:
            return ("_draw_general_find", ("bone_tool", COLOR))
        elif "weapon" in key or "arrow shaft" in key:
            return ("_draw_general_find", ("weapon", COLOR))
        elif "blade" in key:
            return ("_draw_general_find", ("blade", COLOR))
        elif "gate" in key:
            return ("_draw_general_structure", ("gate", COLOR))
        elif "tower" in key:
            return ("_draw_general_structure", ("tower", COLOR))
        elif "fortress" in key or "castle" in key:
            return ("_draw_general_structure", ("fortress", COLOR))
        elif "workshop" in key:
            return ("_draw_general_structure", ("workshop", COLOR))
        elif "dwelling" in key or "house" in key:
            return ("_draw_general_structure", ("dwelling", COLOR))
        elif "road" in key or "pavement" in key:
            return ("_draw_general_structure", ("road", COLOR))
        elif "bridge" in key:
            return ("_draw_general_structure", ("bridge", COLOR))
        elif "terrace" in key:
            return ("_draw_general_landscape", ("terrace", COLOR))
        elif "wall" in key or "rampart" in key:
            return ("_draw_general_structure", ("wall", COLOR))
        elif "posthole" in key:
            return ("_draw_general_structure", ("posthole", COLOR))
        elif "test pit" in key:
            return ("_draw_test_pit", (COLOR,))
        elif "storage pit" in key:
            return ("_draw_general_structure", ("storage_pit", COLOR))
        elif "pit" in key:
            return ("_draw_general_structure", ("pit", COLOR))
        elif "ash layer" in key:
            return ("_draw_ash_layer", (COLOR,))
        elif "burnt" in key:
            return ("_draw_general_landscape", ("burnt", COLOR))
        elif "canal" in key or "water channel" in key:
            return ("_draw_general_landscape", ("canal", COLOR))
        elif "ditch" in key or "moat" in key:
            return ("_draw_general_landscape", ("ditch", COLOR))
        elif "standing stone" in key:
            return ("_draw_general_landscape", ("standing_stone", COLOR))
        elif "stone align" in key:
            return ("_draw_general_landscape", ("alignment", COLOR))
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
        elif "barrow" in key or ("mound" in key and "shell" not in key and "midden" not in key):
            return ("_draw_general_structure", ("mound", COLOR))
        elif "tomb" in key:
            return ("_draw_general_structure", ("tomb", COLOR))
        elif "temple" in key or "shrine" in key:
            return ("_draw_general_structure", ("temple", COLOR))
        elif "kiln" in key or "furnace" in key:
            return ("_draw_general_structure", ("kiln", COLOR))
        elif "well" in key:
            return ("_draw_general_structure", ("well", COLOR))
        elif "skeleton" in key:
            return ("_draw_general_landscape", ("skeleton", COLOR))
        elif "human" in key or "skull" in key:
            return ("_draw_general_landscape", ("skull", COLOR))
        elif "cremation" in key:
            return ("_draw_general_landscape", ("cremation", COLOR))
        elif "burial" in key:
            return ("_draw_general_landscape", ("burial", COLOR))
        elif "hearth" in key or "fire" in key:
            return ("_draw_general_landscape", ("hearth", COLOR))
        elif "midden" in key or "shell" in key:
            return ("_draw_general_landscape", ("midden", COLOR))
        elif "dolmen" in key:
            return ("_draw_general_landscape", ("dolmen", COLOR))
        elif "rock art" in key:
            return ("_draw_general_landscape", ("rock_art", COLOR))
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

    def _draw_pottery_sherd_section(self, painter, s, m, variant, color):
        """
        The three sherd categories, drawn as the wall section each one is.

        These used to be three lumpy outlines with the same field of hatching
        inside, so nothing said which part of the pot a sherd came from. What
        actually distinguishes them is where the profile ends: a rim has a
        lip, a base has a foot, a body is cut at both ends.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.MID))

        if variant == "rim":
            wall = g.poly([(45, 7), (34, 17), (32, 32), (34, 55),
                           (20, 55), (18, 32), (20, 15), (32, 5)])
        elif variant == "base":
            wall = g.poly([(19, 7), (33, 7), (33, 38), (54, 38),
                           (54, 53), (19, 53)])
        else:
            wall = g.poly([(24, 6), (38, 6), (31, 31), (39, 56),
                           (25, 56), (18, 31)])
        painter.drawPath(wall)

        # One hatch direction across all three, so they read as a set.
        _clip_detail(painter, wall)
        painter.setPen(_pen(color, 1.4))
        painter.setBrush(Qt.NoBrush)
        for offset in (0, 11, 21, 32):
            painter.drawPath(g.line(10 + offset, 56, 28 + offset, 22))
        painter.restore()
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_general_landscape(self, painter, s, m, variant, color):
        """
        Human remains, burials and the landscape features.

        Six of these used to be three pictures. Both skull entries were the
        same skull; both burial entries were the same crouched line figure,
        which at marker size read as a numeral. They are drawn as separate
        objects here - a skull against a body in plan, a grave cut against a
        cinerary urn - and the general 지석묘 is a plan so it does not repeat
        the three dolmen elevations.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        ground_tone = QColor(color.red(), color.green(), color.blue(),
                             icon_grid.SOFT)
        edge = _pen(color, 2.6)
        thin = _pen(color, 1.4)
        GROUND = 50

        painter.setPen(edge)
        painter.setBrush(body)

        if variant == "skull":
            # 인골: a skull - cranium, orbits, nasal aperture, jaw.
            painter.drawPath(g.circle(32, 25, 17))
            painter.setBrush(solid)
            painter.drawPath(g.ellipse(25, 23, 4, 5))
            painter.drawPath(g.ellipse(39, 23, 4, 5))
            painter.drawPath(g.poly([(32, 29), (35, 35), (29, 35)]))
            painter.setBrush(body)
            painter.drawPath(g.rect(23, 40, 18, 11, r=4))

        elif variant == "skeleton":
            # 전신 인골: an extended inhumation in plan.
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 12, 8))
            painter.setBrush(body)
            painter.drawPath(g.rect(24, 20, 16, 22, r=3))
            painter.drawPath(g.rect(15, 22, 7, 17, r=3))
            painter.drawPath(g.rect(42, 22, 7, 17, r=3))
            painter.drawPath(g.rect(25, 42, 6, 15, r=3))
            painter.drawPath(g.rect(33, 42, 6, 15, r=3))

        elif variant == "burial":
            # 매장 유구: the grave cut, with what is in it.
            painter.setBrush(ground_tone)
            painter.setPen(_pen(color, 2.0, Qt.DashLine))
            painter.drawPath(g.rect(13, 7, 38, 50))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 18, 7))
            painter.drawPath(g.rect(26, 26, 12, 24, r=5))

        elif variant == "cremation":
            # 화장묘: the urn, and the burnt bone in it.
            urn = g.symmetric([(6, 15), (16, 27), (11, 51)], curved=True)
            painter.drawPath(urn)
            _clip_detail(painter, urn)
            painter.setBrush(solid)
            painter.setPen(thin)
            for cx, cy in ((27, 33), (37, 31), (32, 41)):
                painter.drawPath(g.circle(cx, cy, 4))
            painter.restore()
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.rect(21, 8, 22, 7, r=3))

        elif variant == "hearth":
            # 노지: the burnt centre inside its kerb of stones.
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 32, 13))
            painter.setBrush(body)
            for index in range(6):
                angle = index * math.pi / 3.0
                painter.drawPath(g.circle(32 + 20 * math.cos(angle),
                                          32 + 20 * math.sin(angle), 5.5))

        elif variant == "burnt":
            # 소토 범위: a spread of scorched soil, not a stone.
            painter.setBrush(ground_tone)
            painter.drawPath(g.poly([(11, 21), (30, 11), (50, 18), (57, 34),
                                     (44, 51), (22, 53), (8, 38)]))
            painter.setBrush(solid)
            painter.setPen(thin)
            for cx, cy, rx in ((24, 26, 5), (40, 33, 6), (27, 43, 4)):
                painter.drawPath(g.ellipse(cx, cy, rx, rx * 0.7))

        elif variant == "midden":
            # 패총: the mound, and the shell it is made of.
            painter.setBrush(ground_tone)
            painter.drawPath(g.symmetric([(6, 22), (20, 32), (28, GROUND)],
                                         curved=True))
            painter.setBrush(body)
            painter.setPen(thin)
            for cx, cy in ((22, 40), (32, 32), (42, 40), (32, 45)):
                painter.drawPath(g.ellipse(cx, cy, 6, 4))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(5, GROUND, 59, GROUND))

        elif variant == "ditch":
            # 구 / 해자: the channel itself, as a band with two banks.
            channel = QPainterPath()
            channel.moveTo(*g.pt(32 + 26 * math.cos(math.pi / 3.0),
                                 32 + 26 * math.sin(math.pi / 3.0)))
            g.arc(channel, 32, 32, 26, math.pi / 3.0, 1.33 * math.pi,
                  segments=8)
            channel.lineTo(*g.pt(32 + 11 * math.cos(-1.0 * math.pi / 3.0),
                                 32 + 11 * math.sin(-1.0 * math.pi / 3.0)))
            g.arc(channel, 32, 32, 11, -math.pi / 3.0, -1.33 * math.pi,
                  segments=8)
            channel.closeSubpath()
            painter.setBrush(ground_tone)
            painter.drawPath(channel)

        elif variant == "canal":
            # 수로: a cut channel, with the direction it runs.
            painter.setBrush(ground_tone)
            painter.drawPath(g.rect(5, 21, 54, 22))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 3.0))
            # Two chevrons read as flow; one reads as a play button.
            painter.drawPath(g.poly([(21, 25), (32, 32), (21, 39)],
                                    close=False))
            painter.drawPath(g.poly([(33, 25), (44, 32), (33, 39)],
                                    close=False))

        elif variant == "alignment":
            # 열석: a row of set stones, standing on one line.
            painter.setBrush(body)
            for x, top in ((8, 24), (22, 18), (36, 26), (50, 21)):
                painter.drawPath(g.rect(x, top, 9, GROUND - top, r=4))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(5, GROUND, 59, GROUND))

        elif variant == "dolmen":
            # 지석묘: the capstone in plan, over the chamber it covers -
            # the three named dolmen types already carry the elevations.
            painter.setBrush(body)
            painter.drawPath(g.poly([(9, 19), (33, 10), (56, 20), (52, 43),
                                     (28, 52), (11, 39)]))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.0, Qt.DashLine))
            painter.drawPath(g.rect(24, 25, 18, 14))

        elif variant == "rock_art":
            # 암각화: a rock face with a pecked motif on it.
            face = g.poly([(10, 16), (34, 8), (56, 19), (50, 45),
                           (26, 54), (9, 40)])
            painter.setBrush(ground_tone)
            painter.drawPath(face)
            _clip_detail(painter, face)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.2))
            painter.drawPath(g.circle(30, 29, 13))
            painter.drawPath(g.circle(30, 29, 7))
            painter.setBrush(solid)
            painter.setPen(thin)
            painter.drawPath(g.circle(30, 29, 3))
            painter.drawPath(g.circle(46, 41, 4))
            painter.restore()

        elif variant == "standing_stone":
            # 입석: a menhir, packed at the foot.
            painter.setBrush(body)
            painter.drawPath(g.symmetric([(6, 8), (10, 24), (8, 40), (11, 48)],
                                         curved=False))
            painter.setBrush(solid)
            painter.drawPath(g.ellipse(17, 48, 7, 4))
            painter.drawPath(g.ellipse(47, 48, 7, 4))
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(5, GROUND, 59, GROUND))

        elif variant == "terrace":
            # 단: worked ground, cut back in steps.
            painter.setBrush(ground_tone)
            painter.drawPath(g.poly([(6, 52), (6, 42), (23, 42), (23, 30),
                                     (40, 30), (40, 17), (58, 17), (58, 52)]))
            painter.setPen(_pen(color, 2.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.poly([(6, 42), (23, 42), (23, 30), (40, 30),
                                     (40, 17), (58, 17)], close=False))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_general_structure(self, painter, s, m, variant, color):
        """
        The general structure and feature categories.

        As with the general finds, these are what a user picks when the
        specific type is unknown, and they have to stay clear of the detailed
        Korean entries they sit beside. So the general 무덤 is a section with
        its chamber where 봉토분 is a tiered mound; the general 가마 is a
        domed furnace where 토기가마 is a kiln in profile; 수혈 and 저장혈
        differ by the shape of the cut, which is what defines a storage pit.

        Sections share the ground line at y=50 with the rest of the
        catalogue.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        ground_tone = QColor(color.red(), color.green(), color.blue(),
                             icon_grid.SOFT)
        edge = _pen(color, 2.6)
        thin = _pen(color, 1.4)
        GROUND = 50

        painter.setPen(edge)
        painter.setBrush(body)

        def ground_line(y=GROUND, gap=0):
            # A cut section needs the ground to stop at the lip of the cut.
            # Run straight across and the line reads as a lid on a bowl.
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            if gap:
                painter.drawPath(g.line(5, y, 32 - gap, y))
                painter.drawPath(g.line(32 + gap, y, 59, y))
            else:
                painter.drawPath(g.line(5, y, 59, y))
            painter.setPen(edge)
            painter.setBrush(body)

        if variant == "fortress":
            # 성곽: a curtain wall, crenellated, with its gate.
            painter.drawPath(g.rect(8, 20, 48, 32))
            painter.setBrush(solid)
            for x in (8, 22, 36, 50):
                painter.drawPath(g.rect(x, 13, 6, 7, r=1))
            painter.drawPath(g.rect(27, 38, 10, 14, r=1))

        elif variant == "gate":
            # 문지: two jambs under a lintel, on their threshold.
            painter.setBrush(solid)
            painter.drawPath(g.rect(7, 10, 50, 9, r=1))
            painter.setBrush(body)
            painter.drawPath(g.rect(12, 19, 11, 33))
            painter.drawPath(g.rect(41, 19, 11, 33))
            painter.drawPath(g.rect(7, 52, 50, 6, r=1))

        elif variant == "tower":
            # 망루: a tall crenellated stage.
            painter.drawPath(g.rect(21, 15, 22, 39))
            painter.setBrush(solid)
            for x in (21, 30, 39):
                painter.drawPath(g.rect(x, 8, 6, 7, r=1))
            painter.drawPath(g.rect(28, 24, 8, 8, r=1))
            painter.drawPath(g.rect(28, 38, 8, 8, r=1))

        elif variant == "dwelling":
            # 주거지: a gabled house.
            painter.drawPath(g.poly([(32, 8), (55, 26), (55, 52),
                                     (9, 52), (9, 26)]))
            painter.setBrush(solid)
            painter.drawPath(g.rect(27, 39, 10, 13, r=1))

        elif variant == "workshop":
            # 공방지: the same house with the tools that name it.
            painter.drawPath(g.poly([(32, 8), (55, 26), (55, 52),
                                     (9, 52), (9, 26)]))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.6))
            painter.drawPath(g.line(22, 30, 42, 46))
            painter.drawPath(g.line(42, 30, 22, 46))

        elif variant == "temple":
            # 사찰 / 사당: a hall under a wide gabled roof.
            painter.setBrush(solid)
            painter.drawPath(g.poly([(5, 27), (32, 10), (59, 27)]))
            painter.setBrush(body)
            painter.drawPath(g.rect(13, 27, 38, 25))
            painter.setBrush(solid)
            painter.drawPath(g.rect(27, 39, 10, 13, r=1))

        elif variant == "tomb":
            # 무덤: a mound in section with the chamber it covers.
            painter.setBrush(ground_tone)
            painter.drawPath(g.symmetric([(4, 16), (17, 28), (28, GROUND)],
                                         curved=True))
            painter.setBrush(solid)
            painter.drawPath(g.rect(22, 37, 20, 13, r=1))
            ground_line()

        elif variant == "mound":
            # 분구 / 봉분: the mound in plan, hachured off its edge the way a
            # survey drawing shows one. Two plain circles would be the well
            # again in another colour.
            painter.setBrush(body)
            painter.drawPath(g.circle(32, 32, 21))
            painter.setPen(_pen(color, 2.0))
            painter.setBrush(Qt.NoBrush)
            for index in range(6):
                angle = index * math.pi / 3.0
                painter.drawPath(g.line(32 + 21 * math.cos(angle),
                                        32 + 21 * math.sin(angle),
                                        32 + 28 * math.cos(angle),
                                        32 + 28 * math.sin(angle)))
            painter.setPen(edge)
            painter.setBrush(body)

        elif variant == "kiln":
            # 가마 / 노: a domed furnace with its stoking arch.
            painter.setBrush(body)
            painter.drawPath(g.symmetric([(5, 13), (18, 26), (23, GROUND)],
                                         curved=True))
            painter.setBrush(solid)
            painter.drawPath(g.rect(26, 36, 12, 14, r=6))
            ground_line()

        elif variant == "well":
            # 우물: the shaft, dark, inside its kerb.
            painter.drawPath(g.circle(32, 32, 20))
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 32, 11))

        elif variant == "wall":
            # 성벽: a length of walling with its return, coursed.
            wall = g.poly([(7, 17), (57, 17), (57, 28), (18, 28),
                           (18, 53), (7, 53)])
            painter.drawPath(wall)
            _clip_detail(painter, wall)
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for x in (20, 33, 46):
                painter.drawPath(g.line(x, 17, x, 28))
            painter.drawPath(g.line(7, 40, 18, 40))
            painter.restore()

        elif variant == "pit":
            # 수혈: an open bowl-shaped cut.
            painter.setBrush(ground_tone)
            painter.drawPath(g.symmetric([(23, 17), (23, 34), (13, 52)],
                                         curved=True))
            ground_line(17, gap=24)

        elif variant == "storage_pit":
            # 저장혈: the flask profile - a narrow mouth over a wide belly -
            # which is what makes a pit a storage pit.
            painter.setBrush(ground_tone)
            painter.drawPath(g.symmetric([(10, 17), (26, 34), (21, 54)],
                                         curved=True))
            ground_line(17, gap=11)

        elif variant == "posthole":
            # 주혈: the post pipe inside its packing, in plan.
            painter.setBrush(ground_tone)
            painter.drawPath(g.circle(32, 32, 21))
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 32, 9))

        elif variant == "road":
            # 도로 / 포장면: a made surface with its centre line.
            painter.setBrush(ground_tone)
            painter.drawPath(g.rect(5, 19, 54, 26))
            painter.setBrush(solid)
            painter.setPen(thin)
            for x in (12, 28, 44):
                painter.drawPath(g.rect(x, 30, 10, 5, r=2))

        elif variant == "bridge":
            # 교량: a deck on piers, over the water it crosses.
            painter.setBrush(solid)
            painter.drawPath(g.rect(6, 22, 52, 9, r=1))
            painter.setBrush(body)
            painter.drawPath(g.rect(16, 31, 9, 15))
            painter.drawPath(g.rect(39, 31, 9, 15))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.0))
            painter.drawPath(g.poly([(8, 51), (18, 47), (28, 51), (38, 47),
                                     (48, 51), (56, 47)], close=False))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_general_find(self, painter, s, m, variant, color):
        """
        The catalogue's general find categories.

        These are the entries a user reaches for when the specific type is
        unknown - "a stone tool", "a bead", "an iron artefact" - and they used
        to be five shapes shared between sixteen names: one triangle served
        석기, 화살촉 and 긁개, one disc served 장신구, 구슬 and 팔찌, one wheel
        served 화폐, 인장 and 가락바퀴. On a legend that reads as a bug.

        Each is drawn as its own object here, and each is chosen to avoid the
        detailed Korean types it sits beside - the general arrowhead is barbed
        and tanged where 돌화살촉 is stemmed, the general iron artefact is a
        single-edged knife where 철검 has a guard.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        edge = _pen(color, 2.6)
        thin = _pen(color, 1.4)

        painter.setPen(edge)
        painter.setBrush(solid)

        def detail(*shapes):
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for shape in shapes:
                painter.drawPath(shape)
            painter.setPen(edge)
            painter.setBrush(solid)

        if variant == "pottery":
            # 토기: the plainest jar in the series, since the named wares
            # carry the shapes that mean something.
            painter.drawPath(g.symmetric(
                [(8, 8), (10.5, 13), (18, 28), (13, 50), (9.5, 56)],
                curved=True))

        elif variant == "stone_tool":
            # 석기: a worked nodule, scarred all round.
            core = g.poly([(30, 9), (46, 17), (52, 33), (42, 49),
                           (25, 53), (13, 40), (11, 23)])
            painter.drawPath(core)
            _clip_detail(painter, core)
            detail(g.line(30, 9, 28, 30), g.line(13, 23, 30, 32),
                   g.line(52, 33, 32, 34), g.line(25, 53, 29, 36))
            painter.restore()

        elif variant == "arrowhead":
            # 화살촉: barbed and tanged, which is what tells the general
            # point from the stemmed 돌화살촉 and the long-tanged 철촉.
            painter.drawPath(g.poly([(32, 5), (46, 40), (37, 35), (36, 52),
                                     (28, 52), (27, 35), (18, 40)]))

        elif variant == "scraper":
            # 긁개: a flake with one retouched convex edge.
            flake = g.symmetric([(5, 12), (18, 23), (25, 37), (26, 48)],
                                curved=True)
            painter.drawPath(flake)
            _clip_detail(painter, flake)
            detail(g.line(11, 30, 18, 33), g.line(9, 38, 17, 39),
                   g.line(9, 45, 17, 44))
            painter.restore()

        elif variant == "bronze":
            # 청동기: a cast bronze vessel with its ring handles - the
            # bronze weapons, mirrors and bells all have their own entries.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.4))
            painter.drawPath(g.circle(14, 27, 6))
            painter.drawPath(g.circle(50, 27, 6))
            painter.setPen(edge)
            painter.setBrush(body)
            painter.drawPath(g.symmetric(
                [(20, 21), (21, 32), (14, 44)], curved=True))
            painter.setBrush(solid)
            painter.drawPath(g.rect(23, 44, 18, 8, r=2))

        elif variant == "iron":
            # 철기: a single-edged knife - straight back, angled tip, no
            # guard, so it does not collide with 철검 or with 날붙이.
            painter.drawPath(g.poly([(38, 6), (41, 13), (41, 43), (24, 43),
                                     (24, 21)]))
            painter.setBrush(body)
            painter.drawPath(g.rect(28, 43, 9, 14, r=2))

        elif variant == "chisel":
            # 끌: a struck head over a bevelled edge.
            painter.setBrush(body)
            painter.drawPath(g.rect(24, 6, 16, 8, r=2))
            painter.setBrush(solid)
            bar = g.symmetric([(6, 14), (6, 44), (10, 50), (10, 57)],
                              curved=False)
            painter.drawPath(bar)
            _clip_detail(painter, bar)
            detail(g.line(22, 50, 42, 50))
            painter.restore()

        elif variant == "ornament":
            # 장신구: a disc brooch with its pin.
            # The pin runs behind the plate and out both sides, which is
            # what says brooch rather than frying pan.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.4))
            painter.drawPath(g.line(6, 38, 58, 38))
            painter.setPen(edge)
            painter.setBrush(body)
            painter.drawPath(g.circle(32, 30, 18))
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 30, 6))

        elif variant == "bead":
            # 구슬: one bead on its cord, so the perforation is visible
            # without having to knock a hole out of the fill.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.0))
            painter.drawPath(g.line(32, 6, 32, 58))
            painter.setPen(edge)
            painter.setBrush(body)
            painter.drawPath(g.ellipse(32, 32, 16, 21))

        elif variant == "bracelet":
            # 팔찌 / 반지: an annulus, drawn as the stroke it is.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 9.0))
            painter.drawPath(g.circle(32, 32, 21))

        elif variant == "coin":
            # 화폐: the cash coin, square hole and all.
            painter.setBrush(body)
            painter.drawPath(g.circle(32, 32, 23))
            painter.setBrush(solid)
            painter.drawPath(g.rect(25, 25, 14, 14, r=1))

        elif variant == "seal":
            # 인장: the knob, the block, and the cut face.
            painter.setBrush(solid)
            painter.drawPath(g.rect(27, 6, 10, 15, r=4))
            painter.setBrush(body)
            painter.drawPath(g.rect(13, 21, 38, 32, r=2))
            detail(g.line(21, 30, 43, 30), g.line(21, 38, 43, 38),
                   g.line(32, 30, 32, 46))

        elif variant == "whorl":
            # 가락바퀴: the disc, on the spindle that explains it.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.6))
            painter.drawPath(g.line(32, 5, 32, 59))
            painter.setPen(edge)
            painter.setBrush(body)
            painter.drawPath(g.ellipse(32, 36, 23, 11))

        elif variant == "bone_tool":
            # 골각기: a bone point, the joint end left as the grip.
            painter.setBrush(body)
            painter.drawPath(g.symmetric(
                [(1, 6), (4, 22), (7, 40), (11, 50)], curved=True))
            painter.setBrush(solid)
            painter.drawPath(g.ellipse(32, 50, 13, 7))

        elif variant == "needle":
            # 바늘 / 침: a shaft with an eye.
            painter.setBrush(body)
            painter.drawPath(g.symmetric(
                [(6, 6), (6, 42), (1, 58)], curved=False))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.0))
            painter.drawPath(g.ellipse(32, 16, 3, 6))

        elif variant == "animal_bone":
            # 동물유체: a long bone, knuckled at both ends.
            painter.setBrush(body)
            painter.drawPath(g.rect(26, 15, 12, 34, r=3))
            painter.setBrush(solid)
            for cx, cy in ((25, 13), (39, 13), (25, 51), (39, 51)):
                painter.drawPath(g.circle(cx, cy, 7))

        elif variant == "weapon":
            # 무기: a hafted point. The shaft is what makes it a weapon
            # rather than the loose blade next to it.
            painter.setBrush(body)
            painter.drawPath(g.rect(27, 26, 10, 31, r=2))
            painter.setBrush(solid)
            # A short wide head on a stub of shaft is a mallet; the point has
            # to be longer than it is broad to read as a weapon.
            painter.drawPath(g.symmetric([(0, 4), (11, 21), (5, 32)],
                                         curved=False))

        elif variant == "blade":
            # 날붙이: the blade on its own - two edges and a tang, no haft
            # and no guard.
            painter.drawPath(g.symmetric([(0, 6), (11, 20), (10, 40), (4, 45)],
                                         curved=False))
            painter.setBrush(body)
            painter.drawPath(g.rect(27, 45, 10, 12, r=2))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_bronze_weapon_symbol(self, painter, s, m, variant, color):
        """
        The three general bronze weapons.

        These were three leaf blades of almost the same outline. A 과 is not
        a blade on a line with the shaft - it is mounted across one - and
        that is the difference the symbol has to carry.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(solid)

        if variant == "dagger_axe":
            # 동과: the blade is hafted at right angles to the shaft.
            painter.setBrush(body)
            painter.drawPath(g.rect(41, 6, 8, 52, r=1))
            painter.setBrush(solid)
            painter.drawPath(g.poly([(41, 17), (9, 26), (41, 35)]))

        elif variant == "spear":
            # 동모: a leaf blade over its socket.
            painter.drawPath(g.symmetric(
                [(0, 5), (9, 20), (8, 33), (5, 38), (5, 57)], curved=False))

        else:
            # 동검: the waisted Korean blade, with its guard and grip.
            painter.drawPath(g.symmetric(
                [(0, 5), (7, 13), (4, 23), (8, 33), (5, 40)], curved=False))
            painter.setBrush(body)
            painter.drawPath(g.rect(20, 39, 24, 5))
            painter.drawPath(g.rect(28, 44, 8, 10))
            painter.drawPath(g.rect(23, 54, 18, 4))

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
        # Widened by BLADE_SCALE. Stations are left alone, so the tip angle,
        # the tang position and every difference that makes a type a type
        # survive the correction.
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
        widths = [w * BLADE_SCALE for w in profiles.get(variant, profiles["other"])]
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
        # Widened with the daggers, and for the same reason: five points that
        # differ only at the base have to be broad enough for the base to show.
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
        head = g.symmetric([(w * BLADE_SCALE, y)
                            for w, y in shapes.get(variant, shapes["leaf"])])
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
            mound = g.keyhole(head_cy=22, head_r=13, join_y=30,
                              foot_half=14, foot_y=57)
            painter.drawPath(mound)
        elif variant == "enpun":                        # 원분
            mound = g.circle(32, 32, 21)
            painter.drawPath(mound)
        elif variant == "hofun":                        # 방분
            mound = g.rect(13, 13, 38, 38)
            painter.drawPath(mound)
        elif variant == "hotategai":                    # 가리비형: short front
            mound = g.keyhole(head_cy=26, head_r=17, join_y=40,
                              foot_half=13, foot_y=56)
            painter.drawPath(mound)
        elif variant in ("zenpokoho", "makimuku_ho"):
            # 전방후방분: the same mound as 전방후원분 with a square rear, so
            # the two read as a pair. The numbers are the keyhole's, which is
            # what puts the shoulder in the same place in both.
            mound = g.poly([(19, 9), (45, 9), (45, 28), (42, 30),
                            (46, 57), (18, 57), (22, 30), (19, 28)])
            painter.drawPath(mound)
        elif variant == "sohochuen":                    # 쌍방중원분
            mound = g.spindle(r=16, waist_half=10, foot_half=7,
                              top_y=7, bottom_y=57)
            painter.drawPath(mound)
        elif variant == "yosumi":                       # 사우돌출형
            # The corners are the whole point of the type, so the sides have
            # to fall in between them. Sampling a circle, as this used to,
            # only produced an octagon with nothing protruding.
            mound = g.poly([
                (9, 9), (32, 16), (55, 9), (48, 32),
                (55, 55), (32, 48), (9, 55), (16, 32),
            ])
            painter.drawPath(mound)
        elif variant == "daijobo":                      # 대상묘: a low platform
            mound = g.rect(7, 20, 50, 24)
            painter.drawPath(mound)
        else:
            mound = g.circle(32, 32, 25)
            painter.drawPath(mound)

        # 마키무쿠형 is the one that carries a mark: the terraces on the front.
        # Clipped, because a terrace laid out from the bounding box runs past
        # the shoulder of the mound and out into the tile.
        if variant in ("makimuku_en", "makimuku_ho"):
            _clip_detail(painter, mound)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 1.2))
            painter.drawPath(g.line(20, 44, 44, 44))
            painter.drawPath(g.line(18, 50, 46, 50))
            painter.restore()

        painter.setBrush(old_brush)
        painter.setPen(old_pen)

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
            painter.drawPath(g.rect(13, 15, 38, 34))
            painter.setBrush(solid)
            painter.drawPath(g.rect(20, 22, 24, 20))

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
            painter.drawPath(g.rect(17, 18, 30, 28))
            painter.setBrush(ground_tone)
            painter.drawPath(g.rect(24, 25, 16, 14))

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
            painter.setPen(_pen(color, 5.0))
            painter.drawPath(g.rect(10, 10, 44, 44))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.rect(24, 21, 16, 22))

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
            floor(g.rect(12, 12, 40, 40),
                  [(20, 20), (44, 20), (20, 44), (44, 44)])

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
            painter.setPen(_pen(color, 5.0))
            painter.drawPath(g.circle(32, 32, 24))
            painter.setBrush(solid)
            painter.setPen(edge)
            for cx, cy in ((25, 26), (41, 29), (31, 41)):
                painter.drawPath(g.circle(cx, cy, 6.5))

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

        # Pulled in by VESSEL_SCALE, so the series keeps its proportions.
        #                  (half width, y) down the wall, in grid units
        profiles = {
            "comb_pattern":    ((18, 6), (18, 11), (16, 13), (5, 54), (1, 58)),
            "plain_coarse":    ((17, 6), (16, 22), (14, 42), (12, 57)),
            "red_burnished":   ((6, 6), (8, 11), (20, 26), (17, 44), (10, 57)),
            "black_burnished": ((5, 5), (5, 19), (19, 33), (16, 48), (9, 57)),
            "wajil":           ((8, 7), (9, 12), (21, 30), (14, 50), (4, 57)),
            "gyeongjil":       ((10, 6), (11, 11), (20, 30), (15, 47), (11, 56)),
            "storage_jar":     ((8, 6), (10, 11), (20, 26), (17, 46), (11, 56)),
            "siru":            ((19, 8), (16, 28), (12, 50), (11, 56)),
            "celadon":         ((5, 5), (6, 10), (20, 21), (14, 44), (9, 57)),
            "buncheong":       ((4, 5), (4, 19), (18, 37), (12, 57)),
            "white_porcelain": ((8, 7), (20, 20), (21, 33), (18, 46), (9, 57)),
            "onggi":           ((16, 15), (19, 30), (16, 46), (11, 56)),
            "gobae":           ((19, 8), (16, 14), (7, 22)),   # the dish only
        }
        body = g.symmetric([(w * VESSEL_SCALE, y) for w, y
                            in profiles.get(variant, profiles["plain_coarse"])],
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
        Stone, bronze and iron tools as one series of profiles.

        Almost everything here is a ``Grid.symmetric`` call over half-widths
        read down the object, which is what makes a stone dagger, an iron
        sword and a spearhead share a blade angle and a tang width instead of
        each having its own. Where a tool genuinely is not axial - a sickle,
        a bit, a chopper - it is built from arcs and rings on the same grid.

        The counts are held down deliberately. A microblade core drawn with
        nine scars and a lamellar cuirass drawn with a field of scales both
        read as a barcode at marker size, so each keeps the fewest marks that
        still says which type it is.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        edge = _pen(color, 2.6)
        thin = _pen(color, 1.4)

        painter.setPen(edge)
        painter.setBrush(solid)

        def detail(*shapes):
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for shape in shapes:
                painter.drawPath(shape)
            painter.setPen(edge)
            painter.setBrush(solid)

        # ---- 뗀석기 -----------------------------------------------------
        if variant == "handaxe":
            # 주먹도끼: a point at one end, a rounded butt at the other.
            blade = g.symmetric([(2, 6), (12, 20), (15, 36), (10, 52), (4, 57)],
                                curved=False)
            painter.drawPath(blade)
            _clip_detail(painter, blade)
            detail(g.poly([(23, 36), (32, 43), (41, 36)], close=False))
            painter.restore()

        elif variant == "chopper":
            # 찍개: a cobble with one end struck off. The worked edge is a
            # straight cut with two facets, not the row of saw teeth this
            # used to carry.
            cobble = g.poly([(28, 7), (46, 13), (54, 30), (46, 48),
                             (28, 56), (16, 44), (10, 28), (18, 14)])
            painter.drawPath(cobble)
            _clip_detail(painter, cobble)
            detail(g.line(13, 22, 25, 28), g.line(11, 32, 24, 34),
                   g.line(14, 42, 26, 41))
            painter.restore()

        elif variant == "tanged_point":
            # 슴베찌르개: a blade whose tang is a parallel-sided stem.
            painter.drawPath(g.symmetric(
                [(0, 6), (9, 19), (10, 33), (4, 40), (4, 57)], curved=False))

        elif variant == "microblade_core":
            # 좀돌날몸돌: a wedge with a striking platform. Three scars say
            # a worked face; nine said a fence.
            core = g.poly([(12, 12), (52, 12), (34, 56)])
            painter.drawPath(core)
            painter.setBrush(body)
            painter.drawPath(g.rect(10, 6, 44, 6))
            painter.setBrush(solid)
            _clip_detail(painter, core)
            detail(g.line(21, 15, 29, 50), g.line(31, 15, 34, 54),
                   g.line(41, 15, 39, 50))
            painter.restore()

        # ---- 간석기 -----------------------------------------------------
        elif variant == "polished_dagger":
            # 간돌검: blade, guard and grip, all on the blade series.
            painter.drawPath(g.symmetric(
                [(0, 5), (7, 17), (8, 31), (3, 37)], curved=False))
            painter.setBrush(body)
            painter.drawPath(g.rect(20, 36, 24, 5))
            painter.drawPath(g.rect(27, 41, 10, 12))
            painter.drawPath(g.rect(22, 53, 20, 5))

        elif variant == "semilunar_knife":
            # 반달돌칼: a straight back over a curved edge, drilled twice.
            painter.drawPath(g.symmetric(
                [(22, 22), (21, 29), (15, 38), (2, 44)], curved=True))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            painter.drawPath(g.circle(24, 28, 3))
            painter.drawPath(g.circle(40, 28, 3))

        elif variant == "stone_hoe":
            # 돌괭이: a narrow haft opening into a broad working edge, with
            # the binding across the neck.
            # Broad and squat with a bowed working edge. Given the axe's
            # narrow proportions it read as a bottle, and so did the axe.
            hoe = g.symmetric([(9, 14), (13, 25), (21, 44), (17.5, 53)],
                              curved=True)
            painter.drawPath(hoe)
            _clip_detail(painter, hoe)
            detail(g.line(21, 25, 43, 25), g.line(20, 29, 44, 29))
            painter.restore()

        elif variant == "grinding_slab":
            # 갈판과 갈돌: the quern and the handstone that works it - two
            # stones, so they are drawn apart rather than stacked into one
            # loaf.
            painter.setBrush(body)
            painter.drawPath(g.ellipse(32, 44, 26, 10))
            painter.setBrush(solid)
            painter.drawPath(g.ellipse(32, 27, 12, 7))
            detail(g.line(14, 41, 50, 41))

        elif variant == "stone_arrowhead":
            # 돌화살촉: a triangular point on a single stem.
            painter.drawPath(g.symmetric(
                [(0, 6), (11, 33), (4, 37), (4, 57)], curved=False))

        elif variant == "net_sinker":
            # 어망추: a river pebble with the groove the line was tied into.
            # The groove is a notch at each side, not a line straight
            # through the stone - drawn across, it reads as a division.
            pebble = g.ellipse(32, 32, 17, 24)
            painter.drawPath(pebble)
            _clip_detail(painter, pebble)
            painter.setPen(_pen(color, 2.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(13, 32, 21, 32))
            painter.drawPath(g.line(43, 32, 51, 32))
            painter.restore()

        # ---- 청동기 -----------------------------------------------------
        elif variant in ("coarse_mirror", "fine_mirror"):
            # 다뉴조문경 / 다뉴세문경: one disc with its two loops. The
            # decoration is the only difference, which is how the reference
            # plates separate variants of a type.
            painter.setBrush(body)
            painter.drawPath(g.circle(32, 32, 21))
            painter.setBrush(solid)
            painter.drawPath(g.circle(27, 32, 3))
            painter.drawPath(g.circle(37, 32, 3))
            if variant == "coarse_mirror":
                painter.setPen(_pen(color, 2.2))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(g.circle(32, 32, 15))
            else:
                painter.setPen(thin)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(g.circle(32, 32, 17))
                painter.drawPath(g.circle(32, 32, 13.5))
                painter.drawPath(g.circle(32, 32, 10))

        elif variant == "bronze_rattle":
            # 청동방울: a globular bell, slit down the face, hung by a loop.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.4))
            painter.drawPath(g.circle(32, 12, 6))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.ellipse(32, 38, 19, 20))
            detail(g.line(32, 30, 32, 52))

        elif variant == "bronze_bell":
            # 동탁: a flaring bell body under the same loop.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.4))
            painter.drawPath(g.circle(32, 11, 5))
            painter.setPen(edge)
            painter.setBrush(solid)
            bell = g.symmetric([(4, 17), (10, 30), (17, 48), (18, 53)],
                               curved=True)
            painter.drawPath(bell)
            _clip_detail(painter, bell)
            detail(g.line(15, 47, 49, 47))
            painter.restore()

        # ---- 철기 -------------------------------------------------------
        elif variant == "iron_sword":
            # 철검: the dagger series, drawn long.
            painter.drawPath(g.symmetric(
                [(0, 5), (6, 13), (6, 38), (3, 43)], curved=False))
            painter.setBrush(body)
            painter.drawPath(g.rect(21, 42, 22, 5))
            painter.drawPath(g.rect(28, 47, 8, 11))

        elif variant == "iron_spearhead":
            # 철모: a leaf blade over a socket.
            painter.drawPath(g.symmetric(
                [(0, 5), (10, 21), (9, 34), (6, 39), (6, 57)], curved=False))

        elif variant == "iron_arrowhead":
            # 철촉: a narrow point on a long tang.
            painter.drawPath(g.symmetric(
                [(0, 5), (11, 24), (3.5, 29), (3.5, 57)], curved=False))

        elif variant == "iron_axe":
            # 철부: a socket opening into a splayed edge.
            axe = g.symmetric([(9, 8), (10, 26), (16, 48), (15, 56)],
                              curved=False)
            painter.drawPath(axe)
            _clip_detail(painter, axe)
            detail(g.line(22, 26, 42, 26))
            painter.restore()

        elif variant == "iron_ard":
            # 따비: a share that comes to a point, which is what tells it
            # from the axe's edge.
            painter.drawPath(g.poly([(27, 8), (37, 8), (37, 27), (45, 44),
                                     (32, 57), (19, 44), (27, 27)]))

        elif variant == "iron_sickle":
            # 철겸: a hooked blade. Built as two arcs on one centre so the
            # back and the edge stay concentric.
            hook = QPainterPath()
            hook.moveTo(*g.pt(8, 44))
            g.arc(hook, 34, 44, 26, math.pi, 0.75 * math.pi, segments=6)
            hook.lineTo(*g.pt(34 + 18 * 0.7071, 44 - 18 * 0.7071))
            g.arc(hook, 34, 44, 18, 1.75 * math.pi, -0.75 * math.pi, segments=6)
            hook.closeSubpath()
            painter.drawPath(hook)
            painter.setBrush(body)
            painter.drawPath(g.rect(6, 44, 10, 11))

        elif variant in ("plate_armour", "lamellar_armour"):
            # 판갑 / 찰갑: one cuirass. 판갑 is a few wide riveted plates,
            # 찰갑 is many narrow laced rows - so the band count is the type,
            # and neither needs a field of scales to say so.
            cuirass = g.symmetric([(11, 11), (14, 21), (12.5, 37), (15, 52)],
                                  curved=True)
            painter.drawPath(cuirass)
            _clip_detail(painter, cuirass)
            if variant == "plate_armour":
                detail(g.line(16, 27, 48, 27), g.line(16, 42, 48, 42))
                painter.setBrush(solid)
                painter.setPen(thin)
                painter.drawPath(g.circle(24, 34, 2.5))
                painter.drawPath(g.circle(40, 34, 2.5))
            else:
                detail(g.line(16, 21, 48, 21), g.line(16, 31, 48, 31),
                       g.line(16, 41, 48, 41), g.line(16, 50, 48, 50))
            painter.restore()

        elif variant == "horse_bit":
            # 재갈: two cheek rings on a jointed mouthpiece.
            painter.setBrush(Qt.NoBrush)
            # The rings have to clear the mouthpiece; drawn closer they meet
            # in the middle and the whole thing reads as a bow tie.
            painter.setPen(_pen(color, 4.5))
            painter.drawPath(g.circle(13, 32, 9))
            painter.drawPath(g.circle(51, 32, 9))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.poly([(23, 26), (33, 32), (33, 41), (23, 38)]))
            painter.drawPath(g.poly([(41, 26), (31, 32), (31, 41), (41, 38)]))

        elif variant == "stirrup":
            # 등자: the suspension plate, the hoop and the tread.
            painter.setBrush(body)
            painter.drawPath(g.rect(25, 6, 14, 12))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 4.5))
            painter.drawPath(g.rect(12, 16, 40, 34, r=13))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.rect(14, 43, 36, 9))

        elif variant == "iron_ingot":
            # 철정: a bar ingot, waisted where it was gripped.
            painter.drawPath(g.poly([(19, 7), (45, 7), (38, 32), (45, 57),
                                     (19, 57), (26, 32)]))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Korean ornaments, tiles and other finds
    # ═══════════════════════════════════════════════════════

    def _draw_korean_ornament(self, painter, s, m, variant, color):
        """
        Ornaments, roof tiles and the other finds that are neither vessel
        nor tool.

        The strung ornaments (관옥, 유리구슬) share one cord and one bead
        rhythm; the two roof-ends share a stamped face; the rest are single
        objects built from the same primitives as everything else.

        Counts are the risk in this group. A crown with nine pairs of arms,
        a lotus of nine petals and a strand of eight beads all turn to grey
        at marker size, so each is cut to the fewest that still reads as
        a crown, a lotus, a strand.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        edge = _pen(color, 2.6)
        thin = _pen(color, 1.4)

        painter.setPen(edge)
        painter.setBrush(solid)

        def detail(*shapes):
            painter.setPen(thin)
            painter.setBrush(Qt.NoBrush)
            for shape in shapes:
                painter.drawPath(shape)
            painter.setPen(edge)
            painter.setBrush(solid)

        if variant == "gogok":
            # 곡옥: a thick comma, drilled through the head. Built from two
            # concentric arcs so the crescent keeps its width - at a 12px
            # outline a tapered one closes up and reads as a figure nine.
            painter.drawPath(g.comma(32, 33, 23, head_r=10, tail_r=19,
                                     start=-0.55 * math.pi,
                                     sweep=1.15 * math.pi))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            painter.drawPath(g.circle(28, 16, 3))

        elif variant == "gwanok":
            # 관옥: tubular beads on a cord.
            painter.setPen(_pen(color, 1.6))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.line(6, 32, 58, 32))
            painter.setPen(edge)
            painter.setBrush(solid)
            for x in (11, 26, 41):
                painter.drawPath(g.rect(x, 24, 12, 16, r=6))

        elif variant == "glass_bead":
            # 유리구슬: a strand. Five beads read as a strand; eight read as
            # a smudge.
            painter.setPen(_pen(color, 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(g.poly([(8, 22), (20, 36), (32, 40), (44, 36),
                                     (56, 22)], close=False))
            painter.setPen(edge)
            painter.setBrush(solid)
            for cx, cy in ((11, 26), (22, 37), (32, 40), (42, 37), (53, 26)):
                painter.drawPath(g.circle(cx, cy, 7))

        elif variant == "gold_earring":
            # 금귀걸이: the heavy hoop, its link, and a leaf pendant.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 5.5))
            painter.drawPath(g.circle(32, 18, 13))
            painter.setPen(_pen(color, 2.0))
            painter.drawPath(g.circle(32, 33, 4))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.symmetric([(1, 36), (13, 46), (0, 58)],
                                         curved=True))

        elif variant == "gold_crown":
            # 금관: the band and its three 出-shaped uprights. One pair of
            # arms each - three pairs each was a candelabra.
            painter.setBrush(solid)
            for x in (16, 32, 48):
                # The arms turn up at their tips. Drawn as plain crossbars
                # the three uprights read as a fence.
                painter.drawPath(g.poly([
                    (x - 3, 10), (x + 3, 10), (x + 3, 22), (x + 4, 22),
                    (x + 4, 14), (x + 7, 14), (x + 7, 28), (x + 3, 28),
                    (x + 3, 42), (x - 3, 42), (x - 3, 28), (x - 7, 28),
                    (x - 7, 14), (x - 4, 14), (x - 4, 22), (x - 3, 22),
                ]))
            painter.setBrush(body)
            painter.drawPath(g.rect(10, 41, 44, 11))

        elif variant == "belt_fitting":
            # 대금구: the belt, its plaques and one hanging strap.
            painter.setBrush(body)
            painter.drawPath(g.rect(6, 18, 52, 15))
            painter.setBrush(solid)
            for x in (13, 28, 43):
                painter.drawPath(g.rect(x, 22, 9, 7, r=1))
            painter.drawPath(g.poly([(27, 33), (37, 33), (37, 49), (32, 57),
                                     (27, 49)]))

        elif variant == "mokgan":
            # 목간: a writing slip, notched at the waist and pointed at the
            # foot, with the ink on it.
            slip = g.poly([(24, 7), (40, 7), (40, 23), (37, 26), (40, 29),
                           (40, 49), (32, 58), (24, 49), (24, 29), (27, 26),
                           (24, 23)])
            painter.drawPath(slip)
            _clip_detail(painter, slip)
            detail(g.line(28, 33, 36, 33), g.line(28, 38, 36, 38),
                   g.line(28, 43, 34, 43))
            painter.restore()

        elif variant == "round_roof_tile":
            # 수막새: the round tile end, stamped with its lotus. Six petals,
            # not nine - at 64 units nine is a texture, not a flower.
            painter.setBrush(body)
            painter.drawPath(g.circle(32, 32, 21))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(thin)
            for index in range(6):
                angle = math.pi / 2.0 + index * math.pi / 3.0
                painter.drawPath(g.circle(32 + 12 * math.cos(angle),
                                          32 + 12 * math.sin(angle), 5))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 32, 4.5))

        elif variant == "eaves_roof_tile":
            # 암막새: the eaves tile is a band with a drooping face, which is
            # what tells it from the round 수막새 at a glance.
            painter.setBrush(body)
            face = g.poly([(6, 19), (58, 19), (58, 32), (48, 42),
                           (16, 42), (6, 32)])
            painter.drawPath(face)
            painter.setBrush(solid)
            for x in (18, 32, 46):
                painter.drawPath(g.circle(x, 29, 5))

        elif variant == "floor_brick":
            # 전돌: a square floor tile with its stamped lozenge.
            painter.setBrush(body)
            painter.drawPath(g.rect(12, 12, 40, 40))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(_pen(color, 2.2))
            painter.drawPath(g.poly([(32, 18), (46, 32), (32, 46), (18, 32)]))
            painter.setPen(edge)
            painter.setBrush(solid)
            painter.drawPath(g.circle(32, 32, 5))

        elif variant == "inkstone":
            # 벼루: a slab with the well ground into one end.
            # A big round well beside a straight divider read as a car
            # stereo; the slab keeps a rim and a modest well instead.
            painter.setBrush(body)
            painter.drawPath(g.rect(7, 17, 50, 30, r=4))
            detail(g.rect(11, 21, 42, 22, r=3))
            painter.setBrush(solid)
            painter.drawPath(g.ellipse(19, 32, 6, 6))

        elif variant == "clay_figurine":
            # 토우: a modelled figure - head, body with arms, two legs.
            painter.drawPath(g.circle(32, 13, 7))
            painter.setBrush(body)
            painter.drawPath(g.poly([
                (24, 21), (40, 21), (40, 27), (51, 31), (49, 36), (40, 32),
                (40, 46), (24, 46), (24, 32), (15, 36), (13, 31), (24, 27),
            ]))
            painter.drawPath(g.rect(24, 46, 7, 11))
            painter.drawPath(g.rect(33, 46, 7, 11))

        elif variant == "chimi":
            # 치미: the ridge-end fin, hooked at the top and ribbed.
            # The notched leading edge and the base plate are what make it
            # an architectural ornament rather than a boot.
            fin = g.poly([(13, 51), (17, 35), (24, 21), (34, 11), (44, 7),
                          (47, 14), (40, 19), (46, 23), (38, 29), (38, 40),
                          (41, 51)])
            painter.drawPath(fin)
            _clip_detail(painter, fin)
            detail(g.poly([(21, 50), (24, 34), (32, 20)], close=False),
                   g.poly([(29, 50), (30, 36), (37, 24)], close=False))
            painter.restore()
            painter.setBrush(body)
            painter.drawPath(g.rect(11, 51, 32, 7))

        elif variant == "foundation_stone":
            # 초석: the squared footing and the seat cut for the pillar.
            painter.setBrush(body)
            painter.drawPath(g.rect(8, 32, 48, 21, r=2))
            painter.drawPath(g.rect(17, 23, 30, 9, r=1))
            painter.setBrush(solid)
            painter.drawPath(g.ellipse(32, 23, 11, 5))

        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Human Remains
    # ═══════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Features
    # ═══════════════════════════════════════════════════════

    def _draw_ash_layer(self, painter, s, m, color):
        """
        재층: an ash lens sitting in the deposit it was found in.

        Drawn as a stack of even rules this read as a barcode, and it was
        also the wrong picture - ash arrives as a lens, not as bedding.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.2))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.SOFT))
        painter.drawPath(g.rect(6, 15, 52, 35))
        painter.setBrush(QColor(color))
        painter.drawPath(g.ellipse(32, 32, 20, 8))
        painter.setPen(_pen(color, 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(g.line(13, 22, 24, 22))
        painter.drawPath(g.line(41, 43, 52, 43))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    # ═══════════════════════════════════════════════════════
    #  Drawing methods — Survey / General
    # ═══════════════════════════════════════════════════════

    def _draw_excavation(self, painter, s, m, color):
        """발굴 구역: the open area, laid out on its grid."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.SOFT))
        painter.drawPath(g.rect(7, 7, 50, 50))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color, 1.4, Qt.DashLine))
        for offset in (24, 40):
            painter.drawPath(g.line(7, offset, 57, offset))
            painter.drawPath(g.line(offset, 7, offset, 57))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_north_arrow(self, painter, s, m, color):
        """방위표: the map dart, one half solid so the point reads at a glance."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color))
        painter.drawPath(g.poly([(32, 5), (32, 51), (19, 42)]))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.MID))
        painter.drawPath(g.poly([(32, 5), (45, 42), (32, 51)]))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_scale_bar(self, painter, s, m, color):
        """축척 막대: alternating segments over a baseline."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        solid = QColor(color)
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        painter.setPen(_pen(color, 2.4))
        for index in range(4):
            painter.setBrush(solid if index % 2 == 0 else body)
            painter.drawPath(g.rect(7 + index * 12, 25, 12, 12, r=0))
        painter.setPen(_pen(color, 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(g.line(7, 43, 7, 48))
        painter.drawPath(g.line(55, 43, 55, 48))
        painter.drawPath(g.line(7, 46, 55, 46))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_harris_matrix_context(self, painter, s, m, color):
        """
        해리스 매트릭스 단위: a context and the two relationships that make
        it a matrix - what it lies under, and what it lies over.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        body = QColor(color.red(), color.green(), color.blue(), icon_grid.MID)
        painter.setPen(_pen(color, 1.6))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(g.line(32, 13, 32, 24))
        painter.drawPath(g.line(32, 40, 32, 51))
        painter.setPen(_pen(color, 2.2))
        painter.setBrush(body)
        painter.drawPath(g.rect(20, 6, 24, 8, r=2))
        painter.drawPath(g.rect(20, 50, 24, 8, r=2))
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color))
        painter.drawPath(g.rect(13, 23, 38, 18, r=2))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_stratigraphic_unit(self, painter, s, m, color):
        """
        층위 단위: layers of unequal thickness on tilted contacts.

        Even bands of even spacing are a barcode, not a section - the ground
        does not deposit itself in equal rules.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.2))
        for points, level in (
            ([(10, 17), (54, 14), (54, 26), (10, 29)], icon_grid.SOFT),
            ([(10, 29), (54, 26), (54, 33), (10, 36)], icon_grid.MID),
            ([(10, 36), (54, 33), (54, 48), (10, 51)], icon_grid.SOLID),
        ):
            painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                    level))
            painter.drawPath(g.poly(points))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_survey_point(self, painter, s, m, color):
        """조사 지점: a crosshair over its station."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.MID))
        painter.drawPath(g.circle(32, 32, 21))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color, 2.2))
        painter.drawPath(g.line(32, 6, 32, 58))
        painter.drawPath(g.line(6, 32, 58, 32))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_find_spot(self, painter, s, m, color):
        """유물 출토 지점: the map pin."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color))
        painter.drawPath(g.circle(32, 23, 17))
        painter.drawPath(g.poly([(20, 32), (44, 32), (32, 57)]))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color, 2.6))
        painter.drawPath(g.circle(32, 23, 6))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_trench(self, painter, s, m, color):
        """
        트렌치: a long narrow cut with its section face marked.

        The eleven hatch lines this used to carry read as a barcode; what
        actually says trench is the proportion.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.SOFT))
        painter.drawPath(g.rect(5, 24, 54, 16))
        painter.setBrush(QColor(color))
        painter.drawPath(g.rect(5, 24, 8, 16, r=0))
        painter.setPen(_pen(color, 1.6))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(g.line(13, 24, 13, 40))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_datum_point(self, painter, s, m, color):
        """기준점: the survey triangle over its centre."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.MID))
        painter.drawPath(g.poly([(32, 7), (57, 51), (7, 51)]))
        painter.setBrush(QColor(color))
        painter.drawPath(g.circle(32, 37, 6))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_photo_point(self, painter, s, m, color):
        """사진 촬영 지점: the camera and the view it covers."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 1.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.SOFT))
        painter.drawPath(g.poly([(31, 32), (57, 13), (57, 51)]))
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.MID))
        painter.drawPath(g.rect(7, 21, 25, 22, r=3))
        painter.setBrush(QColor(color))
        painter.drawPath(g.circle(19, 32, 7))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_grid_corner(self, painter, s, m, color):
        """그리드 모서리: the corner two grid lines meet at."""
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.MID))
        painter.drawPath(g.poly([(13, 8), (21, 8), (21, 43), (56, 43),
                                 (56, 51), (13, 51)]))
        painter.setBrush(QColor(color))
        painter.drawPath(g.rect(11, 41, 12, 12, r=2))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_sample_location(self, painter, s, m, color):
        """
        시료 채취 지점: the sample tube itself.

        A dot with a sliver through it says nothing; a stoppered vial with
        something in it says what was done here.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.6))
        tube = g.rect(23, 12, 18, 44, r=8)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.MID))
        painter.drawPath(tube)
        _clip_detail(painter, tube)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.NoPen)
        painter.drawPath(g.rect(23, 32, 18, 24, r=0))
        painter.restore()
        painter.setPen(_pen(color, 2.6))
        painter.setBrush(QColor(color))
        painter.drawPath(g.rect(21, 6, 22, 8, r=2))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

    def _draw_test_pit(self, painter, s, m, color):
        """
        시굴 피트: the small square cut. Dashed against 발굴 구역's solid
        edge, and crossed rather than gridded, so the two do not collide.
        """
        g = icon_grid.Grid(s)
        old_pen, old_brush = painter.pen(), painter.brush()
        painter.setPen(_pen(color, 2.4, Qt.DashLine))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(),
                                icon_grid.SOFT))
        painter.drawPath(g.rect(13, 13, 38, 38))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(_pen(color, 1.6))
        painter.drawPath(g.line(19, 19, 45, 45))
        painter.drawPath(g.line(45, 19, 19, 45))
        painter.setPen(old_pen)
        painter.setBrush(old_brush)

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
