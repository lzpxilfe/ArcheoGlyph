# -*- coding: utf-8 -*-
"""
ArcheoGlyph - Archaeological Symbol Generator for QGIS
A QGIS plugin that generates accurate, standardized symbols from 
archaeological artifact/feature images.
"""


def classFactory(iface):
    """Load ArcheoGlyph class from file archeoglyph.
    
    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .archeoglyph import ArcheoGlyph
    return ArcheoGlyph(iface)
