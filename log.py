# -*- coding: utf-8 -*-
"""
ArchaeoGlyph logging helper.

Routes messages to the QGIS message log when running inside QGIS and falls
back to stderr otherwise (unit tests, command-line experiments).
"""

import sys

_TAG = "ArchaeoGlyph"

try:  # pragma: no cover - only available inside QGIS
    from qgis.core import Qgis, QgsMessageLog

    _LEVELS = {
        "info": Qgis.Info,
        "warning": Qgis.Warning,
        "critical": Qgis.Critical,
    }
except Exception:  # pragma: no cover - plain Python
    Qgis = None
    QgsMessageLog = None
    _LEVELS = {}


def log(message, level="info"):
    """Log a message at ``level`` ("info", "warning" or "critical")."""
    text = str(message)
    if QgsMessageLog is not None:
        try:
            QgsMessageLog.logMessage(text, _TAG, _LEVELS.get(level, _LEVELS["info"]))
            return
        except Exception:
            pass
    sys.stderr.write(f"[{_TAG}] {level}: {text}\n")


def log_exception(context, exc):
    """Log an exception with its context as a warning."""
    log(f"{context}: {type(exc).__name__}: {exc}", level="warning")
