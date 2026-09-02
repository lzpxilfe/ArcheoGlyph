# -*- coding: utf-8 -*-
"""
Pytest bootstrap.

1. Loads the plugin directory as the package ``archeoglyph`` regardless of the
   checkout folder name, so tests can use ``from archeoglyph.generators import x``.
2. Installs lightweight stand-ins for the ``qgis`` modules that image code
   still imports at module level (only ``QSettings`` is used there), so those
   modules import under plain Python. QGIS-facing UI modules are not imported
   by the unit tests.
"""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "archeoglyph"


class _FakeQSettings:
    """Dict-backed QSettings replacement (process-wide store)."""

    _store = {}

    def value(self, key, default=None, type=None):  # noqa: A002 - mirrors Qt API
        value = self._store.get(key, default)
        if type is bool and not isinstance(value, bool):
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        if type is int:
            try:
                return int(value)
            except Exception:
                return default
        return value

    def setValue(self, key, value):
        self._store[key] = value

    def remove(self, key):
        self._store.pop(key, None)

    def contains(self, key):
        return key in self._store

    @classmethod
    def reset(cls):
        cls._store.clear()


def _install_qgis_stubs():
    if "qgis" in sys.modules:
        return
    qgis = types.ModuleType("qgis")
    pyqt = types.ModuleType("qgis.PyQt")
    qtcore = types.ModuleType("qgis.PyQt.QtCore")
    qtcore.QSettings = _FakeQSettings
    core = types.ModuleType("qgis.core")

    class _QgsApplication:
        @staticmethod
        def qgisSettingsDirPath():
            return str(ROOT / ".pytest_qgis_profile") + "/"

    core.QgsApplication = _QgsApplication
    qgis.PyQt = pyqt
    qgis.core = core
    pyqt.QtCore = qtcore
    sys.modules["qgis"] = qgis
    sys.modules["qgis.PyQt"] = pyqt
    sys.modules["qgis.PyQt.QtCore"] = qtcore
    sys.modules["qgis.core"] = core


def _install_plugin_package():
    if PACKAGE_NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)


_install_qgis_stubs()
_install_plugin_package()
