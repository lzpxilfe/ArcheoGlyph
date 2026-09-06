# -*- coding: utf-8 -*-
"""
Translation for the plugin's own strings.

Qt's ``.qm`` route is left in place in ``archeoglyph.py``, but this plugin has
no compiled catalogues and no Qt Linguist tooling in its development
environment, so translations live here as plain Python dictionaries. That
makes them testable without QGIS: every string the UI shows can be checked
for a missing or malformed translation by ``tests/test_i18n.py``.

Nothing here imports QGIS at module level. ``language_from_settings`` is the
only function that touches QSettings, and it degrades to English.
"""

from __future__ import annotations

from typing import Dict, Tuple

LANGUAGE_SETTING = "ArcheoGlyph/language"

#: Language codes with a catalogue, in the order the settings combo shows them.
LANGUAGES: Tuple[Tuple[str, str], ...] = (
    ("auto", "Automatic (follow QGIS)"),
    ("en", "English"),
    ("ko", "한국어"),
)

#: Codes that ``set_language`` accepts. ``auto`` is a setting, not a language.
TRANSLATED_LANGUAGES = ("ko",)

_CATALOGS: Dict[str, Dict[str, str]] = {}
_current = "en"


def available_languages() -> Tuple[Tuple[str, str], ...]:
    """(code, label) pairs for the settings combo, ``auto`` first."""
    return LANGUAGES


def _catalog(code: str) -> Dict[str, str]:
    """Load a catalogue on first use; an unknown code yields an empty one."""
    if code in _CATALOGS:
        return _CATALOGS[code]
    catalog: Dict[str, str] = {}
    if code == "ko":
        from .i18n_ko import CATALOG

        catalog = CATALOG
    _CATALOGS[code] = catalog
    return catalog


def resolve_language(setting: str, locale: str = "") -> str:
    """
    Decide the language from the stored setting and the QGIS locale.

    An explicit ``en``/``ko`` wins. ``auto`` (or anything unrecognised) falls
    back to the locale's first subtag, and to English when that has no
    catalogue. This is pure so it can be tested exhaustively.
    """
    chosen = str(setting or "").strip().lower().replace("-", "_")
    if chosen and chosen != "auto":
        base = chosen.split("_")[0]
        return base if base in TRANSLATED_LANGUAGES else "en"

    base = str(locale or "").strip().lower().replace("-", "_").split("_")[0]
    return base if base in TRANSLATED_LANGUAGES else "en"


def language_from_settings() -> str:
    """Resolve the language from QSettings; English when QGIS is absent."""
    try:
        from qgis.PyQt.QtCore import QSettings

        settings = QSettings()
        return resolve_language(
            str(settings.value(LANGUAGE_SETTING, "auto") or "auto"),
            str(settings.value("locale/userLocale", "") or ""),
        )
    except Exception:
        return "en"


def set_language(code: str) -> str:
    """Set the active language and return the code actually applied."""
    global _current
    _current = resolve_language(code)
    return _current


def apply_settings_language() -> str:
    """Set the active language from QSettings. Called when a dialog opens."""
    global _current
    _current = language_from_settings()
    return _current


def current_language() -> str:
    return _current


def tr(text: str) -> str:
    """
    Translate a UI string.

    The English source text is the key, so an untranslated string still
    displays correctly instead of showing a placeholder.
    """
    if _current == "en":
        return text
    return _catalog(_current).get(text, text)
