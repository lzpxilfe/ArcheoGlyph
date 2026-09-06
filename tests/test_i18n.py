"""
The translation catalogue, checked without QGIS.

Two things can silently break a translated UI: a string the code translates
that has no entry (it shows in English inside a Korean dialog), and an entry
whose placeholders do not match the source (it raises KeyError at format
time, in front of the user). Both are checked here by parsing the source.
"""

import ast
import pathlib
import re

import pytest

from archeoglyph import i18n
from archeoglyph.generators.style_utils import STYLE_OPTIONS
from archeoglyph.generators.template_catalog import TEMPLATE_INFO
from archeoglyph.i18n_ko import CATALOG

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)")
TRANSLATED_SOURCES = [
    "archeoglyph.py",
    "ui/main_dialog.py",
    "ui/settings_dialog.py",
    "generators/template_generator.py",
]


@pytest.fixture(autouse=True)
def _english_by_default():
    """Language is process-global; do not leak a change into other tests."""
    previous = i18n.current_language()
    yield
    i18n.set_language(previous)


def _translated_literals():
    """Every literal passed to tr(...) anywhere in the plugin."""
    found = {}
    for name in TRANSLATED_SOURCES:
        path = ROOT / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            label = getattr(func, "id", None) or getattr(func, "attr", None)
            if label != "tr" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, f"{name}:{first.lineno}")
    return found


def _class_level_ui_strings():
    """
    User-facing text held in class-level tables in the dialogs.

    These reach tr() as a variable, so scanning for tr("...") cannot see them;
    they are collected from the source instead.
    """
    tables = ("MODE_DESCRIPTION", "TEMPLATE_CATEGORY_LABELS")
    found = set()
    tree = ast.parse((ROOT / "ui/main_dialog.py").read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", "") in tables for t in node.targets):
            continue
        for value in ast.walk(node.value):
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.add(value.value)
    return found


def _template_names():
    """
    Template names reach tr() through template_display_name(), so they are
    passed as a variable and scanning for tr("...") cannot see them.
    """
    return list(TEMPLATE_INFO)


# -- language resolution ------------------------------------------------

@pytest.mark.parametrize("setting,locale,expected", [
    ("auto", "ko_KR", "ko"),
    ("auto", "ko", "ko"),
    ("auto", "en_US", "en"),
    ("auto", "", "en"),
    ("auto", "fr_FR", "en"),
    ("ko", "en_US", "ko"),       # explicit choice beats the locale
    ("en", "ko_KR", "en"),
    ("ko_KR", "en_US", "ko"),    # a full locale as the setting
    ("", "ko_KR", "ko"),         # unset behaves like auto
    ("KO", "en_US", "ko"),       # case-insensitive
    ("ko-KR", "", "ko"),         # hyphenated
    ("zz", "ko_KR", "en"),       # unknown explicit choice is not auto
    (None, None, "en"),
])
def test_resolve_language(setting, locale, expected):
    assert i18n.resolve_language(setting, locale) == expected


def test_language_from_settings_is_english_without_qgis():
    assert i18n.language_from_settings() == "en"


def test_auto_is_offered_first_and_every_catalogue_is_listed():
    codes = [code for code, _label in i18n.available_languages()]
    assert codes[0] == "auto"
    for code in i18n.TRANSLATED_LANGUAGES:
        assert code in codes


# -- fallback behaviour -------------------------------------------------

def test_english_returns_the_source_text():
    i18n.set_language("en")
    assert i18n.tr("ArchaeoGlyph Symbol Generator") == "ArchaeoGlyph Symbol Generator"


def test_unknown_string_falls_back_to_the_source_text():
    i18n.set_language("ko")
    assert i18n.tr("a string nobody translated") == "a string nobody translated"


def test_unknown_language_falls_back_to_english():
    assert i18n.set_language("zz") == "en"
    assert i18n.tr("ArchaeoGlyph Symbol Generator") == "ArchaeoGlyph Symbol Generator"


def test_korean_translation_is_used():
    i18n.set_language("ko")
    assert i18n.tr("ArchaeoGlyph Symbol Generator") != "ArchaeoGlyph Symbol Generator"


# -- catalogue health ---------------------------------------------------

def test_no_entry_is_empty():
    empty = sorted(k for k, v in CATALOG.items() if not str(v).strip())
    assert not empty, f"empty Korean translations: {empty}"


def test_every_translated_string_has_an_entry():
    missing = sorted(
        f"{where} {text!r}"
        for text, where in _translated_literals().items()
        if text not in CATALOG
    )
    assert not missing, "strings passed to tr() with no Korean entry:\n" + "\n".join(missing)


def test_no_orphaned_entries():
    """An entry whose source string is no longer in the code is dead weight."""
    used = (
        set(_translated_literals())
        | set(STYLE_OPTIONS)
        | set(_template_names())
        # Language labels reach tr() as a variable too.
        | {label for _code, label in i18n.LANGUAGES}
        | _class_level_ui_strings()
    )
    orphans = sorted(key for key in CATALOG if key not in used)
    assert not orphans, "Korean entries no longer used by the code:\n" + "\n".join(orphans)


def test_placeholders_match_between_source_and_translation():
    """
    A translation is used as a .format() template, so a renamed or dropped
    placeholder raises KeyError in front of the user.
    """
    problems = []
    for source, translation in CATALOG.items():
        if set(PLACEHOLDER.findall(source)) != set(PLACEHOLDER.findall(translation)):
            problems.append(f"{source!r} -> {translation!r}")
    assert not problems, "placeholder mismatch:\n" + "\n".join(problems)


def test_every_template_name_is_translated():
    """
    A template's English name is its identifier; the combo shows tr(name), so
    a name with no entry appears in English inside a Korean list.
    """
    missing = sorted(name for name in _template_names() if name not in CATALOG)
    assert not missing, "templates with no Korean name:\n" + "\n".join(missing)


def test_template_display_names_are_unique():
    """
    Two templates sharing a Korean label are indistinguishable in the combo,
    where only the label is shown.
    """
    i18n.set_language("ko")
    seen = {}
    clashes = []
    for name in _template_names():
        label = i18n.tr(name)
        if label in seen:
            clashes.append(f"{seen[label]} and {name} both show as {label!r}")
        seen[label] = name
    assert not clashes, "\n".join(clashes)


def test_help_documents_are_provided_in_both_languages():
    """
    The two long help pages live outside the catalogue, so nothing else checks
    that the Korean version exists or that it switched.
    """
    from archeoglyph.ui import help_text

    for producer in (help_text.local_sd_setup_html, help_text.help_html):
        i18n.set_language("en")
        english = producer()
        i18n.set_language("ko")
        korean = producer()

        assert english.strip() and korean.strip()
        assert english != korean, f"{producer.__name__} returns English for Korean"
        assert any("가" <= ch <= "힣" for ch in korean), (
            f"{producer.__name__} has no Hangul in its Korean version"
        )
        # Both must stay valid fragments that Qt's rich text can render.
        for document in (english, korean):
            assert document.count("<ul>") == document.count("</ul>")
            assert document.count("<ol>") == document.count("</ol>")
            assert document.count("<li>") == document.count("</li>")
