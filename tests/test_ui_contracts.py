"""
Static contracts for the dialogs.

The UI needs QGIS to run, so it cannot be exercised here, but its source can be
parsed. These checks catch the mistakes that actually happened during this
work: an attribute read after its assignment was edited away, a worker signal
shadowing QThread.finished, and a backend that existed but could not be
selected in the settings dialog.
"""

import ast
import pathlib
import re

from archeoglyph.generators.autotrace import model_store, segment

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIALOGS = ["ui/settings_dialog.py", "ui/main_dialog.py"]
SNAKE_CASE = re.compile(r"^[a-z_][a-z0-9_]*$")
# Inherited Qt slots that are legitimately passed by reference, e.g.
# button.clicked.connect(self.close).
QT_SLOT_REFERENCES = {
    "close", "show", "hide", "accept", "reject", "update", "repaint",
    "deleteLater", "showNormal", "showMaximized",
}


def _classes(path):
    source = (ROOT / path).read_text(encoding="utf-8-sig")
    return source, [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)]


def test_every_dialog_attribute_is_assigned_before_use():
    """
    Any self.<snake_case> that is read must also be assigned somewhere in the
    class. Qt's inherited API is camelCase, so it is naturally excluded.
    """
    problems = []
    for path in DIALOGS:
        _source, classes = _classes(path)
        for cls in classes:
            assigned, read, called = set(), {}, set()
            for node in ast.walk(cls):
                # self.foo(...) is a method call, not an attribute to assign.
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    called.add(node.func.attr)
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    if isinstance(node.ctx, ast.Store):
                        assigned.add(node.attr)
                    else:
                        read.setdefault(node.attr, node.lineno)
            read = {name: line for name, line in read.items() if name not in called}
            defined = {
                n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for statement in cls.body:
                for target in getattr(statement, "targets", []):
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            for name, line in read.items():
                if name in assigned or name in defined:
                    continue
                if not SNAKE_CASE.match(name) or name in QT_SLOT_REFERENCES:
                    continue  # inherited Qt method or slot reference
                problems.append(f"{path}:{line} {cls.name}.{name} is read but never assigned")
    assert not problems, "\n".join(problems)


def test_worker_threads_do_not_shadow_qthread_finished():
    """
    A custom signal named `finished` hides QThread's own, which breaks the
    finished -> deleteLater idiom and confuses lifetime handling.
    """
    offenders = []
    for path in DIALOGS:
        _source, classes = _classes(path)
        for cls in classes:
            bases = {getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases}
            if "QThread" not in bases:
                continue
            for statement in cls.body:
                targets = [t.id for t in getattr(statement, "targets", []) if isinstance(t, ast.Name)]
                if "finished" in targets:
                    offenders.append(f"{path}: {cls.name} defines its own `finished` signal")
    assert not offenders, "\n".join(offenders)


def test_every_mask_backend_can_be_selected_in_settings():
    """
    Each backend the segmentation layer accepts must be offered in the
    settings dialog; otherwise the code path is unreachable.
    """
    source = (ROOT / "ui/settings_dialog.py").read_text(encoding="utf-8-sig")
    offered = set(re.findall(r'mask_backend_combo\.addItem\(.*?"([a-z]+)"\s*\)', source))
    assert offered == set(segment.MASK_BACKENDS), (
        f"settings offers {sorted(offered)}, segmentation accepts {sorted(segment.MASK_BACKENDS)}"
    )


def test_every_downloadable_model_is_offered_and_wired():
    """The model catalogue, the settings combo and the stored key must agree."""
    source = (ROOT / "ui/settings_dialog.py").read_text(encoding="utf-8-sig")
    assert "onnx_model_combo.addItem(spec.label, key)" in source
    assert "ArcheoGlyph/onnx_bg_model" in source

    generator = (ROOT / "generators/contour_generator.py").read_text(encoding="utf-8")
    assert "ArcheoGlyph/onnx_bg_model" in generator, "the generator must read the chosen model"
    assert model_store.DEFAULT_MODEL_KEY in model_store.MODEL_SPECS


def test_settings_keys_written_by_the_ui_are_read_somewhere():
    """
    Every ArcheoGlyph/* key the settings dialog writes must be read by some
    other module, so no setting is silently inert.
    """
    settings_source = (ROOT / "ui/settings_dialog.py").read_text(encoding="utf-8-sig")
    written = set(re.findall(r"setValue\(\s*'(ArcheoGlyph/[a-z_]+)'", settings_source))

    readers = ""
    for path in ROOT.rglob("*.py"):
        if "ui/settings_dialog.py" in str(path) or "/tests/" in str(path):
            continue
        readers += path.read_text(encoding="utf-8-sig")

    inert = sorted(key for key in written if key not in readers)
    # model_refresh_last_checked_utc is bookkeeping for the dialog itself.
    inert = [key for key in inert if not key.endswith("last_checked_utc")]
    assert not inert, f"settings written but never read: {inert}"


def test_translated_combos_are_read_by_data_not_by_label():
    """
    A style or template chosen by its label breaks the moment the label is
    translated: the value is what keys TEMPLATE_INFO and normalize_style.
    """
    source = (ROOT / "ui/main_dialog.py").read_text(encoding="utf-8-sig")
    offenders = []
    for combo in ("style_combo", "template_combo"):
        for method in ("currentText()", "setCurrentText(", "findText("):
            if f"{combo}.{method}" in source:
                offenders.append(f"main_dialog.py uses {combo}.{method} instead of the item data")
    assert not offenders, "\n".join(offenders)


# Call sites whose string argument is shown to the user. Widened deliberately:
# a false positive is a one-word fix, a missed string is an untranslated dialog.
TRANSLATED_WIDGETS = {
    "QLabel", "QPushButton", "QGroupBox", "QCheckBox", "QRadioButton",
    "QAction", "QToolButton", "QCommandLinkButton",
}
TRANSLATED_METHODS = {
    "setText", "setToolTip", "setWindowTitle", "setPlaceholderText", "setTitle",
    "setStatusTip", "setWhatsThis", "setSuffix", "setPrefix", "setLabelText",
    "setInformativeText", "setDetailedText",
}
# owner.method -> positional argument indices carrying user-facing text
TRANSLATED_DIALOG_ARGS = {
    ("QMessageBox", "warning"): (1, 2),
    ("QMessageBox", "information"): (1, 2),
    ("QMessageBox", "critical"): (1, 2),
    ("QMessageBox", "question"): (1, 2),
    ("QInputDialog", "getText"): (1, 2),
    ("QFileDialog", "getOpenFileName"): (1,),
    ("QFileDialog", "getSaveFileName"): (1,),
    ("QFileDialog", "getExistingDirectory"): (1,),
}


def _untranslated_at_call_sites(path):
    """Bare string literals at user-facing call sites, with their line."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    found = []

    def check(node, why):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip():
            found.append((node.lineno, why, node.value))
        elif isinstance(node, ast.JoinedStr):
            found.append((node.lineno, why, "<f-string>"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in TRANSLATED_WIDGETS and node.args:
            check(node.args[0], func.id)
        elif isinstance(func, ast.Attribute):
            key = (getattr(func.value, "id", None), func.attr)
            if key in TRANSLATED_DIALOG_ARGS:
                for index in TRANSLATED_DIALOG_ARGS[key]:
                    if index < len(node.args):
                        check(node.args[index], f"{key[0]}.{key[1]} arg{index}")
            elif func.attr == "addItem" and node.args:
                check(node.args[0], "addItem")
            elif func.attr == "addItems" and node.args:
                if isinstance(node.args[0], (ast.List, ast.Tuple)):
                    for element in node.args[0].elts:
                        check(element, "addItems")
            elif func.attr == "addTab" and len(node.args) > 1:
                check(node.args[1], "addTab")
            elif func.attr in TRANSLATED_METHODS and node.args:
                check(node.args[0], func.attr)
    return found


def test_every_user_facing_string_goes_through_tr():
    """
    A literal at a user-facing call site is a string that will show in English
    inside a translated dialog. f-strings are flagged too: they cannot be
    translated at all and must become tr("...{name}...").format(name=...).
    """
    problems = []
    for path in DIALOGS + ["archeoglyph.py"]:
        for line, why, text in _untranslated_at_call_sites(path):
            problems.append(f"{path}:{line} ({why}) {text[:60]!r} is not inside tr(...)")
    assert not problems, (
        "user-facing strings that bypass translation:\n" + "\n".join(problems)
    )
