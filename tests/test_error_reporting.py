"""
Failures in the generation path must say something.

The expensive bugs in this plugin have all been silent ones: a step fails, the
code returns None, and the user sees a worse symbol with no way to tell why.
This walks the source for `except Exception` handlers that swallow a failure
without logging, and requires each remaining one to be listed with a reason.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGGERS = {"log", "log_exception", "logMessage"}

# Handlers that are deliberately silent: (module, what the handler guards).
ALLOWED_SILENT = {
    # Probing for something optional is not a failure.
    ("generators/contour_generator.py", "the QGIS profile directory probe"),
    ("generators/symbol_result.py", "importing Qt outside QGIS"),
    # A logging call that fails must not raise from inside a logging guard.
    ("generators/huggingface_generator.py", "logging the prompt influence"),
    ("generators/huggingface_generator.py", "logging the image parameters"),
    # A score refinement that cannot be computed just leaves the score as is.
    ("generators/autotrace/sam_backend.py", "an optional mask score bonus"),
}


def _silent_handlers(path):
    """Broad handlers whose whole body is `pass` or a bare `return`."""
    source = (ROOT / path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            broad = handler.type is None or getattr(handler.type, "id", "") in (
                "Exception", "BaseException"
            )
            if not broad:
                continue
            body = handler.body
            swallows = all(isinstance(st, ast.Pass) for st in body) or (
                len(body) == 1 and isinstance(body[0], ast.Return)
            )
            if not swallows:
                continue
            calls = {
                getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                for call in ast.walk(handler)
                if isinstance(call, ast.Call)
            }
            if calls & LOGGERS:
                continue
            found.append(handler.lineno)
    return found


def test_generation_failures_are_reported():
    """
    Every silent broad handler in the generators must be accounted for.

    The count per module is checked rather than the line numbers, so ordinary
    edits do not churn the test while a newly added silent handler still
    fails it.
    """
    expected = {}
    for module, _reason in ALLOWED_SILENT:
        expected[module] = expected.get(module, 0) + 1

    problems = []
    for path in sorted(ROOT.joinpath("generators").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        found = _silent_handlers(relative)
        allowed = expected.get(relative, 0)
        if len(found) > allowed:
            problems.append(
                f"{relative}: {len(found)} silent handlers at lines {found}, "
                f"{allowed} accounted for. Log the failure, or add it to "
                f"ALLOWED_SILENT with a reason."
            )
    assert not problems, "\n".join(problems)


def test_the_allowlist_has_no_stale_entries():
    """An entry for a handler that is no longer silent hides a real one."""
    stale = []
    for module, reason in ALLOWED_SILENT:
        if not _silent_handlers(module):
            stale.append(f"{module} ({reason}) no longer has a silent handler")
    assert not stale, "\n".join(stale)
