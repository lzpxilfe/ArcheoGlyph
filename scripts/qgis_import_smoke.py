#!/usr/bin/env python3
"""
Import smoke test: every shipped module must at least compile and import.

QGIS-free modules are imported for real; modules that need QGIS are only
compiled, so this runs in CI without a QGIS installation.
"""

import compileall
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "archeoglyph"

QGIS_FREE_MODULES = [
    "symbol_breaks",
    "i18n",
    "i18n_ko",
    "diagnostics",
    "generators.symbol_result",
    "generators.svg_sanitize",
    "generators.shape_match",
    "generators.ink_centerline",
    "generators.raster_vectorize",
    "generators.image_ops",
    "generators.style_utils",
    "generators.template_catalog",
    "generators.subject_terms",
    "generators.style_control_utils",
    "generators.autotrace.options",
    "generators.autotrace.io",
    "generators.autotrace.segment",
    "generators.autotrace.geometry",
    "generators.autotrace.colors",
    "generators.autotrace.enhance",
    "generators.autotrace.structure",
    "generators.autotrace.round_motif",
    "generators.autotrace.lines",
    "generators.autotrace.pipeline",
    "generators.autotrace.svg_builder",
    "generators.autotrace.model_store",
]


def _load_package():
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)


def main():
    ok = compileall.compile_dir(
        str(ROOT), quiet=1, rx=__import__("re").compile(r"(\.git|__pycache__|\.pytest)")
    )
    if not ok:
        print("FAILED: some modules do not compile")
        return 1

    sys.path.insert(0, str(ROOT.parent))
    _load_package()
    failures = []
    for name in QGIS_FREE_MODULES:
        try:
            importlib.import_module(f"{PACKAGE}.{name}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    for failure in failures:
        print("FAILED import:", failure)
    if failures:
        return 1
    print(f"OK: {len(QGIS_FREE_MODULES)} QGIS-free modules import cleanly; all sources compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
