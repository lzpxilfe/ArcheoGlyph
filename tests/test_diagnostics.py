"""Runtime diagnostics: the report must be accurate and never raise."""

import pathlib

from archeoglyph import diagnostics
from archeoglyph.generators.autotrace import model_store


def test_package_status_reports_installed_and_missing_packages():
    status = diagnostics.package_status()
    assert status["cv2"]["available"] is True       # required, present in CI
    assert status["numpy"]["available"] is True
    # Every entry explains what it is for, so a missing one is actionable.
    for entry in status.values():
        assert entry["label"] and entry["purpose"]
        assert isinstance(entry["available"], bool)


def test_package_status_covers_every_optional_backend():
    keys = set(diagnostics.OPTIONAL_PACKAGES)
    for expected in ("onnxruntime", "vtracer", "huggingface_hub", "google.genai", "cv2"):
        assert expected in keys


def test_versions_are_reported_without_importing_heavy_packages():
    import sys

    before = set(sys.modules)
    status = diagnostics.package_status()
    # Reporting must not pull torch or transformers into the process.
    newly_imported = set(sys.modules) - before
    assert not {"torch", "transformers"} & newly_imported
    # Installed packages report a version read from their metadata.
    assert status["numpy"]["version"]
    assert status["cv2"]["version"]


def test_model_status_lists_the_catalogue_and_detects_installs(tmp_path):
    rows = diagnostics.model_status(str(tmp_path))
    assert {row["key"] for row in rows} == set(model_store.MODEL_SPECS)
    assert all(row["installed"] is False for row in rows)

    spec = model_store.MODEL_SPECS[model_store.DEFAULT_MODEL_KEY]
    target = pathlib.Path(model_store.model_path(spec, str(tmp_path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0" * spec.size)

    rows = diagnostics.model_status(str(tmp_path))
    installed = [row for row in rows if row["installed"]]
    assert len(installed) == 1
    assert installed[0]["key"] == model_store.DEFAULT_MODEL_KEY
    assert installed[0]["path"] == str(target)


def test_collect_and_format_produce_a_readable_report(tmp_path):
    report = diagnostics.collect(str(tmp_path))
    text = diagnostics.format_report(report)

    assert report["plugin_version"]
    assert "ArchaeoGlyph" in text
    assert "Packages:" in text
    assert "Background-removal models:" in text
    assert "Ink centreline:" in text
    # Outside QGIS the report says so rather than failing.
    assert "QGIS: not available" in text
    # Missing packages name what they are needed for.
    for line in text.splitlines():
        if "missing" in line:
            assert "needed for" in line


def test_report_text_never_raises_without_a_base_directory():
    text = diagnostics.report_text()
    assert "ArchaeoGlyph" in text and "Packages:" in text
