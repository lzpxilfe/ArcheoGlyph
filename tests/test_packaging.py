"""
What actually ships.

The plugin is installed from the ZIP that scripts/package_plugin.py builds, so
a module that exists in the repository but not in the archive simply is not
there for users - the failure appears as a missing feature, not an error.
"""

import pathlib
import subprocess
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git", ".github", "__pycache__", "tests", "scripts", "dist", "build",
    ".pytest_cache", ".pytest_qgis_profile",
}


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    output = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/package_plugin.py"), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    built = sorted(output.glob("*.zip"))
    assert built, f"no archive was produced: {result.stdout}"
    with zipfile.ZipFile(built[-1]) as zf:
        yield set(zf.namelist())


def _source_files(suffix):
    for path in ROOT.rglob(f"*{suffix}"):
        if set(path.relative_to(ROOT).parts) & SKIP_DIRS:
            continue
        yield path.relative_to(ROOT).as_posix()


def test_every_plugin_module_is_shipped(archive):
    missing = sorted(
        name for name in _source_files(".py")
        if f"ArcheoGlyph/{name}" not in archive
    )
    assert not missing, "modules missing from the plugin archive:\n" + "\n".join(missing)


def test_the_manifest_and_icon_are_shipped(archive):
    for required in ("ArcheoGlyph/metadata.txt", "ArcheoGlyph/resources/icon.svg"):
        assert required in archive, f"{required} is missing from the archive"


def test_development_only_files_are_not_shipped(archive):
    for excluded in ("ArcheoGlyph/requirements-dev.txt", "ArcheoGlyph/.flake8"):
        assert excluded not in archive, f"{excluded} should not be shipped"
    assert not [n for n in archive if "/tests/" in n or n.startswith("ArcheoGlyph/scripts/")]


def test_the_shipped_version_matches_the_code():
    """
    metadata.txt is what QGIS shows and defaults.py is what the plugin reports
    in diagnostics and in symbol provenance; if they drift, a bug report names
    the wrong version.
    """
    from archeoglyph.defaults import PLUGIN_VERSION

    manifest = (ROOT / "metadata.txt").read_text(encoding="utf-8")
    declared = next(
        line.split("=", 1)[1].strip()
        for line in manifest.splitlines() if line.startswith("version=")
    )
    assert declared == PLUGIN_VERSION
