#!/usr/bin/env python3
"""
Build the installable QGIS plugin ZIP.

The archive contains the plugin folder only (no tests, scripts or CI files)
and is written deterministically so the same source always produces the same
bytes and checksum.
"""

import argparse
import hashlib
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLDER_NAME = "ArcheoGlyph"  # QGIS plugin directory name

INCLUDE_SUFFIXES = {".py", ".txt", ".md", ".svg", ".png", ".cff", ".yml"}
EXCLUDE_DIRS = {
    ".git", ".github", "__pycache__", "tests", "scripts", "dist", "build",
    ".pytest_cache", ".pytest_qgis_profile",
}
EXCLUDE_NAMES = {"requirements-dev.txt", ".flake8"}
DETERMINISTIC_DATE = (2026, 1, 1, 0, 0, 0)


def iter_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(ROOT).parts)
        if parts & EXCLUDE_DIRS or path.name in EXCLUDE_NAMES:
            continue
        if path.suffix.lower() not in INCLUDE_SUFFIXES:
            continue
        yield path


def plugin_version():
    for line in (ROOT / "metadata.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("version="):
            return line.split("=", 1)[1].strip()
    return "0.0.0"


def build(output_dir):
    version = plugin_version()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"archeoglyph-{version}.zip"

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in iter_files():
            relative = path.relative_to(ROOT)
            info = zipfile.ZipInfo(str(pathlib.PurePosixPath(FOLDER_NAME, *relative.parts)))
            info.date_time = DETERMINISTIC_DATE
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target} ({target.stat().st_size} bytes)")
    print(f"sha256 {digest}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist"), help="output directory")
    args = parser.parse_args()
    build(pathlib.Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
