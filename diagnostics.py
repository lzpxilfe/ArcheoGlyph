# -*- coding: utf-8 -*-
"""
Runtime diagnostics.

Collects what actually decides whether a feature works: which optional
packages are importable, which models are downloaded, where symbols are
stored. The report is plain text so it can be pasted into an issue.

Nothing here imports QGIS at module level, so it can be tested directly; the
QGIS-dependent parts degrade to "unavailable".
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import Dict, List

from .defaults import PLUGIN_VERSION

# import name -> (label, what stops working without it, distributions to try)
OPTIONAL_PACKAGES = {
    "cv2": ("OpenCV", "Auto Trace and every image operation",
            ("opencv-python-headless", "opencv-python", "opencv-contrib-python")),
    "numpy": ("NumPy", "Auto Trace and every image operation", ("numpy",)),
    "scipy": ("SciPy", "faster ink centreline morphology", ("scipy",)),
    "skimage": ("scikit-image", "more accurate ink centreline thinning", ("scikit-image",)),
    "onnxruntime": ("onnxruntime", "background-removal models",
                    ("onnxruntime", "onnxruntime-gpu")),
    "vtracer": ("vtracer", "smoother tracing of AI raster output", ("vtracer",)),
    "huggingface_hub": ("huggingface_hub", "the Hugging Face backend", ("huggingface_hub",)),
    "google.genai": ("google-genai", "the Gemini backend", ("google-genai",)),
    "torch": ("PyTorch", "the SAM backend", ("torch",)),
    "transformers": ("transformers", "SAM 2/3 from Hugging Face", ("transformers",)),
    "PIL": ("Pillow", "image handling in the Hugging Face backend", ("pillow",)),
}


def _distribution_version(distributions) -> str:
    """
    Version from installed package metadata.

    Metadata is read from disk, so reporting on a heavy package such as torch
    never imports it.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return ""
    for name in distributions:
        try:
            return str(version(name))
        except PackageNotFoundError:
            continue
        except Exception:
            return ""
    return ""


def package_status(probe_versions: bool = True) -> Dict[str, dict]:
    """
    Availability and version of each optional package.

    ``find_spec`` decides availability and package metadata supplies the
    version, so nothing here imports the packages it reports on.
    """
    report = {}
    for name, (label, purpose, distributions) in OPTIONAL_PACKAGES.items():
        try:
            available = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available = False
        version = ""
        if available and probe_versions:
            version = _distribution_version(distributions)
            if not version and name in sys.modules:
                version = str(getattr(sys.modules[name], "__version__", "") or "")
        report[name] = {
            "label": label,
            "purpose": purpose,
            "available": available,
            "version": version,
        }
    return report


def model_status(base_dir: str) -> List[dict]:
    """Which background-removal models are downloaded, and how large."""
    from .generators.autotrace.model_store import MODEL_SPECS, is_installed, model_path

    rows = []
    for key, spec in MODEL_SPECS.items():
        installed = False
        path = ""
        try:
            installed = is_installed(spec, base_dir)
            path = model_path(spec, base_dir) if installed else ""
        except OSError:
            pass
        rows.append({
            "key": key,
            "label": spec.label,
            "installed": installed,
            "size_mb": round(spec.size / (1024 * 1024), 1),
            "path": path,
        })
    return rows


def ink_status() -> dict:
    from .generators.ink_centerline import ink_runtime_status

    return ink_runtime_status()


def qgis_status() -> dict:
    """QGIS version, profile directory and symbol store; empty outside QGIS."""
    info = {"available": False}
    try:
        from qgis.core import Qgis, QgsApplication

        info["available"] = True
        info["qgis_version"] = str(getattr(Qgis, "QGIS_VERSION", ""))
        info["profile"] = QgsApplication.qgisSettingsDirPath()
    except Exception:
        return info

    try:
        from .symbol_manager import SVG_SEARCH_PATH_SETTING, symbol_store_dir
        from qgis.PyQt.QtCore import QSettings

        directory = symbol_store_dir()
        info["symbol_store"] = directory
        info["symbol_count"] = len([
            name for name in os.listdir(directory) if name.lower().endswith((".svg", ".png"))
        ])
        paths = QSettings().value(SVG_SEARCH_PATH_SETTING, [])
        if isinstance(paths, str):
            paths = paths.split("|")
        info["svg_search_path_registered"] = any(
            os.path.normpath(str(p)) == os.path.normpath(directory) for p in (paths or [])
        )
    except Exception:
        pass
    return info


def collect(base_dir: str = "") -> dict:
    """Gather the full report as data."""
    return {
        "plugin_version": PLUGIN_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_status(),
        "models": model_status(base_dir) if base_dir else [],
        "ink": ink_status(),
        "qgis": qgis_status(),
    }


def format_report(report: dict) -> str:
    """Render a report as plain text for the clipboard or an issue."""
    lines = [
        f"ArchaeoGlyph {report.get('plugin_version', '?')}",
        f"Python {report.get('python', '?')} on {report.get('platform', '?')}",
    ]

    qgis = report.get("qgis") or {}
    if qgis.get("available"):
        lines.append(f"QGIS {qgis.get('qgis_version', '?')}")
        if qgis.get("symbol_store"):
            registered = "yes" if qgis.get("svg_search_path_registered") else "no"
            lines.append(
                f"Symbols: {qgis.get('symbol_count', 0)} in {qgis['symbol_store']} "
                f"(registered as an SVG search path: {registered})"
            )
    else:
        lines.append("QGIS: not available (running outside QGIS)")

    lines.append("")
    lines.append("Packages:")
    for entry in (report.get("packages") or {}).values():
        mark = "ok     " if entry["available"] else "missing"
        version = f" {entry['version']}" if entry.get("version") else ""
        suffix = "" if entry["available"] else f"  <- needed for {entry['purpose']}"
        lines.append(f"  [{mark}] {entry['label']}{version}{suffix}")

    models = report.get("models") or []
    if models:
        lines.append("")
        lines.append("Background-removal models:")
        for entry in models:
            mark = "downloaded" if entry["installed"] else "not downloaded"
            lines.append(f"  [{mark}] {entry['label']} ({entry['size_mb']} MB)")

    ink = report.get("ink") or {}
    if ink:
        lines.append("")
        lines.append(f"Ink centreline: {ink.get('message', 'unknown')}")

    return "\n".join(lines)


def report_text(base_dir: str = "") -> str:
    """Convenience: collect and format in one call."""
    return format_report(collect(base_dir))
