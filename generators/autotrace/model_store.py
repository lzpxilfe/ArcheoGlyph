# -*- coding: utf-8 -*-
"""
Verified download and lookup of optional ONNX background-removal models.

Models are content-verified (size + SHA-256) and written atomically, so a
partial or tampered download is never used. QGIS-free; the caller supplies
the base directory (normally the QGIS profile folder).
"""

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

MODELS_SUBDIR = os.path.join("archeoglyph", "models")
_REMBG_RELEASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    filename: str
    url: str
    sha256: str
    size: int
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    note: str = ""


MODEL_SPECS: Dict[str, ModelSpec] = {
    "isnet-general-use": ModelSpec(
        key="isnet-general-use",
        label="ISNet general use (best quality, 170 MB)",
        filename="isnet-general-use.onnx",
        url=_REMBG_RELEASE + "isnet-general-use.onnx",
        sha256="60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        size=178648008,
        input_size=1024,
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        note="Dichotomous image segmentation; handles gradients, shadows and low contrast.",
    ),
    "u2net": ModelSpec(
        key="u2net",
        label="U²-Net (168 MB)",
        filename="u2net.onnx",
        url=_REMBG_RELEASE + "u2net.onnx",
        sha256="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        size=175997641,
        input_size=320,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "u2netp": ModelSpec(
        key="u2netp",
        label="U²-Net small (4.4 MB, fast, lower quality)",
        filename="u2netp.onnx",
        url=_REMBG_RELEASE + "u2netp.onnx",
        sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        size=4574861,
        input_size=320,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}
DEFAULT_MODEL_KEY = "isnet-general-use"


class ModelStoreError(RuntimeError):
    pass


def models_dir(base_dir: str) -> str:
    path = os.path.join(base_dir, MODELS_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def model_path(spec: ModelSpec, base_dir: str) -> str:
    return os.path.join(models_dir(base_dir), spec.filename)


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def is_installed(spec: ModelSpec, base_dir: str) -> bool:
    """Cheap check: file exists with the expected size (hash verified on download)."""
    path = model_path(spec, base_dir)
    try:
        return os.path.isfile(path) and os.path.getsize(path) == int(spec.size)
    except OSError:
        return False


def verify_model(spec: ModelSpec, base_dir: str) -> bool:
    path = model_path(spec, base_dir)
    return os.path.isfile(path) and sha256_of(path) == spec.sha256


def installed_model(base_dir: str, key: Optional[str] = None) -> Optional[Tuple[ModelSpec, str]]:
    """Return (spec, path) for the requested key, or the best installed model."""
    keys = [key] if key in MODEL_SPECS else list(MODEL_SPECS)
    for k in keys:
        spec = MODEL_SPECS[k]
        if is_installed(spec, base_dir):
            return spec, model_path(spec, base_dir)
    return None


def download_model(
    spec: ModelSpec,
    base_dir: str,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    timeout: float = 60.0,
) -> str:
    """
    Download ``spec`` into ``base_dir``, verifying size and SHA-256, and
    publish it atomically. Returns the final path.
    """
    target = model_path(spec, base_dir)
    directory = os.path.dirname(target)
    digest = hashlib.sha256()
    received = 0
    fd, temp_path = tempfile.mkstemp(prefix=spec.filename + ".", suffix=".part", dir=directory)
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "ArchaeoGlyph"})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                if cancel_check is not None and cancel_check():
                    raise ModelStoreError("Download cancelled.")
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                received += len(block)
                if received > int(spec.size):
                    raise ModelStoreError("Download exceeded the expected size.")
                if progress is not None:
                    progress(received, int(spec.size))
            out.flush()
            os.fsync(out.fileno())
        if received != int(spec.size):
            raise ModelStoreError(f"Download incomplete ({received} of {spec.size} bytes).")
        if digest.hexdigest() != spec.sha256:
            raise ModelStoreError("SHA-256 mismatch; the downloaded file was discarded.")
        os.replace(temp_path, target)
        return target
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
