# -*- coding: utf-8 -*-
"""
SymbolResult: the single value type every generator hands to the UI.

QGIS-free. A result carries vector SVG text and/or PNG bytes plus provenance
(``source``), the requested style, human-readable warnings and free-form
metadata (for example the fill/outline colours the SVG was parametrised with).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SymbolResult:
    svg: Optional[str] = None
    raster_png: Optional[bytes] = None
    source: str = "unknown"
    style: str = ""
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_vector(self) -> bool:
        return bool(self.svg and self.svg.strip())

    @property
    def is_empty(self) -> bool:
        return not self.is_vector and not self.raster_png

    @property
    def extension(self) -> str:
        return "svg" if self.is_vector else "png"

    def add_warning(self, message: str) -> None:
        text = str(message or "").strip()
        if text and text not in self.warnings:
            self.warnings.append(text)

    def record_provenance(self, image_path=None, **fields) -> "SymbolResult":
        """
        Record how this symbol was produced (source image, style, model, ...).

        Only the file name of ``image_path`` is kept: the full path can expose
        a directory layout that has nothing to do with the symbol.
        """
        import datetime
        import os

        meta = {
            "source": self.source,
            "style": self.style,
            "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        }
        if image_path:
            meta["input"] = os.path.basename(str(image_path))
        meta.update({k: v for k, v in fields.items() if v not in (None, "")})
        self.meta.update({k: v for k, v in meta.items() if v not in (None, "")})
        return self

    def content_hash(self) -> str:
        """Stable short hash of the payload, used for on-disk file names."""
        h = hashlib.sha1()
        if self.is_vector:
            h.update(b"svg:")
            h.update(self.svg.encode("utf-8"))
        elif self.raster_png:
            h.update(b"png:")
            h.update(bytes(self.raster_png))
        return h.hexdigest()[:16]

    def payload_bytes(self) -> bytes:
        if self.is_vector:
            return self.svg.encode("utf-8")
        return bytes(self.raster_png or b"")

    @classmethod
    def coerce(cls, obj: Any, source: str = "unknown", style: str = "") -> Optional["SymbolResult"]:
        """
        Accept legacy generator return values (SVG text, PNG bytes, QImage,
        QPixmap) and wrap them. Returns None for None.
        """
        if obj is None:
            return None
        if isinstance(obj, SymbolResult):
            if obj.source == "unknown" and source:
                obj.source = source
            if not obj.style and style:
                obj.style = style
            return obj
        if isinstance(obj, str):
            return cls(svg=obj, source=source, style=style)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return cls(raster_png=bytes(obj), source=source, style=style)
        png = _qt_image_to_png(obj)
        if png is not None:
            return cls(raster_png=png, source=source, style=style)
        raise TypeError(f"Cannot convert {type(obj).__name__} to SymbolResult")


def _qt_image_to_png(obj: Any) -> Optional[bytes]:
    """Encode a QImage/QPixmap as PNG bytes. Returns None if obj is neither."""
    try:
        from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice
    except Exception:  # pragma: no cover - plain Python
        return None

    image = obj
    if hasattr(obj, "toImage") and not hasattr(obj, "pixelColor"):
        image = obj.toImage()
    if not hasattr(image, "save"):
        return None
    if hasattr(image, "isNull") and image.isNull():
        return None

    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    ok = image.save(buffer, "PNG")
    buffer.close()
    if not ok:
        return None
    return bytes(data)
