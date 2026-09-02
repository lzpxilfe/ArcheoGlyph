# -*- coding: utf-8 -*-
"""Auto Trace options: a plain dataclass so the pipeline never reads QSettings."""

from dataclasses import dataclass, replace
from typing import Optional

DETAIL_MODES = ("fast", "precise")
ROUND_STRATEGIES = ("image_first", "hybrid", "structure_first")
INPUT_KINDS = ("auto", "photo", "drawing")


@dataclass
class AutoTraceOptions:
    style: str = "Simple Symbol"
    color: Optional[str] = None
    symmetry: bool = False
    force_lowres_upscale: bool = False
    detail_mode: str = "fast"
    round_strategy: str = "image_first"
    factuality: int = 72
    symbolic_looseness: int = 34
    exaggeration: int = 22
    synthetic_structure: bool = False
    input_kind: str = "auto"
    seed: int = 0

    def normalized(self):
        """Return a copy with enum-like fields clamped to their valid values."""
        detail = str(self.detail_mode or "fast").strip().lower()
        strategy = str(self.round_strategy or "image_first").strip().lower()
        kind = str(self.input_kind or "auto").strip().lower()
        return replace(
            self,
            detail_mode=detail if detail in DETAIL_MODES else "fast",
            round_strategy=strategy if strategy in ROUND_STRATEGIES else "image_first",
            input_kind=kind if kind in INPUT_KINDS else "auto",
            factuality=_clamp_pct(self.factuality, 72),
            symbolic_looseness=_clamp_pct(self.symbolic_looseness, 34),
            exaggeration=_clamp_pct(self.exaggeration, 22),
            color=(str(self.color).strip() or None) if self.color else None,
        )


def _clamp_pct(value, default):
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        v = int(default)
    return max(0, min(100, v))
