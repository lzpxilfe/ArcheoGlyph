# -*- coding: utf-8 -*-
"""
Shared style-control constants and helpers.
Centralizes slider defaults, QSettings keys, and formatting.
"""

STYLE_CONTROL_FACTUALITY = "factuality"
STYLE_CONTROL_SYMBOLIC_LOOSENESS = "symbolic_looseness"
STYLE_CONTROL_EXAGGERATION = "exaggeration"

STYLE_CONTROL_ORDER = (
    STYLE_CONTROL_FACTUALITY,
    STYLE_CONTROL_SYMBOLIC_LOOSENESS,
    STYLE_CONTROL_EXAGGERATION,
)

STYLE_CONTROL_KEYS = {
    STYLE_CONTROL_FACTUALITY: "ArcheoGlyph/style_factuality",
    STYLE_CONTROL_SYMBOLIC_LOOSENESS: "ArcheoGlyph/style_symbolic_looseness",
    STYLE_CONTROL_EXAGGERATION: "ArcheoGlyph/style_exaggeration",
}

STYLE_CONTROL_DEFAULTS = {
    STYLE_CONTROL_FACTUALITY: 72,
    STYLE_CONTROL_SYMBOLIC_LOOSENESS: 34,
    STYLE_CONTROL_EXAGGERATION: 22,
}

STYLE_CONTROL_MIN = 0
STYLE_CONTROL_MAX = 100


def _clamp_percent(value, default):
    """Clamp any numeric input into [0, 100]."""
    try:
        parsed = int(round(float(value)))
    except Exception:
        parsed = int(default)
    return max(STYLE_CONTROL_MIN, min(STYLE_CONTROL_MAX, parsed))


def resolve_style_controls(
    settings=None,
    factuality=None,
    symbolic_looseness=None,
    exaggeration=None,
):
    """
    Resolve style controls from explicit args first, then QSettings, then defaults.
    """
    provided_values = {
        STYLE_CONTROL_FACTUALITY: factuality,
        STYLE_CONTROL_SYMBOLIC_LOOSENESS: symbolic_looseness,
        STYLE_CONTROL_EXAGGERATION: exaggeration,
    }

    controls = {}
    for name in STYLE_CONTROL_ORDER:
        default = STYLE_CONTROL_DEFAULTS[name]
        provided = provided_values[name]
        if provided is not None:
            controls[name] = _clamp_percent(provided, default)
            continue

        if settings is not None:
            controls[name] = _clamp_percent(
                settings.value(STYLE_CONTROL_KEYS[name], default),
                default,
            )
            continue

        controls[name] = int(default)

    return controls


def save_style_controls(settings, controls):
    """Persist style controls to QSettings."""
    if settings is None:
        return

    normalized = resolve_style_controls(
        settings=None,
        factuality=(controls or {}).get(STYLE_CONTROL_FACTUALITY),
        symbolic_looseness=(controls or {}).get(STYLE_CONTROL_SYMBOLIC_LOOSENESS),
        exaggeration=(controls or {}).get(STYLE_CONTROL_EXAGGERATION),
    )
    for name in STYLE_CONTROL_ORDER:
        settings.setValue(STYLE_CONTROL_KEYS[name], int(normalized[name]))


def style_controls_short_text(controls):
    """Compact UI text for currently active controls."""
    normalized = resolve_style_controls(
        settings=None,
        factuality=(controls or {}).get(STYLE_CONTROL_FACTUALITY),
        symbolic_looseness=(controls or {}).get(STYLE_CONTROL_SYMBOLIC_LOOSENESS),
        exaggeration=(controls or {}).get(STYLE_CONTROL_EXAGGERATION),
    )
    return (
        f"F{normalized[STYLE_CONTROL_FACTUALITY]} / "
        f"S{normalized[STYLE_CONTROL_SYMBOLIC_LOOSENESS]} / "
        f"E{normalized[STYLE_CONTROL_EXAGGERATION]}"
    )


def style_controls_prompt_hint(controls, prefix="control hint"):
    """Readable prompt hint used by AI generators."""
    normalized = resolve_style_controls(
        settings=None,
        factuality=(controls or {}).get(STYLE_CONTROL_FACTUALITY),
        symbolic_looseness=(controls or {}).get(STYLE_CONTROL_SYMBOLIC_LOOSENESS),
        exaggeration=(controls or {}).get(STYLE_CONTROL_EXAGGERATION),
    )
    return (
        f"{prefix}: factuality {normalized[STYLE_CONTROL_FACTUALITY]}/100, "
        f"symbol looseness {normalized[STYLE_CONTROL_SYMBOLIC_LOOSENESS]}/100, "
        f"exaggeration {normalized[STYLE_CONTROL_EXAGGERATION]}/100"
    )
