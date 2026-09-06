#!/usr/bin/env python3
"""
Render every built-in template and lay them out as one page.

The templates are drawn with QPainter, which is not available outside QGIS, so
this replays the recording from qt_recorder as SVG. It is the only way to see
what the catalogue actually looks like without installing QGIS - and most of
these symbols have never been looked at by anyone.

Usage:
    python3 scripts/render_templates.py [--output PATH]
"""

import argparse
import html
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qt_recorder as qr  # noqa: E402

PACKAGE = "archeoglyph"     # the folder is ArcheoGlyph; the package is not

SIZE = 256
CATEGORY_LABELS = [
    ("artifacts", "유물 · Artifacts"),
    ("structures", "유구 · 건물 · Structures"),
    ("features", "유구 흔적 · Features"),
    ("remains", "인골 · 동물유체 · Remains"),
    ("survey", "조사 · 기록 · Survey"),
]


def _load_package():
    """Import the plugin under its package name, as qgis_import_smoke does."""
    import importlib.util

    if PACKAGE in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)


def _stub_qgis():
    """Install the QGIS stubs the template module imports at load time."""
    import types

    if "qgis" in sys.modules:
        return
    qgis = types.ModuleType("qgis")
    pyqt = types.ModuleType("qgis.PyQt")
    for name, members in (
        ("QtCore", ("Qt", "QBuffer", "QByteArray", "QIODevice", "QPointF",
                    "QRect", "QRectF", "QSize", "QSettings")),
        ("QtGui", ("QImage", "QColor", "QPainter", "QPainterPath", "QPolygonF",
                   "QPen", "QFont", "QPixmap")),
        ("QtSvg", ("QSvgGenerator", "QSvgRenderer")),
    ):
        module = types.ModuleType(f"qgis.PyQt.{name}")
        for member in members:
            setattr(module, member, type(member, (), {}))
        setattr(pyqt, name, module)
        sys.modules[f"qgis.PyQt.{name}"] = module
    core = types.ModuleType("qgis.core")
    for name in ("QgsApplication", "QgsStyle", "QgsMessageLog", "Qgis"):
        setattr(core, name, type(name, (), {}))
    qgis.PyQt, qgis.core = pyqt, core
    sys.modules.update({"qgis": qgis, "qgis.PyQt": pyqt, "qgis.core": core})


# ---------------------------------------------------------------- SVG emitting

def _n(value):
    """Trim a float so the markup stays readable."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _path_data(path):
    parts = []
    for command in path.commands:
        head = command[0]
        if head == "M":
            parts.append(f"M {_n(command[1])} {_n(command[2])}")
        elif head == "L":
            parts.append(f"L {_n(command[1])} {_n(command[2])}")
        elif head == "Q":
            c = command[1:]
            parts.append(f"Q {_n(c[0])} {_n(c[1])} {_n(c[2])} {_n(c[3])}")
        elif head == "C":
            c = command[1:]
            parts.append(
                f"C {_n(c[0])} {_n(c[1])} {_n(c[2])} {_n(c[3])} {_n(c[4])} {_n(c[5])}"
            )
        elif head == "close":
            parts.append("Z")
        elif head == "rect":
            r = command[1]
            parts.append(
                f"M {_n(r.left())} {_n(r.top())} H {_n(r.right())} "
                f"V {_n(r.bottom())} H {_n(r.left())} Z"
            )
        elif head == "ellipse":
            r = command[1]
            rx, ry = r.width() / 2.0, r.height() / 2.0
            cx, cy = r.left() + rx, r.top() + ry
            parts.append(
                f"M {_n(cx - rx)} {_n(cy)} "
                f"A {_n(rx)} {_n(ry)} 0 1 0 {_n(cx + rx)} {_n(cy)} "
                f"A {_n(rx)} {_n(ry)} 0 1 0 {_n(cx - rx)} {_n(cy)} Z"
            )
    return " ".join(parts)


def _paint_attrs(brush, pen, closed=True):
    """fill/stroke attributes for the brush and pen in effect."""
    attrs = []
    if closed and isinstance(brush, qr.Color):
        attrs.append(f'fill="{brush.hex()}"')
        if brush.alpha < 255:
            attrs.append(f'fill-opacity="{brush.opacity():.3f}"')
    else:
        attrs.append('fill="none"')
    colour = pen.color()
    attrs.append(f'stroke="{colour.hex()}"')
    attrs.append(f'stroke-width="{_n(pen.width)}"')
    attrs.append('stroke-linejoin="round"')
    attrs.append('stroke-linecap="round"')
    dashes = pen.dash_array()
    if dashes:
        attrs.append(f'stroke-dasharray="{dashes}"')
    return " ".join(attrs)


def _element(kind, payload, brush, pen):
    if kind == "path":
        data = _path_data(payload)
        if not data:
            return ""
        rule = ' fill-rule="evenodd"' if payload.subtracted_from else ""
        return f'<path d="{data}" {_paint_attrs(brush, pen)}{rule}/>'
    if kind == "polygon":
        points = " ".join(f"{_n(x)},{_n(y)}" for x, y in payload.pairs)
        return f'<polygon points="{points}" {_paint_attrs(brush, pen)}/>'
    if kind == "rect":
        r = payload
        return (
            f'<rect x="{_n(r.left())}" y="{_n(r.top())}" '
            f'width="{_n(r.width())}" height="{_n(r.height())}" '
            f'{_paint_attrs(brush, pen)}/>'
        )
    if kind == "ellipse":
        r = payload
        return (
            f'<ellipse cx="{_n(r.center().x())}" cy="{_n(r.center().y())}" '
            f'rx="{_n(r.width() / 2.0)}" ry="{_n(r.height() / 2.0)}" '
            f'{_paint_attrs(brush, pen)}/>'
        )
    if kind == "line":
        c = payload
        return (
            f'<line x1="{_n(c[0])}" y1="{_n(c[1])}" x2="{_n(c[2])}" y2="{_n(c[3])}" '
            f'{_paint_attrs(brush, pen, closed=False)}/>'
        )
    if kind == "point":
        c = payload
        return (
            f'<circle cx="{_n(c[0])}" cy="{_n(c[1])}" r="{_n(max(0.6, pen.width / 2))}" '
            f'fill="{pen.color().hex()}"/>'
        )
    if kind == "arc":
        x1, y1, rx, ry, large, sweep, x2, y2 = qr.arc_endpoints(*payload)
        return (
            f'<path d="M {_n(x1)} {_n(y1)} A {_n(rx)} {_n(ry)} 0 {large} {sweep} '
            f'{_n(x2)} {_n(y2)}" {_paint_attrs(brush, pen, closed=False)}/>'
        )
    if kind == "text":
        x, y, text = payload
        return (
            f'<text x="{_n(x)}" y="{_n(y)}" font-size="20" '
            f'fill="{pen.color().hex()}">{html.escape(text)}</text>'
        )
    return ""


def render(name, colour):
    """The SVG body for one template, or "" when it draws nothing."""
    from archeoglyph.generators import template_generator as tg
    from archeoglyph.generators.template_generator import TemplateGenerator

    saved = {
        attr: getattr(tg, attr)
        for attr in ("QColor", "QPen", "QPainterPath", "QPolygonF", "QPointF",
                     "QRectF", "Qt")
    }
    tg.QColor, tg.QPen, tg.QPainterPath = qr.Color, qr.Pen, qr.Path
    tg.QPolygonF, tg.QPointF, tg.QRectF, tg.Qt = (
        qr.PolygonF, qr.PointF, qr.RectF, qr.Qt
    )
    painter = qr.Painter()
    try:
        generator = TemplateGenerator.__new__(TemplateGenerator)
        generator._paint_template(painter, name, qr.Color(colour), SIZE)
    finally:
        for attr, value in saved.items():
            setattr(tg, attr, value)

    return "".join(_element(*call) for call in painter.calls)


# ---------------------------------------------------------------- the page

def build_page():
    from archeoglyph import i18n
    from archeoglyph.generators.template_catalog import TEMPLATE_INFO
    from archeoglyph.i18n_ko import CATALOG

    i18n.set_language("ko")

    by_category = {key: [] for key, _label in CATEGORY_LABELS}
    empty = []
    for name, info in TEMPLATE_INFO.items():
        colour = info.get("default_color", "#8B4513")
        body = render(name, colour)
        if not body.strip():
            empty.append(name)
        korean = CATALOG.get(name, name)
        by_category.setdefault(info.get("category", "artifacts"), []).append(
            (name, korean, colour, body)
        )
    return by_category, empty


def page_html(by_category):
    """
    A typological plate: the symbols laid out the way a report figure is.

    The catalogue exists to be judged at map scale, so the page can drop every
    tile to 10 mm - the size these are actually drawn for - which is the only
    view that answers "does this read as a symbol?".
    """
    sections = []
    total = sum(len(v) for v in by_category.values())
    for key, label in CATEGORY_LABELS:
        entries = by_category.get(key) or []
        if not entries:
            continue
        korean_label, english_label = label.split(" · ", 1)
        cards = []
        for name, korean, colour, body in sorted(entries, key=lambda e: e[1]):
            search = html.escape((korean + " " + name).lower())
            cards.append(
                f'<figure class="sym" data-q="{search}">'
                f'<div class="plate">'
                f'<svg viewBox="0 0 {SIZE} {SIZE}" role="img" '
                f'aria-label="{html.escape(korean)}">{body}</svg></div>'
                f'<figcaption><span class="ko">{html.escape(korean)}</span>'
                f'<span class="en">{html.escape(name)}</span></figcaption>'
                f'</figure>'
            )
        sections.append(
            f'<section><h2><span class="cat-ko">{html.escape(korean_label)}</span>'
            f'<span class="cat-en">{html.escape(english_label)}</span>'
            f'<span class="count">{len(entries)}</span></h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    return (
        _TEMPLATE
        .replace("__SECTIONS__", "".join(sections))
        .replace("__TOTAL__", str(total))
    )


_TEMPLATE = """<title>ArchaeoGlyph 심볼 도판</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Gothic+A1:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --ground: #e7e9e4;
  --plate: #fbfbf9;
  --ink: #191d1b;
  --muted: #61696a;
  --rule: #ccd2cd;
  --accent: #3e6b5e;
  --tile: 118px;
  --sans: "Gothic A1", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #101413;
    --plate: #191e1c;
    --ink: #e4e9e5;
    --muted: #8b9490;
    --rule: #262d2a;
    --accent: #6fa894;
  }
}
:root[data-theme="dark"] {
  --ground: #101413;
  --plate: #191e1c;
  --ink: #e4e9e5;
  --muted: #8b9490;
  --rule: #262d2a;
  --accent: #6fa894;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
}
header {
  position: sticky; top: 0; z-index: 5;
  background: var(--ground);
  border-bottom: 1px solid var(--rule);
  padding: 22px 24px 14px;
}
.masthead { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
h1 { font-size: 19px; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
.tally {
  font-family: var(--mono); font-size: 12px; color: var(--accent);
  font-variant-numeric: tabular-nums;
}
.lede {
  margin: 6px 0 14px; color: var(--muted); font-size: 13px; max-width: 64ch;
  text-wrap: pretty;
}
.controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
#q {
  flex: 1 1 240px; max-width: 340px;
  padding: 8px 11px; font-family: var(--sans); font-size: 14px;
  color: var(--ink); background: var(--plate);
  border: 1px solid var(--rule); border-radius: 3px;
}
#q::placeholder { color: var(--muted); }
#q:focus-visible, .scale:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 1px;
}
.scale {
  font-family: var(--mono); font-size: 12px; letter-spacing: 0.02em;
  padding: 8px 12px; cursor: pointer;
  color: var(--ink); background: var(--plate);
  border: 1px solid var(--rule); border-radius: 3px;
}
.scale[aria-pressed="true"] {
  background: var(--accent); border-color: var(--accent); color: var(--plate);
}
main { padding: 4px 24px 64px; }
section { margin-top: 30px; }
h2 {
  display: flex; align-items: baseline; gap: 9px; margin: 0 0 14px;
  padding-bottom: 7px; border-bottom: 1px solid var(--rule);
  font-size: 14px; font-weight: 700;
}
.cat-en {
  font-family: var(--mono); font-size: 11px; font-weight: 400;
  color: var(--muted); letter-spacing: 0.03em;
}
.count {
  margin-left: auto; font-family: var(--mono); font-size: 11px;
  color: var(--accent); font-variant-numeric: tabular-nums;
}
.grid {
  display: grid; gap: 16px 12px;
  grid-template-columns: repeat(auto-fill, minmax(var(--tile), 1fr));
}
.sym { margin: 0; min-width: 0; }
.plate {
  background: var(--plate); border: 1px solid var(--rule);
  aspect-ratio: 1; display: grid; place-items: center; padding: 7%;
}
.plate svg { width: 100%; height: 100%; display: block; }
figcaption {
  margin-top: 6px; display: flex; flex-direction: column; gap: 0;
  overflow-wrap: anywhere;
  /* Names run one to three lines; a floor keeps the rows scanning evenly. */
  min-height: 3.6em;
}
.ko { font-size: 12.5px; font-weight: 500; line-height: 1.35; }
.en {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
  line-height: 1.4;
}
/* Marker scale: 10 mm is about 38 px, the size these are drawn for. */
body.at-marker-size { --tile: 38px; }
body.at-marker-size .plate { padding: 0; border-color: transparent; }
body.at-marker-size figcaption { display: none; }
body.at-marker-size .grid { gap: 14px; }
.sym[hidden] { display: none; }
#none { color: var(--muted); font-size: 14px; padding: 34px 0; }
@media (prefers-reduced-motion: no-preference) {
  .scale, #q { transition: background-color .12s ease, border-color .12s ease; }
}
</style>
<header>
  <div class="masthead">
    <h1>ArchaeoGlyph 심볼 도판</h1>
    <span class="tally">__TOTAL__ types</span>
  </div>
  <p class="lede">플러그인에 내장된 심볼 전부입니다. 색은 각 항목의 기본값이고 QGIS에서
     바꿀 수 있습니다. 이 심볼들은 지도에서 5-10 mm로 읽히도록 그렸으니,
     실제 크기로 보면 판단이 빠릅니다.</p>
  <div class="controls">
    <input id="q" type="search" placeholder="검색 - 지석묘, 토기, dagger" autocomplete="off">
    <button class="scale" id="scale" type="button" aria-pressed="false">실제 크기 10 mm</button>
  </div>
</header>
<main>__SECTIONS__<p id="none" hidden>일치하는 심볼이 없습니다.</p></main>
<script>
  const box = document.getElementById('q');
  const symbols = [...document.querySelectorAll('.sym')];
  const sections = [...document.querySelectorAll('section')];
  const none = document.getElementById('none');

  box.addEventListener('input', () => {
    const query = box.value.trim().toLowerCase();
    let shown = 0;
    for (const symbol of symbols) {
      const hit = !query || symbol.dataset.q.includes(query);
      symbol.hidden = !hit;
      if (hit) shown++;
    }
    for (const section of sections) {
      section.hidden = !section.querySelector('.sym:not([hidden])');
    }
    none.hidden = shown > 0;
  });

  const scale = document.getElementById('scale');
  scale.addEventListener('click', () => {
    const on = document.body.classList.toggle('at-marker-size');
    scale.setAttribute('aria-pressed', String(on));
    scale.textContent = on ? '원래 크기로' : '실제 크기 10 mm';
  });
</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist" / "templates.html"))
    args = parser.parse_args()

    _stub_qgis()
    _load_package()
    by_category, empty = build_page()
    if empty:
        print(f"WARNING: {len(empty)} templates rendered nothing: {empty[:5]}")

    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page_html(by_category), encoding="utf-8")
    total = sum(len(v) for v in by_category.values())
    print(f"{target} ({total} templates, {target.stat().st_size // 1024} KB)")
    return 1 if empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
