# -*- coding: utf-8 -*-
"""
Contract for the real-photograph trace fixtures.

Auto Trace used to be exercised only on images this project draws itself
(``tests/synthetic.py``). Those are still the right thing for unit tests -
they are deterministic and need no network - but a drawing whose ground truth
the author already knows cannot show whether the tracer copes with a real
museum photograph: perspective, drop shadows, several objects in one frame,
a label card beside the find.

So the real inputs are catalogued here by provenance rather than committed.
``tests/fixtures/real_sources.json`` records where each photograph comes from,
under which licence, and the sha256 of the file that was measured. The schema
test below always runs; the trace regression runs only for whoever has the
files, pointed at by ``ARCHEOGLYPH_REAL_FIXTURES``.

Nothing here downloads anything. Keeping the bytes out of the repository is
deliberate: 국립문화유산연구원 holds copyright over the excavation-report plates
this project must not reproduce, and even freely licensed photographs do not
belong in a plugin package that users install.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

MANIFEST = Path(__file__).parent / "fixtures" / "real_sources.json"
REQUIRED_KEYS = {"id", "label", "museum_name", "template", "page",
                 "license", "credit", "sha256", "bytes", "pixels"}


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _fixture_dir():
    raw = os.environ.get("ARCHEOGLYPH_REAL_FIXTURES", "").strip()
    return Path(raw) if raw else None


def test_the_manifest_describes_every_source_completely():
    doc = _manifest()
    sources = doc["sources"]
    assert sources, "the manifest must not be empty"
    for entry in sources:
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, f"{entry.get('id')} is missing {sorted(missing)}"
        assert entry["sha256"] and len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
        assert len(entry["pixels"]) == 2 and all(p > 0 for p in entry["pixels"])
        assert entry["page"].startswith("https://")


def test_every_source_carries_a_licence_that_permits_this_use():
    doc = _manifest()
    allowed = set(doc["allowed_licenses"])
    assert allowed, "the allow-list must name the licences this project accepts"
    for entry in doc["sources"]:
        assert entry["license"] in allowed, (
            f"{entry['id']} cites {entry['license']!r}, which is not on the "
            f"allow-list {sorted(allowed)} - an image whose licence has not "
            f"been checked must not become a fixture")
        assert entry["credit"], f"{entry['id']} must name whom to credit"


def test_source_ids_are_unique():
    ids = [e["id"] for e in _manifest()["sources"]]
    assert len(ids) == len(set(ids))


def test_no_image_bytes_are_committed():
    """The manifest is provenance; the photographs stay outside the repo."""
    fixtures = MANIFEST.parent
    strays = [p.name for p in fixtures.iterdir()
              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif")]
    assert not strays, f"image files must not be committed: {strays}"


@pytest.mark.parametrize("entry", _manifest()["sources"], ids=lambda e: e["id"])
def test_the_cached_photograph_is_the_one_that_was_measured(entry):
    """Guards against a fixture directory drifting from the manifest."""
    base = _fixture_dir()
    if base is None:
        pytest.skip("set ARCHEOGLYPH_REAL_FIXTURES to run against real photos")
    path = base / f"{entry['id']}.jpg"
    if not path.exists():
        pytest.skip(f"{path.name} not cached locally")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"], (
        f"{path.name} is not the file the manifest describes; re-download it "
        f"from {entry['page']}")


@pytest.mark.parametrize("entry", _manifest()["sources"], ids=lambda e: e["id"])
def test_a_real_photograph_still_traces_to_a_usable_symbol(entry):
    base = _fixture_dir()
    if base is None:
        pytest.skip("set ARCHEOGLYPH_REAL_FIXTURES to run against real photos")
    path = base / f"{entry['id']}.jpg"
    if not path.exists():
        pytest.skip(f"{path.name} not cached locally")
    cv2 = pytest.importorskip("cv2")

    from archeoglyph.generators.autotrace.options import AutoTraceOptions
    from archeoglyph.generators.autotrace.pipeline import run_autotrace
    from archeoglyph.generators.autotrace.segment import select_mask

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"could not decode {path}"
    svg = run_autotrace(
        bgr,
        AutoTraceOptions(style="Simple Symbol"),
        lambda proc: select_mask(proc, backend="opencv"),
    )
    assert svg and svg.lstrip().startswith("<svg"), "trace produced no SVG"
    assert "<path" in svg, "trace produced an SVG with no geometry"
