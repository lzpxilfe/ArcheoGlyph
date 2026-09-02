import os

from archeoglyph import symbol_manager as sm
from archeoglyph.generators.symbol_result import SymbolResult


def test_store_writes_content_addressed_files_in_profile_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "symbol_store_dir", lambda: str(tmp_path))
    manager = sm.SymbolManager()
    result = SymbolResult(svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>', source="test")

    path, stored = manager.store(result)
    assert os.path.dirname(path) == str(tmp_path)
    assert path.endswith(f"{result.content_hash()}.svg")
    assert open(path, "rb").read() == result.payload_bytes()

    # Same content -> same file, no duplicate; different content -> new file.
    path_again, _ = manager.store(result)
    assert path_again == path
    other, _ = manager.store(SymbolResult(raster_png=b"\x89PNG\r\n", source="test"))
    assert other != path and other.endswith(".png")
    assert len(os.listdir(tmp_path)) == 2


def test_store_rejects_empty_result(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "symbol_store_dir", lambda: str(tmp_path))
    manager = sm.SymbolManager()
    try:
        manager.store(SymbolResult())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_merge_search_paths_adds_once_and_preserves_existing():
    from archeoglyph.symbol_manager import merge_search_paths

    existing = ["/opt/qgis/svg", "/home/user/my svgs"]
    updated = merge_search_paths(existing, "/home/user/profile/archeoglyph/symbols")
    assert updated == existing + ["/home/user/profile/archeoglyph/symbols"]

    # Already present (even spelled with a trailing slash or "..") -> no change.
    assert merge_search_paths(updated, "/home/user/profile/archeoglyph/symbols/") is None
    assert merge_search_paths(updated, "/home/user/profile/other/../archeoglyph/symbols") is None


def test_merge_search_paths_accepts_qgis_string_and_empty_forms():
    from archeoglyph.symbol_manager import merge_search_paths

    assert merge_search_paths("", "/a/b") == ["/a/b"]
    assert merge_search_paths(None, "/a/b") == ["/a/b"]
    # QGIS sometimes stores the list as a single pipe-separated string.
    assert merge_search_paths("/opt/svg|/srv/svg", "/a/b") == ["/opt/svg", "/srv/svg", "/a/b"]
    assert merge_search_paths("/opt/svg|/a/b", "/a/b") is None
