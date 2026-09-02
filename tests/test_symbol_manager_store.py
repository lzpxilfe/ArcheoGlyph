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
