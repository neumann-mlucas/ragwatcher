from pathlib import Path

import pytest

from ragwatcher import manifest
from ragwatcher.errors import SchemaMismatch
from ragwatcher.manifest import FileEntry, Manifest


def _fp() -> dict[str, object]:
    return {
        "embed_model": "test",
        "embed_dim": None,
        "chunk_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "chunk_max_chars": 1500,
        "chunk_semantic_breakpoint_percentile": None,
        "chunk_semantic_buffer_size": None,
        "store_backend": "simple",
        "context_enabled": False,
        "context_prompt_sha256": None,
    }


def test_new_manifest_has_schema_and_fingerprint():
    m = Manifest.new(_fp())
    assert m.schema_version == manifest.CURRENT_SCHEMA
    assert m.config_fingerprint == _fp()
    assert m.files == {}


def test_roundtrip(tmp_path: Path):
    m = Manifest.new(_fp())
    m.files["/a.md"] = FileEntry(sha256="abc", mtime_ns=1, size=10, doc_ids=["x"], chunk_count=1)
    path = tmp_path / "manifest.json"
    manifest.save(path, m)
    loaded = manifest.load(path)
    assert loaded is not None
    assert loaded.files["/a.md"].sha256 == "abc"
    assert loaded.files["/a.md"].doc_ids == ["x"]


def test_load_missing_returns_none(tmp_path: Path):
    assert manifest.load(tmp_path / "nope.json") is None


def test_load_corrupt_moves_aside(tmp_path: Path):
    bad = tmp_path / "manifest.json"
    bad.write_text("not json")
    assert manifest.load(bad) is None
    assert not bad.exists()
    assert any("manifest.json.bad" in p.name for p in tmp_path.iterdir())


def test_fingerprint_mismatch_raises():
    m = Manifest.new(_fp())
    other = _fp() | {"embed_model": "different"}
    with pytest.raises(SchemaMismatch):
        m.check_fingerprint(other)


def test_fingerprint_match_ok():
    m = Manifest.new(_fp())
    m.check_fingerprint(_fp())


def test_atomic_write_no_tmp_leftover(tmp_path: Path):
    m = Manifest.new(_fp())
    path = tmp_path / "manifest.json"
    manifest.save(path, m)
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".manifest.")]
    assert leftovers == []
