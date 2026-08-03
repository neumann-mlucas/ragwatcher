from ragwatcher.chunking import get_splitter, prefix_metadata
from ragwatcher.config import ChunkCfg


def test_get_splitter_recursive():
    cfg = ChunkCfg()
    sp = get_splitter(cfg)
    assert sp is not None


def test_late_falls_back_to_recursive():
    # Late chunking is a placeholder that falls back until a user opts into bge-m3.
    cfg = ChunkCfg(strategy="late")
    sp = get_splitter(cfg)
    assert sp is not None


def test_prefix_metadata_puts_filename_and_heading():
    text = "body body body"
    out = prefix_metadata({"file_name": "notes.md", "heading_path": "Chapter 1"}, text)
    assert "file: notes.md" in out
    assert "heading: Chapter 1" in out
    assert out.endswith(text)


def test_prefix_metadata_noop_when_empty():
    text = "body"
    assert prefix_metadata({}, text) == text
