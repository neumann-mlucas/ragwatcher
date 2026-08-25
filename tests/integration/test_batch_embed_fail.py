"""When the batch embed/insert step fails, previously indexed chunks + the
manifest that references them must survive intact."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ragwatcher.config import Settings
from ragwatcher.index import RagIndex

pytestmark = pytest.mark.slow

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "notes"


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    dst = tmp_path / "notes"
    shutil.copytree(FIXTURE, dst)
    return dst


def test_failed_batch_leaves_prior_state_intact(notes_dir: Path) -> None:
    settings = Settings()
    # First sync succeeds — establishes real chunks + manifest.
    with RagIndex(notes_dir, settings) as idx:
        idx.sync()
        first_files = {p: e.doc_ids[:] for p, e in idx.manifest.files.items()}
        assert first_files
        first_alpha = idx.retrieve("what is alpha", top_k=2)
        assert any("alpha.md" in h.source for h in first_alpha.returned)

    # Second sync: mutate a file so it becomes an update, then force the
    # batch insert to raise. Old chunks and manifest entries must survive.
    (notes_dir / "alpha.md").write_text("# alpha\n\nchanged content that would need re-embed\n")

    with RagIndex(notes_dir, settings) as idx:
        # Warm the retriever so _index_obj exists — insert_nodes is what we monkeypatch.
        idx.retrieve("bootstrap", top_k=1)
        assert idx._index_obj is not None

        def boom(_nodes):
            raise RuntimeError("simulated embed backend failure")

        idx._index_obj.insert_nodes = boom  # type: ignore[method-assign]

        result = idx.sync()
        assert result.errors, "batch failure should surface as per-file errors"

    # Third open: the previous chunks should still be queryable.
    with RagIndex(notes_dir, settings) as idx:
        after = idx.retrieve("what is alpha", top_k=2)
        top_sources = [h.source for h in after.returned]
        assert any("alpha.md" in s for s in top_sources), top_sources
        # Manifest still references chunks that exist in the store.
        for p, e in idx.manifest.files.items():
            assert e.chunk_count > 0, f"{p} lost its chunks"
