"""Integration test — real fastembed model on tiny corpus. Slow."""

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


def test_end_to_end_query_hits_alpha(notes_dir: Path) -> None:
    settings = Settings()
    with RagIndex(notes_dir, settings) as idx:
        result = idx.sync()
        assert result.plan.as_summary()["add"] == 2

        r = idx.retrieve("what is alpha", top_k=2)
        top_sources = [h.source for h in r.returned]
        assert any("alpha.md" in s for s in top_sources), top_sources


def test_manifest_survives_restart(notes_dir: Path) -> None:
    settings = Settings()
    with RagIndex(notes_dir, settings) as idx:
        idx.sync()
        first_files = set(idx.manifest.files)

    with RagIndex(notes_dir, settings) as idx2:
        assert set(idx2.manifest.files) == first_files
        # Second sync is a no-op
        r = idx2.sync()
        assert r.plan.is_noop()


def test_query_debug_shape(notes_dir: Path) -> None:
    settings = Settings()
    with RagIndex(notes_dir, settings) as idx:
        idx.sync()
        r = idx.retrieve("alpha", top_k=1, debug=True)
        assert r.stages.keys() >= {"dense_candidates", "reranked", "returned"}
        assert r.timings_ms.keys() >= {"retrieve", "rerank", "total"}
        if r.returned:
            assert r.returned[0].scores.keys() == {"dense", "bm25", "rrf", "rerank"}
