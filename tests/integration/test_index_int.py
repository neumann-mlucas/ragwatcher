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


def test_update_removes_old_chunks(notes_dir: Path) -> None:
    """Regression: _delete_doc_ids was calling vs.delete(node_id) — a silent
    no-op — so old chunks accumulated on every update. Fixed by delete_nodes()."""
    settings = Settings()
    with RagIndex(notes_dir, settings) as idx:
        idx.sync()
        before = idx.store.persisted_node_count()
        assert before > 0

        (notes_dir / "alpha.md").write_text("# alpha\n\ntotally different body\n")
        idx.sync()
        after = idx.store.persisted_node_count()

        # Old alpha chunks gone, new ones in — net count must equal manifest.
        expected = sum(e.chunk_count for e in idx.manifest.files.values())
        assert after == expected, (before, after, expected)


def test_delete_removes_chunks(notes_dir: Path) -> None:
    settings = Settings()
    with RagIndex(notes_dir, settings) as idx:
        idx.sync()
        (notes_dir / "alpha.md").unlink()
        idx.sync()
        expected = sum(e.chunk_count for e in idx.manifest.files.values())
        assert idx.store.persisted_node_count() == expected
        assert "alpha.md" not in " ".join(idx.manifest.files)


def test_hybrid_bm25_populated_with_lance_backend(notes_dir: Path) -> None:
    """Regression: with the default LanceDB backend, the docstore stayed empty
    (LanceDB owns the text) and BM25 fell back to zero hits — hybrid retrieval
    was silently dense-only. Fixed by store_nodes_override=True on the index."""
    settings = Settings()
    assert settings.store.backend == "lance"
    with RagIndex(notes_dir, settings) as idx:
        idx.sync()
        docstore = idx.store.storage_context().docstore
        assert len(list(docstore.docs.values())) > 0, "docstore must mirror nodes for BM25"

        r = idx.retrieve("alpha", top_k=2, debug=True)
        # BM25 half of hybrid must contribute (was 0 before the fix).
        assert r.stages["bm25_candidates"] > 0


def test_query_debug_shape(notes_dir: Path) -> None:
    settings = Settings()
    with RagIndex(notes_dir, settings) as idx:
        idx.sync()
        r = idx.retrieve("alpha", top_k=1, debug=True)
        assert r.stages.keys() >= {"dense_candidates", "reranked", "returned"}
        assert r.timings_ms.keys() >= {"retrieve", "rerank", "total"}
        if r.returned:
            assert r.returned[0].scores.keys() == {"dense", "bm25", "rrf", "rerank"}
