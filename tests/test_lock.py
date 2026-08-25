"""Writer-only lock: many readers coexist; writers serialize."""

from pathlib import Path

import pytest

from ragwatcher.config import Settings
from ragwatcher.errors import LockHeld
from ragwatcher.index import RagIndex, _WriterLock


def test_two_readers_coexist(tmp_path: Path):
    (tmp_path / "a.md").write_text("hi")
    settings = Settings()
    with RagIndex(tmp_path, settings), RagIndex(tmp_path, settings):
        pass


def test_lock_released_on_exit(tmp_path: Path):
    (tmp_path / "a.md").write_text("hi")
    settings = Settings()
    with RagIndex(tmp_path, settings):
        pass
    with RagIndex(tmp_path, settings):
        pass


def test_reset_retriever_invalidates_bm25(tmp_path: Path):
    """Regression: _reset_retriever was `pass`, so BM25 kept its build-time
    snapshot after every sync. New files never appeared in the BM25 half of
    hybrid retrieval until the process was restarted."""
    (tmp_path / "a.md").write_text("hi")
    settings = Settings()
    with RagIndex(tmp_path, settings) as idx:
        idx._bm25 = object()  # simulate a live BM25 retriever
        idx._reset_retriever()
        assert idx._bm25 is None


def test_close_is_idempotent(tmp_path: Path):
    """Regression: cli.serve had both a signal handler and a `finally` clause
    calling close(); the second close re-took the writer lock (60s stall risk)."""
    (tmp_path / "a.md").write_text("hi")
    settings = Settings()
    idx = RagIndex(tmp_path, settings).__enter__()
    idx.close()
    idx.close()  # second call must be a no-op, not another lock acquire


def test_exclusive_serializes_writers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A second writer waiting past the timeout raises LockHeld."""
    path = tmp_path / "lock"
    a = _WriterLock(path)
    b = _WriterLock(path)

    # Shrink the exclusive timeout so the test doesn't sit for a minute.
    monkeypatch.setattr(_WriterLock, "_EXCLUSIVE_TIMEOUT_SEC", 0.2)

    with a.exclusive():
        with pytest.raises(LockHeld):
            with b.exclusive():
                pass
