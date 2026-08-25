"""Writer-only lock: many readers coexist; writers serialize."""

from pathlib import Path

import pytest

from ragwatcher.config import Settings
from ragwatcher.errors import LockHeld
from ragwatcher.index import RagIndex, _RWLock


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


def test_exclusive_serializes_writers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A second writer waiting past the timeout raises LockHeld."""
    path = tmp_path / "lock"
    a = _RWLock(path)
    b = _RWLock(path)
    a.acquire_shared()
    b.acquire_shared()

    # Shrink the exclusive timeout so the test doesn't sit for a minute.
    monkeypatch.setattr(_RWLock, "_EXCLUSIVE_TIMEOUT_SEC", 0.2)

    with a.exclusive():
        with pytest.raises(LockHeld):
            with b.exclusive():
                pass
