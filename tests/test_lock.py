from pathlib import Path

import pytest

from ragwatcher.config import Settings
from ragwatcher.errors import LockHeld
from ragwatcher.index import RagIndex


def test_second_instance_raises_lock_held(tmp_path: Path):
    (tmp_path / "a.md").write_text("hi")
    settings = Settings()
    with RagIndex(tmp_path, settings) as _idx:
        with pytest.raises(LockHeld):
            RagIndex(tmp_path, settings).__enter__()


def test_lock_released_on_exit(tmp_path: Path):
    (tmp_path / "a.md").write_text("hi")
    settings = Settings()
    with RagIndex(tmp_path, settings):
        pass
    # Should re-acquire cleanly
    with RagIndex(tmp_path, settings):
        pass
