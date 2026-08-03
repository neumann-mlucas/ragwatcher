"""Pure diff test — no embed, no model."""
from pathlib import Path

import pytest

from ragwatcher.config import Settings
from ragwatcher.index import RagIndex


@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text("# A\n\nalpha content")
    (tmp_path / "b.txt").write_text("beta content")
    return tmp_path


def _plan(idx: RagIndex, full: bool = False):
    return idx._plan_sync(full=full)


def test_all_add_when_empty(tiny_dir: Path):
    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        plan = _plan(idx)
        names = sorted(p.name for p in plan.add)
        assert names == ["a.md", "b.txt"]
        assert plan.update == []
        assert plan.delete == []


def test_noop_second_run(tiny_dir: Path):
    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        # Fake manifest entries so plan says no-op via fast-path
        from ragwatcher.manifest import FileEntry, sha256_file
        for p in tiny_dir.iterdir():
            if p.suffix in {".md", ".txt"}:
                st = p.stat()
                idx.manifest.files[str(p)] = FileEntry(
                    sha256=sha256_file(p),
                    mtime_ns=st.st_mtime_ns,
                    size=st.st_size,
                    chunk_count=1,
                )
        plan = _plan(idx)
        assert plan.is_noop()


def test_delete_on_missing_file(tiny_dir: Path):
    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        from ragwatcher.manifest import FileEntry
        idx.manifest.files["/nonexistent/gone.md"] = FileEntry(
            sha256="x", mtime_ns=0, size=0, doc_ids=["d"], chunk_count=1
        )
        plan = _plan(idx)
        assert "/nonexistent/gone.md" in plan.delete


def test_update_on_content_change(tiny_dir: Path):
    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        from ragwatcher.manifest import FileEntry
        target = tiny_dir / "a.md"
        st = target.stat()
        idx.manifest.files[str(target)] = FileEntry(
            sha256="stale-hash",
            mtime_ns=st.st_mtime_ns - 1,  # force mtime miss so sha compared
            size=st.st_size,
            chunk_count=1,
        )
        plan = _plan(idx)
        assert target in plan.update


def test_full_rehashes(tiny_dir: Path):
    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        from ragwatcher.manifest import FileEntry, sha256_file
        for p in tiny_dir.iterdir():
            if p.is_file():
                st = p.stat()
                idx.manifest.files[str(p)] = FileEntry(
                    sha256=sha256_file(p),
                    mtime_ns=st.st_mtime_ns,
                    size=st.st_size,
                    chunk_count=1,
                )
        plan = _plan(idx, full=True)
        assert len(plan.update) == 2
