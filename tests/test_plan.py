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


def test_full_skips_recent_error_files(tiny_dir: Path):
    """--full re-scans everything except files whose last_error is fresh:
    a single broken PDF must not re-consume the entire embed budget every
    rescan. --retry-errors bypasses the skip."""
    from datetime import UTC, datetime

    from ragwatcher.manifest import FileEntry, sha256_file

    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        target = tiny_dir / "a.md"
        st = target.stat()
        idx.manifest.files[str(target)] = FileEntry(
            sha256=sha256_file(target),
            mtime_ns=st.st_mtime_ns,
            size=st.st_size,
            chunk_count=1,
            last_error={"type": "ReadError", "message": "boom"},
            last_error_at=datetime.now(UTC).isoformat(),
        )

        plan = idx._plan_sync(full=True)
        assert target not in plan.update, "recent-error file must be skipped on --full"

        plan_retry = idx._plan_sync(full=True, retry_errors=True)
        assert target in plan_retry.update, "--retry-errors must include it"


def test_incremental_also_skips_recent_error_files(tiny_dir: Path):
    """Non-full sync must ALSO skip recent-error files when the file's
    mtime bumps (fast_hit fails). Prevents the watcher retry-loop that
    pinned the service at 99% CPU on a truncated PDF."""
    from datetime import UTC, datetime

    from ragwatcher.manifest import FileEntry

    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        target = tiny_dir / "a.md"
        st = target.stat()
        idx.manifest.files[str(target)] = FileEntry(
            sha256="stale-hash",
            mtime_ns=st.st_mtime_ns - 1,  # fail fast_hit so entry goes to sha compare
            size=st.st_size,
            chunk_count=1,
            last_error={"type": "ReadError", "message": "boom"},
            last_error_at=datetime.now(UTC).isoformat(),
        )

        plan = idx._plan_sync(full=False)
        assert target not in plan.update, "incremental must skip recent-error too"


def test_full_reincludes_stale_error_files(tiny_dir: Path):
    """Old last_error (past cutoff) doesn't gate the file — it retries."""
    from datetime import UTC, datetime, timedelta

    from ragwatcher.manifest import FileEntry, sha256_file

    settings = Settings()
    with RagIndex(tiny_dir, settings) as idx:
        target = tiny_dir / "a.md"
        st = target.stat()
        old = datetime.now(UTC) - timedelta(hours=idx.config.sync.retry_error_after_hours + 1)
        idx.manifest.files[str(target)] = FileEntry(
            sha256=sha256_file(target),
            mtime_ns=st.st_mtime_ns,
            size=st.st_size,
            chunk_count=1,
            last_error={"type": "ReadError", "message": "old"},
            last_error_at=old.isoformat(),
        )
        plan = idx._plan_sync(full=True)
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
