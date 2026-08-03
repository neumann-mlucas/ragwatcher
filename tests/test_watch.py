"""Watcher: debounce + ignore rules. Uses real watchdog Observer."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ragwatcher.watch import Watcher, _is_ignored


def _make_watcher(tmp_path: Path, sync, debounce: float = 0.15) -> Watcher:
    return Watcher(tmp_path, sync=sync, debounce_sec=debounce, rescan_interval_sec=0)


def test_burst_of_events_coalesces_to_one_sync(tmp_path: Path) -> None:
    calls: list[float] = []
    fired = threading.Event()

    def sync() -> None:
        calls.append(time.monotonic())
        fired.set()

    w = _make_watcher(tmp_path, sync, debounce=0.15)
    w.start()
    try:
        for i in range(6):
            (tmp_path / f"f{i}.txt").write_text(str(i))
            time.sleep(0.01)
        assert fired.wait(timeout=3.0), "sync never ran"
        time.sleep(0.4)  # give any stragglers time to coalesce or fire again
    finally:
        w.stop()

    assert 1 <= len(calls) <= 2, f"expected ~1 coalesced sync, got {len(calls)}"


def test_dotfiles_ignored_at_handler(tmp_path: Path) -> None:
    fired = threading.Event()

    def sync() -> None:
        fired.set()

    w = _make_watcher(tmp_path, sync, debounce=0.1)
    w.start()
    try:
        (tmp_path / ".hidden.txt").write_text("x")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "y.pyc").write_text("x")
        assert not fired.wait(timeout=0.6), "dotfiles/pycache should not enqueue"
    finally:
        w.stop()


def test_stop_is_prompt(tmp_path: Path) -> None:
    w = _make_watcher(tmp_path, sync=lambda: None, debounce=0.1)
    w.start()
    t0 = time.monotonic()
    w.stop(timeout=2.0)
    assert time.monotonic() - t0 < 2.0


def test_sync_exception_does_not_kill_worker(tmp_path: Path) -> None:
    call_count = {"n": 0}
    ok_after_raise = threading.Event()

    def sync() -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        ok_after_raise.set()

    w = _make_watcher(tmp_path, sync, debounce=0.1)
    w.start()
    try:
        (tmp_path / "a.txt").write_text("1")
        time.sleep(0.4)
        (tmp_path / "b.txt").write_text("2")
        assert ok_after_raise.wait(timeout=3.0), "worker died after exception"
    finally:
        w.stop()


def test_is_ignored_rules(tmp_path: Path) -> None:
    assert _is_ignored(tmp_path / ".git" / "HEAD", tmp_path)
    assert _is_ignored(tmp_path / "__pycache__" / "x.pyc", tmp_path)
    assert _is_ignored(tmp_path / "node_modules" / "pkg", tmp_path)
    assert _is_ignored(Path("/somewhere/else"), tmp_path)
    assert not _is_ignored(tmp_path / "notes" / "a.md", tmp_path)
