"""Filesystem watcher + debounced sync queue.

Single-worker thread. Watchdog handler enqueues; worker drains.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ragwatcher.logging import get_logger

log = get_logger("watch")

SyncFn = Callable[[], Any]


class Watcher:
    def __init__(
        self,
        data_dir: Path,
        sync: SyncFn,
        debounce_sec: float = 2.0,
        rescan_interval_sec: int = 300,
    ) -> None:
        self.data_dir = data_dir
        self._sync = sync
        self._debounce = debounce_sec
        self._rescan_interval = rescan_interval_sec
        self._q: queue.Queue[str] = queue.Queue()
        self._observer: Any = None
        self._worker: threading.Thread | None = None
        self._rescan: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        handler = _Handler(self._q, self.data_dir)
        obs = Observer()
        obs.schedule(handler, str(self.data_dir), recursive=True)
        obs.start()
        self._observer = obs
        self._worker = threading.Thread(target=self._drain, daemon=True, name="rag-sync")
        self._worker.start()
        if self._rescan_interval > 0:
            self._rescan = threading.Thread(target=self._periodic, daemon=True, name="rag-rescan")
            self._rescan.start()
        log.info("watch_started", extra={"dir": str(self.data_dir)})

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=timeout)
        self._q.put("__stop__")
        if self._worker:
            self._worker.join(timeout=timeout)
        log.info("watch_stopped")

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item == "__stop__":
                return
            # Debounce: drain everything else in the window
            deadline = time.monotonic() + self._debounce
            while time.monotonic() < deadline:
                try:
                    more = self._q.get(timeout=max(0.01, deadline - time.monotonic()))
                    if more == "__stop__":
                        return
                except queue.Empty:
                    break
            try:
                self._sync()
            except Exception as e:
                log.exception("sync_failed", extra={"err": str(e)})

    def _periodic(self) -> None:
        while not self._stop.wait(self._rescan_interval):
            log.debug("periodic_rescan_enqueue")
            self._q.put("__rescan__")


class _Handler(FileSystemEventHandler):
    def __init__(self, q: queue.Queue[str], data_dir: Path) -> None:
        self._q = q
        self._data_dir = data_dir

    def on_any_event(self, event: FileSystemEvent) -> None:
        src = event.src_path
        if isinstance(src, bytes):
            src = src.decode("utf-8", errors="replace")
        p = Path(src)
        if _is_ignored(p, self._data_dir):
            return
        self._q.put(str(p))


def _is_ignored(p: Path, root: Path) -> bool:
    try:
        rel = p.relative_to(root)
    except ValueError:
        return True
    parts = rel.parts
    if not parts:
        return True
    if any(part.startswith(".") for part in parts):
        return True
    if any(part == "__pycache__" or part == "node_modules" for part in parts):
        return True
    return False
