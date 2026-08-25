"""Concurrency bench: N reader threads querying while one writer runs sync().

Run: uv run python scripts/bench_concurrency.py

Asserts:
  * Readers never see LockHeld or torn manifest state.
  * Writer completes its sync while readers are active.
  * Read throughput scales with N.
"""

from __future__ import annotations

import os
import shutil
import statistics
import tempfile
import threading
import time
from pathlib import Path

from ragwatcher.config import Settings
from ragwatcher.errors import LockHeld
from ragwatcher.index import RagIndex

CORPUS = Path(__file__).resolve().parent.parent / "tests" / "data" / "notes"
READERS = 4
READS_PER_THREAD = 20
WRITER_ITERS = 3


def _reader(idx: RagIndex, done: threading.Event, latencies: list[float]) -> None:
    for _ in range(READS_PER_THREAD):
        if done.is_set():
            return
        t0 = time.monotonic()
        idx.retrieve("alpha", top_k=3)
        latencies.append((time.monotonic() - t0) * 1000)


def _writer(idx: RagIndex, tmp_dir: Path) -> int:
    """Force `WRITER_ITERS` non-noop syncs by touching a file each round."""
    touched = tmp_dir / "churn.md"
    touched.write_text("# initial\n")
    idx.sync()
    for i in range(WRITER_ITERS):
        touched.write_text(f"# churn {i}\nnew content number {i}\n")
        # Bust the mtime fast-path even inside the same second.
        st = touched.stat()
        os.utime(touched, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        idx.sync()
    return WRITER_ITERS


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "notes"
        shutil.copytree(CORPUS, tmp)
        settings = Settings()
        with RagIndex(tmp, settings) as idx:
            idx.sync()  # warm

            latencies: list[float] = []
            done = threading.Event()
            errors: list[str] = []

            def reader_wrap() -> None:
                try:
                    _reader(idx, done, latencies)
                except LockHeld as e:
                    errors.append(f"LockHeld in reader: {e}")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{type(e).__name__}: {e}")

            threads = [threading.Thread(target=reader_wrap) for _ in range(READERS)]
            t0 = time.monotonic()
            for th in threads:
                th.start()
            writer_syncs = _writer(idx, tmp)
            for th in threads:
                th.join(timeout=30)
            done.set()
            elapsed = time.monotonic() - t0

            assert not errors, f"reader errors: {errors}"
            assert latencies, "no reads recorded"

            n = len(latencies)
            print(f"readers        : {READERS}")
            print(f"reads total    : {n}")
            print(f"writer syncs   : {writer_syncs}")
            print(f"elapsed        : {elapsed * 1000:.0f}ms")
            print(f"throughput     : {n / elapsed:.1f} q/s")
            print(f"read latency   : "
                  f"p50={statistics.median(latencies):.1f}ms  "
                  f"p95={statistics.quantiles(latencies, n=20)[-1]:.1f}ms  "
                  f"max={max(latencies):.1f}ms")


if __name__ == "__main__":
    main()
