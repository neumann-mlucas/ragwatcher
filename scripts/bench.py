"""Retrieval benchmark on the fixture corpus.

Run: uv run python scripts/bench.py

Reports: recall@1, recall@3 for a handful of hand-labelled queries.
Enough to catch regressions when swapping embed / rerank / chunking.
"""

from __future__ import annotations

import time
from pathlib import Path

from ragwatcher.config import Settings
from ragwatcher.index import RagIndex

QUERIES = [
    ("what is alpha", "alpha.md"),
    ("beta testing", "beta.md"),
    ("greek letter after alpha", "beta.md"),
]


def main() -> None:
    corpus = Path(__file__).resolve().parent.parent / "tests" / "data" / "notes"
    settings = Settings()
    hits_at_1 = 0
    hits_at_3 = 0
    t0 = time.monotonic()
    with RagIndex(corpus, settings) as idx:
        idx.sync()
        for q, expected in QUERIES:
            r = idx.retrieve(q, top_k=3)
            sources = [Path(h.source).name for h in r.returned]
            if sources and sources[0] == expected:
                hits_at_1 += 1
            if expected in sources:
                hits_at_3 += 1
    dt = int((time.monotonic() - t0) * 1000)
    n = len(QUERIES)
    print(f"queries        : {n}")
    print(f"recall@1       : {hits_at_1}/{n} = {hits_at_1 / n:.0%}")
    print(f"recall@3       : {hits_at_3}/{n} = {hits_at_3 / n:.0%}")
    print(f"total duration : {dt}ms")


if __name__ == "__main__":
    main()
