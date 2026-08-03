"""MCP tool surface round-trip via in-memory client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.client import Client

from ragwatcher.index import RetrievalHit, RetrievalResult, SyncPlan, SyncResult
from ragwatcher.server import build_server

pytestmark = pytest.mark.asyncio


class FakeIndex:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.store = MagicMock()
        self.store.storage_context().docstore.docs.get.return_value = None

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        path_glob: str | None = None,
        min_score: float | None = None,
        ext: list[str] | None = None,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=question,
            returned=[
                RetrievalHit(
                    rank=1,
                    doc_id="d1",
                    source="a.md",
                    snippet=f"hit for {question}",
                    score=0.9,
                    scores={"dense": 0.9, "bm25": 0.5, "rrf": 0.8, "rerank": 0.95},
                    stage_survived="rerank",
                    neighbors=[],
                )
            ],
            dropped=[],
            stages={"dense_candidates": 5, "reranked": 3, "returned": 1},
            timings_ms={"retrieve": 1, "rerank": 1, "total": 2},
        )

    def sync(self, full: bool = False, show_progress: bool = False) -> SyncResult:
        self.sync_calls += 1
        return SyncResult(plan=SyncPlan(add=[Path("x")]), errors={}, duration_ms=1)

    def sources(self) -> list[dict[str, Any]]:
        return [{"path": "/a.md"}, {"path": "/sub/b.txt"}, {"path": "/c.py"}]

    def stats(self) -> dict[str, Any]:
        return {"docs": 2, "chunks": 5}


def _text(result: Any) -> str:
    return "".join(c.text for c in result.content if getattr(c, "text", None))


async def test_list_tools_exposes_all() -> None:
    srv = build_server(FakeIndex())  # type: ignore[arg-type]
    async with Client(srv) as c:
        r = await c.list_tools()
        names = {t.name for t in r.tools}
    assert names == {"query", "query_json", "reindex", "list_sources", "stats", "get_chunk"}


async def test_query_returns_formatted_hits() -> None:
    srv = build_server(FakeIndex())  # type: ignore[arg-type]
    async with Client(srv) as c:
        r = await c.call_tool("query", {"question": "alpha"})
    assert not r.is_error
    text = _text(r)
    assert "hit for alpha" in text
    assert "a.md" in text


async def test_query_json_returns_structured() -> None:
    srv = build_server(FakeIndex())  # type: ignore[arg-type]
    async with Client(srv) as c:
        r = await c.call_tool("query_json", {"question": "beta", "top_k": 1})
    assert not r.is_error
    payload = r.structured_content or json.loads(_text(r))
    assert payload["query"] == "beta"
    assert payload["returned"][0]["doc_id"] == "d1"


async def test_reindex_triggers_sync() -> None:
    idx = FakeIndex()
    srv = build_server(idx)  # type: ignore[arg-type]
    async with Client(srv) as c:
        r = await c.call_tool("reindex", {})
    assert not r.is_error
    assert idx.sync_calls == 1


async def test_list_sources_glob_filter() -> None:
    srv = build_server(FakeIndex())  # type: ignore[arg-type]
    async with Client(srv) as c:
        all_r = await c.call_tool("list_sources", {})
        md_r = await c.call_tool("list_sources", {"glob": "*.md"})
    all_text = _text(all_r) + json.dumps(all_r.structured_content or {})
    md_text = _text(md_r) + json.dumps(md_r.structured_content or {})
    assert "/c.py" in all_text
    assert "/c.py" not in md_text
    assert "/a.md" in md_text


async def test_stats_ok() -> None:
    srv = build_server(FakeIndex())  # type: ignore[arg-type]
    async with Client(srv) as c:
        r = await c.call_tool("stats", {})
    assert not r.is_error


async def test_get_chunk_missing_returns_null() -> None:
    srv = build_server(FakeIndex())  # type: ignore[arg-type]
    async with Client(srv) as c:
        r = await c.call_tool("get_chunk", {"doc_id": "nope"})
    assert not r.is_error
