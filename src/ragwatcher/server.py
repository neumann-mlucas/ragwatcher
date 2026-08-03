"""MCP tool surface. Wraps RagIndex."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from ragwatcher.index import RagIndex
from ragwatcher.logging import get_logger

log = get_logger("server")


def build_server(index: RagIndex, name: str = "ragwatcher") -> MCPServer:
    app: MCPServer = MCPServer(name)

    @app.tool()
    def query(
        question: str,
        top_k: int = 5,
        path_glob: str | None = None,
        min_score: float | None = None,
        ext: list[str] | None = None,
    ) -> str:
        """Return top-k retrieval results as formatted text."""
        result = index.retrieve(
            question=question,
            top_k=top_k,
            path_glob=path_glob,
            min_score=min_score,
            ext=ext,
        )
        return _format_hits(result.returned)

    @app.tool()
    def query_json(
        question: str,
        top_k: int = 5,
        path_glob: str | None = None,
        min_score: float | None = None,
        ext: list[str] | None = None,
    ) -> dict[str, Any]:
        """Structured retrieval payload."""
        r = index.retrieve(
            question=question,
            top_k=top_k,
            path_glob=path_glob,
            min_score=min_score,
            ext=ext,
        )
        return {
            "query": r.query,
            "returned": [
                {
                    "rank": h.rank,
                    "doc_id": h.doc_id,
                    "source": h.source,
                    "snippet": h.snippet,
                    "score": h.score,
                }
                for h in r.returned
            ],
        }

    @app.tool()
    def reindex(full: bool = False) -> dict[str, Any]:
        """Trigger sync. Returns summary."""
        result = index.sync(full=full)
        return {
            "plan": result.plan.as_summary(),
            "errors": len(result.errors),
            "duration_ms": result.duration_ms,
        }

    @app.tool()
    def list_sources(glob: str | None = None) -> list[str]:
        """List indexed file paths, optionally filtered by fnmatch glob."""
        import fnmatch

        paths = [s["path"] for s in index.sources()]
        if glob:
            paths = [p for p in paths if fnmatch.fnmatch(p, glob)]
        return paths

    @app.tool()
    def stats() -> dict[str, Any]:
        """Index summary."""
        return index.stats()

    @app.tool()
    def get_chunk(doc_id: str) -> dict[str, Any] | None:
        """Return raw chunk text + metadata for a doc_id."""
        assert index.store is not None
        docstore = index.store.storage_context().docstore
        node = docstore.docs.get(doc_id)
        if node is None:
            return None
        return {
            "doc_id": doc_id,
            "text": node.get_content(),
            "metadata": dict(node.metadata),
        }

    return app


def _format_hits(hits: list[Any]) -> str:
    if not hits:
        return "no results"
    lines = []
    for h in hits:
        lines.append(f"### {h.rank}. {h.source} (score={h.score:.3f})")
        lines.append(h.snippet.strip())
        lines.append("")
    return "\n".join(lines)
