"""RagIndex — composes everything. Only place that mutates state."""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from ragwatcher import chunking, embed, manifest, readers
from ragwatcher.config import Settings
from ragwatcher.errors import EmbedError, LockHeld, ReadError, StoreError
from ragwatcher.logging import get_logger
from ragwatcher.manifest import CURRENT_SCHEMA, FileEntry, Manifest, build_fingerprint, sha256_file
from ragwatcher.rerank import Reranker
from ragwatcher.store import Store, make_store

log = get_logger("index")


@dataclass
class SyncPlan:
    add: list[Path] = field(default_factory=list)
    update: list[Path] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)  # abs paths as strings

    def is_noop(self) -> bool:
        return not (self.add or self.update or self.delete)

    def as_summary(self) -> dict[str, int]:
        return {"add": len(self.add), "update": len(self.update), "delete": len(self.delete)}


@dataclass
class SyncResult:
    plan: SyncPlan
    errors: dict[str, str]
    duration_ms: int


@dataclass
class RetrievalHit:
    rank: int
    doc_id: str
    source: str
    snippet: str
    score: float
    scores: dict[str, float]
    stage_survived: str
    neighbors: list[str]


@dataclass
class RetrievalResult:
    query: str
    returned: list[RetrievalHit]
    dropped: list[dict[str, Any]]
    stages: dict[str, int]
    timings_ms: dict[str, int]


class RagIndex:
    def __init__(self, data_dir: Path, config: Settings) -> None:
        self.data_dir = data_dir.resolve()
        self.config = config
        self.persist_dir = self.data_dir / config.store.path
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.persist_dir / "lock"))
        self._locked = False
        self.store: Store | None = None
        self.manifest: Manifest | None = None
        self._reranker = Reranker(config.rerank)
        self._index_obj: Any = None
        self._bm25: Any = None

    def __enter__(self) -> RagIndex:
        try:
            self._lock.acquire(timeout=0)
            self._locked = True
        except FileLockTimeout as e:
            raise LockHeld(str(self.persist_dir / "lock")) from e
        self._open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        # sync() already persists + saves manifest under the invariant guard.
        # close() persists again to catch mutations outside sync (none today),
        # but only writes the manifest if the invariant still holds.
        persist_ok = True
        if self.store is not None:
            try:
                self.store.persist()
                self._check_store_invariant()
            except StoreError as e:
                persist_ok = False
                log.warning("persist_failed", extra={"err": str(e)})
        if self.manifest is not None and persist_ok:
            manifest.save(self.persist_dir / "manifest.json", self.manifest)
        if self._locked:
            self._lock.release()
            self._locked = False

    def _open(self) -> None:
        from llama_index.core import Settings as LISettings
        from llama_index.core.llms.mock import MockLLM

        LISettings.embed_model = embed.get_embedding(self.config.embed)
        LISettings.llm = MockLLM()  # retrieval only; avoids "LLM disabled" stdout warning

        self.store = make_store(self.config.store, self.persist_dir)
        fp = build_fingerprint(self.config)

        m = manifest.load(self.persist_dir / "manifest.json")
        if m is None:
            self.manifest = Manifest.new(fp)
        else:
            m.check_fingerprint(fp)  # raises SchemaMismatch on drift
            self.manifest = m

        # Recovery: if manifest claims chunks but vector store is empty on
        # disk, the store was clobbered. Reset the manifest so the next sync
        # re-embeds instead of no-op-ing on mtime.
        expected = sum(e.chunk_count for e in self.manifest.files.values())
        if expected > 0 and self.store.persisted_node_count() == 0:
            log.warning(
                "store_empty_reset_manifest",
                extra={"expected_chunks": expected, "persist_dir": str(self.persist_dir)},
            )
            self.manifest = Manifest.new(fp)

    # ---------------- sync ----------------

    def sync(
        self, full: bool = False, dry_run: bool = False, show_progress: bool = False
    ) -> SyncResult:
        assert self.manifest is not None and self.store is not None
        t0 = time.monotonic()
        plan = self._plan_sync(full=full)
        errors: dict[str, str] = {}

        if dry_run or plan.is_noop():
            return SyncResult(plan=plan, errors={}, duration_ms=int((time.monotonic() - t0) * 1000))

        targets = plan.add + plan.update
        progress = _maybe_progress(len(targets)) if show_progress else None
        for p in targets:
            try:
                self._embed_file(p)
            except (ReadError, EmbedError, StoreError) as e:
                errors[str(p)] = str(e)
                self._record_error(p, e)
                log.warning("file_failed", extra={"file": str(p), "err": str(e)})
            if progress:
                progress[0].advance(progress[1])
        if progress:
            progress[0].stop()

        for path_str in plan.delete:
            self._delete_file(path_str)

        if plan.add or plan.update or plan.delete:
            # Persist store first, then save manifest — so manifest never
            # references doc_ids that failed to reach the vector store.
            self.store.persist()
            self._check_store_invariant()
            manifest.save(self.persist_dir / "manifest.json", self.manifest)
            self._reset_retriever()

        return SyncResult(plan=plan, errors=errors, duration_ms=int((time.monotonic() - t0) * 1000))

    def _plan_sync(self, full: bool) -> SyncPlan:
        assert self.manifest is not None
        disk = self._scan_disk()
        plan = SyncPlan()

        for p in disk:
            key = str(p)
            entry = self.manifest.files.get(key)
            if entry is None:
                plan.add.append(p)
                continue
            st = p.stat()
            fast_hit = not full and entry.mtime_ns == st.st_mtime_ns and entry.size == st.st_size
            if fast_hit:
                continue
            digest = sha256_file(p)
            if digest != entry.sha256 or full:
                plan.update.append(p)

        disk_keys = {str(p) for p in disk}
        for key in self.manifest.files:
            if key not in disk_keys:
                plan.delete.append(key)

        return plan

    def _scan_disk(self) -> list[Path]:
        cfg = self.config.files
        out: list[Path] = []
        for path in _walk(self.data_dir, cfg.follow_symlinks):
            if path.is_dir():
                continue
            rel = path.relative_to(self.data_dir).as_posix()
            if any(part.startswith(".") for part in path.relative_to(self.data_dir).parts):
                continue
            if _matches_any(rel, cfg.exclude_globs):
                continue
            if cfg.include_globs and not _matches_any(rel, cfg.include_globs):
                continue
            if path.suffix.lower() not in readers.SUPPORTED_EXTS:
                continue
            try:
                if path.stat().st_size > cfg.max_file_bytes:
                    continue
            except OSError:
                continue
            out.append(path)
        return out

    def _embed_file(self, path: Path) -> None:
        assert self.store is not None and self.manifest is not None
        text, meta = readers.read(path)
        if not text.strip():
            return
        st = path.stat()
        digest = sha256_file(path)
        base_meta = {
            "file_path": str(path),
            "file_name": path.name,
            "ext": path.suffix.lower(),
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
            "sha256": digest,
        }
        base_meta.update(meta)
        prefixed = chunking.prefix_metadata(base_meta, text)
        doc = Document(text=prefixed, metadata=base_meta, doc_id=digest)

        splitter = chunking.get_splitter(self.config.chunk)
        nodes = splitter.get_nodes_from_documents([doc])
        for n in nodes:
            n.metadata.update(base_meta)

        # Delete old chunks first (update path)
        old_entry = self.manifest.files.get(str(path))
        if old_entry and old_entry.doc_ids:
            self._delete_doc_ids(old_entry.doc_ids)

        try:
            if self._index_obj is None:
                self._index_obj = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=self.store.storage_context(),
                )
            else:
                self._index_obj.insert_nodes(nodes)
        except Exception as e:
            raise EmbedError(f"embed/upsert failed for {path}: {e}") from e

        doc_ids = [n.node_id for n in nodes]
        self.manifest.files[str(path)] = FileEntry(
            sha256=digest,
            mtime_ns=st.st_mtime_ns,
            size=st.st_size,
            doc_ids=doc_ids,
            chunk_count=len(nodes),
            last_error=None,
            last_error_at=None,
        )
        log.info("file_embedded", extra={"file": str(path), "chunks": len(nodes)})

    def _delete_file(self, path_str: str) -> None:
        assert self.manifest is not None
        entry = self.manifest.files.pop(path_str, None)
        if entry:
            self._delete_doc_ids(entry.doc_ids)

    def _delete_doc_ids(self, doc_ids: list[str]) -> None:
        import contextlib

        assert self.store is not None
        vs = self.store.vector_store()
        for did in doc_ids:
            with contextlib.suppress(Exception):
                vs.delete(did)

    def _record_error(self, path: Path, exc: Exception) -> None:
        assert self.manifest is not None
        from datetime import UTC, datetime

        entry = self.manifest.files.setdefault(
            str(path),
            FileEntry(sha256="", mtime_ns=0, size=0),
        )
        entry.last_error = {"type": type(exc).__name__, "message": str(exc)}
        entry.last_error_at = datetime.now(UTC).isoformat()

    # ---------------- retrieve ----------------

    def _ensure_retriever(self) -> Any:
        assert self.store is not None
        if self._index_obj is not None:
            return self._index_obj

        from llama_index.core import load_index_from_storage

        ctx = self.store.storage_context()

        # Empty store → build an empty index bound to the storage context.
        # Do NOT feed docstore nodes here: that path silently re-embeds
        # everything and hides real load failures.
        if self.store.persisted_node_count() == 0:
            self._index_obj = VectorStoreIndex(nodes=[], storage_context=ctx)
            return self._index_obj

        try:
            self._index_obj = load_index_from_storage(ctx)
        except (ValueError, KeyError) as e:
            raise StoreError(
                f"failed to load index from {self.persist_dir}: {e}. "
                "Run `ragwatcher index --full` to rebuild."
            ) from e
        return self._index_obj

    def _reset_retriever(self) -> None:
        # Force rebuild w/ new nodes
        pass

    def _check_store_invariant(self) -> None:
        """Loud failure if manifest claims chunks but store is empty *on disk*.

        Uses persisted_node_count (not in-memory stats) — the original
        SimpleStore bug produced healthy-looking in-memory counts while
        persist silently wrote a 0-byte vector_store.json.
        """
        assert self.store is not None and self.manifest is not None
        expected = sum(e.chunk_count for e in self.manifest.files.values())
        if expected == 0:
            return
        got = self.store.persisted_node_count()
        if got == 0:
            raise StoreError(
                f"store invariant violated: manifest claims {expected} chunks "
                f"but persisted vector store has 0 nodes "
                f"(persist_dir={self.persist_dir})"
            )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        path_glob: str | None = None,
        ext: list[str] | None = None,
        recency_days: int | None = None,
        min_score: float | None = None,
        neighbor_window: int | None = None,
        debug: bool = False,
    ) -> RetrievalResult:
        assert self.manifest is not None
        cfg = self.config
        k = top_k or cfg.retrieve.top_k
        k = min(k, cfg.retrieve.top_k_max)
        _ = neighbor_window if neighbor_window is not None else cfg.retrieve.neighbor_window
        # neighbor expansion wired in P3 alongside PrevNextNodePostprocessor
        mult = cfg.rerank.top_k_input_multiplier if cfg.rerank.enabled else 1
        candidate_k = max(k * mult, k)

        stages: dict[str, int] = {}
        timings: dict[str, int] = {}
        t_all = time.monotonic()

        idx = self._ensure_retriever()
        t0 = time.monotonic()
        dense = idx.as_retriever(similarity_top_k=candidate_k)
        try:
            dense_hits: list[NodeWithScore] = dense.retrieve(question)
        except Exception as e:
            raise StoreError(f"retrieve failed: {e}") from e
        stages["dense_candidates"] = len(dense_hits)

        bm25_hits: list[NodeWithScore] = []
        if cfg.retrieve.hybrid:
            bm25_hits = self._bm25_retrieve(question, candidate_k)
        stages["bm25_candidates"] = len(bm25_hits)

        fused = _rrf_fuse([dense_hits, bm25_hits], candidate_k)
        stages["fused"] = len(fused)

        filtered, dropped_filter = _apply_filters(
            fused, path_glob=path_glob, exts=ext, recency_days=recency_days
        )
        stages["after_filters"] = len(filtered)
        timings["retrieve"] = int((time.monotonic() - t0) * 1000)

        # Rerank
        t1 = time.monotonic()
        rerank_scores: dict[str, float] = {}
        if cfg.rerank.enabled and filtered:
            passages = [h.node.get_content() for h in filtered]
            scores = self._reranker.score(question, passages)
            for h, s in zip(filtered, scores, strict=False):
                rerank_scores[h.node.node_id] = float(s)
            filtered.sort(key=lambda h: rerank_scores.get(h.node.node_id, 0.0), reverse=True)
        stages["reranked"] = len(filtered)
        timings["rerank"] = int((time.monotonic() - t1) * 1000)

        # Truncate + min_score
        top = filtered[:k]
        if min_score is not None:
            top = [h for h in top if rerank_scores.get(h.node.node_id, h.score or 0.0) >= min_score]
        stages["returned"] = len(top)

        returned: list[RetrievalHit] = []
        for i, h in enumerate(top, 1):
            fp = h.node.metadata.get("file_path", "?")
            neighbors: list[str] = []  # populated in P3 alongside PrevNext postprocessor
            final = rerank_scores.get(h.node.node_id, h.score or 0.0)
            returned.append(
                RetrievalHit(
                    rank=i,
                    doc_id=h.node.node_id,
                    source=fp,
                    snippet=h.node.get_content()[:400],
                    score=final,
                    scores={
                        "dense": _lookup_score(dense_hits, h.node.node_id),
                        "bm25": _lookup_score(bm25_hits, h.node.node_id),
                        "rrf": h.score or 0.0,
                        "rerank": rerank_scores.get(h.node.node_id, 0.0),
                    },
                    stage_survived="reranked" if cfg.rerank.enabled else "fused",
                    neighbors=neighbors,
                )
            )

        timings["total"] = int((time.monotonic() - t_all) * 1000)
        return RetrievalResult(
            query=question,
            returned=returned,
            dropped=dropped_filter,
            stages=stages,
            timings_ms=timings,
        )

    def _bm25_retrieve(self, question: str, top_k: int) -> list[NodeWithScore]:
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
        except ImportError:
            return []
        assert self.store is not None
        if self._bm25 is None:
            try:
                docstore = self.store.storage_context().docstore
                nodes = list(docstore.docs.values())
                if not nodes:
                    return []
                self._bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
            except Exception as e:
                log.warning("bm25_init_failed", extra={"err": str(e)})
                return []
        try:
            hits: list[NodeWithScore] = self._bm25.retrieve(question)
            return hits
        except Exception as e:
            log.warning("bm25_retrieve_failed", extra={"err": str(e)})
            return []

    # ---------------- stats ----------------

    def stats(self, include_errors: bool = False) -> dict[str, Any]:
        assert self.manifest is not None and self.store is not None
        s = self.store.stats()
        files = len(self.manifest.files)
        chunk_total = sum(e.chunk_count for e in self.manifest.files.values())
        error_files = [p for p, e in self.manifest.files.items() if e.last_error]
        out: dict[str, Any] = {
            "dir": str(self.data_dir),
            "backend": s.backend,
            "persist": s.persist_path,
            "files": files,
            "chunks": chunk_total,
            "nodes_in_store": s.node_count,
            "errors": len(error_files),
            "schema_version": CURRENT_SCHEMA,
        }
        if include_errors:
            out["error_details"] = [
                {"path": p, **(self.manifest.files[p].last_error or {})} for p in error_files
            ]
        return out

    def sources(self, errors_only: bool = False, stale_only: bool = False) -> list[dict[str, Any]]:
        assert self.manifest is not None
        out: list[dict[str, Any]] = []
        for p, e in self.manifest.files.items():
            if errors_only and not e.last_error:
                continue
            if stale_only:
                if not Path(p).exists():
                    continue
            out.append(
                {
                    "path": p,
                    "chunks": e.chunk_count,
                    "sha256": e.sha256,
                    "size": e.size,
                    "last_error": e.last_error,
                }
            )
        return out


def _maybe_progress(total: int) -> tuple[Any, Any] | None:
    """Rich progress bar. Suppressed if stderr not a tty (see A.7)."""
    import sys

    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from ragwatcher.logging import stderr_console

    if not sys.stderr.isatty() or total == 0:
        return None
    prog = Progress(
        TextColumn("[dim]indexing[/dim]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=stderr_console(),
        transient=True,
    )
    prog.start()
    task_id = prog.add_task("sync", total=total)
    return (prog, task_id)


def _walk(root: Path, follow_symlinks: bool) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and (follow_symlinks or not p.is_symlink())]


def _matches_any(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def _rrf_fuse(
    lists: list[list[NodeWithScore]], top_k: int, k_const: int = 60
) -> list[NodeWithScore]:
    scored: dict[str, tuple[NodeWithScore, float]] = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            nid = h.node.node_id
            add = 1.0 / (k_const + rank)
            if nid in scored:
                prev_node, prev_score = scored[nid]
                scored[nid] = (prev_node, prev_score + add)
            else:
                scored[nid] = (h, add)
    fused = []
    for node, score in scored.values():
        node.score = score
        fused.append(node)
    fused.sort(key=lambda h: h.score or 0.0, reverse=True)
    return fused[:top_k]


def _apply_filters(
    hits: list[NodeWithScore],
    path_glob: str | None,
    exts: list[str] | None,
    recency_days: int | None,
) -> tuple[list[NodeWithScore], list[dict[str, Any]]]:
    kept: list[NodeWithScore] = []
    dropped: list[dict[str, Any]] = []
    cutoff_ns: int | None = None
    if recency_days is not None:
        cutoff_ns = int(time.time() * 1e9) - recency_days * 86_400 * 10**9

    for h in hits:
        meta = h.node.metadata
        fp = meta.get("file_path", "")
        if path_glob and not fnmatch.fnmatch(fp, path_glob):
            dropped.append({"doc_id": h.node.node_id, "reason": "path_glob"})
            continue
        if exts:
            ext = meta.get("ext", Path(fp).suffix.lower())
            if ext not in {e.lower() for e in exts}:
                dropped.append({"doc_id": h.node.node_id, "reason": "ext"})
                continue
        if cutoff_ns is not None:
            mt = int(meta.get("mtime_ns", 0))
            if mt < cutoff_ns:
                dropped.append({"doc_id": h.node.node_id, "reason": "recency_days"})
                continue
        kept.append(h)
    return kept, dropped


def _lookup_score(hits: list[NodeWithScore], node_id: str) -> float:
    for h in hits:
        if h.node.node_id == node_id:
            return float(h.score or 0.0)
    return 0.0
