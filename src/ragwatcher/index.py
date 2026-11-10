"""RagIndex — composes everything. Only place that mutates state."""

from __future__ import annotations

import fnmatch
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore

from ragwatcher import chunking, embed, manifest, readers
from ragwatcher.config import Settings
from ragwatcher.errors import EmbedError, LockHeld, ReadError, StoreError
from ragwatcher.logging import get_logger
from ragwatcher.manifest import CURRENT_SCHEMA, FileEntry, Manifest, build_fingerprint, sha256_file
from ragwatcher.rerank import Reranker
from ragwatcher.store import Store, make_store

log = get_logger("index")

# Stable id for the single VectorStoreIndex we persist. llama-index otherwise
# mints a fresh UUID per VectorStoreIndex() call and appends it to
# index_store.json; that's how we ended up with 3 indices in the same store
# and `load_index_from_storage` failing with "Expected to load a single index".
_INDEX_ID = "ragwatcher"

# Min seconds between store+manifest checkpoints during a sync write phase.
_CHECKPOINT_SEC = 30.0


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


class _WriterLock:
    """Cross-process writer lock. Readers don't lock — LanceDB is
    transactional (default backend); SimpleStore is single-process only.

    `exclusive()` serializes syncs across processes: one writer at a time.
    Blocks up to `_EXCLUSIVE_TIMEOUT_SEC` waiting for a competing writer,
    then raises LockHeld.

    Windows fallback: `filelock.FileLock`. Unix: stdlib `fcntl.flock`.
    """

    _EXCLUSIVE_TIMEOUT_SEC = 60.0

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def exclusive(self) -> Any:
        return _ExclusiveScope(self)


class _ExclusiveScope:
    def __init__(self, lock: _WriterLock) -> None:
        self._lock = lock
        self._fd: int | None = None
        self._fallback: FileLock | None = None

    def __enter__(self) -> None:
        if sys.platform.startswith("win"):
            self._fallback = FileLock(str(self._lock.path))
            try:
                self._fallback.acquire(timeout=_WriterLock._EXCLUSIVE_TIMEOUT_SEC)
            except FileLockTimeout as e:
                raise LockHeld(str(self._lock.path)) from e
            return
        import fcntl

        self._fd = os.open(str(self._lock.path), os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + _WriterLock._EXCLUSIVE_TIMEOUT_SEC
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError as e:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise LockHeld(str(self._lock.path)) from e
                time.sleep(0.05)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fallback is not None:
            self._fallback.release()
            self._fallback = None
        elif self._fd is not None:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class RagIndex:
    def __init__(self, data_dir: Path, config: Settings) -> None:
        self.data_dir = data_dir.resolve()
        self.config = config
        self.persist_dir = self.data_dir / config.store.path
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # Writer-only cross-process lock. Readers don't take it — LanceDB is
        # transactional; SimpleStore is single-process. Windows falls back to
        # filelock (same file, same LockHeld semantics on contention).
        self._lock = _WriterLock(self.persist_dir / "lock")
        self._write_lock = threading.Lock()  # in-process serialization for sync()
        self.store: Store | None = None
        self.manifest: Manifest | None = None
        self._reranker = Reranker(config.rerank)
        self._index_obj: Any = None
        self._bm25: Any = None
        self._bm25_lock = threading.Lock()
        # (inode, mtime_ns) of manifest.json as of our last load/save; a
        # different value under the writer lock means another process wrote.
        self._manifest_sig: tuple[int, int] | None = None

    def __enter__(self) -> RagIndex:
        self._open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        # Nothing to flush: sync() checkpoints store + manifest under the
        # writer lock before returning. Re-persisting here would overwrite
        # other processes' writes with stale state, and taking the lock from
        # a signal handler that interrupted sync() on this thread deadlocks.
        return None

    def _open(self) -> None:
        from llama_index.core import Settings as LISettings
        from llama_index.core.llms.mock import MockLLM

        LISettings.embed_model = embed.get_embedding(self.config.embed)
        LISettings.llm = MockLLM()  # retrieval only; avoids "LLM disabled" stdout warning
        self._load_state()

    def _load_state(self) -> None:
        self.store = make_store(self.config.store, self.persist_dir)
        fp = build_fingerprint(self.config)

        mpath = self.persist_dir / "manifest.json"
        m = manifest.load(mpath)
        self._manifest_sig = _file_sig(mpath)
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
        self,
        full: bool = False,
        dry_run: bool = False,
        show_progress: bool = False,
        retry_errors: bool = False,
    ) -> SyncResult:
        assert self.manifest is not None and self.store is not None
        t0 = time.monotonic()
        plan = self._plan_sync(full=full, retry_errors=retry_errors)
        if dry_run or plan.is_noop():
            return SyncResult(plan=plan, errors={}, duration_ms=int((time.monotonic() - t0) * 1000))

        # Everything below mutates store + manifest: serialize across threads
        # (in-process lock) and processes (flock). Another process may have
        # written since we loaded, so reload + re-plan under the lock —
        # writing back our stale in-memory state would clobber theirs.
        with self._write_lock, self._lock.exclusive():
            if self._reload_if_stale():
                plan = self._plan_sync(full=full, retry_errors=retry_errors)
                if plan.is_noop():
                    return SyncResult(
                        plan=plan, errors={}, duration_ms=int((time.monotonic() - t0) * 1000)
                    )
            errors = self._apply_plan(plan, show_progress)
        return SyncResult(plan=plan, errors=errors, duration_ms=int((time.monotonic() - t0) * 1000))

    def _apply_plan(self, plan: SyncPlan, show_progress: bool) -> dict[str, str]:
        """Stage + embed + commit `plan`. Caller holds both writer locks."""
        assert self.manifest is not None and self.store is not None
        errors: dict[str, str] = {}
        targets = plan.add + plan.update
        progress = _maybe_progress(len(targets)) if show_progress else None
        total = len(targets)

        # Stage 1: read + chunk every file. No embed yet — per-file errors here
        # never poison the embed chunk they'd otherwise share.
        staged: list[tuple[Path, list[BaseNode], FileEntry]] = []
        for i, p in enumerate(targets, 1):
            try:
                s = self._stage_file(p)
                if s is not None:
                    staged.append(s)
            except (ReadError, EmbedError, StoreError) as e:
                # Semantic splitter path can raise EmbedError from _stage_file
                # (it calls the embed model at chunk time). Recursive splitter
                # only raises ReadError; keeping both keeps sync from aborting
                # when a non-default chunk strategy hits a bad file.
                errors[str(p)] = str(e)
                self._record_error(p, e)
                log.warning("file_failed", extra={"file": str(p), "err": str(e)})
            if progress:
                progress[0].advance(progress[1])
            # Structured progress for --json / long non-TTY runs where the
            # Rich bar is suppressed. Every 25 files or on the last file.
            if i % 25 == 0 or i == total:
                log.info(
                    "stage_progress",
                    extra={"done": i, "total": total, "staged": len(staged), "failed": len(errors)},
                )
        if progress:
            progress[0].stop()

        # Guardrail: log the write-phase shape so a --full on a big corpus
        # doesn't kick off in silence.
        node_est = sum(len(nodes) for _, nodes, _ in staged)
        chunk_size = self.config.sync.files_per_chunk
        pending = [staged[i : i + chunk_size] for i in range(0, len(staged), chunk_size)]
        log.info(
            "write_phase_starting",
            extra={
                "files": len(staged),
                "nodes_est": node_est,
                "delete_files": len(plan.delete),
                "files_per_chunk": chunk_size,
                "chunks": len(pending),
            },
        )

        successful_files = 0
        uncommitted = False
        last_ckpt = time.monotonic()
        while pending:
            chunk = pending.pop(0)
            chunk_nodes = [n for _, nodes, _ in chunk for n in nodes]
            try:
                self._insert_chunk(chunk_nodes)
            except Exception as e:
                # Rollback partial inserts so the store never holds orphans.
                self._delete_doc_ids([n.node_id for n in chunk_nodes])
                if len(chunk) > 1:
                    # Retry file by file: one bad file must not stamp
                    # last_error on every file it shared a batch with.
                    log.warning(
                        "chunk_embed_failed_retry_per_file",
                        extra={"err": str(e), "files": len(chunk)},
                    )
                    pending[:0] = [[f] for f in chunk]
                    continue
                p = chunk[0][0]
                errors[str(p)] = f"embed failed: {e}"
                self._record_error(p, EmbedError(str(e)))
                log.warning("file_embed_failed", extra={"file": str(p), "err": str(e)})
                continue

            # Chunk succeeded: drop old chunks for updates, replace the
            # manifest entries.
            for p, _nodes, _entry in chunk:
                old = self.manifest.files.get(str(p))
                if old and old.doc_ids:
                    self._delete_doc_ids(old.doc_ids)
            for p, nodes, entry in chunk:
                self.manifest.files[str(p)] = entry
                log.info("file_embedded", extra={"file": str(p), "chunks": len(nodes)})
                successful_files += 1
            uncommitted = True

            # Checkpoint on a time cadence, not per chunk: persist rewrites
            # the whole docstore/index_store JSON, so per-chunk is O(N^2).
            # ponytail: a mid-sync kill can leave up to _CHECKPOINT_SEC of
            # inserted Lance rows untracked by the manifest (re-added as
            # duplicates next sync). Make node ids deterministic + upsert if
            # that shows up in practice.
            if time.monotonic() - last_ckpt >= _CHECKPOINT_SEC:
                self._checkpoint()
                uncommitted = False
                last_ckpt = time.monotonic()
                log.info(
                    "chunk_committed",
                    extra={"done_files": successful_files, "staged_files": len(staged)},
                )

        for path_str in plan.delete:
            self._delete_file(path_str)
        if uncommitted or plan.delete:
            self._checkpoint()
        elif errors:
            # Errors-only tail: save manifest so last_error survives.
            self._save_manifest()

        if successful_files or plan.delete:
            self._reset_retriever()
        return errors

    def _checkpoint(self) -> None:
        assert self.store is not None
        self.store.persist()
        self._check_store_invariant()
        self._save_manifest()

    def _save_manifest(self) -> None:
        assert self.manifest is not None
        path = self.persist_dir / "manifest.json"
        manifest.save(path, self.manifest)
        self._manifest_sig = _file_sig(path)

    def _reload_if_stale(self) -> bool:
        """Reload store + manifest if another process saved since we did.

        Every writer saves the manifest right after persisting the store
        (see _checkpoint), so the manifest's (inode, mtime) is the change
        signal for both. Call under the exclusive lock.
        """
        if _file_sig(self.persist_dir / "manifest.json") == self._manifest_sig:
            return False
        log.info("reload_external_changes", extra={"persist_dir": str(self.persist_dir)})
        self._load_state()
        self._index_obj = None
        self._reset_retriever()
        return True

    def _insert_chunk(self, nodes: list[BaseNode]) -> None:
        if nodes:
            # Always insert into the loaded/persisted index. Building a fresh
            # VectorStoreIndex here would replace the persisted struct with
            # one holding only these nodes.
            self._ensure_retriever().insert_nodes(nodes)

    def _plan_sync(self, full: bool, retry_errors: bool = False) -> SyncPlan:
        assert self.manifest is not None
        disk = self._scan_disk()
        plan = SyncPlan()
        cutoff_hours = self.config.sync.retry_error_after_hours
        skipped_recent_errors = 0

        for p in disk:
            key = str(p)
            entry = self.manifest.files.get(key)
            if entry is None:
                plan.add.append(p)
                continue

            # Skip repeat-broken files on any sync unless the caller asked
            # to retry, but only while content is unchanged since the error.
            # Without this a single unreadable file (or a truncated PDF whose
            # mtime bumps under a sync tool) re-consumes the whole embed
            # budget every rescan and keeps the watcher in a retry loop. An
            # edited file is always retried.
            if (
                not retry_errors
                and entry.last_error
                and entry.last_error_sha256
                and _error_is_recent(entry.last_error_at, cutoff_hours)
                and sha256_file(p) == entry.last_error_sha256
            ):
                skipped_recent_errors += 1
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

        if skipped_recent_errors:
            log.info(
                "plan_skipped_recent_errors",
                extra={"count": skipped_recent_errors, "cutoff_hours": cutoff_hours},
            )
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

    def _stage_file(self, path: Path) -> tuple[Path, list[BaseNode], FileEntry] | None:
        """Read+chunk one file, returning nodes + a preliminary manifest entry.

        No embed here — the batch embed happens once per sync so FastEmbed can
        pack many files into each backend call.
        """
        text, meta = readers.read(path, timeout_sec=self.config.files.read_timeout_sec)
        if not text.strip():
            return None
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
        # Fixed key set: LanceDB pins the metadata struct schema at first
        # insert and rejects later rows carrying a new key.
        base_meta.update({k: meta.get(k, d) for k, d in readers.META_DEFAULTS.items()})
        prefixed = chunking.prefix_metadata(base_meta, text)
        doc = Document(text=prefixed, metadata=base_meta, doc_id=digest)

        splitter = chunking.get_splitter(self.config.chunk)
        nodes = splitter.get_nodes_from_documents([doc])
        for n in nodes:
            n.metadata.update(base_meta)

        entry = FileEntry(
            sha256=digest,
            mtime_ns=st.st_mtime_ns,
            size=st.st_size,
            doc_ids=[n.node_id for n in nodes],
            chunk_count=len(nodes),
            last_error=None,
            last_error_at=None,
        )
        return (path, nodes, entry)

    def _delete_file(self, path_str: str) -> None:
        assert self.manifest is not None
        entry = self.manifest.files.pop(path_str, None)
        if entry:
            self._delete_doc_ids(entry.doc_ids)

    def _delete_doc_ids(self, doc_ids: list[str]) -> None:
        # doc_ids are node_ids (see _stage_file). vs.delete() takes ref_doc_id
        # (=Document.doc_id=sha256), which never matches; delete_nodes() is the
        # right API and both SimpleVectorStore and LanceDBVectorStore support it.
        # Also drop them from the docstore (BM25 source) and index struct —
        # store_nodes_override mirrors every node there.
        # Not VectorStoreIndex.delete_nodes: its struct delete raises KeyError
        # on ids never inserted (rollback path), skipping the docstore cleanup.
        assert self.store is not None
        if not doc_ids:
            return
        try:
            idx = self._ensure_retriever()
            self.store.vector_store().delete_nodes(doc_ids)
            ctx = self.store.storage_context()
            for nid in doc_ids:
                idx.index_struct.nodes_dict.pop(nid, None)
                ctx.docstore.delete_document(nid, raise_error=False)
            ctx.index_store.add_index_struct(idx.index_struct)
        except Exception as e:
            # Loud, not silent: silent no-op here is the exact class of bug
            # (`vs.delete` mismatch) that let old chunks accumulate for months.
            log.warning(
                "delete_nodes_failed",
                extra={"err": str(e), "count": len(doc_ids), "backend": self.store.backend},
            )

    def _record_error(self, path: Path, exc: Exception) -> None:
        assert self.manifest is not None
        from datetime import UTC, datetime

        entry = self.manifest.files.setdefault(
            str(path),
            FileEntry(sha256="", mtime_ns=0, size=0),
        )
        entry.last_error = {"type": type(exc).__name__, "message": str(exc)}
        entry.last_error_at = datetime.now(UTC).isoformat()
        try:
            entry.last_error_sha256 = sha256_file(path)
        except OSError:
            entry.last_error_sha256 = None

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
        # ponytail: pre-existing stores that predate store_nodes_override
        # have an empty docstore + non-empty vector store. `--full` fixes
        # them but re-embeds everything. Add a `rebuild-docstore` CLI that
        # reads the LanceDB rows back into the docstore without re-embed
        # if the migration cost becomes real.
        if self.store.persisted_node_count() == 0:
            self._index_obj = VectorStoreIndex(
                nodes=[], storage_context=ctx, store_nodes_override=True
            )
            self._index_obj.set_index_id(_INDEX_ID)
            return self._index_obj

        try:
            # Explicit index_id: from the second sync forward the store has
            # exactly one index under `_INDEX_ID`, so this is a single load.
            self._index_obj = load_index_from_storage(
                ctx, index_id=_INDEX_ID, store_nodes_override=True
            )
        except ValueError:
            # Legacy path: older persist accumulated per-sync UUID index
            # stubs (see _INDEX_ID note), each owning a slice of the nodes.
            # Merge them into the largest, drop the rest, re-tag _INDEX_ID.
            from llama_index.core import load_indices_from_storage

            try:
                loaded = load_indices_from_storage(ctx, store_nodes_override=True)
            except (ValueError, KeyError) as e:
                raise StoreError(
                    f"failed to load index from {self.persist_dir}: {e}. "
                    "Run `ragwatcher index --full` to rebuild."
                ) from e
            if not loaded:
                raise StoreError(
                    f"no indices in {self.persist_dir}. Run `ragwatcher index --full` to rebuild."
                )
            best = max(loaded, key=lambda idx: len(idx.index_struct.nodes_dict))
            dropped_ids: list[str] = []
            for idx in loaded:
                if idx is best:
                    continue
                best.index_struct.nodes_dict.update(idx.index_struct.nodes_dict)
                old_id = idx.index_struct.index_id
                try:
                    ctx.index_store.delete_index_struct(old_id)
                    dropped_ids.append(old_id)
                except Exception as e:
                    log.warning("index_stub_drop_failed", extra={"id": old_id, "err": str(e)})
            best.set_index_id(_INDEX_ID)
            # Not persisted here: this also runs inside sync()'s write lock
            # (non-reentrant). The next sync checkpoint persists it; until
            # then each open redoes this cheap in-memory migration.
            log.info(
                "index_migrated_to_stable_id",
                extra={
                    "kept_nodes": len(best.index_struct.nodes_dict),
                    "dropped": dropped_ids,
                    "index_id": _INDEX_ID,
                },
            )
            self._index_obj = best
        return self._index_obj

    def _reset_retriever(self) -> None:
        # BM25 snapshots the docstore at build time; drop it so the next
        # _bm25_retrieve rebuilds against the post-sync nodes. _index_obj is
        # mutated in place by insert_nodes/delete_nodes, so no reset needed.
        with self._bm25_lock:
            self._bm25 = None

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
        # ponytail: BM25 rebuilds from a full docstore walk after every sync
        # (see _reset_retriever). O(N) at build time — cheap up to ~10k nodes,
        # hurts past ~500k. Swap for an on-disk BM25 index if a big corpus
        # measures as bm25_init-dominated in retrieve latency.
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
        except ImportError:
            return []
        assert self.store is not None
        if self._bm25 is None:
            with self._bm25_lock:
                if self._bm25 is None:  # double-checked; two readers can race here
                    try:
                        docstore = self.store.storage_context().docstore
                        nodes = list(docstore.docs.values())
                        if not nodes:
                            return []
                        self._bm25 = BM25Retriever.from_defaults(
                            nodes=nodes, similarity_top_k=top_k
                        )
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


def _file_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns)


def _error_is_recent(last_error_at: str | None, cutoff_hours: int) -> bool:
    """True if `last_error_at` is a parseable ISO timestamp within cutoff."""
    if not last_error_at or cutoff_hours <= 0:
        return False
    from datetime import UTC, datetime

    try:
        err_dt = datetime.fromisoformat(last_error_at)
    except (TypeError, ValueError):
        return False
    if err_dt.tzinfo is None:
        err_dt = err_dt.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - err_dt).total_seconds() / 3600
    return age_hours < cutoff_hours


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
