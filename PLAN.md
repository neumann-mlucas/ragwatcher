# ragwatcher — Action Plan & Todo

Companion to `SPEC.md`. Refines open items, orders work, sets acceptance criteria.

---

## Part A — Spec refinements (deltas to SPEC.md)

### A.1 Reconciled decisions (spec §18 open questions)

| Q | Decision | Notes |
|---|---|---|
| MCP primary vs CLI peer | **CLI peer** (decided in spec §18) | `cli.py` is the entry; `serve` is one subcommand. MCP tools call into `RagIndex` directly. |
| LLM answer synthesis | **No** | Non-goal. MCP client owns generation. Revisit only if the CLI needs `--answer` for shell scripting; even then, use a shell pipe. |
| Multi-directory | **No** | One instance = one dir. Multiple dirs → multiple processes. Keeps lock + config trivial. |
| Rename package | **No** | Keep `ragwatcher`. Existing MCP configs / muscle memory > naming purity. |
| TUI | **Dropped** (spec §11) | MCP client is primary UI; CLI + `rich` (tables, live, progress) + jq/fzf/`$EDITOR` covers every workflow a TUI would have. Reconsider only if a concrete workflow is proven CLI-hostile. |

### A.2 Python version

- **Decided**: `requires-python = ">=3.11"`. Local dev on 3.14 (`.python-version` kept). CI matrix `3.11, 3.12, 3.13`. Add 3.14 to CI once transitive deps ship wheels.
- Rationale: Ubuntu 22.04 baseline = 3.10 EOL → 3.11 floor covers current LTS distros.

### A.2.1 License
- **Decided**: MIT. `LICENSE` committed. `pyproject.toml` → `license = "MIT"`, `license-files = ["LICENSE"]`.

### A.2.2 Fresh start (no legacy port)
- **Decided**: no compatibility with `~/.local/bin/ragwatcher`. No manifest v1 migration. No shebang wrapper preservation.
- Consequences applied below:
  - PLAN A.4 (v1 → v2 migration): **dropped**.
  - Spec §16 steps 1, 7's "existing SimpleVectorStore persist dirs still load", and the `scripts/ragwatcher` shebang: **ignore**.
  - Manifest ships as v1 (of the new package). No `.bak` rotation needed except on future schema bumps.
  - P1 stops being "extract". Becomes "build modules per spec".
  - `test_migration.py`: dropped (no v1 to migrate).

### A.3 Config precedence — table merge rules

Spec says "low → high": defaults → user → per-dir → env → CLI. Clarify:
- **Scalars & lists**: higher precedence wins (replace, not merge).
- **Tables**: shallow-merge (per-key). Nested tables recurse.
- **Env vars**: `RAGWATCHER__EMBED__MODEL=...` (double-underscore = table separator; single = word separator inside key).
- **Per-dir `.ragwatcher.toml`**: loaded from **`<DIR>/.ragwatcher.toml` only** — no subdir walk, no upward walk. Root-only. Simplifies discovery + eliminates ambiguity when nested dirs conflict.
- Trust check on root-only file: on Unix, skip load w/ WARN if `owner != euid`. On Windows: load unconditionally (footgun tolerated, local tool).

### A.4 Manifest schema — pinned fields

**Dropped**: no v1 migration (fresh start, A.2.2). Manifest ships as v1 of the new package.

**Canonical shape** (single source of truth; spec §8 alignment):

```json
{
  "schema_version": 1,
  "created_at": "<iso8601>",
  "updated_at": "<iso8601>",
  "config_fingerprint": { ... see A.4.1 ... },
  "files": {
    "<abs_path>": {
      "sha256": "...",
      "mtime_ns": 0,
      "size": 0,
      "doc_ids": ["..."],
      "chunk_count": 0,
      "last_error": null,
      "last_error_at": null,
      "context_hashes": null
    }
  }
}
```

### A.4.1 `config_fingerprint` — pinned fields

Mutation of **any** field below = fingerprint change = index refuses to serve without `index --full` or `purge`.

```python
fingerprint = {
    # embedding
    "embed_model": str,          # e.g. "BAAI/bge-small-en-v1.5"
    "embed_dim": int | None,     # matryoshka truncation (nomic, mxbai); null = model native
    # chunking
    "chunk_strategy": str,       # "recursive" | "semantic" | "late"
    "chunk_size": int,
    "chunk_overlap": int,
    "chunk_max_chars": int,
    "chunk_semantic_breakpoint_percentile": int | None,  # only if strategy == "semantic"
    "chunk_semantic_buffer_size": int | None,             # only if strategy == "semantic"
    # storage
    "store_backend": str,        # "lance" | "simple" | "qdrant"
    # contextual retrieval (P4+)
    "context_enabled": bool,
    "context_prompt_sha256": str | None,   # sha256 of template text; null when disabled
}
```

Fields **excluded** (mutation does NOT invalidate the index):
- rerank model (rerank is post-retrieval)
- `top_k`, `top_k_multiplier`, `min_score` (query-time knobs)
- BM25 tuning (rebuilt at load)
- watcher / server / log / files.* config
- `neighbor_window` (query-time)

Fingerprint stored as sorted JSON; comparison is byte-equal.

### A.5 Vector store abstraction (`store.py`)

Small protocol — no ABC, just a `Protocol`:

```python
class Store(Protocol):
    def upsert(self, nodes: list[TextNode]) -> None: ...
    def delete(self, doc_ids: list[str]) -> None: ...
    def as_retriever(self, top_k: int, filters: MetadataFilters | None) -> BaseRetriever: ...
    def persist(self) -> None: ...
    def stats(self) -> StoreStats: ...
```

Backends: `LanceStore`, `SimpleStore` (existing llama-index), `QdrantStore` (extra). Each is <100 LOC — thin wrapper around llama-index vector store.

### A.6 Lock semantics

- `filelock.FileLock(<persist>/lock)`. Advisory, cross-platform.
- Acquire in `RagIndex.__init__` with `timeout=0` → raise `LockHeld` immediately if busy. Exit code 3.
- Release in `__exit__` / signal handler.
- `stats`, `sources`, `query` (one-shot), `doctor` acquire **shared read**: separate `<persist>/lock.read` counter file? **No — skip.** Simpler: read-only commands don't lock. Race with a running `serve` is safe because llama-index storage reads are snapshot-based; worst case is a stale read. `ponytail: no shared locks, add if concurrent writes ever corrupt manifest`.

### A.7 stdio-MCP + logging + progress

- MCP stdio protocol uses **stdout** for JSON-RPC frames. Any stdout write outside frames = protocol break.
- Rule: **all logs, progress, warnings go to stderr, always**. stdout is reserved for CLI data output + MCP frames.
- Progress bar disabled when: `not sys.stderr.isatty()` OR `--json` OR `--log-format=json` OR `transport=stdio`.
- Enforce via a `console = rich.console.Console(stderr=True, ...)` singleton.

### A.8 Watcher × rescan × sync coordination

- Single-worker sync queue: watchdog handler enqueues, background thread drains.
- Periodic rescan submits a `full=False` sync job; skipped if queue already has one pending.
- Debounce: coalesce events within `debounce_sec` window into one sync call.
- Sync during query: allowed; query uses current retriever snapshot; new nodes visible on next query.

### A.9 FastEmbed model cache

- Use `platformdirs.user_cache_dir("ragwatcher")/models`.
- Set `FASTEMBED_CACHE_PATH` env var before importing fastembed.
- Rationale: keeps model blobs out of `~/.cache/fastembed` (their default), so `ragwatcher purge --cache` can wipe cleanly.

### A.10 Error taxonomy → exit code table

| Exception | Exit | CLI message |
|---|---|---|
| `LockHeld` | 3 | "another ragwatcher holds <path>/lock" |
| `SchemaMismatch` | 4 | "index built with <old fingerprint>; run `ragwatcher index --full` or `purge`" |
| `ReadError` | 1 (logged, not fatal per-file) | logged at WARN, file skipped |
| `EmbedError` | 1 | "embedding backend failed: <msg>" |
| `StoreError` | 1 | "vector store failed: <msg>" |
| Typer usage | 2 | typer default |
| KeyboardInterrupt | 130 | drain + exit |

### A.11 Contextual retrieval — cache design

- Key: `sha256(chunk_text + embed_model + context_prompt_template)`.
- Store: `manifest.files[path].context_hashes` (parallel to `doc_ids`).
- Regen only when chunk_text OR prompt template changes. LLM endpoint change alone does NOT invalidate (assume same-quality output).
- LLM endpoint config: `[context] endpoint = "http://..."`, OpenAI-compatible chat API. Fail-closed: if endpoint down, embed WITHOUT context, log WARN; do NOT block indexing.

### A.12 Server HTTP mode — health/metrics

- `/health` synchronous (returns cached fields from last sync completion).
- `/metrics` gated by `[server] metrics = true`. Uses `prometheus_client` — behind `[metrics]` extra to avoid dep on baseline install.
- Skip Prometheus in 0.2/0.3; add in 0.6+.

### A.13 New tests missing from spec §14

- `test_lock.py`: two `RagIndex` instances on same dir → second raises `LockHeld`.
- `test_stdio_no_stdout.py`: run `serve --transport stdio` with a fake stdin, assert no non-frame bytes on stdout.
- `test_cli_smoke.py`: `typer.testing.CliRunner` on every subcommand with `--help`.
- `test_fingerprint.py`: mutate each pinned field → manifest load raises `SchemaMismatch`; non-pinned field mutations do NOT raise.
- `test_debug_output.py`: `query --debug --json` payload has stable keys per hit (`rank, doc_id, source, scores: {dense, bm25, rrf, rerank}, stage_survived, neighbors[]`) and a `stages` summary block.
- `test_last_error.py`: broken reader fixture → sync completes, manifest records `last_error` + `last_error_at`; next successful sync clears them.

### A.14 Fixture corpus (`tests/data/`)

Tiny, deterministic:
- `notes/alpha.md` — 3 headings, ~200 words.
- `notes/beta.md` — 1 heading, mentions "alpha" for cross-doc test.
- `code/example.py` — 40 lines, one class, one function.
- `docs/tiny.pdf` — 2 pages, generated from `.md` via `reportlab` at test-collect time (or committed binary <20KB).
- `mixed/table.csv` — 5 rows.
- `mixed/data.json` — nested object.
Total corpus <100KB. Embed w/ `bge-small` in <5s.

### A.15a Retrieval forensics — `query --debug` shape

Spec §6.9 names the fields; PLAN pins the JSON schema so tests + downstream tooling can lock on it.

```json
{
  "query": "...",
  "returned": [
    {
      "rank": 1,
      "doc_id": "...",
      "source": "path/to/file.md:120",
      "snippet": "...",
      "scores": {"dense": 0.83, "bm25": 4.11, "rrf": 0.019, "rerank": 0.91},
      "stage_survived": "reranked",
      "neighbors": ["doc_id_prev", "doc_id_next"]
    }
  ],
  "dropped": [
    {"doc_id": "...", "reason": "min_score", "scores": {...}}
  ],
  "stages": {
    "dense_candidates": 20,
    "bm25_candidates": 20,
    "fused": 32,
    "after_filters": 28,
    "reranked": 5,
    "returned": 5
  },
  "timings_ms": {"retrieve": 42, "rerank": 118, "total": 165}
}
```

Non-`--debug` output omits `dropped`, `stages`, `timings_ms`, and the per-hit `scores`/`stage_survived` keys — same top-level shape, additive fields only.

### A.15a1 BM25 persistence

- **Decided**: persist (option B).
- Mechanism: `llama_index.retrievers.bm25.BM25Retriever.persist(<persist>/bm25/)` on sync completion; `BM25Retriever.from_persist_dir(...)` on startup. No new dep.
- Rebuild triggered when: fingerprint changes, or `bm25/` missing/corrupt (fall back to in-mem rebuild, log WARN, re-persist).
- Included in atomic critical section w/ manifest.

### A.15a2 Query timeout

- **Decided**: soft timeout via `concurrent.futures.Future.result(timeout=)` in **HTTP mode only**. stdio: no timeout (client controls).
- Default `[server].query_timeout_sec = 30`. `0` = disable.
- On timeout: return partial results (whatever reranker had at cutoff) w/ `warning: "query_timeout_partial"` field. Worker continues (not killed — Python can't cleanly kill a thread).
- `ponytail: soft timeout only, upgrade to worker-kill if long queries pile up`.

### A.15a3 Reranker default

- **Decided**: keep it light. Ship w/ current `Xenova/ms-marco-MiniLM-L-6-v2` as default (already tiny, CPU, no torch).
- `bge-reranker-v2-m3` demoted to config option `rerank.model = "..."`. Users w/ CPU headroom opt in.
- Reason: `bge-reranker-v2-m3` in fastembed may require sentence-transformers (torch). Baseline install stays torch-free.
- P3.1 task ("swap default reranker") **dropped**. Replaced by P3.1' ("verify bge-reranker-v2-m3 loadable via fastembed; document as opt-in").

### A.15a4 First-run download UX

- `doctor` preflights: check `platformdirs.user_cache_dir("ragwatcher")/models/<embed_model>/` exists.
- If missing: `doctor` prints "downloading <model> (~130MB) — first run only" and warms cache before returning green.
- `serve` on cold cache: same message via structlog INFO before pull. Never silent.

### A.15b `last_error` semantics

- **Set** when a per-file operation (read, chunk, embed, upsert) raises. Store `{type: <ExceptionName>, message: <str>, at: <iso8601>}`.
- **Cleared** on the first successful sync of that file (all four stages pass).
- **Not cleared** on skip (unchanged mtime + hash) — an unchanged broken file stays broken.
- Surfaced by: `stats --errors` (aggregate + last N), `sources --errors` (per-file), `doctor` (count + top offenders).
- Manifest field is `nullable`; migration from v1 fills `null`.

### A.15c CI matrix (spec §14 clarification)

`.github/workflows/ci.yml`:
- Jobs: `lint` (ruff + mypy), `unit` (matrix 3.11-3.14 × ubuntu), `integration` (nightly cron + on `main` merges only).
- Cache: `~/.cache/uv` keyed on `uv.lock` hash.
- `fastembed` model cache: cached across runs keyed on model id.

---

## Part B — Phased action plan

Each phase ends w/ a shipped version + acceptance criteria. Don't start N+1 until N is green.

### Phase 0 — Bootstrap (0.5 day)

**Goal**: repo scaffold, no runtime code moved yet.

- [ ] Rewrite `pyproject.toml` w/ full metadata, deps split into baseline + extras (see A.2, spec §3).
- [ ] Create `src/ragwatcher/{__init__.py,__main__.py,py.typed}`.
- [ ] Add `ruff.toml` (rules: `E,F,I,UP,B,SIM,RUF`; line-length 100).
- [ ] Add `mypy.ini` (`strict`, ignore `llama_index.*` untyped).
- [ ] Add `.github/workflows/ci.yml` skeleton (lint job only).
- [ ] Wire `uv sync --extra dev`.
- [ ] Verify `uv run python -c "import ragwatcher; print(ragwatcher.__version__)"`.

**Accept**: `uv run pytest` runs w/ zero tests. `uv run ruff check` passes. `uv run mypy src` passes.

### Phase 1 — Build core modules (1-1.5 days)

**Goal**: minimum runnable `ragwatcher serve <DIR>` — package-native, no legacy carryover.

Order (file by file, one commit each):
1. [ ] `errors.py` — `LockHeld`, `SchemaMismatch`, `ReadError`, `EmbedError`, `StoreError`.
2. [ ] `logging.py` — stderr rich handler + JSON formatter for structlog (structlog kept per user decision).
3. [ ] `manifest.py` — `FileEntry`, `Manifest`, atomic load/save, canonical shape per A.4, fingerprint helpers per A.4.1.
4. [ ] `readers.py` — ext → reader factory (pdf/docx/epub/md/html/txt/csv/json/yaml).
5. [ ] `chunking.py` — recursive splitter only; hooks for semantic/late deferred to P4.
6. [ ] `embed.py` — FastEmbed factory, cache path via `FASTEMBED_CACHE_PATH` pre-import (A.9).
7. [ ] `rerank.py` — cross-encoder wrapper, lazy load, MiniLM default (bge-reranker-v2-m3 default swap in P3.1).
8. [ ] `store.py` — `Store` protocol (kept per user decision) + `SimpleStore` implementation. LanceStore added in P3.
9. [ ] `watch.py` — `ChangeHandler`, debounce, single-worker sync queue.
10. [ ] `index.py` — `RagIndex` (state, sync, retrieve). Composes 3-9.
11. [ ] `server.py` — `build_server(index) -> FastMCP` w/ tools per spec §10.
12. [ ] `cli.py` — typer app, `serve` subcommand only (other subcommands land in P2).
13. [ ] `__main__.py` — `from ragwatcher.cli import app; app()`.

**Accept**: `uv run ragwatcher serve tests/data/notes` starts, indexes, answers MCP `query` tool via stdio test client. No `print()` in `src/`. Fingerprint written to manifest on first sync.

### Phase 2 — CLI + config + robustness + forensics (1.5 days)

- [ ] `config.py` — pydantic-settings, precedence chain (A.3).
- [ ] Add subcommands: `query`, `index`, `stats`, `sources`, `purge`, `doctor`, `config`, `version`.
- [ ] Query flags per spec §4: `--top-k`, `--path-glob`, `--ext` (repeatable), `--recency-days`, `--min-score`, `--show-neighbors/--no-show-neighbors`, `--debug`, `--format table|snippets`, `--json`.
- [ ] Index flags: `--full`, `--dry-run`, `--json`, `--watch` (stream sync events, no full serve).
- [ ] Stats flags: `--watch --interval SEC` (rich.Live), `--errors`, `--json`.
- [ ] Sources flags: `--errors`, `--stale`, `--json`.
- [ ] Doctor: model cache, disk, config, manifest schema, index integrity, **fingerprint drift**, **watcher backend**.
- [ ] Retrieval forensics: `query --debug` prints per-stage candidate counts + per-hit scores per A.15a; wired into retriever pipeline (capture stage counts + hooks).
- [ ] Rich rendering: `rich.Table` for `--format table`, `rich.Panel` for `snippets`, `rich.Live` for `--watch`.
- [ ] `filelock` integration (A.6). Exit codes per A.10.
- [ ] Progress bar (rich) w/ suppression rules per A.7.
- [ ] Manifest v1 (of new package): `schema_version`, `config_fingerprint`, `last_error`, `last_error_at` fields defined (nullable). Fingerprint mismatch → `SchemaMismatch` (exit 4). No v1 migration (fresh start, A.2.2).
- [ ] Signal handlers (SIGTERM/SIGINT) drain + persist.
- [ ] Shell completion via `--install-completion`.
- [ ] Unit tests: `test_manifest`, `test_plan`, `test_config`, `test_lock`, `test_fingerprint`, `test_cli_smoke`, `test_debug_output`.

**Accept**: `ragwatcher doctor ~/notes` returns green. `ragwatcher query ~/notes "..." --json` emits valid JSON to stdout, nothing else. `query --debug --json` shape matches A.15a. `stats --watch` refreshes in-place. Two parallel `ragwatcher serve` on same dir → second exits w/ code 3. All unit tests pass < 10s total.

### Phase 3 — Quality upgrades (2 days)

- [ ] Verify `bge-reranker-v2-m3` loadable via fastembed; expose as `rerank.model` opt-in. Default stays MiniLM (A.15a3).
- [ ] Add `LanceStore`, make it default. Keep `SimpleStore` as fallback via config.
- [ ] Reader upgrades: markdown/HTML heading extraction → `heading_path` metadata.
- [ ] Code reader: language-aware recursive splitter (whitespace/indent fallback; no tree-sitter yet).
- [ ] CSV/JSON row-aware chunking w/ header prefix.
- [ ] Periodic rescan thread (A.8).
- [ ] Wire `last_error` capture + clear semantics per A.15b (schema field was added in P2; wiring happens here).
- [ ] Metadata-filter retrieval: `ext`, `recency_days`, `path_glob` actually applied at retrieve time (CLI flags landed in P2 as pass-through — this hooks them into `MetadataFilters`).
- [ ] Integration test `test_index_int.py` (fixture corpus, real embed model).
- [ ] Test `test_last_error.py`.
- [ ] Publish 0.2.0 to PyPI (test-pypi first).

**Accept**: On fixture corpus, query "who mentions alpha in beta" returns `beta.md` in top-2. `ragwatcher stats` shows LanceDB backend, chunk count matches expectation. `sources --errors` lists intentionally-broken fixture. Nightly CI integration job passes.

### Phase 4 — Advanced retrieval (opt-in, 2 days)

- [ ] Semantic chunker behind `chunk.strategy = "semantic"`.
- [ ] Late chunking behind `chunk.strategy = "late"` (requires long-context embed model — bge-m3).
- [ ] Contextual retrieval (A.11): `[context]` config block, LLM client, chunk hash cache, fail-open.
- [ ] Test: contextual on/off delta on fixture corpus; recall@3 should not regress.

**Accept**: switching `chunk.strategy` and `re-index --full` produces functional index; benchmark script in `scripts/bench.py` reports metrics without crashing.

### Phase 5 — Later (backlog, as-needed)

- Prometheus `/metrics` + `[metrics]` extra.
- Qdrant + remote embed/rerank via `[remote]` extra.
- Tree-sitter code splitter via `[code]` extra.
- OpenTelemetry (only if requested).
- Concurrency (batch embed, RW lock, LanceStore default) — see Part C, P6.

---

## Part C — Todo list (flat, actionable)

Legend: `[ ]` open, `[~]` blocked, `[x]` done. Prefixed w/ phase.

### P0 — Bootstrap
- [ ] P0.1 Rewrite `pyproject.toml` (deps, extras, entry point, `[tool.ruff]`, `[tool.mypy]`).
- [ ] P0.2 Create `src/ragwatcher/` skeleton (`__init__`, `__main__`, `py.typed`).
- [ ] P0.3 Add `ruff.toml`, `mypy.ini`.
- [ ] P0.4 Add `.github/workflows/ci.yml` (lint only).
- [ ] P0.5 `uv sync --all-extras`; confirm import.

### P1 — Build (fresh, no legacy port)
- [ ] P1.1 `errors.py` (exception hierarchy per A.10).
- [ ] P1.2 `logging.py` (stderr rich + structlog JSON formatter).
- [ ] P1.3 `manifest.py` (schema per A.4, fingerprint per A.4.1, atomic io).
- [ ] P1.4 `readers.py` (ext factory, per-file WARN on failure).
- [ ] P1.5 `chunking.py` (recursive splitter; strategy switch stubbed).
- [ ] P1.6 `embed.py` (FastEmbed w/ platformdirs cache).
- [ ] P1.7 `rerank.py` (cross-encoder, lazy load).
- [ ] P1.8 `store.py` (`Store` protocol + `SimpleStore`).
- [ ] P1.9 `watch.py` (debounce + single-worker queue).
- [ ] P1.10 `index.py` (`RagIndex`: composes 3-9).
- [ ] P1.11 `server.py` (`build_server(index)` w/ FastMCP tools).
- [ ] P1.12 `cli.py` — `serve` only.
- [ ] P1.13 `__main__.py`.
- [ ] P1.14 Smoke: `ragwatcher serve tests/data/notes`; MCP test client query returns hits.

### P2 — CLI + config
- [ ] P2.1 `config.py` (pydantic-settings, precedence).
- [ ] P2.2 Subcommand `query` (+ `--json`, `--path-glob`, `--top-k`, `--min-score`, `--show-neighbors`).
- [ ] P2.3 Subcommand `index` (+ `--full`, `--dry-run`, `--json`).
- [ ] P2.4 Subcommand `stats`, `sources`, `purge --yes`.
- [ ] P2.5 Subcommand `doctor` (validate config, check model cache, verify manifest schema, disk space).
- [ ] P2.6 Subcommand `config --show-sources`.
- [ ] P2.7 Subcommand `version`.
- [ ] P2.8 `filelock` in `RagIndex.__enter__`, exit code 3.
- [ ] P2.9 Signal handlers (SIGTERM/SIGINT).
- [ ] P2.10 Rich progress bar w/ A.7 suppression.
- [ ] P2.11 Manifest fingerprint enforcement + `.bak` on schema bump (no v1 migration — fresh start).
- [ ] P2.12 `--install-completion` support (typer built-in).
- [ ] P2.13 Test: `test_manifest.py`.
- [ ] P2.14 Test: `test_plan.py`.
- [ ] P2.15 Test: `test_config.py`.
- [ ] P2.16 Test: `test_lock.py`.
- [ ] P2.17 Test: `test_cli_smoke.py`.
- [ ] P2.18 Test: `test_stdio_no_stdout.py` (or CI grep `! grep -rn 'print(' src/ragwatcher`).
- [ ] P2.19 CI: add unit job to workflow.

### P3 — Quality
- [ ] P3.1 Verify `bge-reranker-v2-m3` loadable via fastembed; expose as `rerank.model` config option. Default stays MiniLM (A.15a3).
- [ ] P3.2 `LanceStore` backend + make default.
- [ ] P3.3 Markdown heading path metadata.
- [ ] P3.4 HTML heading path metadata.
- [ ] P3.5 Code reader (language-aware recursive splitter, no tree-sitter).
- [ ] P3.6 CSV/JSON row-aware chunking.
- [ ] P3.7 Periodic rescan thread.
- [ ] P3.8 Fixture corpus (`tests/data/`).
- [ ] P3.9 `test_index_int.py` (real embed, small).
- [ ] P3.10 `test_readers.py`, `test_chunking.py` for new formats.
- [ ] P3.11 CI: nightly integration job.
- [ ] P3.12 Publish 0.2.0 → test-pypi → pypi.

### P4 — Advanced retrieval
- [ ] P4.1 Semantic chunker.
- [ ] P4.2 Late chunker (bge-m3).
- [ ] P4.3 `[context]` config + LLM client + cache.
- [ ] P4.4 Metadata filters (`ext`, `recency_days`).
- [ ] P4.5 `min_score` threshold.
- [ ] P4.6 Bench script `scripts/bench.py`.
- [ ] P4.7 Publish 0.3.0.

### P5 — Backlog (as-needed)
- [ ] Prometheus metrics + `[metrics]` extra.
- [ ] Qdrant backend + `[remote]` extra.
- [ ] Tree-sitter code splitter + `[code]` extra.
- [ ] API embed/rerank backends.

### P6 — Concurrency (backlog, motivated by multi-Claude-session use)

**Problem.** `RagIndex` takes an exclusive `filelock.FileLock` in `__enter__`
(`index.py:84`), so a second `ragwatcher serve` blocks on the lock. The
recommended workaround today is one shared HTTP server; P6 makes the lock
model itself concurrent-friendly and speeds up sync.

Do the items in order — each stops paying if the previous made the workload
fast/safe enough. YAGNI applies.

- [ ] P6.1 **Batch embed in `sync()`.** Biggest win, no threads. Collect
      nodes from all files in the plan, call `embed_model.get_text_embedding_batch(texts)`
      once (or in chunks of N), then hand results to `vector_store.add`.
      Rewrite `_embed_file` to *stage* nodes; flush after the file loop in
      `sync()`. Manifest entries written after the flush. FastEmbed already
      parallelizes over CPU inside a batch, so single-thread is enough.
- [ ] P6.2 **RW lock (reader / writer).** Replace `filelock.FileLock` with
      `fcntl.flock(LOCK_SH | LOCK_EX)` (stdlib) or the `readerwriterlock`
      package. `retrieve()` acquires shared, `sync()` acquires exclusive.
      Update `LockHeld` semantics: still raised when a writer holds; readers
      never see it against other readers. Enables N concurrent stdio Claude
      sessions to `query`. Requires proving `SimpleVectorStore.query` is
      read-only and safe (it is — read of dict).
- [ ] P6.3 **LanceStore as default backend.** Stub exists at `store.py:92`.
      LanceDB is transactional (MVCC), so concurrent readers + a single
      writer are safe without app-level locking around the store. Migrate
      `store.default` to `lance`, keep `simple` as a zero-dep fallback.
      Manifest fingerprint bump forces re-embed.
- [ ] P6.4 **Thread pool for read + chunk stage.** Only after P6.1 measures
      as read-bound. `ThreadPoolExecutor(os.cpu_count())` runs
      `readers.read` + `splitter.get_nodes_from_documents` in parallel;
      results feed a single embed thread that drains a queue in batches.
      Needs a `threading.Lock` around `self.manifest.files[...] =` writes.
- [ ] P6.5 **BM25 lazy init race.** `index.py:410-416` — guard `self._bm25`
      construction with a `threading.Lock` (or build eagerly once at first
      sync).
- [ ] P6.6 **`LISettings` audit.** Global mutable in `index.py:110`. If P6.4
      lands, prove that no thread mutates `LISettings.embed_model` mid-run
      (currently only `_open()` writes it, once per process — should be OK).
- [ ] P6.7 **Bench + test.** Add `scripts/bench_concurrency.py`: N reader
      threads doing `retrieve` while one writer runs `sync`. Assert no
      `LockHeld`, no torn state, throughput scales.

**Non-goals for P6.** Multi-process writers, distributed indexing,
async-all-the-way through `index.py` (still sync — SPEC F.12 stands, only
FastMCP is async).

---

## Part D — Non-todos (deliberately deferred)

- Auth / multi-tenant / RBAC.
- Multi-directory single-process.
- Web UI.
- TUI (dropped per SPEC §11 / PLAN A.1).
- LLM answer synthesis.
- OpenTelemetry.
- Windows-first testing (best-effort only; primary is Linux/macOS).
- Cross-user shared indexes.

Each one has a single reason: **YAGNI until someone asks**. Adding earlier = drag on every phase above.

---

## Part E — Architecture

### E.1 Module boundaries (dependency direction)

```
cli.py ─────────┐
                ├─► index.py ──► store.py    (Store protocol; Lance/Simple/Qdrant)
server.py ──────┘            ├─► embed.py    (FastEmbedEmbedding factory)
                             ├─► rerank.py   (cross-encoder, lazy)
                             ├─► chunking.py (splitter factory)
                             ├─► readers.py  (ext → reader map)
                             ├─► watch.py    (Observer + queue)
                             └─► manifest.py (FileEntry, atomic io, fingerprint)

config.py    ── consumed by all; imports none of the above
logging.py   ── consumed by all; imports only stdlib + rich + structlog
errors.py    ── leaf; imports stdlib only
```

**Rules**
- `index.py` is the only module that mutates state. Everything else is pure/factory/adapter.
- No module below `index.py` imports from `cli` or `server`. Enforced by convention + CI grep; skip `import-linter` unless drift shows up.
- `config.py`, `logging.py`, `errors.py` are leaves — no project imports.
- `store.py` defines `Store` protocol; backends live in same file until >200 LOC each, then split to `store_lance.py`, etc.

### E.2 Runtime data flow

**Serve**
```
main() → CLI parses → load config → RagIndex(dir, config) __enter__ (acquires lock)
      → initial sync (readers → chunking → embed → store.upsert → manifest.save)
      → start Observer + rescan thread
      → build_server(index) → FastMCP.run(transport)
      → on shutdown: drain queue, persist, release lock
```

**Query**
```
question → retriever (dense + BM25 fusion via QueryFusionRetriever)
        → top_k*multiplier candidates
        → rerank.score(question, candidates)
        → truncate top_k, filter min_score
        → PrevNextNodePostprocessor (neighbor expansion)
        → format (text | json) → stdout
```

**Sync (single-file diff)**
```
plan_sync(disk_files, manifest_files) → {add: [...], update: [...], delete: [...]}
add|update: read → chunk → embed → store.upsert; manifest[path] = FileEntry(...)
delete:    store.delete(doc_ids); manifest.pop(path)
manifest.save() atomically
```

### E.3 Key invariants (never violate)

1. **stdout is sacred.** MCP frames + explicit CLI data. No `print()`, no log. Logs → stderr via `Console(stderr=True)`.
2. **Manifest write is atomic.** Always `tmp + rename`. Never partial write.
3. **Manifest ↔ store consistency.** Any `store.upsert` is followed by manifest update in the same critical section. Crash between = detected on next sync (doc_ids not in store → re-embed).
4. **Config fingerprint is authoritative.** If it mismatches, refuse to serve. No silent re-embed.
5. **Lock held for the lifetime of `RagIndex`.** No process on same dir escapes it.
6. **No global mutable state** except the rich console and the fastembed model cache path (set once, pre-import).
7. **All external I/O logged w/ duration_ms.** Debuggability.
8. **Reader failures are per-file WARN, not fatal.** One broken PDF doesn't kill sync.

### E.4 Persistence layout

```
<DIR>/
  <files...>
  .ragwatcher.toml            # optional per-dir override (root-only, A.3)
  .rag_index/
    manifest.json             # v1 (new package)
    manifest.json.bak         # written ONLY on schema bump; not per-write (A.2.2)
    lock                      # filelock target
    docstore.json             # llama-index docstore (BM25 source)
    bm25/                     # persisted BM25 postings (A.15a1)
    lance/chunks.lance/       # if store.backend=lance (P3)
    simple_vector_store.json  # if store.backend=simple (mutually exclusive w/ lance/)
    graph_store.json          # llama-index scaffolding, mostly empty
```

Cache (outside data dir, per platformdirs):
```
$XDG_CACHE_HOME/ragwatcher/
  models/                     # fastembed model blobs
  reranker/                   # cross-encoder blobs
```

### E.5 Concurrency model

- **One process per data dir** (enforced by lock).
- Inside process:
  - Main thread: MCP server or CLI request handler.
  - Worker thread: sync queue drainer (single, serial — indexing is I/O + CPU bound; parallelism gets you thrashing, not throughput).
  - Observer thread: watchdog callback → enqueue only.
  - Rescan thread: `time.sleep(interval)` → enqueue only.
- All shared state (`manifest`, `store`) touched only by main + worker; guarded by `threading.Lock` (one lock, whole index; `ponytail: global lock, split if profile shows contention`).
- Query path is read-only against a snapshot retriever — no lock needed to answer queries.

### E.6 Extension points (only these are stable)

- `Store` protocol → new backend = new class implementing 5 methods.
- `chunking.get_splitter(config) -> NodeParser` → new strategy = new branch in factory.
- `embed.get_embedding(config) -> BaseEmbedding` → new model = config value.
- Readers map (`readers.EXT_TO_READER`) → new format = one entry.

Everything else is internal. Break freely across minor versions before 1.0.

---

## Part F — Code guidelines

Grounded in the ponytail rules already in effect. Explicit for this repo.

### F.1 Style

- Python ≥3.11. Prefer `match`/`|` unions/`Self`/`Never` when they help; don't force.
- `ruff` formats + lints. Line length 100. Config in `ruff.toml`.
- Enabled rules: `E, F, I, UP, B, SIM, RUF, N, PL(C,E,W), TRY`. Don't disable individually unless documented in `ruff.toml` w/ 1-line reason.
- Imports sorted by ruff (`I`); no wildcard imports.
- No `# type: ignore` without `# type: ignore[code]  # <reason>`.

### F.2 Typing

- `mypy --strict` on `src/`. Tests are `--no-strict` (allow untyped helpers).
- Public functions: full annotations always. Internal helpers: annotate when non-obvious.
- Prefer `typing.Protocol` over ABC when possible (see `Store` in E.1).
- No `Any` in `src/` unless quarantined behind a stub file (`stubs/llama_index.pyi`) or `cast()` w/ comment.

### F.3 File & function size

- Module ≤ 300 LOC. Split when it grows past that. If you can't split cleanly, the split is wrong — look at the boundary again.
- Function ≤ 40 LOC. Longer usually means it's doing two things.
- One class per file when the class is >100 LOC. Multiple small classes per file OK.

### F.4 Errors

- Custom exceptions in `ragwatcher.errors` (`LockHeld`, `SchemaMismatch`, `ReadError`, `EmbedError`, `StoreError`).
- CLI catches these at the top level, maps to exit codes (A.10), prints to stderr, exits.
- No bare `except:`. No `except Exception:` w/o re-raise unless at a trust boundary (per-file reader, per-query handler).
- Never swallow → always log at WARN or re-raise as a project exception.
- Error messages: **what failed, what path, what to do**. Not "an error occurred".

### F.5 Logging

- `logging.getLogger("ragwatcher.<module>")` — never root.
- Levels: `DEBUG` (internals), `INFO` (lifecycle: sync start/end, query answered), `WARN` (recoverable), `ERROR` (data loss risk or crash).
- Structured payload via `extra={"file": ..., "duration_ms": ...}`. structlog picks it up in JSON mode; text mode formats via rich.
- Never log secrets (API keys, LLM endpoint auth).

### F.6 Tests

- Unit tests: no models, no network, no fixtures over 10KB. `<1s` per test.
- Integration tests: `@pytest.mark.slow`; run in nightly + on-demand only.
- Naming: `test_<module>.py`, function `test_<what>_<expected>`.
- No mocks for real components (embed, store) — spec §14. Fake at boundaries only (LLM endpoint in P4 → `httpx.MockTransport`).
- Every public function in `index.py`, `manifest.py`, `chunking.py`, `config.py` has ≥1 test.
- Ponytail rule: non-trivial logic ships w/ ONE runnable check. For this project = pytest test; no `demo()` blocks in `src/`.

### F.7 Comments

- Default: none.
- Write one only when the WHY is non-obvious: a hidden constraint, a workaround, a surprising invariant, an intentional simplification.
- Deliberate corners: `# ponytail: <ceiling>, <upgrade path>`. Harvested by `/ponytail-debt`.
- No `# TODO` w/o a name + date. Prefer issues.
- No `# NOTE:` walls of prose. If it needs paragraphs, it belongs in the module docstring.

### F.8 Config surface

- Read config **once** at `RagIndex.__init__`. Pass sub-objects (`config.embed`, `config.chunk`) into factories.
- No module reads env vars directly except `config.py` and the pre-import `FASTEMBED_CACHE_PATH` setter.
- Defaults live in the pydantic model, not scattered across modules.

### F.9 Dependencies

- Baseline install must stay CPU-only, no torch.
- New dep? Justify against the ladder (§SPEC.1 rules): stdlib? already installed? one-liner? Add to spec §3 table w/ a reason before importing.
- Pin lower bounds only (`>=X.Y`). No upper caps unless a known break.
- Extras: three (`pdf-pro`, `remote`, `dev`). Add `metrics` and `code` only if P5 backlog items land. Resist more.

### F.10 CLI ergonomics

- Every subcommand: `--help` text w/ 1-line summary + 1 example.
- Data commands (`query`, `stats`, `sources`, `index`) support `--json`. When set: exactly one JSON object to stdout, nothing else on stdout, logs to stderr.
- Errors → stderr always. Exit codes per A.10.
- No interactive prompts except destructive ops (`purge` w/o `--yes`).
- Idempotent: running the same command twice is safe.

### F.11 Git / commits

- Conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, `docs:`).
- Subject ≤ 50 chars.
- Body only when *why* isn't obvious.
- One logical change per commit. Phase 1's "build module X" ships as one commit per module (13 commits, small diffs, easy revert).

### F.12 Don'ts (repo-specific)

- No `print()` in `src/`.
- No `os._exit` outside a `finally:` block that already tried clean shutdown.
- No new dep for a value that fits in ≤10 stdlib lines.
- No interface with one implementation.
- No factory for one product.
- No config key for a value that never changes.
- No async in `src/` unless the underlying library forces it (FastMCP already async — fine to `await` there; do not spread it inward).
