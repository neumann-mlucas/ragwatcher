# ragwatcher — Spec (proper package)

Status: draft
Target: promote single-file `~/.local/bin/ragwatcher` to a maintainable, distributable Python package with first-class CLI and better RAG quality.

Guiding rules:
- Lazy senior dev. Ladder: does it need to exist → stdlib → already-installed dep → new dep. Don't add abstractions for one implementation.
- Every default must work offline on CPU. GPU / API options are opt-in.
- Correctness > cleverness. Boring, testable, deletable.
- Local-first: the MCP client is the primary UI. CLI is the peer. No TUI.

---

## 1. Goals / non-goals

**Goals**
1. Robust local RAG over a directory, incremental, no manual reindex.
2. Serve as MCP tool (stdio + HTTP) *and* be a usable standalone CLI (query, index, stats, purge, serve). CLI is a peer, not a wrapper.
3. Retrieval quality that holds up on mixed corpora (code, notes, PDFs, EPUBs).
4. Retrieval forensics — user can inspect *why* a chunk ranked where it did.
5. Zero-config default (drops into `~/notes`, works). Full config for power users.
6. Single-binary install path (`uv tool install ragwatcher` or `pipx install`).

**Non-goals**
- Multi-tenant / auth / RBAC. Localhost tool.
- Distributed / sharded index.
- Generation (no LLM in the loop — retrieval only, MCP client generates).
- Web UI or TUI. MCP client + CLI covers all interaction.

---

## 2. Package layout (src)

```
ragwatcher/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/ragwatcher/
│   ├── __init__.py          # __version__ = importlib.metadata.version(...)
│   ├── __main__.py          # python -m ragwatcher
│   ├── cli.py               # typer app, subcommands
│   ├── config.py            # pydantic-settings, TOML + env
│   ├── index.py             # RagIndex (state, sync, retrieve)
│   ├── store.py             # vector-store abstraction (LanceDB default)
│   ├── manifest.py          # FileEntry, load/save, versioned schema
│   ├── readers.py           # per-extension loaders + text cleanup
│   ├── chunking.py          # splitters (recursive, semantic, late)
│   ├── embed.py             # embedding backend selector
│   ├── rerank.py            # reranker backend selector
│   ├── watch.py             # watchdog handler + periodic rescan
│   ├── server.py            # FastMCP tools
│   ├── logging.py           # structured (JSON) + human formatters
│   └── py.typed
├── tests/
│   ├── conftest.py
│   ├── test_manifest.py
│   ├── test_plan.py         # pure diff
│   ├── test_readers.py
│   ├── test_chunking.py
│   ├── test_index_int.py    # integration (real fs, small model)
│   └── data/                # tiny corpus fixtures
```

Rationale for `src/` layout: prevents accidental imports of local source instead of installed package (see modern packaging guidance).

---

## 3. Dependencies (justified)

| Package | Purpose | Why |
|---|---|---|
| `llama-index-core` | RAG plumbing | Stays. Best RAG-focused framework; already works. |
| `llama-index-vector-stores-lance` | Vector store | Replace in-memory `SimpleVectorStore` with LanceDB — disk-native, scales past RAM, columnar. |
| `llama-index-embeddings-fastembed` | Embeddings (default) | CPU-friendly, quantized ONNX. |
| `llama-index-readers-file` | Readers | Existing coverage. |
| `llama-index-retrievers-bm25` | Sparse retrieval | Existing. |
| `fastembed` | Embed + rerank runtimes | ONNX, no torch on CPU path. |
| `fastmcp` | MCP transport | Existing. |
| `watchdog` | File events | Existing. |
| `pypdf`, `python-docx`, `EbookLib`, `html2text` | File formats | Existing. |
| `typer` | CLI | Type-hint driven, mature, click-based. (Alt: `cyclopts` — shorter, richer types, but younger. Stick with typer as default.) |
| `pydantic-settings` | Config | TOML + env, validated. Stdlib `tomllib` for the parse. |
| `rich` | Tables, logs, progress, `--watch` refresh | Already pulled transitively by typer. Covers every interactive-output need without a framework. |
| `structlog` | JSON logs | Opt-in via `--log-format=json`; falls back to stdlib. |
| `platformdirs` | XDG paths | Config + cache locations, cross-platform. |
| `filelock` | Per-dir lock | Cross-platform `fcntl`/`msvcrt`. Prevents two instances on same dir. |

**Deferred / consider later**
- `sentence-transformers` — only if user picks a torch-only model.
- `pymupdf` — better PDF than `pypdf`, but AGPL; leave opt-in extra `[pdf-pro]`.
- `unstructured` — heavy, brings 20+ deps. Skip unless a user asks.
- `chromadb` / `qdrant-client` — LanceDB covers embedded case; alt stores behind config, not default.
- `openai`/`cohere` — API reranker/embed only if user opts in via config.
- `textual` — TUI dropped. MCP client is the UI; CLI + rich covers the rest. Reconsider only if a concrete workflow is proven CLI-hostile.

**Extras** (pyproject `[project.optional-dependencies]`):
- `pdf-pro` → pymupdf
- `remote` → qdrant-client, openai, cohere
- `dev` → pytest, pytest-asyncio, ruff, mypy, pytest-cov

Baseline install stays small (CPU, no torch, no API SDKs).

---

## 4. CLI surface

`typer` app, `ragwatcher` entry point.

```
ragwatcher [--config PATH] [--log-level LEVEL] [--log-format text|json] <command>

Commands:
  serve <DIR>                  Start MCP server (stdio or HTTP).
    --transport stdio|http     (default: stdio)
    --host HOST                (default: 127.0.0.1)
    --port PORT
    --no-watch                 Disable filesystem watcher.
    --rescan-interval SEC      Periodic full rescan (default: 300).

  query <DIR> "QUESTION"       One-shot query, print results, exit.
    --top-k N                  (default: 5, max: 50)
    --path-glob GLOB
    --ext EXT                  Filter by extension (repeatable).
    --recency-days N           Only files modified in last N days.
    --min-score FLOAT          Drop reranker scores below threshold.
    --show-neighbors / --no    Prev/next chunks (default on).
    --debug                    Print per-hit dense/bm25/rerank scores,
                               filtered candidates, retrieval stages.
    --json                     JSON output for scripting.
    --format table|snippets    Rich rendering (default: snippets).

  index <DIR>                  One-shot incremental sync, exit.
    --full                     Ignore mtime fast-path, rehash everything.
    --dry-run                  Print plan (add/update/delete counts), no changes.
    --json                     Machine-readable summary.
    --watch                    Stay attached, print sync events until Ctrl-C
                               (useful for debugging without full serve).

  stats <DIR>                  Print index stats.
    --watch [--interval SEC]   Live refresh with rich.Live (default 2s).
    --errors                   Include per-file error rows (last sync).
    --json

  sources <DIR>                List indexed file paths.
    --errors                   Only files that failed to index.
    --stale                    Only files whose config fingerprint drifted.
    --json

  purge <DIR>                  Delete .rag_index/ (confirm required).
    --yes

  doctor [<DIR>]               Diagnostics: model cache, disk space, config,
                               manifest schema version, index integrity,
                               config-fingerprint drift, watcher backend.

  config                       Show effective config (merged sources).
    --show-sources             Annotate each key with origin.

  version
```

**Global ergonomics**
- Exit codes: 0 ok, 1 runtime error, 2 usage, 3 lock held, 4 index schema mismatch.
- `--json` on data-producing commands emits one JSON object to stdout, logs to stderr.
- Shell completion: `ragwatcher --install-completion {bash,zsh,fish}`.
- `RAGWATCHER_*` env vars override config keys (dotted → underscored).
- `NO_COLOR` respected.
- Composable with Unix: `ragwatcher query ... --json | jq`, `ragwatcher query ... --json | fzf --preview 'jq -r .snippet <<< {}'`.

---

## 5. Configuration

**Precedence** (low → high): built-in defaults → user config (`$XDG_CONFIG_HOME/ragwatcher/config.toml`) → per-dir config (`<DIR>/.ragwatcher.toml`) → env → CLI flags.

**Schema** (pydantic-settings):

```toml
[embed]
model = "BAAI/bge-small-en-v1.5"      # or "BAAI/bge-m3" for multilingual, "nomic-embed-text-v1.5"
device = "cpu"                        # cpu|cuda|auto
batch_size = 32
dim = null                            # optional matryoshka truncation (nomic, mxbai)

[rerank]
enabled = true
model = "BAAI/bge-reranker-v2-m3"     # upgrade from Xenova/ms-marco-MiniLM-L-6-v2
top_k_input_multiplier = 4            # fetch top_k * N, rerank down
min_score = null

[chunk]
strategy = "recursive"                # recursive|semantic|late
size = 512
overlap = 64
max_chunk_chars = 1500

[chunk.semantic]                      # if strategy = "semantic"
breakpoint_percentile = 95
buffer_size = 1

[retrieve]
top_k = 5
top_k_max = 50
hybrid = true                         # dense + bm25 fusion
fusion_mode = "reciprocal_rerank"
neighbor_window = 1

[store]
backend = "lance"                     # lance|simple|qdrant
path = ".rag_index"                   # relative to data dir

[watch]
enabled = true
debounce_sec = 2.0
rescan_interval_sec = 300             # periodic full sweep for NFS/FUSE

[server]
transport = "stdio"
host = "127.0.0.1"
port = 8000

[log]
level = "info"
format = "text"                       # text|json

[files]
include_globs = []                    # if empty, all supported exts
exclude_globs = [".git/**", "node_modules/**", "**/__pycache__/**"]
max_file_bytes = 20_000_000           # skip huge files
follow_symlinks = false
```

`doctor` validates the merged config and warns on impossible combos (e.g. rerank enabled but model not downloadable).

---

## 6. RAG pipeline (quality)

### 6.1 Readers
- Keep current set. Add:
  - Markdown/HTML → strip boilerplate, preserve heading path in metadata (`heading_path: "H1 > H2"`).
  - Source code (`.py`, `.js`, `.ts`, `.go`, `.rs`) → language-aware splitter (tree-sitter optional, extra `[code]`; fallback recursive on whitespace).
  - CSV/JSON → row/record-aware chunking with header prefix.
- Per-file metadata always populated: `file_path`, `file_name`, `ext`, `mtime`, `size`, `sha256`, `heading_path`.

### 6.2 Chunking
- **Default**: recursive character splitter, 512 tokens, 64 overlap. Best precision/recall tradeoff per 2026 benchmarks.
- **Opt-in `semantic`**: sentence embeddings + percentile breakpoints. Costs embedding every sentence; recall +~9%.
- **Opt-in `late`**: embed full doc with long-context model, then chunk token embeddings. Requires long-context embed model (BGE-M3). Preserves cross-chunk semantics.
- Chunks always carry filename + heading prefix so filename-mentioning queries hit.

### 6.3 Embeddings
- **Default**: `BAAI/bge-small-en-v1.5` via fastembed (CPU, ~130MB). Current.
- **English quality**: `BAAI/bge-large-en-v1.5`.
- **Multilingual / long-context / hybrid**: `BAAI/bge-m3` (dense + sparse + multi-vec in one model — pairs well with sparse retrieval).
- **Small footprint**: `nomic-embed-text-v1.5` (matryoshka, truncate to 256 dims to shrink index).
- Model change bumps `manifest.schema` → `doctor` warns → user runs `index --full` or `purge`.

### 6.4 Retrieval
- Hybrid: dense (vector store) + BM25 (docstore-backed).
- Fusion: reciprocal-rank fusion (`QueryFusionRetriever`).
- Optional metadata filter via `path_glob` (fnmatch), extension filter, `mtime` recency filter.
- Fetch `top_k * multiplier` candidates → rerank → truncate.

### 6.5 Reranking
- **Default upgrade**: `BAAI/bge-reranker-v2-m3` (Apache 2.0, multilingual, matches Cohere quality at $0). Replaces `Xenova/ms-marco-MiniLM-L-6-v2`.
- CPU fallback: keep MiniLM as a `--rerank-model` option for weak hardware.
- API path (opt-in): `cohere` / `zerank-2` if user provides key.
- Reranker lazy-loaded on first query. Preload flag: `--preload-rerank`.

### 6.6 Contextual retrieval (opt-in, phase 0.4)
- Prepend LLM-generated 1-2 sentence context to each chunk before embedding (Anthropic 2024 technique). Requires user-provided LLM endpoint. Big recall gain, one-time cost per file. Cache generated context in manifest keyed by chunk hash so re-embed on model swap doesn't re-call the LLM.

### 6.7 Neighbor expansion
- Keep `PrevNextNodePostprocessor`, window configurable.

### 6.8 Query-time knobs (exposed)
- `top_k`, `path_glob`, `ext`, `recency_days`, `min_score`, `debug`.

### 6.9 Retrieval forensics (`query --debug`)
Replaces the debugging role a TUI would have played.
Prints, per hit:
- rank, doc_id, source
- dense score, BM25 score, RRF score, rerank score
- reason it survived filters (or was dropped)
- neighbor doc_ids included

Plus a summary block: candidate counts at each stage (dense → BM25 → fused → filtered → reranked → returned) and the count filtered by each stage. Makes quality regressions inspectable without a UI.

---

## 7. Vector store

Abstract behind `store.py` with three backends:
- `lance` (default) — LanceDB, on-disk, columnar, mmap. Handles > RAM corpora. Cheap incremental writes.
- `simple` — current llama-index in-memory, JSON persist. Kept for < 10k chunks; zero-dep smoke tests.
- `qdrant` — remote/self-hosted for multi-tenant / larger scale. Behind `[remote]` extra.

Store choice recorded in manifest; `doctor` refuses to load mismatched store.

---

## 8. Manifest & index schema

`manifest.json` becomes:

```json
{
  "schema_version": 2,
  "created_at": "...",
  "config_fingerprint": {
    "embed_model": "...",
    "chunk_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "store_backend": "lance"
  },
  "files": {
    "<abs_path>": {
      "sha256": "...",
      "mtime_ns": 0,
      "size": 0,
      "doc_ids": ["..."],
      "chunk_count": 0,
      "last_error": null,
      "last_error_at": null,
      "context_hashes": ["..."]   // only if contextual retrieval enabled
    }
  }
}
```

- Fingerprint mismatch on load → surface via `doctor`, block `serve` unless `--force`.
- Manifest write is atomic (tmp + rename), plus a `.bak` rotation on schema bump.
- Migrations live in `manifest.py::migrate(v_from, v_to, m)`.
- Per-file `last_error` powers `sources --errors` and `stats --errors`.

---

## 9. Filesystem watcher

- Keep debounced `watchdog` handler.
- Add **periodic full rescan** (default 300s) — belt-and-braces for NFS/SMB/FUSE where inotify misses events.
- Handle rename/move as delete+add via `dest_path`.
- Skip `.rag_index/`, hidden dirs, files > `max_file_bytes`, files matching `exclude_globs`.
- Batch: drain events during debounce window, one sync per batch.

---

## 10. MCP server

Keep FastMCP. Tools:
- `query(question, top_k, path_glob, min_score, ext)` → formatted results.
- `query_json(...)` → structured JSON payload (for programmatic clients).
- `reindex(full: bool = false)` → sync summary.
- `list_sources(glob=None)` → list.
- `stats()` → summary.
- `get_chunk(doc_id)` → return raw chunk + neighbors, for follow-up drill-in.

HTTP mode: add `GET /health` (`{ok, files, last_sync_at, index_schema}`) and `GET /metrics` (opt-in Prometheus text if user enables).

---

## 11. Interactive output (in place of a TUI)

Local-first + MCP client as primary UI ⇒ no TUI. Every workflow a TUI would enable is covered by CLI + `rich`:

| Workflow | CLI equivalent |
|---|---|
| See scores + snippets nicely | `query` uses `rich.Table` / `rich.Panel`, colorized |
| Watch initial embed | `rich.progress` bar on `index` |
| See which files failed | `stats --errors`, `sources --errors` |
| Debug retrieval quality | `query --debug` (stage-by-stage scores) |
| Live sync monitor | `stats --watch` (`rich.Live` refresh) |
| Interactive filter / preview | `query --json \| fzf --preview 'jq -r .snippet <<< {}'` |
| Jump to source | print `path:line`; MCP client or `$EDITOR +line file` opens it |

Composability with jq/fzf/`$EDITOR` beats a bespoke UI because it works over SSH, inside tmux, inside CI logs, and inside another agent's tool output. Reconsider TUI only if a concrete workflow is proven CLI-hostile.

---

## 12. Logging & observability

- Default: rich-formatted text to stderr.
- `--log-format json` → structlog JSON to stderr (fields: `event`, `ts`, `level`, `dir`, `op`, `file`, `duration_ms`).
- Per-op timing logged at INFO: sync duration, per-file embed time, query latency, rerank latency.
- HTTP `--metrics` flag exposes `/metrics` Prometheus text (query count, latency histogram, sync outcomes, index size).
- No OpenTelemetry until someone asks — YAGNI.

---

## 13. Robustness

- **Locking**: `filelock` on `<persist>/lock`. Refuse startup if held; exit code 3.
- **Signals**: SIGTERM/SIGINT → drain in-flight sync, persist manifest, stop watcher, exit 0. Keep documented `os._exit` only as last resort in `finally`.
- **Progress**: `rich.progress` bar during initial embed and full reindex (goes to stderr; suppressed in `--json` and stdio-MCP modes).
- **Backpressure**: cap in-flight `insert()` batch size; stream file reads.
- **Corruption**: manifest unreadable → move to `manifest.bad.<ts>` and re-scan (log at WARN, not silent).
- **Model download**: pre-flight check in `doctor`; log clear message if fastembed cache miss triggers download on `serve` start.
- **Query timeout**: configurable, default 30s; return partial results with warning field.
- **Error taxonomy**: `ReadError`, `EmbedError`, `StoreError`, `LockHeld`, `SchemaMismatch` — mapped to exit codes for CLI, error fields for JSON output.
- **Per-file error persistence**: read/embed failures recorded in manifest (`last_error`, `last_error_at`) so `stats --errors` shows chronic offenders across restarts.

---

## 14. Testing

- `pytest`, no framework layers.
- **Unit** (fast, no models):
  - `test_manifest.py`: load/save/migrate/corrupt.
  - `test_plan.py`: `_plan_sync` diff — add / update / delete / rename / mtime-only bump.
  - `test_chunking.py`: splitter boundaries, filename prefix, heading metadata.
  - `test_readers.py`: each format → non-empty text.
  - `test_config.py`: precedence chain, env override, per-dir override.
- **Integration** (`@pytest.mark.slow`, real embed model, small):
  - `test_index_int.py`: tmpdir corpus → sync → query → assert relevant file in top-3.
  - Manifest round-trip after restart.
  - Watcher: touch file → poll manifest updates within N seconds.
  - `query --debug` output shape stable.
- **No mocks for embed/store** — mocked retrieval is useless. Use `bge-small` on tiny fixtures (< 5s per run).
- CI: GitHub Actions matrix on py3.11/3.12/3.13, `ruff`, `mypy --strict`, unit tests always, integration nightly.

---

## 15. Distribution

- `uv build` → wheel + sdist.
- `pyproject.toml` with `[project.scripts] ragwatcher = "ragwatcher.cli:app"`.
- Publish to PyPI. Install paths:
  - `uv tool install ragwatcher` (recommended)
  - `pipx install ragwatcher`
- Reproducible builds via `uv.lock`.
- Version from `importlib.metadata`, not string constant.

---

## 16. Build order (fresh package, no legacy carryover)

No compatibility with the legacy `~/.local/bin/ragwatcher` shebang or its manifest. Users of the old script run `purge && index --full` once against the new package. See PLAN A.2.2 for the decision.

1. `pyproject.toml`, `src/` layout, entry point, ruff/mypy configs.
2. Build core modules (errors → logging → manifest → readers → chunking → embed → rerank → store → watch → index → server → cli → `__main__`). See PLAN Phase 1.
3. Add `query`, `index`, `stats`, `sources`, `purge`, `doctor`, `config`, `version` subcommands. See PLAN Phase 2.
4. Manifest ships as v1 (new package). Fields: `schema_version`, `config_fingerprint`, per-file `last_error`. Fingerprint mismatch → `SchemaMismatch` (exit 4).
5. Reranker default stays `Xenova/ms-marco-MiniLM-L-6-v2` (light, no torch). `bge-reranker-v2-m3` opt-in via `rerank.model`.
6. Default vector store swaps to LanceDB in P3. `SimpleStore` remains as `store.backend = "simple"` fallback.
7. Add lock file + periodic rescan + progress bar + `--debug` retrieval forensics.
8. Add tests. Publish 0.2.0.
9. Contextual retrieval as 0.4.

---

## 17. Roadmap (phased)

**0.2 — Package & CLI (1-2 days)**
Split into package, pyproject, typer CLI, config file, lock, progress bar, log level, JSON output, `doctor`, `--debug`, `stats --watch`, unit tests, PyPI.

**0.3 — RAG quality (2-3 days)**
BGE-reranker-v2-m3 default, LanceDB store, per-format readers with heading metadata, code splitter, better logging, integration tests, periodic rescan, per-file error persistence.

**0.4 — Advanced retrieval (opt-in, 2-3 days)**
Semantic + late chunking behind config, BGE-M3 for multilingual, contextual retrieval (with user-supplied LLM), metadata filters (`ext`, `recency_days`), min-score threshold.

**0.5+ — as-needed**
Prometheus metrics, remote vector stores (qdrant), API embed/rerank backends, tree-sitter code splitter.

Ship 0.2 first. Everything after is validated against real use.

---

## 18. Open questions

1. Keep MCP as the primary interface or make CLI query a peer? (Decided: peer. CLI-first, MCP is one transport.)
2. Include an LLM answer synthesis mode as opt-in? (Recommend: no. Non-goal; MCP client does that.)
3. Support multiple directories per instance? (Recommend: no. Run multiple processes; keeps lock model trivial.)
4. Ship as `ragwatcher` or rename now that CLI is first-class? (Recommend: keep name.)

---

## Sources

- [Python Packaging in 2026 — uv, Poetry, and the modern ecosystem](https://andrewodendaal.com/python-packaging-2026-uv-poetry-modern-ecosystem/)
- [Python Packaging Best Practices: setuptools, Poetry, and Hatch in 2026](https://dasroot.net/posts/2026/01/python-packaging-best-practices-setuptools-poetry-hatch/)
- [Building a Python Library in 2026](https://stephenlf.dev/blog/python-library-in-2026/)
- [Cyclopts vs Typer comparison](https://cyclopts.readthedocs.io/en/latest/vs_typer/README.html)
- [Typer alternatives](https://typer.tiangolo.com/alternatives/)
- [Best Embedding Models for RAG (2026) — MTEB, cost, self-hosting](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/)
- [Best Embedding Models in 2026 — Mixpeek](https://mixpeek.com/curated-lists/best-embedding-models)
- [RAG Chunking Strategies 2026 — 8 methods compared](https://denser.ai/blog/rag-chunking-strategies/)
- [Best Chunking Strategies for RAG in 2026 — Firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Best Rerankers for RAG in 2026 — 7 models compared](https://futureagi.com/blog/best-rerankers-for-rag-2026/)
- [Best Reranker Models for RAG: Open-Source vs API (2026)](https://docs.bswen.com/blog/2026-02-25-best-reranker-models/)
- [LangChain vs LlamaIndex vs Haystack for RAG 2026](https://gigagpu.com/langchain-vs-llamaindex-vs-haystack-2026/)
- [Haystack vs LangChain vs LlamaIndex for Production RAG 2026](https://www.icertglobal.com/community/haystack-vs-langchain-vs-llamaindex-for-production-rag-2026)
- [Vector Database Comparison 2026: Chroma vs Qdrant vs pgvector vs LanceDB](https://4xxi.com/articles/vector-database-comparison/)
- [Best Vector Databases in 2026 — Encore](https://encore.dev/articles/best-vector-databases)
