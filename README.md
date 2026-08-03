# ragwatcher

Local RAG over a directory. Incremental. Served as MCP tool + CLI.

- Zero-config default (drops into `~/notes`, works).
- Offline, CPU-only baseline (FastEmbed ONNX). GPU / API models opt-in.
- CLI is a peer to the MCP server, not a wrapper.

## Install

```
uv tool install ragwatcher
# or
pipx install ragwatcher
```

Optional extras: `[lance]` (LanceDB store), `[pdf-pro]` (pymupdf), `[remote]` (qdrant/openai/cohere).

## Usage

```
ragwatcher serve ~/notes                  # MCP server (stdio)
ragwatcher index ~/notes                  # one-shot sync
ragwatcher query ~/notes "what is X"      # one-shot query
ragwatcher stats ~/notes                  # index summary
ragwatcher doctor ~/notes                 # health check
ragwatcher purge ~/notes --yes            # wipe .rag_index/
```

Every data-producing command supports `--json` for scripting.

## Configuration

Precedence (low → high): defaults → `$XDG_CONFIG_HOME/ragwatcher/config.toml` → `<DIR>/.ragwatcher.toml` → `RAGWATCHER_*` env vars → CLI flags.

See `SPEC.md` § 5 for the full schema.

## Development

```
uv sync --extra dev
uv run pytest -q --ignore=tests/integration
uv run ruff check src tests
uv run mypy src
```

Integration tests (real embed model, ~3s):

```
uv run pytest tests/integration -m slow
```

## License

MIT
