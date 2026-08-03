"""Typer CLI. Peer to MCP server, not a wrapper."""

from __future__ import annotations

import json as _json
import signal
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from ragwatcher import __version__
from ragwatcher.config import Settings, apply_cli_overrides, load
from ragwatcher.errors import LockHeld, SchemaMismatch
from ragwatcher.index import RagIndex
from ragwatcher.logging import get_logger, setup, stderr_console

app = typer.Typer(
    name="ragwatcher",
    help="Local RAG over a directory. Incremental. Served as MCP + CLI.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)

log = get_logger("cli")

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_LOCK = 3
EXIT_SCHEMA = 4

_state: dict[str, object] = {}


def _resolve_settings(data_dir: Path | None) -> Settings:
    cli_overrides = _state.get("overrides", {}) or {}
    if not isinstance(cli_overrides, dict):
        cli_overrides = {}
    user_cfg = _state.get("config_path")
    settings, _origins = load(
        data_dir=data_dir,
        user_cfg=user_cfg if isinstance(user_cfg, Path) else None,
    )
    return apply_cli_overrides(settings, cli_overrides)


@app.callback()
def _root(
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to user config.toml")
    ] = None,
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="debug|info|warn|error")
    ] = None,
    log_format: Annotated[str | None, typer.Option("--log-format", help="text|json")] = None,
) -> None:
    overrides: dict[str, object] = {}
    if log_level:
        overrides["log.level"] = log_level
    if log_format:
        overrides["log.format"] = log_format
    _state["overrides"] = overrides
    _state["config_path"] = config


def _setup_logging(settings: Settings) -> None:
    setup(level=settings.log.level, fmt=settings.log.format)


def _open_index(data_dir: Path) -> RagIndex:
    settings = _resolve_settings(data_dir)
    _setup_logging(settings)
    try:
        return RagIndex(data_dir, settings).__enter__()
    except LockHeld as e:
        typer.echo(f"lock held: {e}", err=True)
        raise typer.Exit(EXIT_LOCK) from None
    except SchemaMismatch as e:
        typer.echo(f"schema mismatch: {e}", err=True)
        raise typer.Exit(EXIT_SCHEMA) from None


# ---------------- serve ----------------


@app.command()
def serve(
    directory: Annotated[Path, typer.Argument(help="Data directory to index")],
    transport: Annotated[str, typer.Option("--transport", help="stdio|http")] = "stdio",
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    no_watch: Annotated[bool, typer.Option("--no-watch", help="Disable fs watcher")] = False,
    rescan_interval: Annotated[
        int, typer.Option("--rescan-interval", help="Periodic full rescan seconds")
    ] = 300,
) -> None:
    """Start the MCP server."""
    from ragwatcher.server import build_server
    from ragwatcher.watch import Watcher

    directory = directory.resolve()
    if not directory.is_dir():
        typer.echo(f"not a directory: {directory}", err=True)
        raise typer.Exit(EXIT_USAGE)

    idx = _open_index(directory)
    try:
        show_prog = transport != "stdio" and idx.config.log.format != "json"
        result = idx.sync(show_progress=show_prog)
        log.info(
            "initial_sync_done",
            extra={"plan": result.plan.as_summary(), "duration_ms": result.duration_ms},
        )

        watcher: Watcher | None = None
        if not no_watch and idx.config.watch.enabled:
            watcher = Watcher(
                data_dir=directory,
                sync=lambda: idx.sync(),
                debounce_sec=idx.config.watch.debounce_sec,
                rescan_interval_sec=rescan_interval,
            )
            watcher.start()

        srv = build_server(idx)

        def _shutdown(*_: object) -> None:
            log.info("shutdown_signal")
            if watcher:
                watcher.stop()
            idx.close()
            sys.exit(EXIT_OK)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        if transport == "stdio":
            srv.run()
        elif transport == "http":
            srv.run(transport="streamable-http", host=host, port=port)
        else:
            typer.echo(f"unknown transport: {transport}", err=True)
            raise typer.Exit(EXIT_USAGE)
    finally:
        idx.close()


# ---------------- query ----------------


@app.command()
def query(
    directory: Annotated[Path, typer.Argument()],
    question: Annotated[str, typer.Argument()],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    path_glob: Annotated[str | None, typer.Option("--path-glob")] = None,
    ext: Annotated[list[str] | None, typer.Option("--ext")] = None,
    recency_days: Annotated[int | None, typer.Option("--recency-days")] = None,
    min_score: Annotated[float | None, typer.Option("--min-score")] = None,
    show_neighbors: Annotated[bool, typer.Option("--show-neighbors/--no-show-neighbors")] = True,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    fmt: Annotated[str, typer.Option("--format", help="table|snippets")] = "snippets",
    json_out: Annotated[bool, typer.Option("--json", help="JSON to stdout")] = False,
) -> None:
    """One-shot query."""
    idx = _open_index(directory.resolve())
    try:
        result = idx.retrieve(
            question=question,
            top_k=top_k,
            path_glob=path_glob,
            ext=ext,
            recency_days=recency_days,
            min_score=min_score,
            neighbor_window=idx.config.retrieve.neighbor_window if show_neighbors else 0,
            debug=debug,
        )
        if json_out:
            payload: dict[str, object] = {
                "query": result.query,
                "returned": [
                    {
                        "rank": h.rank,
                        "doc_id": h.doc_id,
                        "source": h.source,
                        "snippet": h.snippet,
                        "score": h.score,
                        **(
                            {
                                "scores": h.scores,
                                "stage_survived": h.stage_survived,
                                "neighbors": h.neighbors,
                            }
                            if debug
                            else {}
                        ),
                    }
                    for h in result.returned
                ],
            }
            if debug:
                payload["dropped"] = result.dropped
                payload["stages"] = result.stages
                payload["timings_ms"] = result.timings_ms
            typer.echo(_json.dumps(payload, indent=2))
        else:
            _render_hits(result, fmt=fmt, debug=debug)
    finally:
        idx.close()


def _render_hits(result, fmt: str, debug: bool) -> None:  # type: ignore[no-untyped-def]
    console = stderr_console()
    if not result.returned:
        typer.echo("no results")
        return
    if fmt == "table":
        t = Table(title=f"Results for {result.query!r}")
        t.add_column("#")
        t.add_column("Score")
        t.add_column("Source")
        t.add_column("Snippet", overflow="fold")
        for h in result.returned:
            t.add_row(str(h.rank), f"{h.score:.3f}", h.source, h.snippet[:200])
        console.print(t)
    else:
        for h in result.returned:
            typer.echo(f"[{h.rank}] {h.source}  score={h.score:.3f}")
            typer.echo(h.snippet.strip())
            typer.echo("")
    if debug:
        console.print(Panel.fit(str(result.stages), title="stages", border_style="dim"))
        console.print(Panel.fit(str(result.timings_ms), title="timings (ms)", border_style="dim"))


# ---------------- index ----------------


@app.command("index")
def index_cmd(
    directory: Annotated[Path, typer.Argument()],
    full: Annotated[bool, typer.Option("--full")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
    watch: Annotated[bool, typer.Option("--watch", help="Stream events until Ctrl-C")] = False,
) -> None:
    """Incremental sync. Exits after unless --watch."""
    from ragwatcher.watch import Watcher

    idx = _open_index(directory.resolve())
    try:
        show_prog = not json_out and idx.config.log.format != "json"
        result = idx.sync(full=full, dry_run=dry_run, show_progress=show_prog)
        plan_summary = result.plan.as_summary()
        summary: dict[str, object] = {
            "plan": plan_summary,
            "errors": len(result.errors),
            "duration_ms": result.duration_ms,
        }
        if json_out:
            typer.echo(_json.dumps(summary, indent=2))
        else:
            typer.echo(
                f"add={plan_summary['add']} update={plan_summary['update']} "
                f"delete={plan_summary['delete']} errors={len(result.errors)} "
                f"took={result.duration_ms}ms"
            )
        if watch and not dry_run:
            watcher = Watcher(
                data_dir=directory.resolve(),
                sync=lambda: idx.sync(),
                debounce_sec=idx.config.watch.debounce_sec,
                rescan_interval_sec=idx.config.watch.rescan_interval_sec,
            )
            watcher.start()
            try:
                signal.pause()
            except KeyboardInterrupt:
                watcher.stop()
    finally:
        idx.close()


# ---------------- stats ----------------


@app.command()
def stats(
    directory: Annotated[Path, typer.Argument()],
    watch: Annotated[bool, typer.Option("--watch")] = False,
    interval: Annotated[float, typer.Option("--interval")] = 2.0,
    errors: Annotated[bool, typer.Option("--errors")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Index stats."""
    idx = _open_index(directory.resolve())
    try:
        if watch:
            import time as _t

            from rich.live import Live

            with Live(
                _render_stats(idx.stats(include_errors=errors)), refresh_per_second=1
            ) as live:
                while True:
                    _t.sleep(interval)
                    live.update(_render_stats(idx.stats(include_errors=errors)))
        else:
            s = idx.stats(include_errors=errors)
            if json_out:
                typer.echo(_json.dumps(s, indent=2))
            else:
                stderr_console().print(_render_stats(s))
    except KeyboardInterrupt:
        return
    finally:
        idx.close()


def _render_stats(s: dict) -> Table:  # type: ignore[type-arg]
    t = Table(title="ragwatcher stats", show_header=False, box=None)
    for k, v in s.items():
        display = f"{len(v)} items" if isinstance(v, list) else str(v)
        t.add_row(str(k), display)
    return t


# ---------------- sources ----------------


@app.command()
def sources(
    directory: Annotated[Path, typer.Argument()],
    errors: Annotated[bool, typer.Option("--errors")] = False,
    stale: Annotated[bool, typer.Option("--stale")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List indexed sources."""
    idx = _open_index(directory.resolve())
    try:
        items = idx.sources(errors_only=errors, stale_only=stale)
        if json_out:
            typer.echo(_json.dumps(items, indent=2))
        else:
            for it in items:
                typer.echo(it["path"])
    finally:
        idx.close()


# ---------------- purge ----------------


@app.command()
def purge(
    directory: Annotated[Path, typer.Argument()],
    yes: Annotated[bool, typer.Option("--yes", help="Confirm delete")] = False,
) -> None:
    """Delete .rag_index/. Requires --yes."""
    import shutil

    settings = _resolve_settings(directory.resolve())
    _setup_logging(settings)
    persist = directory.resolve() / settings.store.path
    if not persist.exists():
        typer.echo(f"nothing to purge: {persist}")
        return
    if not yes:
        typer.echo(f"would delete {persist} (pass --yes to confirm)", err=True)
        raise typer.Exit(EXIT_USAGE)
    shutil.rmtree(persist)
    typer.echo(f"purged {persist}")


# ---------------- doctor ----------------


@app.command()
def doctor(
    directory: Annotated[Path | None, typer.Argument()] = None,
) -> None:
    """Diagnostics."""
    from ragwatcher import embed

    settings = _resolve_settings(directory.resolve() if directory else None)
    _setup_logging(settings)
    console = stderr_console()
    t = Table(title="ragwatcher doctor", show_header=False, box=None)
    t.add_row("version", __version__)
    t.add_row("embed_model", settings.embed.model)
    t.add_row("store.backend", settings.store.backend)
    t.add_row("cache_dir", embed.cache_dir())
    model_dir = Path(embed.cache_dir())
    t.add_row("cache_present", "yes" if model_dir.exists() else "no")

    if directory:
        d = directory.resolve()
        t.add_row("data_dir", str(d))
        persist = d / settings.store.path
        t.add_row("persist_dir", str(persist))
        m_path = persist / "manifest.json"
        t.add_row("manifest", "present" if m_path.exists() else "absent")
        if m_path.exists():
            from ragwatcher.manifest import build_fingerprint
            from ragwatcher.manifest import load as _mload

            m = _mload(m_path)
            if m:
                current_fp = build_fingerprint(settings)
                match = m.config_fingerprint == current_fp
                t.add_row("fingerprint", "match" if match else "DRIFT")
    console.print(t)


# ---------------- config ----------------


@app.command("config")
def config_cmd(
    directory: Annotated[Path | None, typer.Argument()] = None,
    show_sources: Annotated[bool, typer.Option("--show-sources")] = False,
) -> None:
    """Show effective config."""
    settings, origins = load(data_dir=directory.resolve() if directory else None)
    settings = apply_cli_overrides(settings, _state.get("overrides", {}))  # type: ignore[arg-type]
    dump = settings.model_dump()
    if show_sources:
        typer.echo("# origins:")
        for k, src in origins.items():
            typer.echo(f"#   {k} <- {src}")
    typer.echo(_json.dumps(dump, indent=2))


# ---------------- version ----------------


@app.command()
def version() -> None:
    """Print version and exit."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
