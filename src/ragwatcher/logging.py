"""Logging setup — stderr only. stdout is reserved for CLI data + MCP frames."""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog
from rich.console import Console
from rich.logging import RichHandler

_console: Console | None = None


def stderr_console() -> Console:
    """Singleton rich console bound to stderr."""
    global _console
    if _console is None:
        _console = Console(stderr=True, highlight=False)
    return _console


def setup(level: str = "info", fmt: Literal["text", "json"] = "text") -> None:
    """Configure stdlib + structlog. Always writes to stderr."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handler: logging.Handler
    if fmt == "json":
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = RichHandler(
            console=stderr_console(),
            show_time=True,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(log_level)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
            if fmt == "json"
            else structlog.dev.ConsoleRenderer(colors=False),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "event": record.getMessage(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        for k, v in record.__dict__.items():
            if k in _STD_LOG_RECORD_KEYS:
                continue
            payload[k] = v
        return json.dumps(payload, default=str)


_STD_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ragwatcher.{name}")
