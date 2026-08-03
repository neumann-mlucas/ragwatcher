"""Manifest = on-disk state.

Schema v1 (new package, no legacy migration). Atomic writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragwatcher.errors import SchemaMismatch

CURRENT_SCHEMA = 1


@dataclass
class FileEntry:
    sha256: str
    mtime_ns: int
    size: int
    doc_ids: list[str] = field(default_factory=list)
    chunk_count: int = 0
    last_error: dict[str, str] | None = None
    last_error_at: str | None = None
    context_hashes: list[str] | None = None


@dataclass
class Manifest:
    schema_version: int
    created_at: str
    updated_at: str
    config_fingerprint: dict[str, Any]
    files: dict[str, FileEntry] = field(default_factory=dict)

    @classmethod
    def new(cls, fingerprint: dict[str, Any]) -> Manifest:
        now = _iso_now()
        return cls(
            schema_version=CURRENT_SCHEMA,
            created_at=now,
            updated_at=now,
            config_fingerprint=dict(fingerprint),
            files={},
        )

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_fingerprint": self.config_fingerprint,
            "files": {p: asdict(e) for p, e in self.files.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        raw = json.loads(text)
        v = int(raw.get("schema_version", 0))
        if v != CURRENT_SCHEMA:
            raw = migrate(v, CURRENT_SCHEMA, raw)
        files = {p: FileEntry(**e) for p, e in raw.get("files", {}).items()}
        return cls(
            schema_version=raw["schema_version"],
            created_at=raw["created_at"],
            updated_at=raw.get("updated_at", raw["created_at"]),
            config_fingerprint=raw.get("config_fingerprint", {}),
            files=files,
        )

    def bump_updated(self) -> None:
        self.updated_at = _iso_now()

    def check_fingerprint(self, current: dict[str, Any]) -> None:
        if _fp_bytes(self.config_fingerprint) != _fp_bytes(current):
            raise SchemaMismatch(
                "config fingerprint drift; run `ragwatcher index --full` or `purge`"
            )


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _fp_bytes(fp: dict[str, Any]) -> bytes:
    return json.dumps(fp, sort_keys=True).encode()


def load(path: Path) -> Manifest | None:
    if not path.exists():
        return None
    try:
        return Manifest.from_json(path.read_text())
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        bad = path.with_suffix(path.suffix + f".bad.{int(datetime.now().timestamp())}")
        path.rename(bad)
        return None


def save(path: Path, m: Manifest) -> None:
    """Atomic: tmp file + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m.bump_updated()
    text = m.to_json()
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def migrate(v_from: int, v_to: int, m: dict[str, Any]) -> dict[str, Any]:
    """No migrations yet (fresh start). Bump adds entries here."""
    raise SchemaMismatch(f"cannot migrate manifest v{v_from} → v{v_to}")


def sha256_file(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def build_fingerprint(cfg: Any) -> dict[str, Any]:
    """Extract canonical fingerprint fields per PLAN A.4.1.

    cfg is a pydantic Settings instance; access via attr.
    """
    ch = cfg.chunk
    semantic = ch.strategy == "semantic"
    ctx_enabled = getattr(getattr(cfg, "context", object()), "enabled", False)
    ctx_prompt = getattr(getattr(cfg, "context", object()), "prompt", None)
    return {
        "embed_model": cfg.embed.model,
        "embed_dim": cfg.embed.dim,
        "chunk_strategy": ch.strategy,
        "chunk_size": ch.size,
        "chunk_overlap": ch.overlap,
        "chunk_max_chars": ch.max_chunk_chars,
        "chunk_semantic_breakpoint_percentile": (
            ch.semantic.breakpoint_percentile if semantic else None
        ),
        "chunk_semantic_buffer_size": (ch.semantic.buffer_size if semantic else None),
        "store_backend": cfg.store.backend,
        "context_enabled": ctx_enabled,
        "context_prompt_sha256": (
            hashlib.sha256(ctx_prompt.encode()).hexdigest() if ctx_enabled and ctx_prompt else None
        ),
    }
