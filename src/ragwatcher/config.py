"""Config layers: defaults → user TOML → per-dir TOML → env → CLI flags.

pydantic-settings for validation. TOML via stdlib `tomllib`.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal

import platformdirs
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedCfg(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    batch_size: int = 32
    dim: int | None = None


class RerankCfg(BaseModel):
    enabled: bool = True
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    top_k_input_multiplier: int = 4
    min_score: float | None = None


class SemanticCfg(BaseModel):
    breakpoint_percentile: int = 95
    buffer_size: int = 1


class ChunkCfg(BaseModel):
    strategy: Literal["recursive", "semantic", "late"] = "recursive"
    size: int = 512
    overlap: int = 64
    max_chunk_chars: int = 1500
    semantic: SemanticCfg = Field(default_factory=SemanticCfg)


class RetrieveCfg(BaseModel):
    top_k: int = 5
    top_k_max: int = 50
    hybrid: bool = True
    fusion_mode: str = "reciprocal_rerank"
    neighbor_window: int = 1


class StoreCfg(BaseModel):
    backend: Literal["lance", "simple", "qdrant"] = "lance"
    path: str = ".rag_index"


class WatchCfg(BaseModel):
    enabled: bool = True
    debounce_sec: float = 2.0
    rescan_interval_sec: int = 300


class ServerCfg(BaseModel):
    transport: Literal["stdio", "http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    query_timeout_sec: int = 30
    metrics: bool = False


class LogCfg(BaseModel):
    level: str = "info"
    format: Literal["text", "json"] = "text"


class FilesCfg(BaseModel):
    include_globs: list[str] = Field(default_factory=list)
    exclude_globs: list[str] = Field(
        default_factory=lambda: [".git/**", "node_modules/**", "**/__pycache__/**"]
    )
    max_file_bytes: int = 20_000_000
    follow_symlinks: bool = False


class ContextCfg(BaseModel):
    enabled: bool = False
    endpoint: str | None = None
    prompt: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGWATCHER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    embed: EmbedCfg = Field(default_factory=EmbedCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    chunk: ChunkCfg = Field(default_factory=ChunkCfg)
    retrieve: RetrieveCfg = Field(default_factory=RetrieveCfg)
    store: StoreCfg = Field(default_factory=StoreCfg)
    watch: WatchCfg = Field(default_factory=WatchCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)
    log: LogCfg = Field(default_factory=LogCfg)
    files: FilesCfg = Field(default_factory=FilesCfg)
    context: ContextCfg = Field(default_factory=ContextCfg)


def load(
    data_dir: Path | None = None, user_cfg: Path | None = None
) -> tuple[Settings, dict[str, str]]:
    """Return merged settings + per-key origin map.

    Order (low → high): defaults, user cfg, per-dir cfg, env (via BaseSettings), CLI overlays applied later.
    """
    origins: dict[str, str] = {}
    merged: dict[str, Any] = {}

    if user_cfg is None:
        user_cfg = Path(platformdirs.user_config_dir("ragwatcher")) / "config.toml"
    if user_cfg.exists():
        _deep_merge(merged, _load_toml(user_cfg), origins, str(user_cfg))

    if data_dir is not None:
        per = data_dir / ".ragwatcher.toml"
        if per.exists() and _trust_ok(per):
            _deep_merge(merged, _load_toml(per), origins, str(per))

    settings = Settings(**merged)
    return settings, origins


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(
    dst: dict[str, Any], src: dict[str, Any], origins: dict[str, str], source_label: str
) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v, origins, source_label)
        else:
            dst[k] = v
            origins[k] = source_label


def _trust_ok(path: Path) -> bool:
    """Skip per-dir cfg if owner != euid (Unix). Always trust on Windows."""
    if sys.platform.startswith("win"):
        return True
    try:
        st = path.stat()
        return st.st_uid == os.geteuid()
    except OSError:
        return False


def apply_cli_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """Overlay CLI flag values onto settings. Returns new Settings.

    `overrides` uses dotted keys, e.g. {"log.level": "debug"}.
    """
    if not overrides:
        return settings
    dump = settings.model_dump()
    for dotted, val in overrides.items():
        if val is None:
            continue
        parts = dotted.split(".")
        node = dump
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return Settings(**dump)
