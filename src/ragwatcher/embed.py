"""Embedding factory. FastEmbed default. Sets cache path pre-import."""

from __future__ import annotations

import os
from functools import lru_cache

import platformdirs

_CACHE_DIR = str(platformdirs.user_cache_dir("ragwatcher") + os.sep + "models")
os.environ.setdefault("FASTEMBED_CACHE_PATH", _CACHE_DIR)


def cache_dir() -> str:
    return _CACHE_DIR


@lru_cache(maxsize=4)
def _make(model: str, dim: int | None) -> object:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    kwargs: dict[str, object] = {"model_name": model, "cache_dir": _CACHE_DIR}
    return FastEmbedEmbedding(**kwargs)


def get_embedding(cfg: object) -> object:
    """Return an llama-index BaseEmbedding instance."""
    from ragwatcher.config import EmbedCfg

    assert isinstance(cfg, EmbedCfg), f"expected EmbedCfg, got {type(cfg)}"
    return _make(cfg.model, cfg.dim)
