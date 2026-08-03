"""Reranker. Lazy-load, cross-encoder via fastembed rerank if available."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ragwatcher.config import RerankCfg
from ragwatcher.logging import get_logger

log = get_logger("rerank")


class Reranker:
    def __init__(self, cfg: RerankCfg) -> None:
        self.cfg = cfg
        self._impl: Any = None

    def _ensure(self) -> None:
        if self._impl is not None or not self.cfg.enabled:
            return
        self._impl = _load_impl(self.cfg.model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        self._ensure()
        if self._impl is None:
            return [0.0] * len(passages)
        result: list[float] = self._impl(query, passages)
        return result


@lru_cache(maxsize=2)
def _load_impl(model: str) -> Any:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        enc = TextCrossEncoder(model_name=model)

        def _score(q: str, ps: list[str]) -> list[float]:
            return list(enc.rerank(q, ps))

        return _score
    except (ImportError, ValueError) as e:
        log.warning("rerank_load_failed", extra={"model": model, "err": str(e)})
        return None
