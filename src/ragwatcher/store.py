"""Vector store abstraction. P1 ships SimpleStore. LanceStore in P3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ragwatcher.config import StoreCfg
from ragwatcher.errors import StoreError

if TYPE_CHECKING:
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.vector_stores.types import BasePydanticVectorStore


@dataclass
class StoreStats:
    backend: str
    node_count: int
    persist_path: str


class Store(Protocol):
    backend: str

    def storage_context(self) -> StorageContext: ...
    def vector_store(self) -> BasePydanticVectorStore: ...
    def persist(self) -> None: ...
    def stats(self) -> StoreStats: ...
    def clear(self) -> None: ...


class SimpleStore:
    """In-memory llama-index SimpleVectorStore w/ JSON persist."""

    backend = "simple"

    def __init__(self, persist_dir: Path) -> None:
        from llama_index.core import StorageContext
        from llama_index.core.vector_stores import SimpleVectorStore

        self._persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
        except Exception:
            vs = SimpleVectorStore()
            self._ctx = StorageContext.from_defaults(vector_store=vs)

    def storage_context(self) -> StorageContext:
        return self._ctx

    def vector_store(self) -> BasePydanticVectorStore:
        return self._ctx.vector_store

    def persist(self) -> None:
        try:
            self._ctx.persist(persist_dir=str(self._persist_dir))
        except Exception as e:
            raise StoreError(f"persist failed: {e}") from e

    def stats(self) -> StoreStats:
        vs = self._ctx.vector_store
        count = 0
        data = getattr(vs, "data", None)
        if data is not None:
            embedding_dict = getattr(data, "embedding_dict", {})
            count = len(embedding_dict)
        return StoreStats(
            backend=self.backend, node_count=count, persist_path=str(self._persist_dir)
        )

    def clear(self) -> None:
        import shutil

        if self._persist_dir.exists():
            shutil.rmtree(self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)


def make_store(cfg: StoreCfg, persist_dir: Path) -> Store:
    if cfg.backend == "simple":
        return SimpleStore(persist_dir)
    if cfg.backend == "lance":
        return LanceStore(persist_dir)
    if cfg.backend == "qdrant":
        raise NotImplementedError("QdrantStore is [remote] extra, not in baseline")
    raise ValueError(f"unknown store backend: {cfg.backend}")


class LanceStore:
    """LanceDB-backed vector store. Optional — install [lance] extra."""

    backend = "lance"

    def __init__(self, persist_dir: Path) -> None:
        try:
            from llama_index.core import StorageContext
            from llama_index.vector_stores.lancedb import LanceDBVectorStore
        except ImportError as e:
            raise StoreError(
                "lance backend requires the [lance] extra: uv add ragwatcher[lance]"
            ) from e

        self._persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._lance_dir = persist_dir / "lance"
        self._vs = LanceDBVectorStore(uri=str(self._lance_dir), table_name="chunks")
        try:
            self._ctx = StorageContext.from_defaults(
                persist_dir=str(persist_dir), vector_store=self._vs
            )
        except Exception:
            self._ctx = StorageContext.from_defaults(vector_store=self._vs)

    def storage_context(self) -> StorageContext:
        return self._ctx

    def vector_store(self) -> BasePydanticVectorStore:
        vs: BasePydanticVectorStore = self._vs
        return vs

    def persist(self) -> None:
        try:
            self._ctx.persist(persist_dir=str(self._persist_dir))
        except Exception as e:
            raise StoreError(f"lance persist failed: {e}") from e

    def stats(self) -> StoreStats:
        count = 0
        try:
            tbl = self._vs._table  # noqa: SLF001
            if tbl is not None:
                count = int(tbl.count_rows())
        except Exception:
            count = 0
        return StoreStats(
            backend=self.backend, node_count=count, persist_path=str(self._persist_dir)
        )

    def clear(self) -> None:
        import shutil

        if self._persist_dir.exists():
            shutil.rmtree(self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
