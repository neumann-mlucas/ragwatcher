"""Chunking. P1 ships recursive splitter only. Semantic/late deferred to P4."""

from __future__ import annotations

from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.interface import NodeParser

from ragwatcher.config import ChunkCfg


def get_splitter(cfg: ChunkCfg) -> NodeParser:
    if cfg.strategy == "recursive":
        return SentenceSplitter(
            chunk_size=cfg.size,
            chunk_overlap=cfg.overlap,
        )
    if cfg.strategy == "semantic":
        from llama_index.core import Settings as LISettings
        from llama_index.core.node_parser import SemanticSplitterNodeParser

        return SemanticSplitterNodeParser(
            buffer_size=cfg.semantic.buffer_size,
            breakpoint_percentile_threshold=cfg.semantic.breakpoint_percentile,
            embed_model=LISettings.embed_model,
        )
    if cfg.strategy == "late":
        # Late chunking = long-context embed + post-chunk token embeddings.
        # ponytail: falls back to recursive until a user asks for real late chunking;
        # requires long-context embed model (bge-m3) not in default install.
        return SentenceSplitter(chunk_size=cfg.size, chunk_overlap=cfg.overlap)
    raise ValueError(f"unknown chunk.strategy: {cfg.strategy}")


def prefix_metadata(metadata: dict[str, Any], text: str) -> str:
    """Prepend filename + heading path so filename queries hit."""
    prefix_parts = []
    if fn := metadata.get("file_name"):
        prefix_parts.append(f"file: {fn}")
    if hp := metadata.get("heading_path"):
        prefix_parts.append(f"heading: {hp}")
    if not prefix_parts:
        return text
    return "\n".join(prefix_parts) + "\n\n" + text
