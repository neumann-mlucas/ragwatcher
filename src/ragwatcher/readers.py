"""File readers: ext → (text, metadata).

Per-file failures raise ReadError. Index.py logs WARN and continues.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragwatcher.errors import ReadError

Reader = Callable[[Path], tuple[str, dict[str, Any]]]


def _plain(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), {}
    except OSError as e:
        raise ReadError(f"cannot read {path}: {e}") from e


def _pdf(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ReadError(f"pypdf missing: {e}") from e
    try:
        reader = PdfReader(str(path))
        parts = [(p.extract_text() or "") for p in reader.pages]
        return "\n\n".join(parts), {"page_count": len(reader.pages)}
    except Exception as e:
        raise ReadError(f"pdf read failed {path}: {e}") from e


def _docx(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from docx import Document
    except ImportError as e:
        raise ReadError(f"python-docx missing: {e}") from e
    try:
        d = Document(str(path))
        return "\n".join(p.text for p in d.paragraphs), {}
    except Exception as e:
        raise ReadError(f"docx read failed {path}: {e}") from e


def _epub(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        import html2text
        from ebooklib import ITEM_DOCUMENT, epub
    except ImportError as e:
        raise ReadError(f"ebooklib/html2text missing: {e}") from e
    try:
        book = epub.read_epub(str(path))
        h = html2text.HTML2Text()
        h.ignore_links = False
        parts = [
            h.handle(item.get_content().decode("utf-8", errors="replace"))
            for item in book.get_items_of_type(ITEM_DOCUMENT)
        ]
        return "\n\n".join(parts), {}
    except Exception as e:
        raise ReadError(f"epub read failed {path}: {e}") from e


def _html(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        import html2text
    except ImportError as e:
        raise ReadError(f"html2text missing: {e}") from e
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        h = html2text.HTML2Text()
        h.ignore_links = False
        text = h.handle(raw)
        heading = _first_heading(text)
        meta: dict[str, Any] = {"heading_path": heading} if heading else {}
        return text, meta
    except Exception as e:
        raise ReadError(f"html read failed {path}: {e}") from e


def _md(path: Path) -> tuple[str, dict[str, Any]]:
    text, _ = _plain(path)
    heading = _first_heading(text)
    meta: dict[str, Any] = {"heading_path": heading} if heading else {}
    return text, meta


def _csv(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return "", {}
        header = rows[0]
        lines = [", ".join(header)]
        for row in rows[1:]:
            lines.append(", ".join(f"{h}={v}" for h, v in zip(header, row, strict=False)))
        return "\n".join(lines), {"row_count": len(rows) - 1}
    except OSError as e:
        raise ReadError(f"csv read failed {path}: {e}") from e


def _json(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return json.dumps(obj, indent=2, ensure_ascii=False), {}
    except (OSError, json.JSONDecodeError) as e:
        raise ReadError(f"json read failed {path}: {e}") from e


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("# ").strip() or None
    return None


EXT_TO_READER: dict[str, Reader] = {
    ".txt": _plain,
    ".md": _md,
    ".markdown": _md,
    ".rst": _plain,
    ".log": _plain,
    ".py": _plain,
    ".js": _plain,
    ".ts": _plain,
    ".tsx": _plain,
    ".jsx": _plain,
    ".go": _plain,
    ".rs": _plain,
    ".java": _plain,
    ".c": _plain,
    ".h": _plain,
    ".cpp": _plain,
    ".sh": _plain,
    ".yaml": _plain,
    ".yml": _plain,
    ".toml": _plain,
    ".json": _json,
    ".csv": _csv,
    ".html": _html,
    ".htm": _html,
    ".pdf": _pdf,
    ".docx": _docx,
    ".epub": _epub,
}


SUPPORTED_EXTS = frozenset(EXT_TO_READER.keys())


def read(path: Path) -> tuple[str, dict[str, Any]]:
    """Read text + metadata for supported ext, else ReadError."""
    reader = EXT_TO_READER.get(path.suffix.lower())
    if reader is None:
        raise ReadError(f"unsupported extension: {path.suffix}")
    return reader(path)
