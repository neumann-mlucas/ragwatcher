"""Reject stdout writes outside CLI JSON output or MCP frames."""
import re
from pathlib import Path

_PRINT_CALL = re.compile(r"(?<![\w.])print\s*\(")


def test_no_print_in_src():
    src = Path(__file__).resolve().parent.parent / "src" / "ragwatcher"
    offenders = []
    for py in src.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _PRINT_CALL.search(stripped):
                offenders.append(f"{py.relative_to(src)}:{lineno}: {stripped}")
    assert not offenders, "print() forbidden in src/:\n" + "\n".join(offenders)
