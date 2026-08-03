from pathlib import Path

import pytest

from ragwatcher import readers
from ragwatcher.errors import ReadError

FIXTURE = Path(__file__).parent / "data"


def test_md_reader_gets_heading():
    text, meta = readers.read(FIXTURE / "notes" / "alpha.md")
    assert "Alpha is" in text
    assert meta.get("heading_path") == "Alpha"


def test_txt_via_plain_ext(tmp_path: Path):
    f = tmp_path / "n.txt"
    f.write_text("plain text body")
    text, _ = readers.read(f)
    assert text.strip() == "plain text body"


def test_csv_row_prefixed():
    text, meta = readers.read(FIXTURE / "mixed" / "table.csv")
    assert "name=alice" in text
    assert meta.get("row_count") == 4


def test_json_roundtrip():
    text, _ = readers.read(FIXTURE / "mixed" / "data.json")
    assert "ragwatcher" in text
    assert "python" in text


def test_python_source_readable():
    text, _ = readers.read(FIXTURE / "code" / "example.py")
    assert "class Widget" in text


def test_unsupported_ext_raises(tmp_path: Path):
    bad = tmp_path / "unknown.xyz"
    bad.write_text("data")
    with pytest.raises(ReadError):
        readers.read(bad)
