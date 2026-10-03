"""Text I/O helpers that always pin UTF-8 explicitly, with stable line endings.

`Path.read_text()`/`Path.write_text()` and plain `open()` fall back to
`locale.getpreferredencoding()` when no `encoding` is given: UTF-8 on
Linux/macOS, but commonly cp1252 on Windows. Every text file this project
reads or writes (config JSON, normalized JSONL, CSV/JSONL tables, markdown
reports) is UTF-8, and some configs contain non-ASCII characters (e.g. the
Hebrew month aliases in `config/preprocessing.json`), so relying on the
platform default silently breaks on Windows.

Writes also pin `newline="\\n"`: text-mode writes translate `\n` to
`os.linesep` by default, which is `\r\n` on Windows. Since every generated
artifact's SHA-256 checksum is reported for reproducibility, letting the
platform silently rewrite line endings would make those checksums differ
across platforms even when the content is identical. (The raw source CSVs
under `data/raw/` are outside this project's control and can still be
CRLF-converted on checkout by `core.autocrlf`; that affects their own
checksum but not anything these helpers write.)

Every text read/write in this package should go through these helpers
rather than calling `read_text`/`write_text`/`open` directly, so this stays
fixed in one place.
"""
import csv
import json
from pathlib import Path


def read_text(path, encoding="utf-8"):
    return Path(path).read_text(encoding=encoding)


def write_text(path, content):
    Path(path).write_text(content, encoding="utf-8", newline="\n")


def read_json(path):
    return json.loads(read_text(path))


def read_jsonl(path):
    text = read_text(path)
    return [json.loads(line) for line in text.splitlines()] if text else []


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv_rows(path, encoding="utf-8"):
    """Header and rows (dicts of strings) of a CSV file. Pass encoding="utf-8-sig" for files
    that start with a byte-order mark. Rows with more or fewer fields than the header raise."""
    with Path(path).open(encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV rows in {path}")
    return columns, rows
