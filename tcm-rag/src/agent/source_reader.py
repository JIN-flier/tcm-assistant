"""Bounded excerpts from corpus source files selected by the agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def decode_source(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


class SourceReader:
    def __init__(self, data_root: Path, *, excerpt_chars: int = 4000, padding_chars: int = 600) -> None:
        self.data_root = data_root.resolve()
        self.excerpt_chars = excerpt_chars
        self.padding_chars = padding_chars

    def _safe_path(self, source_file: str) -> Path:
        relative = Path(source_file)
        if relative.is_absolute():
            raise ValueError("absolute source paths are not allowed")
        # Audit records normally store ``raw/book.txt``.  Also tolerate the
        # equivalent ``data/raw/book.txt`` form when data_root is ``data``.
        if relative.parts and relative.parts[0] == self.data_root.name:
            relative = Path(*relative.parts[1:])
        path = (self.data_root / relative).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("source path escapes the configured data root")
        return path

    def read_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "chunk_id": hit["chunk_id"], "source_file": hit.get("source_file"),
            "status": "not_read", "encoding": None, "start": None, "end": None,
            "text": None, "error": None,
        }
        source_file = hit.get("source_file")
        if not source_file:
            result["error"] = "database hit has no source_file"
            return result
        try:
            path = self._safe_path(source_file)
            raw = path.read_bytes()
            text, encoding = decode_source(raw)
            start, end = self._locate(text, hit)
            result.update(
                status="read", encoding=encoding, start=start, end=end,
                text=text[start:end],
            )
        except (OSError, ValueError) as error:
            result["error"] = str(error)
        return result

    def _locate(self, text: str, hit: dict[str, Any]) -> tuple[int, int]:
        source_start, source_end = hit.get("source_start"), hit.get("source_end")
        if isinstance(source_start, int) and isinstance(source_end, int):
            start = max(0, source_start - self.padding_chars)
            end = min(len(text), source_end + self.padding_chars)
        elif isinstance(hit.get("line_start"), int) and isinstance(hit.get("line_end"), int):
            lines = text.splitlines(keepends=True)
            first = max(0, hit["line_start"] - 1)
            last = min(len(lines), hit["line_end"] + 1)
            start = sum(len(line) for line in lines[:first])
            end = sum(len(line) for line in lines[:last])
        else:
            needle = hit.get("text", "") if hit.get("text_kind") == "original" else ""
            found = text.find(needle) if needle else -1
            if found < 0:
                raise ValueError("no source offsets and chunk text was not found in the raw file")
            start = max(0, found - self.padding_chars)
            end = min(len(text), found + len(needle) + self.padding_chars)
        if end - start > self.excerpt_chars:
            end = start + self.excerpt_chars
        return start, end
