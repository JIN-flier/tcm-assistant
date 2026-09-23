#!/usr/bin/env python3
"""Build provenance-preserving JSONL audit records for tcmoc source files.

This program is deliberately deterministic: it never calls an LLM and it never
modifies ``data/raw``.  Ambiguous values remain null/unknown and are accompanied
by an explanation in ``field_sources`` or ``classification.evidence``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
CATALOG_FILENAME = re.compile(
    r"^\((?P<code>[\d.]+)-(?P<catalog_id>[\d.]+)\)\.(?P<category>[^.]+)\.《(?P<title>[^》]+)》(?P<tail>.*)$"
)
NUMBERED_FILENAME = re.compile(r"^(?P<number>\d+)[.-](?P<title>.+)$")
HEADER_FIELD = re.compile(r"^(书名|作者|朝代|年份)[：:]\s*(.+?)\s*$", re.MULTILINE)
VOLUME_PATTERN = re.compile(r"[（(]([一二三四五六七八九十百\d]+)卷(?:、续集([一二三四五六七八九十百\d]+)卷)?[）)]")

# These indicators are intentionally narrow.  A match is recorded as evidence;
# it is a triage signal, not a bibliographical claim.
MODERN_TITLE_TERMS = ("近现代", "名老中医", "临证经验", "医疗经验", "实验录", "名师垂教", "方法", "应用")
REFERENCE_TITLE_TERMS = ("医籍考", "目录", "辞典", "索引", "汇编")
COMMENTARY_TITLE_TERMS = ("集注", "注", "解", "释", "疏", "校注", "评文")
CASE_TITLE_TERMS = ("医案", "验案", "病案")
TRADITIONAL_HINTS = set("醫藥經傷證針灸臟腑學錄診療衛產婦兒")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decode_source(raw: bytes) -> tuple[str, str, bool]:
    """Decode bytes without hiding loss: a replacement result is flagged."""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return raw.decode(encoding), encoding, True
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace"), "utf-8", False


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip() + "\n"


def safe_value(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    return None if value in {"", "不详", "未知", "无"} else value


def chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100}
    if value in digits:
        return digits[value]
    if "十" in value:
        left, _, right = value.partition("十")
        return (digits.get(left, 1) if left else 1) * 10 + digits.get(right, 0)
    return None


def parse_filename(path: Path) -> dict[str, Any]:
    stem = path.stem
    catalog = CATALOG_FILENAME.match(stem)
    if catalog:
        parts = catalog.group("category").split("-")
        volume = VOLUME_PATTERN.search(catalog.group("tail"))
        return {
            "title": catalog.group("title"), "catalog_title": catalog.group("title"),
            "catalog_id": catalog.group("catalog_id"), "category_code": catalog.group("code"),
            "category": parts, "volume_match": volume, "source": "catalog_filename",
        }
    numbered = NUMBERED_FILENAME.match(stem)
    return {
        "title": numbered.group("title") if numbered else stem,
        "catalog_title": None, "catalog_id": None, "category_code": None,
        "category": None, "volume_match": None, "source": "filename",
    }


def parse_header(text: str) -> dict[str, str | None]:
    fields: dict[str, str | None] = {"书名": None, "作者": None, "朝代": None, "年份": None}
    # Only inspect the early bibliographic block, avoiding content quotations.
    for name, value in HEADER_FIELD.findall(text[:5000]):
        if fields[name] is None:
            fields[name] = safe_value(value)
    title_match = re.search(r"^<篇名>\s*(.+?)\s*$", text[:5000], re.MULTILINE)
    if not fields["书名"] and title_match:
        fields["书名"] = safe_value(title_match.group(1))
    return fields


def year_from_header(value: str | None) -> int | None:
    if not value:
        return None
    years = [int(v) for v in re.findall(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", value)]
    return years[0] if len(years) == 1 else None


def classify(title: str, year: int | None, text: str, category: list[str] | None) -> dict[str, Any]:
    evidence: list[str] = []
    if category:
        evidence.append("catalog_filename.category")
    if year and year >= 1912:
        evidence.append(f"header.year={year} (>=1912)")
        return {"text_type": "modern_book", "method": "rule", "confidence": "high", "evidence": evidence}
    term = next((term for term in MODERN_TITLE_TERMS if term in title), None)
    if term:
        return {"text_type": "modern_book", "method": "title_rule", "confidence": "medium", "evidence": [f"title contains {term}"]}
    term = next((term for term in REFERENCE_TITLE_TERMS if term in title), None)
    if term:
        return {"text_type": "reference", "method": "title_rule", "confidence": "medium", "evidence": [f"title contains {term}"]}
    term = next((term for term in CASE_TITLE_TERMS if term in title), None)
    if term:
        return {"text_type": "medical_case", "method": "title_rule", "confidence": "high", "evidence": [f"title contains {term}"]}
    term = next((term for term in COMMENTARY_TITLE_TERMS if term in title), None)
    if term:
        return {"text_type": "commentary", "method": "title_rule", "confidence": "medium", "evidence": [f"title contains {term}"]}
    if len(text.strip()) < 200:
        return {"text_type": "unknown", "method": "content_rule", "confidence": "low", "evidence": ["text shorter than 200 characters"]}
    return {"text_type": "ancient_classical", "method": "default_rule", "confidence": "low", "evidence": ["no modern/reference/case/commentary indicator"]}


def character_type(text: str) -> str:
    sample = text[:20000]
    traditional = sum(char in TRADITIONAL_HINTS for char in sample)
    return "traditional" if traditional >= 8 else "simplified"


def volume_info(filename_data: dict[str, Any], text: str) -> tuple[bool, int | None, str | None]:
    match = filename_data["volume_match"] or VOLUME_PATTERN.search(text[:5000])
    if not match:
        return False, None, None
    count = chinese_number(match.group(1))
    supplement = chinese_number(match.group(2) or "") or 0
    return True, (count + supplement) if count else None, "catalog_filename" if filename_data["volume_match"] else "header_text"


def make_record(path: Path, raw_root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text, encoding, encoding_ok = decode_source(raw)
    normalized = normalize_text(text)
    filename_data = parse_filename(path)
    header = parse_header(text)
    detected_title = header["书名"] or filename_data["title"]
    title_source = "header.书名" if header["书名"] else filename_data["source"]
    year = year_from_header(header["年份"])
    has_volume, estimated_volumes, volume_source = volume_info(filename_data, text)
    empty_line_ratio = (sum(not line.strip() for line in text.splitlines()) / max(1, len(text.splitlines())))
    replacement_count = text.count("�")
    # Long runs of CJK characters or missing-glyph markers are suspicious.  Do
    # not treat a common dashed page separator as an OCR problem.
    suspicious_ocr = bool(re.search(r"[□�]{3,}|([\u4e00-\u9fff])\1{8,}", text))
    classification = classify(detected_title, year, text, filename_data["category"])
    field_sources = {
        "detected_title": title_source,
        "catalog_title": "catalog_filename" if filename_data["catalog_title"] else None,
        "author": "header.作者" if header["作者"] else None,
        "dynasty": "header.朝代" if header["朝代"] else None,
        "year": "header.年份 (unambiguous Gregorian year)" if year else None,
        "catalog_id": "catalog_filename" if filename_data["catalog_id"] else None,
        "category": "catalog_filename" if filename_data["category"] else None,
        "estimated_volumes": volume_source,
    }
    notes = []
    if not encoding_ok:
        notes.append("decode required replacement characters")
    if replacement_count:
        notes.append(f"replacement characters: {replacement_count}")
    if classification["text_type"] == "unknown":
        notes.append("text type needs manual review")
    status = "needs_review" if notes or suspicious_ocr else "accepted"
    record = {
        "schema_version": SCHEMA_VERSION,
        "document_id": f"TCM_{sha256(raw)[:16].upper()}",
        "source_file": path.relative_to(raw_root.parent).as_posix(),
        "source_sha256": sha256(raw),
        "normalized_text_sha256": sha256(normalized.encode("utf-8")),
        "detected_title": detected_title,
        "catalog_title": filename_data["catalog_title"],
        "text_type": classification["text_type"],
        "language_style": (
            "modern_chinese" if classification["text_type"] == "modern_book"
            else "classical_chinese" if classification["text_type"] in {"ancient_classical", "commentary", "medical_case"}
            else "unknown"
        ),
        "author": header["作者"], "dynasty": header["朝代"], "year": year,
        "catalog_id": filename_data["catalog_id"], "category_code": filename_data["category_code"],
        "category": filename_data["category"], "character_type": character_type(text),
        "has_volume": has_volume, "estimated_volumes": estimated_volumes,
        "quality": {"encoding_ok": encoding_ok, "detected_encoding": encoding,
                    "replacement_char_count": replacement_count, "empty_line_ratio": round(empty_line_ratio, 5),
                    "suspected_ocr": suspicious_ocr, "byte_count": len(raw), "character_count": len(text)},
        "classification": classification, "field_sources": field_sources,
        "status": status, "notes": notes,
    }
    return record


def apply_duplicate_metadata(records: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["normalized_text_sha256"]].append(record)
    for fingerprint, group in groups.items():
        if len(group) > 1:
            group_id = f"DUP_{fingerprint[:16].upper()}"
            canonical = min(item["document_id"] for item in group)
            for item in group:
                item["duplicate_group"] = group_id
                item["duplicate_of"] = None if item["document_id"] == canonical else canonical
        else:
            group[0]["duplicate_group"] = None
            group[0]["duplicate_of"] = None


def report(records: list[dict[str, Any]], raw_root: Path) -> dict[str, Any]:
    counts = Counter(record["text_type"] for record in records)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(), "raw_root": raw_root.as_posix(),
        "total_files": len(records), "by_text_type": dict(sorted(counts.items())),
        "with_detected_title": sum(bool(r["detected_title"]) for r in records),
        "with_author": sum(bool(r["author"]) for r in records),
        "with_dynasty": sum(bool(r["dynasty"]) for r in records),
        "with_year": sum(r["year"] is not None for r in records),
        "with_catalog_metadata": sum(r["catalog_id"] is not None for r in records),
        "with_volume": sum(r["has_volume"] for r in records),
        "needs_review": sum(r["status"] == "needs_review" for r in records),
        "encoding_issues": sum(not r["quality"]["encoding_ok"] for r in records),
        "suspected_ocr": sum(r["quality"]["suspected_ocr"] for r in records),
        "duplicate_files": sum(r["duplicate_of"] is not None for r in records),
        "exact_duplicate_groups": len({r["duplicate_group"] for r in records if r["duplicate_group"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/audit"))
    args = parser.parse_args()
    raw_root = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(path for path in raw_root.iterdir() if path.is_file())
    records = [make_record(path, raw_root) for path in files]
    apply_duplicate_metadata(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "documents.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "report.json").write_text(
        json.dumps(report(records, raw_root), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
