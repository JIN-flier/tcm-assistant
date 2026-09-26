#!/usr/bin/env python3
"""Import completed enriched translation records into the TCM PostgreSQL schema.

Run ``migrate.py`` first.  This importer is safe to run repeatedly: immutable
rows are identified by their source/content hashes, so records already present
in the database are skipped and newly appended JSONL records are inserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
import psycopg
from psycopg.types.json import Jsonb


DEFAULT_INPUT = Path("data/enriched/translations.jsonl")
NODE_RANKS = {
    "document": 0,
    "volume": 1,
    "chapter": 2,
    "section": 3,
    "subsection": 4,
    "entry": 5,
    "paragraph": 99,
}
TRAILING_NUMBER = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class ImportRecord:
    """A validated completed record from ``translations.jsonl``."""

    line_number: int
    value: dict[str, Any]


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest used by the enrichment pipeline."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonb(value: dict[str, Any]) -> Jsonb:
    """Adapt a dictionary explicitly as JSONB for psycopg."""
    return Jsonb(value)


def required_string(item: dict[str, Any], name: str, line_number: int) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"line {line_number}: {name!r} must be a non-empty string")
    return value


def optional_int(value: Any, name: str, line_number: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"line {line_number}: {name!r} must be an integer or null")
    return value


def read_records(path: Path) -> Iterator[ImportRecord]:
    """Yield only valid, completed records; failed translation attempts are skipped."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number}: each JSONL value must be an object")
            if item.get("status") != "completed":
                yield ImportRecord(line_number, item)
                continue

            for name in (
                "document_id",
                "unit_id",
                "source_file",
                "source_sha256",
                "original_text",
                "normalized_original",
                "normalized_text_sha256",
                "modern_translation",
            ):
                required_string(item, name, line_number)
            if sha256_text(item["normalized_original"]) != item["normalized_text_sha256"]:
                raise ValueError(f"line {line_number}: normalized_text_sha256 does not match normalized_original")
            if not isinstance(item.get("node_path"), list):
                raise ValueError(f"line {line_number}: node_path must be a list")
            if not isinstance(item.get("document_metadata"), dict):
                raise ValueError(f"line {line_number}: document_metadata must be an object")
            yield ImportRecord(line_number, item)


def source_sequence(source_node_id: str, node_type: str) -> int:
    """Make a stable, collision-resistant sibling sequence from parser node IDs.

    ``translations.jsonl`` retains structural IDs but not the parser's sibling
    sequence.  The parser IDs end in a per-type counter, so reserving the last
    two digits for the type rank keeps IDs such as ``V0001`` and ``C0001``
    distinct under the same parent while staying deterministic across imports.
    """
    match = TRAILING_NUMBER.search(source_node_id)
    number = int(match.group(1)) if match else int(hashlib.sha256(source_node_id.encode()).hexdigest()[:12], 16)
    # ``nodes.sequence`` is a PostgreSQL INTEGER.  Parser counters are much
    # smaller in practice; modulo keeps malformed external IDs in range too.
    number %= 20_000_000
    return number * 100 + NODE_RANKS.get(node_type, 98)


def fetch_or_insert_id(connection: psycopg.Connection, statement: str, params: tuple[Any, ...], lookup: str, lookup_params: tuple[Any, ...]) -> str:
    """Insert an immutable row if new, then return its primary-key UUID."""
    row = connection.execute(statement, params).fetchone()
    if row:
        return str(row[0])
    existing = connection.execute(lookup, lookup_params).fetchone()
    if not existing:
        raise RuntimeError("row was not inserted and could not be found")
    return str(existing[0])


def ensure_document(connection: psycopg.Connection, item: dict[str, Any], line_number: int) -> str:
    metadata = item["document_metadata"]
    document_id = item["document_id"]
    canonical_title = metadata.get("detected_title") or metadata.get("catalog_title") or document_id
    if not isinstance(canonical_title, str):
        raise ValueError(f"line {line_number}: document title must be a string when supplied")
    category_path = metadata.get("category")
    if category_path is not None and (not isinstance(category_path, list) or not all(isinstance(part, str) for part in category_path)):
        raise ValueError(f"line {line_number}: document_metadata.category must be a string list or null")
    year = optional_int(metadata.get("year"), "document_metadata.year", line_number)
    provenance = {
        "import_source": "data/enriched/translations.jsonl",
        "document_metadata_sources": item.get("document_metadata_sources", {}),
    }
    return fetch_or_insert_id(
        connection,
        """INSERT INTO tcm.documents
              (source_id, canonical_title, catalog_title, author, dynasty, year,
               catalog_id, category_code, category_path, text_type, character_type, metadata)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (source_id) DO NOTHING RETURNING id""",
        (
            document_id,
            canonical_title,
            metadata.get("catalog_title"),
            metadata.get("author"),
            metadata.get("dynasty"),
            year,
            metadata.get("catalog_id"),
            metadata.get("category_code"),
            category_path,
            metadata.get("text_type") or "unknown",
            metadata.get("character_type"),
            jsonb(provenance),
        ),
        "SELECT id FROM tcm.documents WHERE source_id=%s",
        (document_id,),
    )


def ensure_document_version(connection: psycopg.Connection, document_id: str, item: dict[str, Any]) -> str:
    source_sha256 = item["source_sha256"]
    return fetch_or_insert_id(
        connection,
        """INSERT INTO tcm.document_versions
              (document_id, source_file, source_sha256, metadata)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (source_sha256) DO NOTHING RETURNING id""",
        (document_id, item["source_file"], source_sha256, jsonb({"import_source": "data/enriched/translations.jsonl"})),
        "SELECT id FROM tcm.document_versions WHERE source_sha256=%s",
        (source_sha256,),
    )


def ensure_node(
    connection: psycopg.Connection,
    document_version_id: str,
    source_node_id: str,
    parent_id: str | None,
    node_type: str,
    title: str | None,
    level: int,
    source_start: int | None,
    source_end: int | None,
    line_start: int | None,
    line_end: int | None,
) -> str:
    return fetch_or_insert_id(
        connection,
        """INSERT INTO tcm.nodes
              (document_version_id, source_node_id, parent_id, node_type, title, level, sequence,
               source_start, source_end, line_start, line_end, parser_metadata, metadata)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (document_version_id, source_node_id) DO NOTHING RETURNING id""",
        (
            document_version_id,
            source_node_id,
            parent_id,
            node_type,
            title,
            level,
            0 if parent_id is None else source_sequence(source_node_id, node_type),
            source_start,
            source_end,
            line_start,
            line_end,
            jsonb({"method": "translations_jsonl_import"}),
            jsonb({"import_source": "data/enriched/translations.jsonl"}),
        ),
        "SELECT id FROM tcm.nodes WHERE document_version_id=%s AND source_node_id=%s",
        (document_version_id, source_node_id),
    )


def ensure_text(
    connection: psycopg.Connection,
    node_id: str,
    text_kind: str,
    content: str,
    source_text_id: str | None,
    provenance: dict[str, Any],
    model_provider: str | None = None,
    model_name: str | None = None,
    prompt_version: str | None = None,
    prompt_sha256: str | None = None,
) -> str:
    content_sha256 = sha256_text(content)
    return fetch_or_insert_id(
        connection,
        """INSERT INTO tcm.node_texts
              (node_id, text_kind, language_code, content, content_sha256, source_text_id,
               model_provider, model_name, prompt_version, prompt_sha256, provenance)
           VALUES (%s, %s, 'zh-Hans', %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (node_id, text_kind, language_code, content_sha256) DO NOTHING RETURNING id""",
        (
            node_id,
            text_kind,
            content,
            content_sha256,
            source_text_id,
            model_provider,
            model_name,
            prompt_version,
            prompt_sha256,
            jsonb(provenance),
        ),
        """SELECT id FROM tcm.node_texts
           WHERE node_id=%s AND text_kind=%s AND language_code='zh-Hans' AND content_sha256=%s""",
        (node_id, text_kind, content_sha256),
    )


def ensure_chunk(
    connection: psycopg.Connection,
    document_version_id: str,
    node_id: str,
    source_text_id: str,
    text: str,
) -> str:
    """Store a single full-paragraph retrieval chunk for an imported text artifact."""
    text_sha256 = sha256_text(text)
    return fetch_or_insert_id(
        connection,
        """INSERT INTO tcm.chunks
              (document_version_id, node_id, source_text_id, sequence, char_start, char_end,
               text, text_sha256, metadata)
           VALUES (%s, %s, %s, 0, 0, %s, %s, %s, %s)
           ON CONFLICT (source_text_id, sequence, text_sha256) DO NOTHING RETURNING id""",
        (
            document_version_id,
            node_id,
            source_text_id,
            len(text),
            text,
            text_sha256,
            jsonb({"import_source": "data/enriched/translations.jsonl", "chunking": "one_paragraph_per_text"}),
        ),
        """SELECT id FROM tcm.chunks
           WHERE source_text_id=%s AND sequence=0 AND text_sha256=%s""",
        (source_text_id, text_sha256),
    )


def import_completed_record(connection: psycopg.Connection, record: ImportRecord, counts: Counter[str]) -> None:
    item = record.value
    if item.get("status") != "completed":
        counts["skipped_not_completed"] += 1
        return

    document_id = ensure_document(connection, item, record.line_number)
    document_version_id = ensure_document_version(connection, document_id, item)
    path = item["node_path"]
    if not path or path[0].get("node_id") != item["document_id"]:
        path = [{"node_id": item["document_id"], "type": "document", "title": item["document_metadata"].get("detected_title")}] + path

    parent_id: str | None = None
    for level, path_node in enumerate(path):
        source_node_id = required_string(path_node, "node_id", record.line_number)
        node_type = path_node.get("type")
        if not isinstance(node_type, str) or not node_type:
            raise ValueError(f"line {record.line_number}: node_path type must be a non-empty string")
        title = path_node.get("title")
        if title is not None and not isinstance(title, str):
            raise ValueError(f"line {record.line_number}: node_path title must be a string or null")
        parent_id = ensure_node(
            connection,
            document_version_id,
            source_node_id,
            parent_id,
            node_type,
            title,
            level,
            None,
            None,
            None,
            None,
        )

    node_id = ensure_node(
        connection,
        document_version_id,
        item["unit_id"],
        parent_id,
        "paragraph",
        None,
        len(path),
        optional_int(item.get("source_start"), "source_start", record.line_number),
        optional_int(item.get("source_end"), "source_end", record.line_number),
        optional_int(item.get("line_start"), "line_start", record.line_number),
        optional_int(item.get("line_end"), "line_end", record.line_number),
    )
    original_id = ensure_text(
        connection,
        node_id,
        "original",
        item["original_text"],
        None,
        {"import_source": "data/enriched/translations.jsonl"},
    )
    normalized_id = ensure_text(
        connection,
        node_id,
        "normalized",
        item["normalized_original"],
        original_id,
        {"import_source": "data/enriched/translations.jsonl", "normalized_text_sha256": item["normalized_text_sha256"]},
    )
    translation = item.get("translation_provenance", {})
    if not isinstance(translation, dict):
        raise ValueError(f"line {record.line_number}: translation_provenance must be an object when supplied")
    modern_id = ensure_text(
        connection,
        node_id,
        "modern_translation",
        item["modern_translation"],
        normalized_id,
        {
            "import_source": "data/enriched/translations.jsonl",
            "translation_notes": item.get("translation_notes", []),
            "translation_provenance": translation,
        },
        translation.get("method"),
        translation.get("model"),
        translation.get("prompt_version"),
        translation.get("prompt_sha256"),
    )
    # These two chunk kinds match the default original/modern retrieval channels
    # in src/rag/answer.py.  Embeddings are deliberately left to a separate
    # embedding pipeline so the importer has no model or GPU dependency.
    ensure_chunk(connection, document_version_id, node_id, original_id, item["original_text"])
    ensure_chunk(connection, document_version_id, node_id, modern_id, item["modern_translation"])
    counts["processed_completed_records"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"JSONL input path (default: {DEFAULT_INPUT})")
    parser.add_argument("--database-url", help="Overrides DATABASE_URL from .env")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count JSONL records without connecting or writing")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    summary: Counter[str] = Counter()
    if args.dry_run:
        for record in read_records(args.input):
            summary["completed_records"] += record.value.get("status") == "completed"
            summary["not_completed_records"] += record.value.get("status") != "completed"
        print(json.dumps(dict(summary), ensure_ascii=False, sort_keys=True))
        return

    load_dotenv()
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is required in .env, environment, or --database-url")
    with psycopg.connect(database_url) as connection:
        with connection.transaction():
            schema_present = connection.execute("SELECT to_regclass('tcm.documents') IS NOT NULL").fetchone()[0]
            if not schema_present:
                raise RuntimeError("TCM schema is absent; run src/database/migrate.py first")
            connection.execute("SELECT pg_advisory_xact_lock(hashtext('tcm-translations-import'))")
            for record in read_records(args.input):
                import_completed_record(connection, record, summary)
                summary["input_records"] += 1
    print(json.dumps(dict(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
