#!/usr/bin/env python3
"""Print basic, read-only health and content statistics for the TCM database."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv
import psycopg


QUERIES = {
    "schema_migrations": "SELECT count(*) FROM public.schema_migrations",
    "documents": "SELECT count(*) FROM tcm.documents",
    "document_versions": "SELECT count(*) FROM tcm.document_versions",
    "nodes": "SELECT count(*) FROM tcm.nodes",
    "chunks": "SELECT count(*) FROM tcm.chunks",
    "embedding_models": "SELECT count(*) FROM tcm.embedding_models",
    "chunk_embeddings": "SELECT count(*) FROM tcm.chunk_embeddings",
    "documents_with_translations": """SELECT count(DISTINCT dv.document_id)
        FROM tcm.node_texts nt
        JOIN tcm.nodes n ON n.id=nt.node_id
        JOIN tcm.document_versions dv ON dv.id=n.document_version_id
        WHERE nt.text_kind='modern_translation'""",
}


def rows_to_dicts(rows: Iterable[tuple[Any, ...]], columns: list[str]) -> list[dict[str, Any]]:
    """Convert simple psycopg tuple rows into JSON-friendly dictionaries."""
    return [dict(zip(columns, row, strict=True)) for row in rows]


def collect(connection: psycopg.Connection) -> dict[str, Any]:
    """Collect concise counts and breakdowns without modifying the database."""
    schema_present = connection.execute("SELECT to_regclass('tcm.documents') IS NOT NULL").fetchone()[0]
    if not schema_present:
        raise RuntimeError("TCM schema is absent; run src/database/migrate.py first")

    result = {name: connection.execute(statement).fetchone()[0] for name, statement in QUERIES.items()}
    result["node_texts_by_kind"] = rows_to_dicts(
        connection.execute("SELECT text_kind, count(*) FROM tcm.node_texts GROUP BY text_kind ORDER BY text_kind").fetchall(),
        ["text_kind", "count"],
    )
    result["documents_by_text_type"] = rows_to_dicts(
        connection.execute("SELECT text_type, count(*) FROM tcm.documents GROUP BY text_type ORDER BY text_type").fetchall(),
        ["text_type", "count"],
    )
    result["processing_runs_by_stage_status"] = rows_to_dicts(
        connection.execute("""SELECT stage, status, count(*) FROM tcm.processing_runs
                              GROUP BY stage, status ORDER BY stage, status""").fetchall(),
        ["stage", "status", "count"],
    )
    result["embedding_models_detail"] = rows_to_dicts(
        connection.execute("""SELECT provider, model_name, dimensions, distance_metric, count(ce.chunk_id)
                              FROM tcm.embedding_models em
                              LEFT JOIN tcm.chunk_embeddings ce ON ce.embedding_model_id=em.id
                              GROUP BY em.id, em.provider, em.model_name, em.dimensions, em.distance_metric
                              ORDER BY em.provider, em.model_name, em.dimensions""").fetchall(),
        ["provider", "model_name", "dimensions", "distance_metric", "embedding_count"],
    )
    return result


def print_table(result: dict[str, Any]) -> None:
    """Render scalar counts and small breakdowns for terminal use."""
    for name in (
        "schema_migrations",
        "documents",
        "document_versions",
        "nodes",
        "documents_with_translations",
        "chunks",
        "embedding_models",
        "chunk_embeddings",
    ):
        print(f"{name}: {result[name]}")
    for name in ("node_texts_by_kind", "documents_by_text_type", "processing_runs_by_stage_status", "embedding_models_detail"):
        print(f"\n{name}:")
        for row in result[name]:
            print("  " + " | ".join(f"{key}={value}" for key, value in row.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", help="Overrides DATABASE_URL from .env")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    load_dotenv()
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is required in .env, environment, or --database-url")
    with psycopg.connect(database_url) as connection:
        # Make accidental future changes to the inspection queries fail closed.
        connection.execute("SET TRANSACTION READ ONLY")
        result = collect(connection)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    else:
        print_table(result)


if __name__ == "__main__":
    main()
