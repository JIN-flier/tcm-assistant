#!/usr/bin/env python3
"""Apply append-only PostgreSQL + pgvector migrations for the TCM corpus."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
import psycopg
from psycopg import sql


@dataclass(frozen=True)
class Migration:
    """Describe one immutable, append-only database migration.

    Attributes:
        version: Monotonically increasing migration identifier, such as ``0001``.
        description: Human-readable summary of the schema change.
        statement: SQL statement(s) executed when the migration is first applied.
    """
    version: str
    description: str
    statement: str


MIGRATIONS = [Migration("0001", "initial corpus schema", r"""
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS tcm;

CREATE TABLE IF NOT EXISTS tcm.processing_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), stage TEXT NOT NULL,
  pipeline_version TEXT NOT NULL, configuration JSONB NOT NULL DEFAULT '{}'::jsonb,
  input_manifest_sha256 TEXT, status TEXT NOT NULL DEFAULT 'running',
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ, error_message TEXT,
  CHECK (status IN ('running','completed','failed','cancelled'))
);

-- Logical work. source_id is the stable audit document_id, not a file hash.
CREATE TABLE IF NOT EXISTS tcm.documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), source_id TEXT NOT NULL UNIQUE,
  canonical_title TEXT NOT NULL, catalog_title TEXT, author TEXT, dynasty TEXT, year INTEGER,
  catalog_id TEXT, category_code TEXT, category_path TEXT[], text_type TEXT NOT NULL,
  character_type TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Corrected or newly supplied source files create a version; never overwrite one.
CREATE TABLE IF NOT EXISTS tcm.document_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES tcm.documents(id) ON DELETE RESTRICT,
  source_file TEXT NOT NULL, source_sha256 TEXT NOT NULL, normalized_text_sha256 TEXT,
  source_encoding TEXT, byte_count BIGINT, character_count BIGINT,
  quality JSONB NOT NULL DEFAULT '{}'::jsonb, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  processing_run_id UUID REFERENCES tcm.processing_runs(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_id, source_sha256), UNIQUE (source_sha256)
);

CREATE TABLE IF NOT EXISTS tcm.nodes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_version_id UUID NOT NULL REFERENCES tcm.document_versions(id) ON DELETE CASCADE,
  source_node_id TEXT NOT NULL, parent_id UUID REFERENCES tcm.nodes(id) ON DELETE RESTRICT,
  node_type TEXT NOT NULL, title TEXT, level SMALLINT NOT NULL, sequence INTEGER NOT NULL,
  source_start INTEGER, source_end INTEGER, line_start INTEGER, line_end INTEGER,
  parser_metadata JSONB NOT NULL DEFAULT '{}'::jsonb, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_version_id, source_node_id), UNIQUE (document_version_id, parent_id, sequence),
  CHECK (level >= 0),
  CHECK (source_start IS NULL OR source_end IS NULL OR source_start <= source_end),
  CHECK (line_start IS NULL OR line_end IS NULL OR line_start <= line_end)
);

-- Immutable text artifacts: original, normalized, modern_translation, summary, etc.
CREATE TABLE IF NOT EXISTS tcm.node_texts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), node_id UUID NOT NULL REFERENCES tcm.nodes(id) ON DELETE CASCADE,
  text_kind TEXT NOT NULL, language_code TEXT NOT NULL DEFAULT 'zh-Hans', content TEXT NOT NULL,
  content_sha256 TEXT NOT NULL, source_text_id UUID REFERENCES tcm.node_texts(id) ON DELETE RESTRICT,
  processing_run_id UUID REFERENCES tcm.processing_runs(id) ON DELETE SET NULL,
  model_provider TEXT, model_name TEXT, prompt_version TEXT, prompt_sha256 TEXT,
  review_status TEXT NOT NULL DEFAULT 'unreviewed', provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (node_id, text_kind, language_code, content_sha256),
  CHECK (review_status IN ('unreviewed','approved','rejected','needs_review'))
);

CREATE TABLE IF NOT EXISTS tcm.chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_version_id UUID NOT NULL REFERENCES tcm.document_versions(id) ON DELETE CASCADE,
  node_id UUID NOT NULL REFERENCES tcm.nodes(id) ON DELETE RESTRICT,
  source_text_id UUID NOT NULL REFERENCES tcm.node_texts(id) ON DELETE RESTRICT,
  parent_chunk_id UUID REFERENCES tcm.chunks(id) ON DELETE RESTRICT,
  sequence INTEGER NOT NULL, char_start INTEGER, char_end INTEGER, text TEXT NOT NULL, text_sha256 TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  processing_run_id UUID REFERENCES tcm.processing_runs(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (source_text_id, sequence, text_sha256), CHECK (sequence >= 0),
  CHECK (char_start IS NULL OR char_end IS NULL OR char_start <= char_end)
);

CREATE TABLE IF NOT EXISTS tcm.embedding_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), provider TEXT NOT NULL, model_name TEXT NOT NULL,
  dimensions INTEGER NOT NULL, distance_metric TEXT NOT NULL DEFAULT 'cosine',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (provider, model_name, dimensions), CHECK (dimensions > 0),
  CHECK (distance_metric IN ('cosine','l2','inner_product'))
);

-- Unconstrained vector permits models with different dimensions. Build an HNSW
-- index per model (with an explicit cast) after it is registered.
CREATE TABLE IF NOT EXISTS tcm.chunk_embeddings (
  chunk_id UUID NOT NULL REFERENCES tcm.chunks(id) ON DELETE CASCADE,
  embedding_model_id UUID NOT NULL REFERENCES tcm.embedding_models(id) ON DELETE RESTRICT,
  embedding_type TEXT NOT NULL, embedding vector NOT NULL, content_sha256 TEXT NOT NULL,
  processing_run_id UUID REFERENCES tcm.processing_runs(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (chunk_id, embedding_model_id, embedding_type), CHECK (vector_dims(embedding) > 0)
);

CREATE TABLE IF NOT EXISTS tcm.entities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), canonical_name TEXT NOT NULL, entity_type TEXT NOT NULL,
  normalized_name TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE (canonical_name, entity_type)
);
CREATE TABLE IF NOT EXISTS tcm.chunk_entities (
  chunk_id UUID NOT NULL REFERENCES tcm.chunks(id) ON DELETE CASCADE,
  entity_id UUID NOT NULL REFERENCES tcm.entities(id) ON DELETE CASCADE, mention_text TEXT,
  char_start INTEGER, char_end INTEGER, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (chunk_id, entity_id, char_start),
  CHECK (char_start IS NULL OR char_end IS NULL OR char_start <= char_end)
);

CREATE INDEX IF NOT EXISTS document_versions_document_idx ON tcm.document_versions(document_id);
CREATE INDEX IF NOT EXISTS nodes_tree_idx ON tcm.nodes(document_version_id, parent_id, sequence);
CREATE INDEX IF NOT EXISTS node_texts_lookup_idx ON tcm.node_texts(node_id, text_kind, created_at DESC);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON tcm.chunks(document_version_id);
CREATE INDEX IF NOT EXISTS chunks_node_idx ON tcm.chunks(node_id, sequence);
CREATE INDEX IF NOT EXISTS chunk_embeddings_model_idx ON tcm.chunk_embeddings(embedding_model_id, embedding_type);
CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx ON tcm.documents USING gin(metadata);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx ON tcm.chunks USING gin(metadata);
"""),
Migration("0002", "add publication year", """
    ALTER TABLE tcm.documents
    ADD COLUMN IF NOT EXISTS publication_year INTEGER;

    CREATE INDEX IF NOT EXISTS documents_publication_year_idx
    ON tcm.documents (publication_year);
"""),
Migration("0003", "add trigram lexical retrieval index", """
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx
    ON tcm.chunks USING gin (text gin_trgm_ops);
"""),
]


def checksum(statement: str) -> str:
    """Return the SHA-256 checksum of a migration SQL statement."""
    return hashlib.sha256(statement.encode()).hexdigest()


def apply_migrations(connection: psycopg.Connection) -> None:
    """Apply all pending schema migrations in a transaction-safe, append-only way.

    A bookkeeping table records every applied migration and its checksum. An
    advisory transaction lock prevents multiple processes from applying schema
    migrations concurrently. If an already-applied migration has been modified,
    the function raises an error and requires a new migration to be appended.

    Args:
        connection: An open psycopg PostgreSQL connection.
    """
    connection.execute("""CREATE TABLE IF NOT EXISTS public.schema_migrations (
        version TEXT PRIMARY KEY, description TEXT NOT NULL, checksum_sha256 TEXT NOT NULL,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
    with connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('tcm-schema-migrations'))")
        applied = {row[0]: row[1] for row in connection.execute("SELECT version, checksum_sha256 FROM public.schema_migrations")}
        for migration in MIGRATIONS:
            digest = checksum(migration.statement)
            if old_checksum := applied.get(migration.version):
                if old_checksum != digest:
                    raise RuntimeError(f"Migration {migration.version} changed after application; append a migration instead.")
                print(f"skip {migration.version}: already applied")
                continue
            connection.execute(migration.statement)
            connection.execute("INSERT INTO public.schema_migrations(version, description, checksum_sha256) VALUES (%s,%s,%s)", (migration.version, migration.description, digest))
            print(f"apply {migration.version}: {migration.description}")


def create_hnsw_index(connection: psycopg.Connection, name: str, model_id: str, dimensions: int) -> None:
    """Create or verify an HNSW cosine index for one registered embedding model.

    ``chunk_embeddings.embedding`` intentionally has no fixed vector dimension,
    allowing embeddings from different models to coexist. This function creates
    a model-specific partial HNSW index by casting the vector to the registered
    dimension and restricting the index to one ``embedding_model_id``.

    Args:
        connection: An open psycopg PostgreSQL connection.
        name: PostgreSQL index name; must be a lowercase identifier up to 63 characters.
        model_id: UUID of the row in ``tcm.embedding_models``.
        dimensions: Expected vector dimension for the selected embedding model.

    Raises:
        ValueError: If the index name is invalid, dimensions are non-positive, or
            the model does not exist / has a different registered dimension.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", name):
        raise ValueError("index name must be a lowercase SQL identifier up to 63 characters")
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    row = connection.execute("SELECT dimensions FROM tcm.embedding_models WHERE id=%s", (model_id,)).fetchone()
    if not row or row[0] != dimensions:
        raise ValueError("embedding model is missing or its dimension does not match --dimensions")
    statement = sql.SQL("""CREATE INDEX IF NOT EXISTS {name} ON tcm.chunk_embeddings
        USING hnsw ((embedding::vector({dimensions})) vector_cosine_ops)
        WITH (m=16, ef_construction=64) WHERE embedding_model_id={model_id}""").format(
        name=sql.Identifier(name), dimensions=sql.Literal(dimensions), model_id=sql.Literal(model_id))
    connection.execute(statement)
    print(f"created or verified HNSW index {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", help="Overrides DATABASE_URL from .env")
    parser.add_argument("--dry-run", action="store_true", help="Print migration checksums without connecting")
    parser.add_argument("--create-hnsw-index", action="store_true")
    parser.add_argument("--index-name")
    parser.add_argument("--embedding-model-id")
    parser.add_argument("--dimensions", type=int)
    args = parser.parse_args()
    if args.dry_run:
        for item in MIGRATIONS:
            print(item.version, item.description, checksum(item.statement))
        return
    if args.create_hnsw_index and not all((args.index_name, args.embedding_model_id, args.dimensions)):
        parser.error("--create-hnsw-index requires --index-name, --embedding-model-id, and --dimensions")
    load_dotenv()
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is required in .env, environment, or --database-url")
    # autocommit keeps ordinary statements simple; apply_migrations explicitly
    # opens its own transaction for the migration-critical section.
    with psycopg.connect(database_url, autocommit=True) as connection:
        apply_migrations(connection)
        if args.create_hnsw_index:
            create_hnsw_index(connection, args.index_name, args.embedding_model_id, args.dimensions)


if __name__ == "__main__":
    main()
