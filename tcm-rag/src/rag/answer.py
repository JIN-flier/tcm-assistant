#!/usr/bin/env python3
"""Read-only hybrid RAG over the TCM PostgreSQL/pgvector corpus."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import psycopg


ANALYZER_PROMPT = "Analyze this Chinese TCM-text question for retrieval only. Return a concise search query, semantic paraphrase, keywords, and only explicit metadata filters. Set absent scalar filters to null and absent lists to []. Do not answer or diagnose."
RERANKER_PROMPT = "Rank supplied TCM passages for the question. Return only supplied IDs in relevance order; do not answer or diagnose."
ANSWER_PROMPT = "Answer only from supplied TCM-text passages. Cite factual claims as [S1], [S2], etc. Say when context is insufficient. Do not diagnose, prescribe, or present historical text as personal medical advice."


class Filters(BaseModel):
    title: str | None = Field(...)
    author: str | None = Field(...)
    dynasty: str | None = Field(...)
    category_code: str | None = Field(...)
    text_types: list[str] = Field(...)


class QueryPlan(BaseModel):
    search_query: str
    semantic_query: str
    keywords: list[str] = Field(...)
    query_type: str = Field(...)
    filters: Filters = Field(...)


class RankedID(BaseModel):
    chunk_id: str
    reason: str


class RerankResult(BaseModel):
    ranked: list[RankedID]


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    node_id: str
    sequence: int
    title: str
    node_title: str | None
    text_kind: str
    text: str
    score: float
    channels: list[str] = field(default_factory=list)
    rrf_score: float = 0.0
    rerank_reason: str | None = None


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def chat(model_name: str, schema: type[BaseModel] | None = None):
    from langchain_openai import ChatOpenAI
    kwargs: dict[str, Any] = {"model": model_name, "api_key": os.environ["OPENAI_API_KEY"], "temperature": 0}
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    instance = ChatOpenAI(**kwargs)
    return instance.with_structured_output(schema, method="json_schema") if schema else instance


class QueryEmbedder(Protocol):
    """Minimal interface shared by remote and local embedding backends."""

    def embed_query(self, text: str) -> list[float]: ...


class FlagEmbeddingEmbedder:
    """Embed queries locally with BGE-M3's native FlagEmbedding interface."""

    def __init__(self, model_name_or_path: str, normalize: bool) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as error:
            raise RuntimeError(
                "Local BGE-M3 embeddings require FlagEmbedding. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from error
        self.model = BGEM3FlagModel(
            model_name_or_path,
            normalize_embeddings=normalize,
            use_fp16=os.getenv("RAG_EMBEDDING_USE_FP16", "false").lower() in {"1", "true", "yes"},
        )
        self.max_length = int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", "512"))

    def embed_query(self, text: str) -> list[float]:
        # BGE-M3 does not require a retrieval instruction.  Request only the
        # dense output: the database stores one pgvector per chunk.
        result = self.model.encode(
            [text], batch_size=1, max_length=self.max_length,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        return result["dense_vecs"][0].tolist()


def embeddings(model_name: str, backend: str) -> QueryEmbedder:
    """Build the configured embedding backend without changing stored vectors."""
    if backend == "flag_embedding":
        return FlagEmbeddingEmbedder(
            model_name,
            normalize=os.getenv("RAG_EMBEDDING_NORMALIZE", "true").lower() not in {"0", "false", "no"},
        )
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs: dict[str, Any] = {"model": model_name, "api_key": os.environ["OPENAI_API_KEY"]}
        if base_url := os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)
    raise ValueError("RAG_EMBEDDING_BACKEND must be 'flag_embedding' or 'openai'")


def analyze(question: str, model_name: str) -> QueryPlan:
    result = chat(model_name, QueryPlan).invoke([("system", ANALYZER_PROMPT), ("human", question)])
    return result if isinstance(result, QueryPlan) else QueryPlan.model_validate(result)


def metadata_filters(filters: Filters) -> tuple[str, list[Any]]:
    clauses, params = [], []
    for column, value in (("canonical_title", filters.title), ("author", filters.author), ("dynasty", filters.dynasty), ("category_code", filters.category_code)):
        if value:
            clauses.append(f"d.{column} ILIKE %s")
            params.append(f"%{value}%")
    if filters.text_types:
        clauses.append("d.text_type = ANY(%s)")
        params.append(filters.text_types)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def to_hits(rows: Iterable[tuple[Any, ...]], channel: str) -> list[Hit]:
    return [Hit(str(row[0]), str(row[1]), str(row[2]), row[3], row[4], row[5], row[6], row[7], float(row[8]), [channel]) for row in rows]


class Retriever:
    def __init__(self, connection: psycopg.Connection, model_id: str, dimensions: int) -> None:
        self.connection, self.model_id, self.dimensions = connection, model_id, dimensions

    @staticmethod
    def select(score: str, joins: str, where: str) -> str:
        return f"""SELECT c.id, dv.document_id, c.node_id, c.sequence, d.canonical_title, n.title,
            nt.text_kind, c.text, {score}
            FROM tcm.chunks c JOIN tcm.document_versions dv ON dv.id=c.document_version_id
            JOIN tcm.documents d ON d.id=dv.document_id JOIN tcm.nodes n ON n.id=c.node_id
            JOIN tcm.node_texts nt ON nt.id=c.source_text_id {joins} WHERE {where}"""

    def lexical(self, plan: QueryPlan, limit: int) -> list[Hit]:
        terms = [item for item in dict.fromkeys([*plan.keywords, plan.search_query]) if len(item.strip()) >= 2]
        if not terms:
            return []
        extra, extra_params = metadata_filters(plan.filters)
        # A short Chinese term (for example, "人参") has very low trigram
        # similarity to a full paragraph even when it occurs verbatim.  Keep
        # trigram search as a fuzzy fallback, but make literal occurrences a
        # first-class retrieval condition and rank them ahead of fuzzy hits.
        exact_match = " OR ".join("strpos(c.text, %s) > 0" for _ in terms)
        fuzzy_match = " OR ".join("c.text %% %s" for _ in terms)
        exact_score = "CASE WHEN (" + exact_match + ") THEN 1 ELSE 0 END"
        fuzzy_score = "GREATEST(" + ",".join("similarity(c.text, %s)" for _ in terms) + ")"
        score = f"({exact_score}) + ({fuzzy_score})"
        statement = self.select(score, "", f"(({exact_match}) OR ({fuzzy_match})){extra} ORDER BY 9 DESC LIMIT %s")
        params = [*terms, *terms, *terms, *terms, *extra_params, limit]
        return to_hits(self.connection.execute(statement, params).fetchall(), "lexical")

    def vector(self, vector: list[float], plan: QueryPlan, embedding_type: str, text_kind: str, channel: str, limit: int) -> list[Hit]:
        extra, extra_params = metadata_filters(plan.filters)
        statement = self.select("1 - (ce.embedding <=> %s::vector)", "JOIN tcm.chunk_embeddings ce ON ce.chunk_id=c.id", """ce.embedding_model_id=%s AND ce.embedding_type=%s AND nt.text_kind=%s
            AND vector_dims(ce.embedding)=%s""" + extra + " ORDER BY ce.embedding <=> %s::vector LIMIT %s")
        literal = vector_literal(vector)
        params: list[Any] = [literal, self.model_id, embedding_type, text_kind, self.dimensions, *extra_params, literal, limit]
        return to_hits(self.connection.execute(statement, params).fetchall(), channel)

    def neighbors(self, hit: Hit, radius: int = 1) -> list[Hit]:
        statement = self.select("1.0", "", "c.node_id=%s AND nt.text_kind=%s AND c.sequence BETWEEN %s AND %s ORDER BY c.sequence")
        return to_hits(self.connection.execute(statement, (hit.node_id, hit.text_kind, max(0, hit.sequence-radius), hit.sequence+radius)).fetchall(), "context")


def rrf(result_lists: list[list[Hit]], rrf_k: int = 60, limit: int = 40) -> list[Hit]:
    merged: dict[str, Hit] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, 1):
            existing = merged.setdefault(hit.chunk_id, hit)
            existing.rrf_score += 1 / (rrf_k + rank)
            for channel in hit.channels:
                if channel not in existing.channels:
                    existing.channels.append(channel)
    return sorted(merged.values(), key=lambda item: item.rrf_score, reverse=True)[:limit]


def rerank(question: str, hits: list[Hit], model_name: str, limit: int) -> list[Hit]:
    payload = [{"chunk_id": item.chunk_id, "title": item.title, "section": item.node_title, "text": item.text} for item in hits]
    result = chat(model_name, RerankResult).invoke([("system", RERANKER_PROMPT), ("human", json.dumps({"question": question, "candidates": payload}, ensure_ascii=False))])
    result = result if isinstance(result, RerankResult) else RerankResult.model_validate(result)
    by_id, output = {item.chunk_id: item for item in hits}, []
    for choice in result.ranked:
        if item := by_id.get(choice.chunk_id):
            item.rerank_reason = choice.reason
            output.append(item)
        if len(output) == limit:
            break
    return output or hits[:limit]


def context(retriever: Retriever, hits: list[Hit], maximum_chars: int) -> tuple[str, list[Hit]]:
    chosen, seen, size = [], set(), 0
    for hit in hits:
        for item in [hit, *retriever.neighbors(hit)]:
            if item.chunk_id in seen:
                continue
            rendered = f"[S{len(chosen)+1}] 《{item.title}》" + (f" · {item.node_title}" if item.node_title else "") + f"\n{item.text}\n"
            if size + len(rendered) > maximum_chars:
                continue
            chosen.append(item); seen.add(item.chunk_id); size += len(rendered)
    return "".join(f"[S{i}] 《{item.title}》" + (f" · {item.node_title}" if item.node_title else "") + f"\n{item.text}\n" for i, item in enumerate(chosen, 1)), chosen


def answer(question: str, passages: str, model_name: str) -> str:
    response = chat(model_name).invoke([("system", ANSWER_PROMPT), ("human", f"Question:\n{question}\n\nSources:\n{passages}")])
    return response.content if isinstance(response.content, str) else str(response.content)


def main() -> None:
    # Load .env before parser defaults are evaluated.  Previously variables in
    # .env did not reach the RAG_* argument defaults unless also exported.
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--database-url")
    parser.add_argument("--embedding-model-id", default=os.getenv("RAG_EMBEDDING_MODEL_ID"))
    parser.add_argument("--embedding-dimensions", type=int, default=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1536")))
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-backend", choices=("flag_embedding", "openai"), default=os.getenv("RAG_EMBEDDING_BACKEND", "flag_embedding"))
    parser.add_argument("--chat-model", default=os.getenv("RAG_CHAT_MODEL", "gpt-5-mini"))
    parser.add_argument("--retrieval-limit", type=int, default=30)
    parser.add_argument("--fusion-limit", type=int, default=40)
    parser.add_argument("--rerank-limit", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=12000)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()
    if not args.embedding_model_id or not (args.database_url or os.environ.get("DATABASE_URL")) or not os.environ.get("OPENAI_API_KEY"):
        parser.error("DATABASE_URL, OPENAI_API_KEY, and RAG_EMBEDDING_MODEL_ID are required")
    plan = analyze(args.question, args.chat_model)
    query_vector = embeddings(args.embedding_model, args.embedding_backend).embed_query(plan.semantic_query)
    if len(query_vector) != args.embedding_dimensions:
        parser.error(
            f"embedding dimension mismatch: model returned {len(query_vector)}, "
            f"but --embedding-dimensions is {args.embedding_dimensions}. "
            "It must match the vectors already stored in tcm.chunk_embeddings."
        )
    database_url = args.database_url or os.environ["DATABASE_URL"]
    with psycopg.connect(database_url, autocommit=True) as connection:
        retriever = Retriever(connection, args.embedding_model_id, args.embedding_dimensions)
        lists = [retriever.lexical(plan, args.retrieval_limit),
                 retriever.vector(query_vector, plan, os.getenv("RAG_ORIGINAL_EMBEDDING_TYPE", "original"), "original", "original_vector", args.retrieval_limit),
                 retriever.vector(query_vector, plan, os.getenv("RAG_MODERN_EMBEDDING_TYPE", "modern"), "modern_translation", "modern_vector", args.retrieval_limit)]
        fused = rrf(lists, limit=args.fusion_limit)
        ranked = fused[:args.rerank_limit] if args.no_rerank else rerank(args.question, fused, args.chat_model, args.rerank_limit)
        passages, sources = context(retriever, ranked, args.context_chars)
    result: dict[str, Any] = {"question": args.question, "query_plan": plan.model_dump(), "sources": [asdict(item) for item in sources]}
    result["answer"] = None if args.retrieve_only else answer(args.question, passages, args.chat_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
