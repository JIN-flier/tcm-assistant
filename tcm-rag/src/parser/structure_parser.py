#!/usr/bin/env python3
"""Parse audited classical texts into a source-traceable AST and Markdown.

Rules create all default structure.  ``--llm-fallback`` is opt-in and only asks
an LLM to label unresolved short candidate lines; it never sends a request to
rewrite the source text and cannot alter rule-recognised headings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HEADING_MARKER = re.compile(r"^<(目录|篇名)>\s*(.*?)\s*$")
VOLUME = re.compile(r"^(?:卷(?:第)?[一二三四五六七八九十百千〇零\d]+|第[一二三四五六七八九十百千〇零\d]+卷)")
CHAPTER = re.compile(r"^(?:第[一二三四五六七八九十百千〇零\d]+[篇章节].*|.+?篇第[一二三四五六七八九十百千〇零\d]+|.+?论)$")
SECTION = re.compile(r"^(?:【.+】|[一二三四五六七八九十]+、.+|第[一二三四五六七八九十百千〇零\d]+节.+)$")
SENTENCE_END = re.compile(r"[。！？；：，、]$")
BIBLIOGRAPHIC_METADATA = re.compile(r"^(?:书名|作者|朝代|年份)[：:]")
ATTRIBUTE_PREFIX = re.compile(r"^属性[：:]\s*")
CJK_SPACE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")

NODE_RANK = {"volume": 1, "chapter": 2, "section": 3, "subsection": 4, "entry": 5}
DEFAULT_TEXT_TYPES = {"ancient_classical", "commentary", "medical_case"}


@dataclass
class SourceLine:
    number: int
    start: int
    end: int
    text: str


def clean_line(value: str) -> str:
    """Perform only layout normalisation; do not change Chinese wording."""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\\r\\n", " ").replace("\\n", " ").replace("\\t", " ")
    value = value.replace("\\x", "")  # tcmoc's visible editorial boundary marker
    value = re.sub(r"[ \t]+", " ", value).strip()
    return CJK_SPACE.sub("", value)


def source_lines(text: str) -> list[SourceLine]:
    result: list[SourceLine] = []
    offset = 0
    for number, raw in enumerate(text.splitlines(keepends=True), start=1):
        without_newline = raw.rstrip("\r\n")
        result.append(SourceLine(number, offset, offset + len(without_newline), clean_line(without_newline)))
        offset += len(raw)
    if not result:
        result.append(SourceLine(1, 0, 0, ""))
    return result


def heading_from_rule(line: str) -> tuple[str, str, int, str] | None:
    """Return node type, title, rank and evidence for deterministic headings."""
    marker = HEADING_MARKER.match(line)
    if marker:
        title = marker.group(2).strip()
        if not title:
            return None
        if VOLUME.match(title):
            return "volume", title, 1, f"<{marker.group(1)}> volume pattern"
        return "chapter", title, 2, f"<{marker.group(1)}> marker"
    if VOLUME.match(line):
        return "volume", line, 1, "plain volume pattern"
    if CHAPTER.match(line) and len(line) <= 40:
        return "chapter", line, 2, "plain chapter pattern"
    if SECTION.match(line) and len(line) <= 80:
        return "section", line, 3, "plain section pattern"
    return None


def unresolved_candidates(lines: list[SourceLine]) -> list[dict[str, Any]]:
    """Find conservative possible headings, for opt-in LLM review only."""
    candidates = []
    for index, line in enumerate(lines):
        text = line.text
        if not text or BIBLIOGRAPHIC_METADATA.match(text) or ATTRIBUTE_PREFIX.match(text) or heading_from_rule(text):
            continue
        adjacent_blank = (index == 0 or not lines[index - 1].text) or (index + 1 == len(lines) or not lines[index + 1].text)
        if adjacent_blank and 2 <= len(text) <= 50 and not SENTENCE_END.search(text):
            candidates.append({"line_number": line.number, "text": text})
    return candidates


def llm_heading_decisions(candidates: list[dict[str, Any]], model: str) -> dict[int, tuple[str, int]]:
    """Use LangChain only for unrecognised candidates and validate its JSON."""
    if not candidates:
        return {}
    try:
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError("LLM fallback needs `pip install -r requirements.txt`.") from error
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("LLM fallback requested but OPENAI_API_KEY is not set in .env or the environment.")
    payload = json.dumps(candidates, ensure_ascii=False)
    prompt = """You are a conservative classical-Chinese document-structure classifier.
Classify only the supplied candidate source lines. Do not rewrite, correct, translate,
or invent text. Return JSON only: an array of objects with line_number, type, and rank.
type must be one of volume, chapter, section, subsection, entry, or none. rank must be
1-5 and must match volume=1, chapter=2, section=3, subsection=4, entry=5; use none
when uncertain. Candidates:\n""" + payload
    kwargs: dict[str, Any] = {"model": model, "temperature": 0}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    print("LLM parsing ...")
    response = ChatOpenAI(**kwargs).invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("LLM fallback did not return valid JSON; no decisions were applied.") from error
    allowed = {item["line_number"] for item in candidates}
    decisions: dict[int, tuple[str, int]] = {}
    for item in result if isinstance(result, list) else []:
        line_number, node_type, rank = item.get("line_number"), item.get("type"), item.get("rank")
        if line_number in allowed and node_type in NODE_RANK and rank == NODE_RANK[node_type]:
            decisions[line_number] = (node_type, rank)
    return decisions


def node(node_id: str, node_type: str, title: str | None, line: SourceLine, evidence: str) -> dict[str, Any]:
    return {"node_id": node_id, "type": node_type, "title": title, "source_start": line.start,
            "source_end": line.end, "line_start": line.number, "line_end": line.number,
            "parser": {"method": "llm_fallback" if evidence.startswith("LLM") else "rule", "evidence": evidence}, "children": []}


def parse_document(audit: dict[str, Any], raw_dir: Path, use_llm: bool, model: str) -> tuple[dict[str, Any], str]:
    source_path = raw_dir.parent / audit["source_file"]
    print(source_path)
    raw = source_path.read_bytes()
    text = raw.decode(audit["quality"]["detected_encoding"], errors="replace")
    lines = source_lines(text)
    decisions = llm_heading_decisions(unresolved_candidates(lines), model) if use_llm else {}
    root_line = SourceLine(1, 0, len(text), audit["detected_title"])
    root = node(audit["document_id"], "document", audit["detected_title"], root_line, "audit title")
    root["source_end"] = len(text)
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    paragraph_lines: list[SourceLine] = []
    sequence = Counter()

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        body = "".join(item.text for item in paragraph_lines).strip()
        if body:
            parent = stack[-1][1]
            sequence["paragraph"] += 1
            paragraph = node(f"{audit['document_id']}_P{sequence['paragraph']:05d}", "paragraph", None,
                             paragraph_lines[0], "layout-normalized source lines")
            paragraph["source_end"] = paragraph_lines[-1].end
            paragraph["line_end"] = paragraph_lines[-1].number
            paragraph["text"] = body
            parent["children"].append(paragraph)
        paragraph_lines = []

    for line in lines:
        if not line.text:
            flush_paragraph()
            continue
        if BIBLIOGRAPHIC_METADATA.match(line.text):
            flush_paragraph()
            continue
        marker = HEADING_MARKER.match(line.text)
        if marker and not marker.group(2).strip():
            flush_paragraph()
            continue
        content_text = ATTRIBUTE_PREFIX.sub("", line.text)
        content_line = SourceLine(line.number, line.start, line.end, content_text)
        decision = heading_from_rule(content_text)
        if not decision and line.number in decisions:
            node_type, rank = decisions[line.number]
            decision = (node_type, content_text, rank, "LLM fallback, validated enum response")
        if decision:
            flush_paragraph()
            node_type, title, rank, evidence = decision
            # The source's first <篇名> normally repeats the document title.
            if title == audit["detected_title"] and not root["children"]:
                continue
            while stack and stack[-1][0] >= rank:
                stack.pop()
            parent = stack[-1][1] if stack else root
            # Some sources repeat '<目录>卷一' before each chapter. Merge only
            # immediately repeated volume headings; chapters remain distinct.
            if node_type == "volume" and parent["children"] and parent["children"][-1]["type"] == "volume" and parent["children"][-1]["title"] == title:
                child = parent["children"][-1]
                child["source_end"] = line.end
                child["line_end"] = line.number
            else:
                sequence[node_type] += 1
                child = node(f"{audit['document_id']}_{node_type[:1].upper()}{sequence[node_type]:04d}", node_type, title, line, evidence)
                parent["children"].append(child)
            stack.append((rank, child))
        else:
            paragraph_lines.append(content_line)
    flush_paragraph()

    def extend_bounds(current: dict[str, Any]) -> None:
        """A structural node spans its heading plus every descendant it owns."""
        for child in current["children"]:
            extend_bounds(child)
            current["source_end"] = max(current["source_end"], child["source_end"])
            current["line_end"] = max(current["line_end"], child["line_end"])

    extend_bounds(root)

    structured = {"schema_version": "1.0.0", "document_id": audit["document_id"], "title": audit["detected_title"],
                  "source_file": audit["source_file"], "source_sha256": audit["source_sha256"],
                  "normalization": {"joined_wrapped_lines": True, "removed_literal_escape_markers": ["\\\\x", "\\\\n", "\\\\r\\\\n", "\\\\t"],
                                    "removed_inter_cjk_spaces": True, "llm_fallback_enabled": use_llm}, "nodes": [root]}
    return structured, render_markdown(root)


def render_markdown(root: dict[str, Any]) -> str:
    output = [f"# {root['title']}"]
    def visit(current: dict[str, Any], level: int) -> None:
        for child in current["children"]:
            if child["type"] == "paragraph":
                output.extend(["", child["text"]])
            else:
                output.extend(["", "#" * min(level, 6) + " " + (child["title"] or "")])
                visit(child, level + 1)
    visit(root, 2)
    return "\n".join(output).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-file", type=Path, default=Path("data/audit/documents.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/structured"))
    parser.add_argument("--llm-fallback", action="store_true", help="Enable opt-in LangChain heading labels for unresolved candidates.")
    parser.add_argument("--llm-model", default="nvidia/nemotron-3-ultra-550b-a55b:free")
    parser.add_argument("--include-all", action="store_true", help="Also parse modern and reference records.")
    args = parser.parse_args()
    audits = [json.loads(line) for line in args.audit_file.read_text(encoding="utf-8").splitlines()]
    selected = audits if args.include_all else [item for item in audits if item["text_type"] in DEFAULT_TEXT_TYPES]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir = args.output_dir / "markdown"
    markdown_dir.mkdir(exist_ok=True)
    results = []
    for audit in selected:
        structured, markdown = parse_document(audit, args.raw_dir, args.llm_fallback, args.llm_model)
        results.append(structured)
        (markdown_dir / f"{audit['document_id']}.md").write_text(markdown, encoding="utf-8")
    with (args.output_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    node_counts: Counter[str] = Counter()
    def count_nodes(current: dict[str, Any]) -> None:
        node_counts[current["type"]] += 1
        for child in current["children"]:
            count_nodes(child)
    for result in results:
        for root in result["nodes"]:
            count_nodes(root)
    summary = {"schema_version": "1.0.0", "parsed_documents": len(results), "llm_fallback_enabled": args.llm_fallback,
               "excluded_text_types": [] if args.include_all else sorted(set(item["text_type"] for item in audits) - DEFAULT_TEXT_TYPES),
               "node_counts": dict(sorted(node_counts.items()))}
    (args.output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
