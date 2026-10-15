"""Presentation data for the TUI; citation numbering follows RAG context()."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceCard:
    label: str
    title: str
    metadata: str
    text: str
    cited: bool


def source_cards(result: dict[str, Any]) -> list[SourceCard]:
    cited = set(re.findall(r"\[([SR]\d+)\]", result.get("final_answer", "")))
    cards = []
    # ranked may include candidates omitted by the context budget, and lacks
    # neighbors added to the context. Never assign S labels from ranked.
    for index, source in enumerate((result.get("rag") or {}).get("context_sources", []), 1):
        label = f"S{index}"
        kind = {"original": "古籍原文", "modern_translation": "现代译文"}.get(
            source.get("text_kind"), source.get("text_kind") or "检索片段"
        )
        details = [kind, f"文件：{source.get('source_file') or '未提供'}"]
        if source.get("line_start") is not None:
            details.append(f"行号：{source['line_start']}–{source.get('line_end') or source['line_start']}")
        if source.get("source_start") is not None:
            details.append(f"节点字符范围：{source['source_start']}–{source.get('source_end')}")
        channels = {"lexical": "关键词", "original_vector": "原文向量", "modern_vector": "译文向量", "context": "相邻上下文"}
        if source.get("channels"):
            details.append("召回：" + " / ".join(channels.get(item, item) for item in source["channels"]))
        details.append(f"chunk_id：{source.get('chunk_id', '未知')}")
        title = f"《{source.get('title') or '未命名文献'}》"
        if source.get("node_title"):
            title += f" · {source['node_title']}"
        text = source.get("text") or "（无片段文本）"
        if text.endswith("…"):
            text += "\n\n（此处展示的检索片段可能已被调试输出长度限制截断。）"
        cards.append(SourceCard(label, title, "\n".join(details), text, label in cited))

    for source in (result.get("original_source_read") or {}).get("excerpts", []):
        label = source["source_label"]
        details = ["原始文件回读", f"文件：{source.get('source_file') or '未提供'}"]
        if source.get("status") == "read":
            details.append(f"读取成功 · 编码：{source.get('encoding') or '未知'}")
            details.append(f"回读字符范围：{source.get('start')}–{source.get('end')}（左闭右开）")
            text = source.get("text") or "（空片段）"
        else:
            details.append("读取失败")
            text = source.get("error") or "没有可用的原文片段。"
        cards.append(SourceCard(label, f"《{source.get('title') or '未命名文献'}》", "\n".join(details), text, label in cited))
    return cards


def empty_sources_message(result: dict[str, Any]) -> str:
    rag = result.get("rag") or {}
    if rag.get("status", "skipped") == "skipped":
        return "本次请求未执行 RAG 检索，没有可展示的文献来源。"
    if rag.get("ranked"):
        return "检索到了候选，但没有片段进入回答上下文，因此没有 [S] 来源。"
    return "本次检索未找到可用文献。"
