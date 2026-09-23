# TCM RAG: 第 1 步数据审计

本仓库当前只实现 `TCM-preprocess.md` 所述的第 1 步：对 `data/raw/` 中已筛选的资料生成可复现、可追溯的审计记录。原始文件不会被改写。

运行：

```bash
python3 src/audit/build_audit.py
```

产物：

- `data/audit/documents.jsonl`：一行一个文件的 JSON 审计记录。
- `data/audit/report.json`：总量、类型、元数据覆盖率、质量异常和重复记录统计。

记录包含原始和规范化文本的 SHA-256、书名/作者/年代/目录分类、卷数、编码与 OCR 风险，以及 `field_sources` 和 `classification.evidence`。因此每个非空元数据值都能追溯到文件头、目录式文件名或明确的规则；不能可靠提取的值为 `null`。

`text_type` 的规则优先级为：明确的现代年份、现代书名信号、参考书信号、医案信号、注释信号，再到古籍默认值。默认值的置信度是 `low`，方便在人工复核或后续 LLM 复核时优先处理，而不是把规则结果伪装成书目事实。

重新运行会完整重建审计结果，适合以哈希做后续增量处理。JSONL 使用稳定的键排序；生成时间只存在于统计报告中。

## 第 2 步：古籍结构化与 Markdown

默认解析审计结果中 `ancient_classical`、`commentary` 和 `medical_case` 三类资料：

```bash
python3 src/parser/structure_parser.py
```

输出在 `data/structured/`：`documents.jsonl` 保存统一 AST（每个节点都有原文字符、行号范围和解析证据），`markdown/<document_id>.md` 是排版规范化后的 Markdown。解析会合并不构成段落的换行、移除 tcmoc 中可见的 `\\x` / `\\n` 等布局转义标记，并清除汉字之间的异常空格；不会改写词句。

规则识别 `<目录>`、`<篇名>`、卷、篇章和节标题。仅在规则未识别的短行需要复核时，才可选择开启 LLM 兜底：

```bash
pip install -r requirements.txt
python3 src/parser/structure_parser.py --llm-fallback
```

LLM 经 LangChain 调用，只能从候选行中返回节点类型与层级；其 JSON 响应会经过枚举与行号校验，不能改写正文或覆盖规则标题。使用 `--include-all` 可把现代资料和参考书一并解析。

结果：699/699 文件均有稳定 document_id、原始与规范化 SHA-256、标题和分类证据。初步分类为古籍 611、注释本 36、医案 33、现代资料 14、参考资料 5；未发现完全重复文件。37 份被标记为疑似 OCR 问题，进入 needs_review。

## 第 3 步：原文、规范化原文与现代译文

翻译器读取结构化 JSONL 的段落节点，逐条输出原文、规范化原文、现代汉语译文、结构路径、审计元数据与翻译 provenance；不会覆盖第 1、2 步产物。

在 `.env` 设置 `OPENAI_API_KEY`、可选的 `OPENAI_BASE_URL` 与 `TRANSLATION_LLM_MODEL`，并安装依赖。建议先限制为小样本：

```bash
python3 src/translation/translate_structured.py --max-units 20 --fail-fast
```

全量运行：

```bash
python3 src/translation/translate_structured.py
```

结果写入 `data/enriched/translations.jsonl`。每批按规范化文本总长度（默认 8,000 字符）提交；模型须返回原样的段落 ID 且顺序完全一致，否则该批会被拒绝。再次运行会根据 `unit_id + normalized_text_sha256` 跳过已完成译文，避免重复调用。

