# TCM RAG 与 LangGraph 健康知识 Agent

本项目包含中医语料审计、结构化、翻译、PostgreSQL/pgvector 导入、混合 RAG，以及一个可调试的 LangGraph 健康知识 Agent。

Agent 的定位是“健康咨询 / 中医知识辅助”，不是自动诊断或自动开方系统。它会保存用户明确提供的基本情况，以本地 JSON 作为唯一用户档案来源；会经过风险分流后检索古籍，并由 LLM 决定是否需要回到原始语料文件核对命中位置。默认输出完整调试 JSON。

## 目录与主要入口

```text
src/
├── agent/
│   ├── health_agent.py       # LangGraph 状态机与 CLI
│   ├── profile_store.py      # 本地 JSON 档案、校验、原子写入
│   └── source_reader.py      # 受限的原始文件片段读取
├── rag/answer.py             # 混合检索、RRF、重排和独立 RAG CLI
├── audit/build_audit.py
├── parser/structure_parser.py
├── translation/translate_structured.py
└── database/
    ├── migrate.py
    ├── import_translations.py
    └── inspect_database.py
```

## 安装与配置

```bash
pip install -r requirements.txt
cp .env.example .env
```

至少配置：

```dotenv
OPENAI_API_KEY=...
RAG_CHAT_MODEL=...
DATABASE_URL=postgresql://...
RAG_EMBEDDING_MODEL_ID=<tcm.embedding_models 中的 UUID>
```

默认使用本地 BGE-M3。`RAG_EMBEDDING_DIMENSIONS`、是否归一化和最大长度必须与入库向量时完全一致。使用 OpenAI 兼容服务时，可额外设置 `OPENAI_BASE_URL`。

## 运行 Agent

### 终端交互界面（TUI）

安装上述依赖后运行：

```bash
python use_agent.py
```

在输入框中输入问题，按 Enter 或点击“提问”。左侧以 Markdown 展示最终回答，右侧展示可展开的 RAG 来源；窄终端会自动改为上下排列。已引用的条目默认展开，包含书名、章节、原文/译文、来源文件、位置和片段。`[S1]` 等编号严格对应实际送入模型的上下文（包含相邻片段），`[R1]` 等对应原始文件回读，并显示读取成功或失败状态。检索片段沿用调试输出长度限制，可能被截断。

请求在后台执行，界面显示等待时间，完成后显示关键词、来源数和 LLM 调用次数。请求失败会保留输入，便于检查配置后重试；未检索、未命中和无上下文来源会分别提示。按 `Ctrl+L` 聚焦输入框，按 `Ctrl+Q` 退出；可以用 Tab 切换焦点，滚动查看长回答和来源。

默认读取仓库根目录的 `.env`，沿用 `use_agent.py` 原有的模型配置；可通过 `TCM_PROFILE_FILE` 和 `TCM_DATA_ROOT` 指定档案和语料目录。每次提交都是独立问题，继续使用本地用户档案，不携带上一轮聊天记录。

### 命令行与 Python API

请从仓库根目录使用模块方式运行：

```bash
python -m src.agent.health_agent \
  "我42岁，女，有高血压。最近想查古籍里对夜间出汗怎么说。"
```

第一次运行时，如果 `data/user/profile.json` 不存在，会自动创建默认档案。每次请求都会重新读取该文件；本轮明确提供的新信息经结构化校验后原子写回。未提及的字段不会被当成 `false`，例如未知妊娠状态保持 `unknown`。

只看最终回答：

```bash
python -m src.agent.health_agent \
  "《伤寒论》中发热、恶风、有汗如何描述？" \
  --final-only
```

使用另一份独立档案：

```bash
python -m src.agent.health_agent \
  "我对青霉素过敏，请记住" \
  --profile-file data/user/demo-profile.json
```

知识检索示例：

```bash
python -m src.agent.health_agent \
  "《本草纲目》如何论述人参？"
```

风险分流示例：

```bash
python -m src.agent.health_agent \
  "我胸口剧痛而且出冷汗，想先按中医辨证看看"
```

这类命中确定性红旗规则的请求会在风险分流后终止普通 RAG/建议流程，并优先提示紧急医疗评估。

Python 调用示例见 [examples/use_agent.py](examples/use_agent.py)，可用 `python -m examples.use_agent` 执行。

## LangGraph 流程

```text
START
  → prepare（读取档案、本地红旗规则）
  → analyze（一次 LLM：意图、档案更新、风险、检索计划）
  → retrieve（按意图选择，直接复用检索计划）
  → respond（一次 LLM：直接回答，或请求回读原文）
      ├─ 直接回答 → END
      └─ read_originals → respond（必须最终回答）→ END

prepare / analyze 判断为紧急风险 → urgent_response（固定提示）→ END
```

常规请求默认只调用 **2 次 LLM**。意图识别、档案抽取、风险分流和关键词/过滤条件分析在第一次调用中完成；第二次调用基于 RAG 证据回答，并同时决定是否有必要回读原文。健康咨询和知识查询即使没有生成检索计划，代码也会用用户原问题构建备用计划并执行检索。

Agent 默认使用 RRF 排序，不额外调用 LLM 重排，也不再单独调用病例分析、建议规划、原文决策、校验和修订模型。独立 RAG CLI 的默认行为保持不变。

| 请求路径 | 默认 LLM 调用次数 |
|---|---:|
| 健康咨询 / 知识查询 / 档案管理 / 寒暄 | 2 |
| 回读原始文件后再回答 | 3 |
| 本地红旗规则直接拦截 | 0 |
| 首次分析识别紧急风险后拦截 | 1 |

如需要 LLM 重排，可显式开启，成功召回候选后额外调用一次：

```bash
python -m src.agent.health_agent "《本草纲目》如何论述人参？" --rerank
```

Python API 对应设置 `AgentSettings(..., use_reranker=True)`。`--no-rerank` 仍然兼容，且现在是默认行为。`llm_call_count` 记录应用层聊天模型调用次数（包含显式重排），不包含 embedding 计算或 SDK 网络重试。

### 原始文件按需读取

RAG 命中包含 `source_file`、节点字符范围和行号。回答模型可以从排序后的候选 `chunk_id` 中选择最多 3 个；只有在需要核对逐字引用、扩展上下文或处理译文歧义时才读取。原文回读最多一轮，读取后必须完成回答，读取失败时也不会再次请求。读取器会：

- 将路径限制在 `TCM_DATA_ROOT`，拒绝绝对路径和目录穿越；
- 优先按 `source_start/source_end` 读取；
- 其次按行号或原文精确定位；
- 每个片段限制长度，不会默认把整本古籍送进上下文；
- 把读取成功、编码、范围或失败原因写入调试输出。

原始文件片段带有独立的 `source_label`，例如 `R1`，最终回答可以用 `[R1]` 引用；RAG 上下文继续使用 `[S1]` 等编号。

## 调试输出

默认 stdout 是一个 JSON 对象，主要字段如下：

```json
{
  "profile": {
    "file": ".../data/user/profile.json",
    "before": {},
    "after": {},
    "changed_fields": ["age", "biological_sex"]
  },
  "intent": {
    "intent": "health_consultation",
    "should_retrieve": true
  },
  "risk_assessment": {},
  "keywords": ["盗汗", "夜间汗出"],
  "query_plan": {},
  "rag": {
    "status": "completed",
    "ranking_method": "rrf",
    "channels": {
      "lexical": [],
      "original_vector": [],
      "modern_vector": []
    },
    "fused": [],
    "ranked": [],
    "context_sources": []
  },
  "original_source_read": {
    "decision": {},
    "excerpts": []
  },
  "llm_call_count": 2,
  "llm_outputs": {
    "analysis": {},
    "responses": [
      {
        "action": "answer",
        "final_answer": "...",
        "chunk_ids": [],
        "reason": "已有证据足够"
      }
    ]
  },
  "graph_trace": [],
  "final_answer": "..."
}
```

三路召回各自保留，方便验证关键词、向量召回和排序。`llm_outputs.analysis` 保存首次合并分析的模型结果；`responses` 保存回答模型的结构化输出，回读时有两条记录。原来的病例候选、建议规划和独立校验输出字段已移除。跳过检索的分支仍返回完整的空 `rag` 对象，可以安全访问 `rag.ranked`。

为控制日志体积，调试 JSON 中每个 chunk 文本默认截到 1,200 字符；真正传给生成模型的 `rag_context` 仍受 `--context-chars` 独立控制。

## 用户档案

自动创建的原始 JSON 形态为：

```json
{
  "schema_version": "1.0",
  "user_id": "local-user",
  "updated_at": null,
  "name": null,
  "age": null,
  "biological_sex": "unknown",
  "height_cm": null,
  "weight_kg": null,
  "pregnancy_status": "unknown",
  "medical_history": [],
  "current_medications": [],
  "allergies": [],
  "tcm_context": {
    "sleep": null,
    "appetite": null,
    "stool": null,
    "urination": null,
    "cold_heat": null,
    "sweating": null,
    "tongue": null,
    "pulse": null
  }
}
```

写入采用临时文件 + `os.replace`，最终文件权限设为 `0600`。该实现适合单机单用户原型；生产环境仍需补充加密、身份认证、并发控制、保留周期和删除机制。

## 独立运行混合 RAG

```bash
python -m src.rag.answer \
  "《伤寒论》里发热、恶风、有汗如何描述？" \
  --retrieve-only
```

检索流程为：查询分析 → lexical/original vector/modern vector 三路召回 → RRF → reranker → 相邻上下文扩展。`src/rag/answer.py` 也导出 `retrieve(...)`，供 Agent 复用同一实现。

## 数据处理流程

### 1. 审计

```bash
python src/audit/build_audit.py
```

生成 `data/audit/documents.jsonl` 和 `data/audit/report.json`，不会修改 `data/raw/`。

### 2. 结构解析

```bash
python src/parser/structure_parser.py
```

可选 LLM 标题兜底：

```bash
python src/parser/structure_parser.py --llm-fallback
```

### 3. 翻译与规范化

小样本：

```bash
python src/translation/translate_structured.py --max-units 20 --fail-fast
```

全量：

```bash
python src/translation/translate_structured.py
```

### 4. PostgreSQL + pgvector

预览并执行追加式迁移：

```bash
python src/database/migrate.py --dry-run
python src/database/migrate.py
```

导入翻译产物：

```bash
python src/database/import_translations.py
```

为一个 embedding 模型创建 HNSW 索引：

```bash
python src/database/migrate.py \
  --create-hnsw-index \
  --index-name bge_m3_1024_cosine_hnsw \
  --embedding-model-id e71f04de-a279-4aff-b017-011c2e6e4137 \
  --dimensions 1024
```

## 当前安全边界与未实现项

- 确定性红旗规则和 LLM 风险分流都在任何古籍建议之前执行。
- 当前仓库没有经过审核的现代安全知识库、药物-中药相互作用库或剂量库，回答提示词仍限制个体化方剂、剂量、服法和配药指令，只允许无剂量的历史知识说明。
- 精简流程已移除独立的 LLM 校验和修订节点；当前回答不经过第二模型审核。原文读取最多一轮，避免无限循环。
- 本次没有实现文档第 14 节“Agent 评测体系”和第 15 节“专门安全测试集”；也没有加入相关指标、数据集或评测脚本。
