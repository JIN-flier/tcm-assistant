https://github.com/lab99x/tcmoc本身已经定义了一些元数据，但是内容比较混乱，第一步就是分类和整理元数据。因此我们的数据模型增加 RAG 所需要的结构层、文本版本和 provenance。:chatgpt-content-reference{index="0"}

整体工程建议是：

```text
tcm-rag/
├── data/
│   ├── raw/                 # tcmoc 原始数据，永不修改
│   ├── audit/               # 第1步审计结果
│   ├── structured/          # 第2步结构解析
│   ├── enriched/            # 第3步翻译、规范化、元数据
│   └── evaluation/          # 第6步评测集
│
├── src/
│   ├── audit/
│   ├── parser/
│   ├── normalization/
│   ├── translation/
│   ├── metadata/
│   ├── chunking/
│   ├── embedding/
│   ├── database/
│   ├── retrieval/
│   ├── reranker/
│   ├── generation/
│   └── evaluation/
│
├── migrations/
├── prompts/
├── tests/
└── configs/
```

---

# 第 1 步：数据审计与分类

这一阶段**不要翻译、不要 embedding、不要切固定长度 Chunk**。目标只有一个：搞清楚这 700 个文件到底是什么。

`tcmoc` 自己已经按照《中国中医古籍总目》设计了 12 个一级分类，例如医经、基础理论、伤寒金匮、本草、方书、临证各科等，并在 Issue 中已经为不少文件整理了古籍编号、类别、作者和年代，例如《本草纲目》被标为本草类、李时珍、明代。:chatgpt-content-reference{index="1"}

## 1.1 每个文件生成审计记录

先生成：

```json
{
  "source_file": "013-本草纲目.txt",
  "source_sha256": "...",

  "detected_title": "本草纲目",
  "catalog_title": "本草纲目",

  "text_type": "ancient_classical",
  "language_style": "classical_chinese",

  "author": "李时珍",
  "dynasty": "明",
  "year": 1578,

  "catalog_id": "02511.1",
  "category_code": "6.2.3",
  "category": [
    "本草",
    "综合本草",
    "明代本草"
  ],

  "character_type": "simplified",
  "has_volume": true,
  "estimated_volumes": 52,

  "quality": {
    "encoding_ok": true,
    "replacement_char_count": 0,
    "empty_line_ratio": 0.13,
    "suspected_ocr": false
  },

  "status": "accepted",
  "notes": []
}
```

## 1.2 `text_type` 一定要先分类

至少使用：

```text
ancient_classical
ancient_vernacular
modern_book
medical_case
commentary
reference
unknown
```

这样后面：

```text
ancient_classical
        ↓
需要古文翻译

modern_book
        ↓
不需要翻译
```

避免把《名老中医之路》这种现代资料也送进去做古译今。

## 1.3 分类不要完全依赖 LLM

推荐三层判定：

```text
规则
 ↓
已有 tcmoc 元数据
 ↓
LLM 辅助分类
```

例如先利用：

```text
文件名
书名
作者
朝代
《中国中医古籍总目》编号
```

能确定的就不要让 LLM 猜。

LLM 主要处理：

```text
不知道年代
不知道是古籍还是现代资料
书名和正文不一致
可能是注释本
```

## 1.4 做文件指纹

这一步非常重要。

计算：

```python
sha256(raw_bytes)
```

以及正文规范化之后：

```python
sha256(normalized_text)
```

这样以后可以识别：

```text
完全重复
轻度修改版本
同书不同版本
```

后续 pipeline 也可以根据 hash 实现增量处理。

## 1.5 第一阶段数据库先不用 PostgreSQL

先输出：

```text
data/audit/documents.jsonl
```

一行一书。

同时生成统计报告：

```text
总文件数
古籍数量
现代资料数量
无法识别数量
有作者数量
有年代数量
有卷结构数量
疑似重复数量
编码异常数量
```

### 第一步验收标准

做到：

```text
100% 文件有 document_id
100% 文件有 text_type
100% 文件有 source hash
>95% 文件能够确定书名
无法确定的字段明确为 null
所有字段都有来源，而不是静默猜测
```

---

# 第 2 步：古籍数据结构化解析

这是整个项目最重要的数据工程环节。

目标不是：

```text
TXT → 每500字切块
```

而是：

```text
TXT
 ↓
Book
 ↓
Volume
 ↓
Chapter
 ↓
Section
 ↓
Entry
 ↓
Paragraph
```

`tcmoc` 自己希望最终正文用 Markdown 标题表达结构，例如：

```markdown
# 海药本草

## 玉石部卷第一

### 玉屑
```

这正好可以成为我们的结构模型。:chatgpt-content-reference{index="2"}

## 2.1 先定义统一 AST

无论原始书格式是什么，都先转换成统一树：

```json
{
  "document_id": "TCM_000013",
  "title": "本草纲目",

  "nodes": [
    {
      "node_id": "TCM_000013_V001",
      "type": "volume",
      "title": "卷一",

      "children": [
        {
          "node_id": "TCM_000013_V001_C001",
          "type": "chapter",
          "title": "序例",

          "children": []
        }
      ]
    }
  ]
}
```

推荐 node type：

```text
document
volume
chapter
section
subsection
entry
paragraph
```

不是每本书都必须有全部层级。

比如《伤寒论》：

```text
document
└── chapter
    └── clause
```

《本草纲目》：

```text
document
└── volume
    └── category
        └── herb
            └── subsection
```

## 2.2 Parser 使用规则优先、LLM 兜底

例如匹配卷标题：

```python
VOLUME_PATTERNS = [
    r"^卷[一二三四五六七八九十百千]+",
    r"^卷第[一二三四五六七八九十]+",
    r"^第[一二三四五六七八九十]+卷"
]
```

匹配篇：

```python
r".+篇第[一二三四五六七八九十]+"
r"^辨.+病脉证并治"
```

匹配药物条目则可能结合：

```text
行长度
上下文
标题格式
本草词典
```

规则无法识别的区域再交给 LLM：

```text
“判断下面这些行中哪些是标题，
并返回标题层级，不得修改正文。”
```

LLM **只判断结构，不重写文本**。

## 2.3 永远保留字符位置

每个 Node 保存：

```json
{
  "source_start": 10331,
  "source_end": 10882
}
```

甚至保存：

```text
line_start
line_end
```

这样以后任何 Chunk 都能够追溯：

```text
数据库 Chunk
→ structured JSON
→ 原始 TXT
```

## 2.4 Chunk 在结构识别后生成

先找天然语义单元。

例如：

```text
人参
└── 主治
```

如果只有 430 字：

```text
直接作为一个 Chunk
```

如果有 3800 字：

```text
保持 subsection 不变
再按照 paragraph / sentence
切成 500~800 字
```

初始参数可以：

```yaml
chunking:
  target_chars: 600
  min_chars: 200
  max_chars: 1200
  overlap_chars: 80
```

但是有一条比长度重要：

> **不能为了凑 600 字跨结构边界。**

因此：

```text
结构边界 > 长度规则
```

## 2.5 Parent-Child 同时生成

例如：

```json
{
  "section_id": "BCGM_RENSHEN_ZHUZHI",
  "section_text": "完整主治部分……"
}
```

以及：

```json
{
  "chunk_id": "BCGM_RENSHEN_ZHUZHI_001",
  "parent_section_id": "BCGM_RENSHEN_ZHUZHI",
  "text": "其中一段……"
}
```

以后：

```text
Chunk 用于搜索
Section 用于给 LLM 阅读
```

### 第二步验收标准

拿 20 本不同类型的书人工检查：

```text
本草
伤寒
内经
方书
医案
针灸
现代资料
```

至少检查：

```text
标题有没有被吃掉
卷是否识别正确
不同药物有没有混进一个 Chunk
不同条文有没有错误合并
原始字符是否可以100%追溯
```

这一阶段宁可不识别，也不要错误识别。

---

# 第 3 步：原文 / 规范化原文 / 现代翻译 + 元数据

这一阶段输入已经不是 TXT，而是第二步的结构化节点。

核心原则：

```text
original_text 永不修改
```

然后产生两个派生文本：

```text
original_text
     │
     ├── normalized_text
     │
     └── modern_text
```

## 3.1 original_text

就是原始文本。

例如：

```text
太阳病，项背强几几，无汗恶风，葛根汤主之。
```

禁止覆盖。

---

## 3.2 normalized_text

只允许做**确定性的文字规范化**。

例如：

```text
Unicode NFC
全半角统一
异常空格
连续空行
乱码字符
标点规范
```

但是：

```text
異 → 异
醫 → 医
```

是否转换，需要谨慎。

我更建议：

```text
original_text = 原来的繁简

normalized_text = 保持原繁简，只规范字符

search_text = 可额外生成简体检索版本
```

即：

```text
original
normalized
search_normalized
modern
```

避免繁简转换反过来污染学术引用。

---

# 3.3 modern_text

只针对：

```text
ancient_classical
```

运行。

Prompt 最好固定版本：

```text
prompts/translation/v1.txt
```

模型必须严格执行：

```text
忠实翻译
不得添加原文没有的病理解释
传统中医术语尽量保留
不得自动对应现代医学诊断
人名、方名、药名保持不变
存在歧义则标记 uncertain
```

让输出采用 JSON：

```json
{
  "modern_text": "...",

  "uncertain_terms": [
    {
      "term": "几几",
      "reason": "该词存在训诂解释差异"
    }
  ],

  "translation_confidence": 0.83
}
```

不要让模型直接返回自由文本。

---

# 3.4 建议增加一个 `summary`

不是替代原文，而是帮助检索：

```json
{
  "summary": "讨论太阳病无汗、恶风并伴项背拘急时葛根汤的应用。"
}
```

以后 embedding 可以实验：

```text
modern_text

vs

title + metadata + modern_text

vs

summary + modern_text
```

---

# 3.5 Metadata 分两类

**Document metadata：**

```json
{
  "title": "伤寒论",
  "author": "张仲景",
  "dynasty": "东汉",
  "category": "伤寒金匮",
  "catalog_id": "...",
  "edition": null
}
```

**Chunk metadata：**

```json
{
  "volume": "...",
  "chapter": "辨太阳病脉证并治",
  "section": "第三十一条",
  "hierarchy_path": [
    "伤寒论",
    "辨太阳病脉证并治",
    "第三十一条"
  ]
}
```

`tcmoc` 已经提出 `title / author / era / date / version / category / chartype / tags` 这些基础字段，我们可以直接兼容。:chatgpt-content-reference{index="3"}

---

# 3.6 Entity Extraction

V1 建议只做最重要的：

```text
herb
formula
symptom
syndrome
disease
acupoint
treatment
person
```

例如：

```json
{
  "entities": [
    {
      "name": "葛根汤",
      "canonical_name": "葛根汤",
      "type": "formula"
    },
    {
      "name": "恶风",
      "type": "symptom"
    }
  ]
}
```

但是 entity extraction 不应该阻塞第一版 RAG。

可以先：

```text
结构 → 翻译 → RAG
```

之后补 Entity。

---

# 3.7 provenance 必须保存

每个机器生成字段保存：

```json
{
  "translation": {
    "model": "...",
    "prompt_version": "translation_v1",
    "generated_at": "...",
    "input_hash": "...",
    "review_status": "unreviewed"
  }
}
```

未来换模型：

```text
translation_v1
translation_v2
```

就可以重新生成，而不会影响原始数据。

### 第三步验收标准

随机抽取至少：

```text
本草 50 Chunk
伤寒 50
内经 50
方书 50
医案 50
```

重点人工检查：

```text
有没有凭空增加信息
方剂名有没有被改
药名有没有错误解释
古文否定词有没有翻错
数值/剂量有没有改变
原文和译文是否严格对应
```

---

# 第 4 步：PostgreSQL + pgvector

这里建议数据库只承担：

```text
结构化数据
文本
metadata
全文/关键词检索
vector
```

第一版不需要 Milvus。

## 4.1 Schema

核心表建议这样：

```text
documents
nodes
chunks
chunk_embeddings
entities
chunk_entities
processing_runs
```

其中 `documents`：

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    source_id TEXT UNIQUE NOT NULL,

    title TEXT NOT NULL,
    author TEXT,
    dynasty TEXT,
    year INTEGER,

    category_code TEXT,
    category_path TEXT[],

    text_type TEXT NOT NULL,
    edition TEXT,

    source_file TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,

    metadata JSONB NOT NULL DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT now()
);
```

结构节点：

```sql
CREATE TABLE nodes (
    id UUID PRIMARY KEY,

    document_id UUID NOT NULL
        REFERENCES documents(id),

    parent_id UUID
        REFERENCES nodes(id),

    node_type TEXT NOT NULL,
    title TEXT,

    level INTEGER NOT NULL,
    sequence INTEGER NOT NULL,

    source_start INTEGER,
    source_end INTEGER,

    metadata JSONB DEFAULT '{}'
);
```

Chunk：

```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY,

    document_id UUID NOT NULL
        REFERENCES documents(id),

    node_id UUID NOT NULL
        REFERENCES nodes(id),

    parent_chunk_id UUID,

    sequence INTEGER NOT NULL,

    original_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    search_text TEXT NOT NULL,
    modern_text TEXT,
    summary TEXT,

    previous_chunk_id UUID,
    next_chunk_id UUID,

    char_count INTEGER,

    metadata JSONB DEFAULT '{}'
);
```

---

# 4.2 Embedding 单独建表

不要直接：

```text
original_embedding
modern_embedding
summary_embedding
```

全塞 `chunks`。

推荐：

```sql
CREATE TABLE chunk_embeddings (
    chunk_id UUID NOT NULL
        REFERENCES chunks(id),

    embedding_type TEXT NOT NULL,

    model TEXT NOT NULL,

    dimensions INTEGER NOT NULL,

    embedding vector(1024),

    PRIMARY KEY (
        chunk_id,
        embedding_type,
        model
    )
);
```

其中：

```text
embedding_type =
original
modern
summary
```

好处是以后换 embedding model 不需要改表。

---

# 4.3 pgvector 使用 HNSW

目前 pgvector 官方同时支持 HNSW 和 IVFFlat；官方说明 HNSW 通常有更好的 speed/recall tradeoff，但构建较慢、占内存更多。对于你的数据规模，我会直接从 HNSW 开始。余弦距离对应 `vector_cosine_ops`，查询时 `<=>` 表示 cosine distance。:chatgpt-content-reference{index="4"}

例如：

```sql
CREATE INDEX chunk_embedding_hnsw
ON chunk_embeddings
USING hnsw (
    embedding vector_cosine_ops
);
```

查询：

```sql
SELECT
    chunk_id,
    1 - (embedding <=> $1) AS similarity
FROM chunk_embeddings
WHERE embedding_type = 'modern'
ORDER BY embedding <=> $1
LIMIT 30;
```

pgvector 也支持 metadata filter + vector search；不过官方提醒 ANN 检索时过滤条件可能在近邻扫描后应用，因此过滤很强时需要仔细测试召回率。:chatgpt-content-reference{index="5"}

---

# 4.4 中文“BM25”这里要特别注意

这里我会修正我们之前简化的说法：

> PostgreSQL 原生 FTS 不等于真正的中文 BM25，而且默认中文分词并不理想。

因此第一版如果坚持纯 PostgreSQL，可以做：

```text
Vector retrieval
+
Entity exact retrieval
+
中文 lexical retrieval
```

中文 lexical retrieval 有几种路线。

简单 V1：

```text
应用层分词
→ 将空格分隔后的 token 写入 search_tokens
→ PostgreSQL tsvector
```

或者使用：

```text
pg_trgm
```

做词组/原文命中。

如果以后必须要真正高质量 BM25，可以增加专门的中文全文搜索方案，而不需要改现有 PostgreSQL 数据模型。

---

# 第 5 步：实现具体 RAG

代码层我建议拆成：

```text
retrieval/
├── query_analyzer.py
├── lexical.py
├── original_vector.py
├── modern_vector.py
├── fusion.py
└── filters.py

reranker/
└── reranker.py

context/
└── builder.py

generation/
├── prompt.py
└── generator.py
```

完整请求流程：

```text
Question
   ↓
Query Analyzer
   ↓
Query Expansion
   ↓
Metadata Filters
   ↓
┌─────────────────────────┐
│ lexical retrieval       │
│ original vector         │
│ modern vector           │
└─────────────────────────┘
   ↓
RRF
   ↓
Top 40
   ↓
Reranker
   ↓
Top 8
   ↓
Context Expansion
   ↓
Dedup
   ↓
LLM
```

---

## 5.1 Query Analyzer

用户：

> 《伤寒论》里对于发热、恶风、有汗是怎么描述的？

输出：

```json
{
  "query": "发热 恶风 有汗",
  "semantic_query": "太阳病出现发热、怕风并且自汗的相关论述",

  "filters": {
    "title": "伤寒论"
  },

  "keywords": [
    "发热",
    "恶风",
    "汗出",
    "自汗"
  ],

  "entities": [],

  "query_type": "symptom_search"
}
```

这个 LLM 只进行 Query Understanding。

---

# 5.2 三路召回

第一路：

```text
lexical
```

例如：

```text
汗出
恶风
桂枝汤
```

第二路：

```text
embedding(original_text)
```

第三路：

```text
embedding(modern_text)
```

初始：

```text
lexical Top 30
original Top 30
modern Top 30
```

---

# 5.3 RRF

三个系统的 similarity score 不能直接相加。

使用：

```text
RRF(d) =
Σ 1 / (k + rank_i(d))
```

例如：

```python
def rrf(result_lists, k=60):
    scores = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, 1):
            scores.setdefault(chunk.id, 0)
            scores[chunk.id] += 1 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
```

pgvector 官方文档本身也明确提到，向量搜索与 PostgreSQL 全文搜索组合时可以使用 RRF 或 cross-encoder 做 hybrid search。:chatgpt-content-reference{index="6"}

---

# 5.4 Reranker

RRF 后：

```text
Top 40
```

交给 reranker。

输入：

```text
query
+
title
+
hierarchy_path
+
modern_text
+
original_text
```

最终：

```text
Top 6~10
```

不要让生成模型自己替代 reranker。

---

# 5.5 Parent Context Expansion

假设命中：

```text
《本草纲目》
人参
主治
chunk 2/4
```

数据库自动取：

```text
chunk 1
chunk 2
chunk 3
```

或者直接读取：

```text
parent section
```

但要限制最终 context token。

推荐：

```text
核心命中块：完整
相邻块：按需要
父标题：始终保留
```

因此每个 Context：

```text
《本草纲目》
卷十二 > 草部 > 人参 > 主治

【原文】
...

【现代释义】
...
```

---

# 5.6 去重

同一段可能：

```text
original_vector 命中
modern_vector 也命中
lexical 也命中
```

最终必须按：

```text
chunk_id
```

去重。

如果相邻 Chunk 大量重叠，还需要做 semantic overlap dedup。

---

# 5.7 Generator

最终 Prompt 不让模型自由发挥：

```text
你是中医古籍文献检索助手。

只依据提供的文献上下文回答。

规则：
1. 不得捏造文献内容。
2. 每个主要结论应注明出处。
3. 引用古籍时使用 original_text。
4. modern_text 仅用于辅助理解。
5. 原文与现代释义冲突时，以原文为准。
6. 不得将传统医学概念自动解释为现代医学诊断。
7. 找不到充分证据时明确说明。
```

最终输出：

```text
结论

……

相关古籍原文

《伤寒论·辨太阳病脉证并治》
“……”

释义

……

来源
……
```

---

# 第 6 步：通过评测集验证

这一阶段不能只评价“回答看起来不错”。

必须拆开：

```text
Retrieval Evaluation

和

Generation Evaluation
```

否则你不知道到底哪里有问题。

---

## 6.1 建立 Gold Dataset

建议先做：

```text
300 道问题
```

不是全自动生成。

可以让 LLM 生成候选，但最终人工确认：

```json
{
  "question_id": "Q001",

  "question": "项背拘急、无汗、恶风时古籍使用什么方？",

  "expected_documents": [
    "伤寒论"
  ],

  "relevant_chunks": [
    "SHL_031"
  ],

  "expected_entities": [
    "葛根汤"
  ],

  "answer_reference": "葛根汤",

  "question_type": "semantic_to_classical"
}
```

---

## 6.2 问题类型一定要分层

建议覆盖：

| 类型 | 目的 |
|---|---|
| 原文 → 来源 | 测 lexical |
| 方剂名 → 条文 | 测精确搜索 |
| 中药 → 主治 | 测实体搜索 |
| 现代白话 → 古文 | 测 modern embedding |
| 症状 → 方剂 | 测语义检索 |
| 指定书籍 | 测 metadata filter |
| 指定朝代 | 测 metadata |
| 跨书检索 | 测全库 recall |
| 易混淆问题 | 测 reranker |
| 无答案问题 | 测 hallucination |

---

# 6.3 Retrieval Metrics

重点测：

```text
Recall@5
Recall@10
Recall@20
MRR
nDCG@10
```

例如：

```text
Recall@10 =

正确 Chunk 是否出现在前 10
```

对 RAG 来说 Recall 往往比 Precision 更优先，因为后面还有 reranker。

第一阶段目标可以设：

```text
Recall@20 > 0.90
```

具体阈值以后根据真实数据调整。

---

# 6.4 做 Ablation Test

这是评测最有价值的地方。

不要只测最终系统。

同时运行：

```text
A:
Original Vector

B:
Modern Vector

C:
Original + Modern

D:
Lexical + Original + Modern

E:
D + Reranker

F:
E + Query Expansion
```

形成：

| 系统 | Recall@10 | MRR | nDCG |
|---|---:|---:|---:|
| Original vector | | | |
| Modern vector | | | |
| 双 Vector | | | |
| Hybrid | | | |
| Hybrid + rerank | | | |

这样你才能回答一个很重要的问题：

> 花大量成本把文言文翻译成现代文，到底提升了多少？

如果：

```text
Original Recall@10 = 72%
Modern = 84%
Dual = 91%
```

翻译就证明有价值。

---

# 6.5 Generation Evaluation

再对最终答案测：

```text
Citation Correctness
Faithfulness
Answer Relevance
Context Utilization
Unsupported Claims
```

尤其加入一个指标：

```text
Citation Coverage
```

即：

> 回答里的事实是否都能对应到具体古籍 Chunk。

---

# 6.6 错误分类

所有失败样本不要只记录：

```text
wrong
```

而要标记：

```text
STRUCTURE_ERROR
TRANSLATION_ERROR
METADATA_ERROR
RETRIEVAL_MISS
RERANK_ERROR
CONTEXT_ERROR
GENERATION_ERROR
NO_DATA
```

例如用户问：

```text
太阳中风汗出……
```

正确 Chunk 根本没进入 Top 50：

```text
RETRIEVAL_MISS
```

正确 Chunk rank 3，但是 reranker 排到 40：

```text
RERANK_ERROR
```

上下文正确，但 LLM 答错：

```text
GENERATION_ERROR
```

这样才能知道应该改哪一层。

---

# 六步之间最终的数据流

整个项目最终应该成为：

```text
                 tcmoc
                   │
                   ▼
        ┌────────────────────┐
        │ 1. Corpus Audit    │
        └─────────┬──────────┘
                  │
                  ▼
        documents.jsonl
                  │
                  ▼
      ┌───────────────────────┐
      │ 2. Structure Parsing  │
      └───────────┬───────────┘
                  │
                  ▼
        structured JSON
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 3. Enrichment               │
   │ normalization               │
   │ translation                 │
   │ metadata                    │
   │ entities                    │
   │ chunking                    │
   └──────────────┬───────────────┘
                  │
                  ▼
            corpus.jsonl
                  │
                  ▼
      ┌──────────────────────┐
      │ 4. PostgreSQL        │
      │ + pgvector           │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ 5. Hybrid RAG        │
      │ Retrieval            │
      │ RRF                  │
      │ Reranker             │
      │ Context Builder      │
      │ Generator            │
      └──────────┬───────────┘
                 │
                 ▼
              Answer
                 │
                 ▼
       ┌───────────────────┐
       │ 6. Evaluation     │
       └─────────┬─────────┘
                 │
          Error Analysis
                 │
    ┌────────────┼─────────────┐
    ▼            ▼             ▼
 Parsing    Translation    Retrieval
```

## 我建议的实际开发顺序

不要直接对 700 本书批量跑。

先选 **5 本结构差异大的文献作为开发集**，例如一本文献以条文为主、一本本草、一本方书、一本医案，再加一份现代资料。首先把 `raw TXT → structured JSON → enriched JSONL → PostgreSQL → RAG → evaluation` **完整跑通一次**。

等这 5 本的数据模型稳定以后，再扩大到 50 本，最后才处理全部约 700 本。

这里最值得现在开始写代码的是 **Step 1 + Step 2**。因为一旦 `document / node / chunk` 三层数据契约确定下来，第三步的 LLM Pipeline、第四步 SQL schema、第五步检索接口都会围绕同一套 ID 和结构运行，不容易到后面推倒重来。
