# Qwen3-8B 中医/医疗领域 SFT 评估方案（配套脚本）

本目录在**不修改原来的 SFT 工程**的前提下，增加对 **Base vs LoRA SFT** 的成对评估。依赖用户已有的真实 `raw/`、`processed/training/{train,validation,test}.jsonl`、Qwen3-8B 权重以及 LoRA checkpoint。

> **重要**：本评估包只包含脚本和说明，没有原始数据、模型权重，也没有已经算出的测试分数。`MedChatZH` 是综合医疗对话数据，不应直接把整体表现称为“中医辨证准确率”。

## A. 为什么可以重建“剩余的 MedChatZH”

原始 `scripts/prepare_data.sh` 使用：

- `MedChatZH_train.json`，**不是** `MedChatZH_valid.json` 或 2M 混合数据。
- `MEDCHAT_SAMPLE_RATE=0.1963`、`SEED=42`（默认值）。
- `normalize_sft.py` 先 `instruction + "\n\n" + input` 形成 question，清理 `output` 中现有 `<think>...</think>`。
- 用 `blake2b(f"{seed}:{question + '\n' + answer}", digest_size=8)` 生成伪随机数，**小于 0.1963 的样本纳入原始训练候选池**，反之为本次测试候选。
- 然后 3 类领域数据做**完整问答行级**精确去重，再混合通用数据，最后划分 train/validation/test；train/val/test 的比例默认 99.8% / 0.1% / 0.1%。

因此**必须使用原始文件及实际 SFT 时的 seed/rate**；不能仅从未出现在 `normalized/medchatzh.jsonl` 的字符串推断安全。尤其训练集可能有同问题不同答案、跨数据集相似问题。此脚本先恢复抽样外记录，然后逐行扫描**真实** `$TRAIN` `$VAL` `$TEST` 的 user 提问，排除规范化后的相同问题。可选保守近文本拷贝过滤，不能完全检出语义改写。

如果当时曾更改 `SEED`、`MEDCHAT_SAMPLE_RATE`、原始数据版本或实际训练文件，而这里没有同步，评测集**不能保证训练隔离**。

## B. 一键运行

在解压出的工程根目录：

```bash
source .venv/bin/activate
source scripts/env.sh
# 若模型路径不同，请先 export MODEL=/data/models/Qwen3-8B，再重新 source scripts/env.sh
# 安装新增轻量依赖（本环境通常已有 ijson/orjson；near-copy 和文本指标需要 rapidfuzz）
uv pip install ijson rapidfuzz openai

export EVAL_ADAPTER="$OUTPUT_ROOT/v4-20261001-115824/checkpoint-302237"  # 修改为真正使用的 checkpoint
export EVAL_LIMIT=2000          # 第一次可设 100，用来 smoke test
export EVAL_POOL_FACTOR=10      # 初始候选池 = 2000*10，过滤不足时适当调大
export EVAL_FUZZY=1            # 1: 开启保守近文本去重；0: 仅规范化后完全一致

bash scripts/run_medchat_eval.sh prepare
bash scripts/run_medchat_eval.sh generate
bash scripts/run_medchat_eval.sh report
```

也可以 `bash scripts/run_medchat_eval.sh all`。推理使用 `enable_thinking=False`、`do_sample=False`、SDPA，与原 `benchmark_base_vs_lora.py` 保持一致；同一 PEFT 模型上通过 `disable_adapter()` 获取 Base。单卡逐样本生成，LoRA/Base 顺序交替，有效控制设备上的加载差异，不保证严格的吞吐 benchmarking。脚本在结果文件已存在时支持 `--resume`，配置/输入不一致会拒绝继续。

运行目录 `evaluation/medchat_holdout/`：

```text
medchat_unseen.jsonl        # 抽样外、交叉过滤后的未见问答与参考答案
medchat_unseen.audit.json   # 排除与保留数量，检查审计日志
base_vs_sft.jsonl           # 同题 Base/LoRA 回答、tokens/s、输出长度
base_vs_sft.meta.json       # 模型、数据内容哈希和固定生成参数
blind_review.csv            # 匿名 A/B 临床质量人工打分表
blind_review.key.jsonl      # 真实模型身份映射，切勿提供给评审者
report.json                 # 机读指标
report.md                   # 人读评测摘要
```

注意：旧训练没有在最终 JSONL 保存源数据 ID，因此**这里通过还原原始抽样函数+全量实际 split 的问题去重重建测试集**，不是按原始文件行号随意取“后80%”。`audit.json` 请保留。`prepare` 会覆盖原来的评测样本，因此**开始评测后不要更改抽样条件或反复重建再择优报告**。

## C. 核心指标与医学安全限制

1. **可自动、客观复现**：原始答复对齐的字符 ROUGE-L F1、字符 2-gram F1；Base/SFT 配对差值、配对 bootstrap 95% CI；重复回答比例；推理 token 数和速度。这两项 F1 只是**文本重叠度**，不是医疗正确率或辨证分类 F1。
2. **第三方模型盲评**（可选）：医学正确性（1–5）、完整性（1–5）、明确危险建议（布尔）、SFT 胜率（平局按 0.5）。模型打分会犯错，不能单凭它宣称模型已具备可靠问诊能力。
3. **专业人工盲评**（推荐 200 例以上）：对匿名回答 A、B 评判正确性/完整性和危险建议，解盲后输出 SFT 胜率与各评分对比。若同一病例多位评审，建议单独报告一致性；本脚本只处理一份已汇总评分的 CSV。
4. **真正 Accuracy**：MedChatZH 原始开放式问答没有可直接求 `预测标签==金标准标签` 的标准分类标签，故本项目**不会伪造 Accuracy**。若需要客观准确率，应使用独立、经过核对的中医知识选择题或证型标签集，例如 TCMBench 相关客观题；另建对应 gold-label 评测，先去重并确认授权。

不要把 MedChatZH 参考答案当作医学金标准。可能出现 SFT 医学上更正确、ROUGE 却下降的现象。尤其任何剂量、禁忌、急症识别和延误转诊等错误，应由医学专业人员复核。

### 使用独立评审模型（OpenAI-compatible 服务）

**此步骤会将病例和模型回答发送到你指定的 API/服务**，如果包含个人医疗信息，务必先匿名化并遵循数据权限约束。不要使用待测的 Base/LoRA 本身担当唯一裁判。

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export JUDGE_MODEL=your-independent-medical-judge
export JUDGE_API_KEY=EMPTY
python scripts/judge_medchat_pairs.py \
  --input "$PROJECT_ROOT/evaluation/medchat_holdout/base_vs_sft.jsonl" \
  --output "$PROJECT_ROOT/evaluation/medchat_holdout/judge.jsonl" \
  --resume

python scripts/report_medchat_eval.py \
  --input "$PROJECT_ROOT/evaluation/medchat_holdout/base_vs_sft.jsonl" \
  --judge "$PROJECT_ROOT/evaluation/medchat_holdout/judge.jsonl" \
  --output-prefix "$PROJECT_ROOT/evaluation/medchat_holdout/report_with_judge"
```

判分模型通过固定 seed 随机匿名交换 A/B 顺序；评分参考问诊问题、原数据集参考答案（仅供辅助）以及 A/B 回答。

### 专业人工盲评

`blind_review.csv` 按随机顺序隐藏模型身份。向专业评审**只提供 CSV，不提供 `.key.jsonl`**。填：

- `winner`：`A` / `B` / `tie`
- `A_correctness`、`B_correctness`、`A_coverage`、`B_coverage`：1–5 整数
- `A_unsafe`、`B_unsafe`：`true` / `false`
- `notes`：可记录错误及危险建议的具体原因

填完后：

```bash
python scripts/report_medchat_eval.py \
  --input "$PROJECT_ROOT/evaluation/medchat_holdout/base_vs_sft.jsonl" \
  --human-review "$PROJECT_ROOT/evaluation/medchat_holdout/blind_review.csv" \
  --review-key "$PROJECT_ROOT/evaluation/medchat_holdout/blind_review.key.jsonl" \
  --output-prefix "$PROJECT_ROOT/evaluation/medchat_holdout/report_with_human"
```

临床评分是评审判断的评分，并非金标准准确率。真实多轮问诊还需要**基于标准化模拟患者的交互测试**；MedChatZH 单轮输入/输出问答无法测出连续追问的完整度、状态跟踪和最终诊断可靠性。

## D. 重要限制与推荐顺序

- 建议先 `EVAL_LIMIT=100` 全流程 smoke test，确认数据、显存、LoRA 真正加载；之后用另一个 `EVAL_DIR` 重新固定正式的 2,000 条测评样本，防止混写结果。
- `--fuzzy` 只针对前/后 18 字相似且文本相似度很高的**近文本拷贝**，不能阻止改写、同一病例的多轮或语义相似泄漏。这一限制必须写入论文/报告；如有必要另做 embedding 相似检索加人工审阅。
- 若正式中医问诊垂域是目标，需人工标注 `domain` 并按中医/现代医学/通用任务分别报告；否则 MedChatZH 综合平均值不是“中医准确率”。
- 通用能力是否退化，应再用**与训练无重叠**的 C-Eval / CMMLU 非医疗子集做一组独立测试。本项目没有提供或伪造这些外部基准的测试结果。
- 长输入默认跳过（而不是截断）；检查 generation `hit_max_tokens` 是否常触顶。所有比较必须使用相同上下文长度和解码设置。
- 当前 `--pool-factor` 选取的是 stable-hash top-K 候选。过滤后只有较少样本可用时，增大 pool-factor（例如 20）。

## E. 原有工程脚本保持不变

仅新增：

```text
EVALUATION.md
scripts/prepare_medchat_eval.py
scripts/generate_medchat_pairs.py
scripts/judge_medchat_pairs.py
scripts/report_medchat_eval.py
scripts/run_medchat_eval.sh
```

不要求重新训练，不改 Qwen3 空 `<think>` SFT 的旧数据，不改 base/adapter，也不修改 `scripts/prepare_data.sh`。建议在正式评估前保存生成数据的 `sha256sum`、环境依赖版本及 LoRA 路径。

## F. 可选：使用独立标注的 MCQ 计算真正准确率

评估工程另提供 `scripts/eval_labeled_mcq.py`。它**不把 MedChatZH 开放式答案冒充成选择题标签**。先准备一份不与原训练数据重叠、已核对正确选项的独立 JSONL（来自正规可用的评测集或人工标注），例如：

```json
{"id":"example-001","question":"以下哪项属于中医四诊？","A":"望闻问切","B":"视触叩听","C":"望触叩听","D":"望闻触听","answer":"A"}
```

再执行：

```bash
python scripts/eval_labeled_mcq.py \
  --model "$MODEL" --adapter "$EVAL_ADAPTER" \
  --input "$PROJECT_ROOT/evaluation/independent_mcq.jsonl" \
  --output "$PROJECT_ROOT/evaluation/mcq_predictions.jsonl"
```

会输出 **Base/SFT Accuracy、Macro-F1、两模型仅一方正确的配对样本数量**。该方法利用题目和四个选项后下一 token 的 A/B/C/D logits 强制选一项，因此度量的是**单 token 约束选择准确率**，不是自然生成的格式遵循能力；必须对两模型使用相同提示词。务必检查真实评测题的 Gold label 和跨训练集重叠，且不能拿只有这个示例的一道题推断实际能力。

## G. 运行轻量自测

```bash
python -m unittest discover -s tests -v
```

只检查采样哈希、训练问题排除、基础指标和盲评 JSON 格式；**不运行真实 8B 推理**。运行需要 `rapidfuzz`；真实大数据 JSON 数组还需要 `ijson`。
