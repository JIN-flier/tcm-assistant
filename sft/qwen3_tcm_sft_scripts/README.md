# Qwen3-8B + ms-swift + LoRA 中医 SFT 可复现流程

这套工程把一次完整的中医领域 Qwen3-8B LoRA SFT 流程固化为可复现脚本，覆盖：

1. Python / ms-swift 环境准备
2. 下载 Qwen3-8B
3. 下载 MedChatZH、TCMChat-dataset-600k、Traditional-Chinese-Medicine-Dataset-SFT 与通用中文 SFT 数据
4. 检查原始 JSON 结构
5. 统一为 `messages` 格式
6. 删除显式 CoT
7. 领域数据 exact dedup
8. 加入约 10% 通用中文任务
9. 为 Qwen3 加入空 `<think>\n\n</think>`，保留 non-CoT SFT 形式
10. 划分 train / validation / test
11. 统计最终 token 长度
12. 20-step smoke test
13. 正式 `ms-swift + Qwen3-8B + LoRA` 训练
14. 断点续训
15. 找 best checkpoint
16. 交互推理
17. Base 与 LoRA 回答质量、TTFT、tokens/s 对比
18. 验证 LoRA 权重是否非零、是否真正改变 logits、`ΔW/W` 是否合理

当前方案**不依赖 FlashAttention**，使用：

- Qwen3-8B
- ms-swift
- LoRA
- BF16
- PyTorch SDPA
- Liger Kernel
- `loss_scale=ignore_empty_think`
- `max_length=1024`

> 这套设置来自一次已经成功完成 smoke test 和正式训练的流程。若目标机器的 GPU / PyTorch / ms-swift 版本不同，应先跑 smoke test，再启动全量训练。

---

## 1. 工程结构

```text
qwen3_tcm_sft_repro/
├── README.md
├── .gitignore
└── scripts/
    ├── env.sh
    ├── setup_env.sh
    ├── download_model.sh
    ├── download_datasets.sh
    ├── download_datasets.py
    ├── inspect_json.py
    ├── normalize_sft.py
    ├── deduplicate.py
    ├── filter_by_tokens.py
    ├── filter_overlap.py
    ├── sample_exact.py
    ├── prepare_qwen3_split.py
    ├── token_stats.py
    ├── prepare_data.sh
    ├── smoke_test.sh
    ├── train_lora.sh
    ├── resume_latest.sh
    ├── find_best_checkpoint.py
    ├── infer_lora.sh
    ├── benchmark_base_vs_lora.py
    ├── check_lora_weights.py
    ├── check_lora_effect.py
    └── check_lora_update_ratio.py
```

运行后会生成：

```text
raw/
processed/
output/
.cache/
.tmp/
```

这些大文件目录已写入 `.gitignore`。

---

# 2. 在新机器上复制工程

例如：

```bash
cd /data/users/<user>
cp -r qwen3_tcm_sft_repro TCMdata
cd TCMdata
```

如果希望直接保留当前目录名也可以。

所有核心路径都集中在：

```bash
scripts/env.sh
```

默认值：

```bash
PROJECT_ROOT=<当前工程根目录>
RAW_DIR=$PROJECT_ROOT/raw
PROCESSED_DIR=$PROJECT_ROOT/processed
MODEL=$HOME/data/models/Qwen3-8B
TRAIN=$PROCESSED_DIR/training/train.jsonl
VAL=$PROCESSED_DIR/training/validation.jsonl
TEST=$PROCESSED_DIR/training/test.jsonl
OUTPUT_ROOT=$PROJECT_ROOT/output/qwen3-8b-tcm-lora-v1
CUDA_VISIBLE_DEVICES=0
```

如果新机器的模型希望放在：

```text
/data/models/Qwen3-8B
```

无需修改脚本，可直接：

```bash
export MODEL=/data/models/Qwen3-8B
source scripts/env.sh
```

或者把 `scripts/env.sh` 中的默认 MODEL 改掉。

---

# 3. 创建 Python 环境

本工程继续使用 `uv`。

```bash
bash scripts/setup_env.sh
```

默认创建：

```text
./.venv
```

如果目标机器已经有可用环境，可跳过此步骤，只需确保至少安装：

```text
ms-swift
transformers
huggingface_hub
datasets
ijson
orjson
tqdm
safetensors
peft
liger-kernel
```

激活：

```bash
source .venv/bin/activate
```

检查 GPU：

```bash
nvidia-smi
```

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16:', torch.cuda.is_bf16_supported())
PY
```

---

# 4. Hugging Face 登录

公开仓库通常不强制登录，但登录可以减少匿名下载限流。

```bash
hf auth login
hf auth whoami
```

---

# 5. 加载统一环境变量

每个新 shell / SSH 会话中执行：

```bash
source .venv/bin/activate
source scripts/env.sh
```

查看：

```bash
echo "$PROJECT_ROOT"
echo "$MODEL"
echo "$TRAIN"
echo "$VAL"
echo "$OUTPUT_ROOT"
```

`env.sh` 还处理了曾经遇到的 Matplotlib 权限问题：

```bash
MPLCONFIGDIR=$PROJECT_ROOT/.cache/matplotlib
```

因此不依赖 `~/.config/matplotlib` 可写。

---

# 6. 下载 Qwen3-8B

```bash
source scripts/env.sh
bash scripts/download_model.sh
```

默认下载：

```text
Qwen/Qwen3-8B
```

到：

```text
$MODEL
```

检查：

```bash
du -sh "$MODEL"
ls -lh "$MODEL" | head
```

至少应有：

```text
config.json
model.safetensors.index.json
model-0000x-of-0000x.safetensors
tokenizer.json
```

---

# 7. 下载四组训练数据

## 7.1 推荐：命令行脚本

```bash
bash scripts/download_datasets.sh
```

下载内容：

### MedChatZH

仓库：

```text
tyang816/MedChatZH
```

只下载：

```text
MedChatZH_train.json
MedChatZH_valid.json
```

不下载额外的 `MedChatZH_mix_2M.json`。

### TCMChat-dataset-600k

仓库：

```text
ZJUFanLab/TCMChat-dataset-600k
```

只下载：

```text
sft/**
```

正式 SFT 不把 `pretrain/` 语料直接混进来。

### Traditional-Chinese-Medicine-Dataset-SFT

仓库：

```text
SylvanL/Traditional-Chinese-Medicine-Dataset-SFT
```

完整下载，包含：

```text
SFT_medicalKnowledge_source1_548404.json
SFT_medicalKnowledge_source2_99334.json
SFT_medicalKnowledge_source3_556540.json
SFT_nlpDiseaseDiagnosed_61486.json
SFT_nlpSyndromeDiagnosed_48665.json
SFT_structGeneral_310860.json
SFT_structPrescription_92896.json
_SFT_traditionalTrans_1959542.json
```

### 通用中文任务

使用：

```text
TigerResearch/sft_zh
```

只下载：

```text
tigerbot-alpaca-zh-0.5m.json
```

用于最终约 10% 的通用能力混合。

## 7.2 Python 下载方式

等价地可以运行：

```bash
python scripts/download_datasets.py --raw-dir "$RAW_DIR"
```

两种方式二选一，不要重复下载。

---

# 8. 检查原始数据字段

先不要直接跑全量转换，建议抽查：

```bash
python scripts/inspect_json.py \
  "$RAW_DIR/MedChatZH/MedChatZH_train.json"
```

```bash
python scripts/inspect_json.py \
  "$RAW_DIR/TCMChat/sft/train/medical_case.json"
```

```bash
python scripts/inspect_json.py \
  "$RAW_DIR/TraditionalTCM/SFT_nlpSyndromeDiagnosed_48665.json"
```

```bash
python scripts/inspect_json.py \
  "$RAW_DIR/General/TigerBot/tigerbot-alpaca-zh-0.5m.json"
```

当前流水线预期主要字段为：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

额外字段会被忽略。

---

# 9. 一键完成数据处理

正式处理前确保：

```bash
source .venv/bin/activate
source scripts/env.sh
```

然后：

```bash
bash scripts/prepare_data.sh
```

该脚本依次完成下面步骤。

## 9.1 MedChatZH 子集

默认：

```bash
MEDCHAT_SAMPLE_RATE=0.1963
```

对应约 15 万条 MedChatZH 数据。

如需修改：

```bash
MEDCHAT_SAMPLE_RATE=0.25 bash scripts/prepare_data.sh
```

## 9.2 TCMChat

使用：

```text
sft/train/*.json
```

全部保留。

## 9.3 Traditional TCM

所有 `SFT_*.json` 和 `_SFT_*.json` 全部保留。

这严格复现本次实验。需要注意，其中古今文翻译数据占比较高；未来若追求“问诊能力优先”，可在新的消融实验中降低该子集权重，但不要在复现实验时临时改变。

## 9.4 统一格式

统一为 ms-swift 支持的：

```json
{
  "messages": [
    {"role": "user", "content": "instruction + input"},
    {"role": "assistant", "content": "output"}
  ]
}
```

`normalize_sft.py` 会删除已有显式：

```text
<think>真实推理过程</think>
```

因此不会拿真实 CoT 进行 SFT。

## 9.5 领域 exact dedup

使用 SQLite + BLAKE2b 进行精确去重：

```text
Traditional
+ TCMChat
+ MedChatZH subset
      ↓
domain_all.jsonl
```

这不是 MinHash 近似去重，只消除完全相同的标准化样本。

## 9.6 通用数据 token 过滤

TigerBot 默认先保留：

```text
<= 1000 tokens
```

给后续加入 Qwen3 空 think token 留出余量。

## 9.7 通用数据与领域数据交叉去重

生成：

```text
processed/final/general_unique.jsonl
```

## 9.8 约 10% 通用任务

目标满足：

```text
general / (domain + general) ≈ 10%
```

因此：

```text
general_target = domain_count / 9
```

生成：

```text
processed/final/general_10pct.jsonl
```

如果通用数据实际不足，则使用全部可用通用数据，并打印 warning。

## 9.9 Qwen3 空 think

最终 assistant 内容会变成：

```text
<think>

</think>

最终回答
```

这不是训练真实思维链，而是配合：

```bash
--loss_scale ignore_empty_think
```

使 empty-think 部分不计 loss，降低 non-CoT SFT 对原 Qwen3 reasoning 能力的扰动。

## 9.10 数据划分

默认：

```text
train      99.8%
validation  0.1%
test        0.1%
```

生成：

```text
processed/training/train.jsonl
processed/training/validation.jsonl
processed/training/test.jsonl
```

---

# 10. 检查最终 token 长度

`prepare_data.sh` 最后会自动运行一次。

也可以手工运行：

```bash
python scripts/token_stats.py \
  --dataset "$TRAIN" \
  --model "$MODEL"
```

本次实际训练前统计结果为：

```text
全部 <= 1024 tokens
```

因此正式训练选择：

```bash
--max_length 1024
```

换机器重新处理数据后应再次检查，不要直接假设结果相同。

---

# 11. 为什么不安装 FlashAttention

本次环境中 FlashAttention 源码构建失败，原因是系统没有 `nvcc` / 完整 CUDA Toolkit。

这不是 Qwen3 LoRA SFT 的必要依赖。

最终稳定训练采用：

```bash
--attn_impl sdpa
```

并继续使用：

```bash
--use_liger_kernel true
```

因此本工程**没有安装 FlashAttention 的步骤**，也没有开启：

```bash
--packing true
```

这样更容易在另一台机器上复现。

---

# 12. 先跑 smoke test

所有机器都应先跑：

```bash
bash scripts/smoke_test.sh
```

配置：

```text
Qwen3-8B
LoRA r=16 alpha=32
all-linear
BF16
SDPA
Liger
max_length=1024
20 steps
```

确认：

- 能正常加载 base model
- 能加载数据
- loss 不是 NaN
- 可以反向传播
- checkpoint 能保存
- GPU 显存没有 OOM

另开终端：

```bash
watch -n 2 nvidia-smi
```

只有 smoke test 正常后再正式训练。

---

# 13. 正式训练配置

正式命令已经写入：

```text
scripts/train_lora.sh
```

核心参数：

| 参数 | 值 |
|---|---:|
| Base | Qwen3-8B |
| 方法 | LoRA |
| dtype | BF16 |
| attention | SDPA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| target | all-linear |
| max length | 1024 |
| train batch / GPU | 1 |
| grad accumulation | 16 |
| 单卡 effective batch | 16 |
| learning rate | 1e-4 |
| scheduler | cosine |
| warmup | 3% |
| epochs | 1 |
| eval | 2000 steps |
| save | 2000 steps |
| save total | 4 |
| gradient checkpointing | true |
| Liger | true |
| real CoT | 不训练 |
| empty think | ignore_empty_think |

启动：

```bash
source .venv/bin/activate
source scripts/env.sh
bash scripts/train_lora.sh
```

日志保存在：

```text
$OUTPUT_ROOT/logs/
```

ms-swift 通常会在 output 下再创建版本目录，例如：

```text
output/qwen3-8b-tcm-lora-v1/v4-YYYYMMDD-HHMMSS/
```

checkpoint 形态：

```text
checkpoint-2000/
checkpoint-4000/
...
```

本次实际完整 1 epoch 的最终 step 达到了约 30 万级，因此换机器时不要把 `30k` 当作完整 epoch。

---

# 14. 自定义训练参数

不必改脚本，可以临时覆盖：

```bash
LR=5e-5 \
LORA_RANK=32 \
LORA_ALPHA=64 \
EVAL_STEPS=1000 \
bash scripts/train_lora.sh
```

常用可覆盖变量：

```text
EPOCHS
TRAIN_BS
EVAL_BS
GRAD_ACC
LR
WARMUP_RATIO
LORA_RANK
LORA_ALPHA
MAX_LENGTH
EVAL_STEPS
SAVE_STEPS
LOGGING_STEPS
SAVE_TOTAL_LIMIT
DATASET_NUM_PROC
DATALOADER_NUM_WORKERS
```

---

# 15. 断点续训

自动寻找最新 checkpoint：

```bash
bash scripts/resume_latest.sh
```

或者显式指定：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-180000 \
  bash scripts/train_lora.sh
```

断点续训会恢复：

- LoRA 参数
- optimizer 状态
- LR scheduler
- Trainer global step
- RNG 状态

而不是从 step 0 重启。

---

# 16. checkpoint 中各文件

典型目录：

```text
adapter_config.json
adapter_model.safetensors
additional_config.json
args.json
optimizer.pt
rng_state.pth
scheduler.pt
trainer_state.json
training_args.bin
README.md
```

其中：

```text
adapter_model.safetensors  LoRA 真正训练出的增量权重
adapter_config.json        LoRA r/alpha/target 等配置
args.json                  ms-swift 训练配置
optimizer.pt               optimizer 断点状态
scheduler.pt               scheduler 断点状态
trainer_state.json         step / eval_loss / best checkpoint 等
rng_state.pth              随机状态
training_args.bin          HF Trainer 参数
```

推理重点是 adapter 文件；完整断点续训则不要删除 optimizer/scheduler/state 文件。

---

# 17. 找 best checkpoint

不要默认最后一个 checkpoint 最好。

例如：

```bash
python scripts/find_best_checkpoint.py \
  output/qwen3-8b-tcm-lora-v1/v4-YYYYMMDD-HHMMSS/checkpoint-302237
```

重点查看：

```text
best_metric
best_model_checkpoint
```

由于训练配置：

```bash
--load_best_model_at_end true
--metric_for_best_model loss
--greater_is_better false
```

应优先比较 best checkpoint 与 final checkpoint。

---

# 18. 交互式测试 LoRA

```bash
bash scripts/infer_lora.sh \
  /path/to/best-or-final-checkpoint
```

显式采用：

```text
base Qwen3-8B
+ adapter
+ transformers backend
+ SDPA
+ enable_thinking=false
+ temperature=0
```

推荐测试至少三类：

### 中医辨证

```text
患者女性，42岁，近一个月容易疲倦，食欲下降，饭后腹胀，大便偏溏，舌淡苔白。请从中医辨证角度进行分析，并说明判断依据。
```

### 问诊追问

```text
患者只说“最近经常头晕”，目前没有更多资料。作为问诊助手，你下一步应该重点询问哪些信息？
```

### 通用能力

```text
请用简单语言解释哈希表的工作原理，并说明两种常见哈希冲突处理方法。
```

---

# 19. Base vs LoRA：回答质量与速度

运行：

```bash
python scripts/benchmark_base_vs_lora.py \
  --model "$MODEL" \
  --adapter /path/to/checkpoint \
  --max-new-tokens 512 \
  --output benchmark_results.json
```

同一个进程内对同一个 base model 分别：

```text
disable adapter -> Base
adapter enabled -> LoRA
```

比较：

- 回答内容
- TTFT
- 总耗时
- decode tokens/s

使用 `do_sample=False`，便于可重复比较。

注意：Transformers 单请求推理速度主要用于 Base/LoRA A/B，不等同于未来 vLLM 部署吞吐。

---

# 20. 如果 Base 与 LoRA 回答几乎逐字一致

不能仅凭最终文本判断 LoRA 没有学到。

`temperature=0` / greedy decoding 下，即使 logits 已变化，只要每一步 top-1 token 没换，最终句子仍可完全相同。

按以下顺序诊断。

## 20.1 检查 adapter 是否真的有非零更新

```bash
python scripts/check_lora_weights.py /path/to/checkpoint
```

训练后的 LoRA 应出现大量明显非零 tensor。

例如本次实际检查中，部分 `lora_B`：

```text
max      ~ 0.1 - 0.45
mean_abs ~ 0.01 - 0.02
```

说明 LoRA 确实已经更新，不是全零 adapter。

## 20.2 直接比较 Base 与 LoRA logits

```bash
python scripts/check_lora_effect.py \
  --model "$MODEL" \
  --adapter /path/to/checkpoint
```

重点看：

```text
mean abs diff
max abs diff
L2 norm
Base top-1
LoRA top-1
```

判断：

```text
logit diff == 0
    -> adapter 没有真正参与前向，或加载异常

logit diff != 0，但 top-1 相同
    -> LoRA 已生效，只是 greedy 输出仍相同

Base top-1 != LoRA top-1
    -> LoRA 已直接改变当前 token 决策
```

## 20.3 检查实际 LoRA 更新相对 base 权重有多大

```bash
python scripts/check_lora_update_ratio.py \
  --model "$MODEL" \
  --adapter /path/to/checkpoint
```

计算：

```text
ΔW = (alpha / r) * B @ A
ΔW/W = ||ΔW|| / ||W||
```

比单独查看 A/B 最大值更能说明 LoRA 更新幅度。

---

# 21. 完整推荐执行顺序

新机器从零开始可以直接按照：

```bash
# 1. 环境
bash scripts/setup_env.sh
source .venv/bin/activate
source scripts/env.sh

# 2. HF 登录
hf auth login

# 3. 模型
bash scripts/download_model.sh

# 4. 数据集
bash scripts/download_datasets.sh

# 5. 抽查格式
python scripts/inspect_json.py "$RAW_DIR/MedChatZH/MedChatZH_train.json"
python scripts/inspect_json.py "$RAW_DIR/TCMChat/sft/train/medical_case.json"
python scripts/inspect_json.py "$RAW_DIR/TraditionalTCM/SFT_nlpSyndromeDiagnosed_48665.json"
python scripts/inspect_json.py "$RAW_DIR/General/TigerBot/tigerbot-alpaca-zh-0.5m.json"

# 6. 全部数据处理
bash scripts/prepare_data.sh

# 7. smoke test
bash scripts/smoke_test.sh

# 8. 正式训练
bash scripts/train_lora.sh

# 9. 找 best checkpoint
python scripts/find_best_checkpoint.py /path/to/a/checkpoint

# 10. 交互测试
bash scripts/infer_lora.sh /path/to/best/checkpoint

# 11. Base vs LoRA 质量和速度
python scripts/benchmark_base_vs_lora.py \
  --model "$MODEL" \
  --adapter /path/to/best/checkpoint \
  --output benchmark_results.json

# 12. LoRA 权重诊断
python scripts/check_lora_weights.py /path/to/best/checkpoint
python scripts/check_lora_effect.py --model "$MODEL" --adapter /path/to/best/checkpoint
python scripts/check_lora_update_ratio.py --model "$MODEL" --adapter /path/to/best/checkpoint
```

---

# 22. 数据与模型来源

本流程使用：

- Qwen3-8B: `Qwen/Qwen3-8B`
- MedChatZH: `tyang816/MedChatZH`
- TCMChat-dataset-600k: `ZJUFanLab/TCMChat-dataset-600k`
- Traditional-Chinese-Medicine-Dataset-SFT: `SylvanL/Traditional-Chinese-Medicine-Dataset-SFT`
- 通用中文 SFT: `TigerResearch/sft_zh` 中的 `tigerbot-alpaca-zh-0.5m.json`

截至整理本工程时，上述数据仓库公开页面中相关数据均仍可访问。实际用于商业或公开发布前，应再次核对各仓库的最新许可证、数据来源说明和医学用途限制。

---

# 23. 关键设计决策总结

这次复现实验最重要的不是某一条命令，而是以下设计：

```text
Qwen3-8B
  + LoRA
  + Traditional TCM 全量
  + TCMChat SFT 全量
  + MedChatZH 子集
  + ~10% 通用中文任务
  + 不训练真实 CoT
  + empty think + ignore_empty_think
  + exact dedup
  + max_length 1024
  + BF16
  + SDPA
  + Liger
  + 1 epoch
```

其中 FlashAttention **不是复现的必要条件**。

为了科学比较不同实验版本，建议始终固定：

```text
seed=42
相同 train/val/test
相同 benchmark questions
相同 temperature=0 / do_sample=False
相同 base model
```

然后只修改一个因素，例如：

```text
LoRA rank
学习率
数据配比
是否保留全部古今文翻译
MedChatZH 比例
通用任务比例
```

这样后续才能可靠判断哪一个设计真正提升了中医问诊能力。
