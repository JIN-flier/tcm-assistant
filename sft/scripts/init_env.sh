#!/usr/bin/env bash

# ============================================================
# 项目路径
# ============================================================

export PROJECT="/data/users/jinsiyuan/TCMdata"

# 你当前模型位于 ~/data/models/
export MODEL="$HOME/data/models/Qwen3-8B"

# 最终训练数据
export TRAIN="$PROJECT/processed/training/train.jsonl"
export VAL="$PROJECT/processed/training/validation.jsonl"
export TEST="$PROJECT/processed/training/test.jsonl"

# 训练输出
export OUTPUT="$PROJECT/output/qwen3-8b-tcm-lora-v1"
export LOG_DIR="$OUTPUT/logs"


# ============================================================
# GPU
# ============================================================

# 单卡训练，使用 GPU 0
export CUDA_VISIBLE_DEVICES=0


# ============================================================
# Matplotlib
# ============================================================

# 解决你之前 ~/.config/matplotlib 没权限的问题
export MPLCONFIGDIR="/data/users/jinsiyuan/.cache/matplotlib"


# ============================================================
# PyTorch
# ============================================================

# 减轻显存碎片问题
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# ============================================================
# Tokenizer
# ============================================================

# 避免 tokenizer 多进程 fork 警告
export TOKENIZERS_PARALLELISM=false


# ============================================================
# 缓存 / 临时目录
# ============================================================

#export HF_HOME="/data/users/jinsiyuan/.cache/huggingface"
#export TMPDIR="/data/users/jinsiyuan/tmp"


# ============================================================
# 创建需要的目录
# ============================================================

mkdir -p "$MPLCONFIGDIR"
#mkdir -p "$HF_HOME"
#mkdir -p "$TMPDIR"
mkdir -p "$OUTPUT"
mkdir -p "$LOG_DIR"


# ============================================================
# check
# ============================================================

echo "PROJECT = $PROJECT"
echo "MODEL   = $MODEL"
echo "TRAIN   = $TRAIN"
echo "VAL     = $VAL"
echo "TEST    = $TEST"
echo "OUTPUT  = $OUTPUT"
echo "GPU     = $CUDA_VISIBLE_DEVICES"

test -f "$MODEL/config.json" && echo "MODEL OK"
test -f "$TRAIN" && echo "TRAIN OK"
test -f "$VAL" && echo "VAL OK"
test -f "$TEST" && echo "TEST OK"

wc -l "$TRAIN" "$VAL" "$TEST"





