#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/init_env.sh"

source "/data/users/jinsiyuan/TCMdata/.venv/bin/activate"


echo "============================================================"
echo "Qwen3-8B TCM LoRA Training"
echo "============================================================"
echo "MODEL : $MODEL"
echo "TRAIN : $TRAIN"
echo "VAL   : $VAL"
echo "OUTPUT: $OUTPUT"
echo "GPU   : $CUDA_VISIBLE_DEVICES"
echo "============================================================"


# ------------------------------------------------------------
# 基础检查
# ------------------------------------------------------------

test -f "$MODEL/config.json"
test -f "$TRAIN"
test -f "$VAL"

mkdir -p "$OUTPUT"
mkdir -p "$LOG_DIR"


# ------------------------------------------------------------
# 可选：从 checkpoint 恢复
#
# 使用：
#
# RESUME_FROM_CHECKPOINT=/path/checkpoint-20000 \
# bash scripts/train_lora.sh
# ------------------------------------------------------------

EXTRA_ARGS=()

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    echo "Resume from checkpoint:"
    echo "$RESUME_FROM_CHECKPOINT"

    EXTRA_ARGS+=(
        --resume_from_checkpoint
        "$RESUME_FROM_CHECKPOINT"
    )
fi


# ------------------------------------------------------------
# 日志文件
# ------------------------------------------------------------

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"


# ------------------------------------------------------------
# 正式训练
# ------------------------------------------------------------

swift sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "$TRAIN" \
    --val_dataset "$VAL" \
    \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --group_by_length true \
    \
    --num_train_epochs 1 \
    \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    \
    --max_length 1024 \
    \
    --gradient_checkpointing false \
    --use_liger_kernel true \
    \
    --loss_scale ignore_empty_think \
    \
    --logging_steps 10 \
    \
    --eval_strategy steps \
    --eval_steps 2000 \
    \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 4 \
    \
    --load_best_model_at_end true \
    --metric_for_best_model loss \
    --greater_is_better false \
    \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    \
    --seed 42 \
    \
    --output_dir "$OUTPUT" \
    \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
