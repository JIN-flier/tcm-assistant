#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

EPOCHS="${EPOCHS:-1}"
TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

for f in "$MODEL/config.json" "$TRAIN" "$VAL"; do
  test -f "$f" || { echo "Missing required file: $f" >&2; exit 1; }
done

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
  echo "Resuming from: $RESUME_FROM_CHECKPOINT"
fi

cat <<INFO
============================================================
Qwen3-8B TCM LoRA SFT
MODEL       : $MODEL
TRAIN       : $TRAIN
VAL         : $VAL
OUTPUT_ROOT : $OUTPUT_ROOT
GPU         : $CUDA_VISIBLE_DEVICES
ATTENTION   : SDPA (FlashAttention not required)
EPOCHS      : $EPOCHS
MAX_LENGTH  : $MAX_LENGTH
LoRA        : r=$LORA_RANK alpha=$LORA_ALPHA all-linear
Effective batch (single GPU): $((TRAIN_BS * GRAD_ACC))
============================================================
INFO

swift sft \
  --model "$MODEL" \
  --tuner_type lora \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$TRAIN_BS" \
  --per_device_eval_batch_size "$EVAL_BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --warmup_ratio "$WARMUP_RATIO" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules all-linear \
  --max_length "$MAX_LENGTH" \
  --gradient_checkpointing true \
  --use_liger_kernel true \
  --loss_scale ignore_empty_think \
  --logging_steps "$LOGGING_STEPS" \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --load_best_model_at_end true \
  --metric_for_best_model loss \
  --greater_is_better false \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --seed 42 \
  --output_dir "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
