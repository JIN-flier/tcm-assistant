#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SMOKE_OUTPUT="${SMOKE_OUTPUT:-$PROJECT_ROOT/output/qwen3-8b-tcm-smoke}"
mkdir -p "$SMOKE_OUTPUT"

swift sft \
  --model "$MODEL" \
  --tuner_type lora \
  --dataset "${TRAIN}#2000" \
  --val_dataset "${VAL}#200" \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_steps 20 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 1024 \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --eval_strategy steps \
  --eval_steps 10 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 2 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 8 \
  --use_liger_kernel true \
  --loss_scale ignore_empty_think \
  --output_dir "$SMOKE_OUTPUT"
