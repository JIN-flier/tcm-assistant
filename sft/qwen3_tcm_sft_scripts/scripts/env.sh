#!/usr/bin/env bash
# Shared configuration for the reproducible Qwen3-8B TCM SFT pipeline.
# Override any variable before sourcing this file, e.g.:
#   MODEL=/data/models/Qwen3-8B source scripts/env.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/raw}"
export PROCESSED_DIR="${PROCESSED_DIR:-$PROJECT_ROOT/processed}"
export MODEL="${MODEL:-$HOME/data/models/Qwen3-8B}"

export TRAIN="${TRAIN:-$PROCESSED_DIR/training/train.jsonl}"
export VAL="${VAL:-$PROCESSED_DIR/training/validation.jsonl}"
export TEST="${TEST:-$PROCESSED_DIR/training/test.jsonl}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/output/qwen3-8b-tcm-lora-v1}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.tmp}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$OUTPUT_ROOT" "$LOG_DIR"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$TMPDIR"
