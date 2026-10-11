#!/usr/bin/env bash
# Usage: EVAL_ADAPTER=/path/to/checkpoint bash scripts/run_medchat_eval.sh prepare|generate|report|all
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
PY="${PYTHON:-python}"
ACTION="${1:-all}"
EVAL_DIR="${EVAL_DIR:-$PROJECT_ROOT/evaluation/medchat_holdout}"
EVAL_LIMIT="${EVAL_LIMIT:-2000}"
EVAL_POOL_FACTOR="${EVAL_POOL_FACTOR:-10}"
EVAL_SAMPLE_RATE="${EVAL_SAMPLE_RATE:-0.1963}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_ADAPTER="${EVAL_ADAPTER:-}"
EVAL_INPUT="$EVAL_DIR/medchat_unseen.jsonl"
GENERATIONS="$EVAL_DIR/base_vs_sft.jsonl"
mkdir -p "$EVAL_DIR"

prepare() {
  local flags=()
  if [[ "${EVAL_FUZZY:-1}" == "1" ]]; then flags+=(--fuzzy); fi
  "$PY" "$SCRIPT_DIR/prepare_medchat_eval.py" \
    --raw "$RAW_DIR/MedChatZH/MedChatZH_train.json" \
    --splits "$TRAIN" "$VAL" "$TEST" \
    --output "$EVAL_INPUT" \
    --limit "$EVAL_LIMIT" --pool-factor "$EVAL_POOL_FACTOR" \
    --seed "$EVAL_SEED" --sample-rate "$EVAL_SAMPLE_RATE" \
    "${flags[@]}"
}

generate() {
  [[ -n "$EVAL_ADAPTER" ]] || { echo 'Set EVAL_ADAPTER=/full/path/to/checkpoint' >&2; exit 2; }
  local extra=()
  [[ ! -f "$GENERATIONS" ]] || extra+=(--resume)
  "$PY" "$SCRIPT_DIR/generate_medchat_pairs.py" \
    --input "$EVAL_INPUT" --output "$GENERATIONS" \
    --model "$MODEL" --adapter "$EVAL_ADAPTER" \
    --max-input-tokens "${EVAL_MAX_INPUT_TOKENS:-2048}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-512}" \
    "${extra[@]}"
}

report() {
  local review=()
  [[ -f "$EVAL_DIR/blind_review.csv" ]] || review+=(--export-review "$EVAL_DIR/blind_review.csv")
  "$PY" "$SCRIPT_DIR/report_medchat_eval.py" \
    --input "$GENERATIONS" \
    --output-prefix "$EVAL_DIR/report" \
    --review-limit "${EVAL_REVIEW_LIMIT:-200}" \
    --seed "$EVAL_SEED" "${review[@]}"
}

case "$ACTION" in
  prepare) prepare ;;
  generate) generate ;;
  report) report ;;
  all) prepare; generate; report ;;
  *) echo "Usage: $0 prepare|generate|report|all" >&2; exit 2 ;;
esac
