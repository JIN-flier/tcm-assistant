#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CKPT="${1:-${CKPT:-}}"
if [[ -z "$CKPT" ]]; then
  echo "Usage: bash scripts/infer_lora.sh /path/to/checkpoint" >&2
  exit 1
fi

test -f "$CKPT/adapter_model.safetensors" || {
  echo "Not a LoRA checkpoint: $CKPT" >&2
  exit 1
}

swift infer \
  --model "$MODEL" \
  --adapters "$CKPT" \
  --load_args false \
  --infer_backend transformers \
  --attn_impl sdpa \
  --enable_thinking false \
  --stream true \
  --temperature 0 \
  --max_new_tokens 512
