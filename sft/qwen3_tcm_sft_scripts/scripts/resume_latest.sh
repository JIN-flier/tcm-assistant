#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

LATEST=$(find "$OUTPUT_ROOT" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null \
  | sort -n | tail -n 1 | cut -d' ' -f2-)

if [[ -z "$LATEST" ]]; then
  echo "No checkpoint found below: $OUTPUT_ROOT" >&2
  exit 1
fi

echo "Latest checkpoint: $LATEST"
RESUME_FROM_CHECKPOINT="$LATEST" bash "$SCRIPT_DIR/train_lora.sh"
