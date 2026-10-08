#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
mkdir -p "$(dirname "$MODEL")"

echo "Downloading $MODEL_ID -> $MODEL"
hf download "$MODEL_ID" --local-dir "$MODEL"

echo
du -sh "$MODEL"
test -f "$MODEL/config.json" && echo "config OK"
