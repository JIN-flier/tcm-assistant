#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

command -v hf >/dev/null 2>&1 || {
  echo "hf CLI not found. Run scripts/setup_env.sh or install huggingface_hub." >&2
  exit 1
}

mkdir -p "$RAW_DIR/MedChatZH" "$RAW_DIR/TCMChat" "$RAW_DIR/TraditionalTCM" "$RAW_DIR/General/TigerBot"

echo "[1/4] MedChatZH: download train/valid only (skip the 2M mix file)"
hf download tyang816/MedChatZH \
  MedChatZH_train.json MedChatZH_valid.json README.md \
  --repo-type dataset \
  --local-dir "$RAW_DIR/MedChatZH"

echo "[2/4] TCMChat: download SFT files only (skip pretrain corpus)"
hf download ZJUFanLab/TCMChat-dataset-600k \
  --repo-type dataset \
  --include 'sft/**' README.md \
  --local-dir "$RAW_DIR/TCMChat"

echo "[3/4] Traditional Chinese Medicine SFT: download complete dataset"
hf download SylvanL/Traditional-Chinese-Medicine-Dataset-SFT \
  --repo-type dataset \
  --local-dir "$RAW_DIR/TraditionalTCM"

echo "[4/4] General Chinese SFT: download only TigerBot Alpaca 0.5M file"
hf download TigerResearch/sft_zh \
  tigerbot-alpaca-zh-0.5m.json README.md \
  --repo-type dataset \
  --local-dir "$RAW_DIR/General/TigerBot"

echo
echo "Downloaded sizes:"
du -sh "$RAW_DIR"/* "$RAW_DIR/General/TigerBot" 2>/dev/null || true
