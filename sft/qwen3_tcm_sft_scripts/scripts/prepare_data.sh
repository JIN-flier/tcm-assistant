#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PY="${PYTHON:-python}"
MEDCHAT_SAMPLE_RATE="${MEDCHAT_SAMPLE_RATE:-0.1963}"   # ~150k / 764k
GENERAL_MAX_TOKENS="${GENERAL_MAX_TOKENS:-1000}"
VAL_RATIO="${VAL_RATIO:-0.001}"
TEST_RATIO="${TEST_RATIO:-0.001}"
SEED="${SEED:-42}"

mkdir -p "$PROCESSED_DIR/normalized" "$PROCESSED_DIR/final" "$PROCESSED_DIR/training"

MEDCHAT="$RAW_DIR/MedChatZH/MedChatZH_train.json"
TCMCHAT_DIR="$RAW_DIR/TCMChat/sft/train"
TRAD_DIR="$RAW_DIR/TraditionalTCM"
GENERAL="$RAW_DIR/General/TigerBot/tigerbot-alpaca-zh-0.5m.json"

for f in "$MEDCHAT" "$GENERAL"; do
  test -f "$f" || { echo "Missing file: $f" >&2; exit 1; }
done
test -d "$TCMCHAT_DIR" || { echo "Missing directory: $TCMCHAT_DIR" >&2; exit 1; }
test -d "$TRAD_DIR" || { echo "Missing directory: $TRAD_DIR" >&2; exit 1; }
test -f "$MODEL/config.json" || { echo "Missing model: $MODEL" >&2; exit 1; }

echo "[1/9] Normalize MedChatZH subset"
$PY "$SCRIPT_DIR/normalize_sft.py" \
  --inputs "$MEDCHAT" \
  --output "$PROCESSED_DIR/normalized/medchatzh.jsonl" \
  --sample-rate "$MEDCHAT_SAMPLE_RATE" --seed "$SEED"

echo "[2/9] Normalize all TCMChat SFT train files"
mapfile -t TCMCHAT_FILES < <(find "$TCMCHAT_DIR" -maxdepth 1 -type f -name '*.json' | sort)
$PY "$SCRIPT_DIR/normalize_sft.py" \
  --inputs "${TCMCHAT_FILES[@]}" \
  --output "$PROCESSED_DIR/normalized/tcmchat.jsonl" \
  --sample-rate 1.0 --seed "$SEED"

echo "[3/9] Normalize all Traditional TCM SFT files"
mapfile -t TRAD_FILES < <(find "$TRAD_DIR" -maxdepth 1 -type f \( -name 'SFT_*.json' -o -name '_SFT_*.json' \) | sort)
$PY "$SCRIPT_DIR/normalize_sft.py" \
  --inputs "${TRAD_FILES[@]}" \
  --output "$PROCESSED_DIR/normalized/traditional_tcm.jsonl" \
  --sample-rate 1.0 --seed "$SEED"

echo "[4/9] Exact-deduplicate the domain datasets"
rm -f "$PROCESSED_DIR/dedup.sqlite3" "$PROCESSED_DIR/dedup.sqlite3-wal" "$PROCESSED_DIR/dedup.sqlite3-shm"
$PY "$SCRIPT_DIR/deduplicate.py" \
  --inputs \
    "$PROCESSED_DIR/normalized/traditional_tcm.jsonl" \
    "$PROCESSED_DIR/normalized/tcmchat.jsonl" \
    "$PROCESSED_DIR/normalized/medchatzh.jsonl" \
  --output "$PROCESSED_DIR/final/domain_all.jsonl" \
  --database "$PROCESSED_DIR/dedup.sqlite3"

echo "[5/9] Normalize general Chinese SFT"
$PY "$SCRIPT_DIR/normalize_sft.py" \
  --inputs "$GENERAL" \
  --output "$PROCESSED_DIR/normalized/tigerbot_general.jsonl" \
  --sample-rate 1.0 --seed "$SEED"

echo "[6/9] Keep general samples <= ${GENERAL_MAX_TOKENS} tokens"
$PY "$SCRIPT_DIR/filter_by_tokens.py" \
  --input "$PROCESSED_DIR/normalized/tigerbot_general.jsonl" \
  --output "$PROCESSED_DIR/normalized/tigerbot_general_${GENERAL_MAX_TOKENS}.jsonl" \
  --model "$MODEL" --max-tokens "$GENERAL_MAX_TOKENS"

echo "[7/9] Remove exact overlap between general and domain data"
rm -f "$PROCESSED_DIR/general_overlap.sqlite3" "$PROCESSED_DIR/general_overlap.sqlite3-wal" "$PROCESSED_DIR/general_overlap.sqlite3-shm"
$PY "$SCRIPT_DIR/filter_overlap.py" \
  --reference "$PROCESSED_DIR/final/domain_all.jsonl" \
  --input "$PROCESSED_DIR/normalized/tigerbot_general_${GENERAL_MAX_TOKENS}.jsonl" \
  --output "$PROCESSED_DIR/final/general_unique.jsonl" \
  --database "$PROCESSED_DIR/general_overlap.sqlite3"

echo "[8/9] Sample general data so it is ~10% of the final mixture"
DOMAIN_COUNT=$(wc -l < "$PROCESSED_DIR/final/domain_all.jsonl")
GENERAL_TARGET=$($PY - <<PY
import math
print(math.ceil($DOMAIN_COUNT / 9))
PY
)
GENERAL_AVAILABLE=$(wc -l < "$PROCESSED_DIR/final/general_unique.jsonl")
if (( GENERAL_AVAILABLE < GENERAL_TARGET )); then
  echo "WARNING: only $GENERAL_AVAILABLE general samples available, target was $GENERAL_TARGET." >&2
  GENERAL_TARGET=$GENERAL_AVAILABLE
fi
rm -f "$PROCESSED_DIR/general_sample.sqlite3" "$PROCESSED_DIR/general_sample.sqlite3-wal" "$PROCESSED_DIR/general_sample.sqlite3-shm"
$PY "$SCRIPT_DIR/sample_exact.py" \
  --input "$PROCESSED_DIR/final/general_unique.jsonl" \
  --output "$PROCESSED_DIR/final/general_10pct.jsonl" \
  --count "$GENERAL_TARGET" --seed "$SEED" \
  --database "$PROCESSED_DIR/general_sample.sqlite3"

echo "[9/9] Add empty Qwen3 think block and split train/validation/test"
$PY "$SCRIPT_DIR/prepare_qwen3_split.py" \
  --inputs "$PROCESSED_DIR/final/domain_all.jsonl" "$PROCESSED_DIR/final/general_10pct.jsonl" \
  --output-dir "$PROCESSED_DIR/training" \
  --val-ratio "$VAL_RATIO" --test-ratio "$TEST_RATIO" --seed "$SEED"

echo
echo "Final counts:"
wc -l "$TRAIN" "$VAL" "$TEST"

echo
echo "Final token statistics:"
$PY "$SCRIPT_DIR/token_stats.py" --dataset "$TRAIN" --model "$MODEL"
