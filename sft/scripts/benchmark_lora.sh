source scripts/init_env.sh
source .venv/bin/activate

export CKPT=$(
    find "$OUTPUT" \
        -maxdepth 1 \
        -type d \
        -name 'checkpoint-*' \
    | sort -V \
    | tail -n 1
)

python scripts/benchmark_lora.py \
    --model "$MODEL" \
    --adapter "$CKPT" \
    --max-new-tokens 512 \
    --output "$OUTPUT/benchmark_results.json"
