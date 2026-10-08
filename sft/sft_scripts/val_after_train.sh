export CKPT="./output/qwen3-8b-tcm-lora-v1/v4-20261001-115824/checkpoint-302237"

echo "$CKPT"

swift infer \
    --adapters "$CKPT" \
    --infer_backend transformers \
    --attn_impl sdpa \
    --enable_thinking false \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512
