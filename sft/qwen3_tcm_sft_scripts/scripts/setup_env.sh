#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
}

if [[ ! -d "$VENV_DIR" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Core data + inference/training dependencies. FlashAttention is intentionally omitted.
uv pip install -U \
  huggingface_hub \
  datasets \
  transformers \
  ijson \
  orjson \
  tqdm \
  xxhash \
  safetensors \
  peft \
  packaging \
  psutil \
  ninja \
  liger-kernel

# Let ms-swift resolve a compatible torch backend for this environment.
uv pip install -U ms-swift --torch-backend=auto

python - <<'PY'
import torch, transformers
import swift
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("ms-swift:", swift.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
