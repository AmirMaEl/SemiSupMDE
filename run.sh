#!/usr/bin/env bash
# One command to install, download and run:
#
#   ./run.sh                      evaluate the finetuned model on the KITTI Eigen test split (downloads ~2 GB)
#   ./run.sh eval [model.pth]     evaluate a checkpoint (default: checkpoints/finetuned_KITTI.pth)
#   ./run.sh finetune [args]      vKITTI -> KITTI finetuning (downloads ~40 GB, needs ~35 GB of GPU memory)
#   ./run.sh setup                only create the environment and download the models
#
# Also works without cloning first:  curl -sSL https://raw.githubusercontent.com/AmirMaEl/SemiSupMDE/main/run.sh | bash
# The Python environment goes to .venv/, models to checkpoints/, data to data/ (or KITTI_ROOT / VKITTI_ROOT).
set -euo pipefail

REPO_URL=https://github.com/AmirMaEl/SemiSupMDE.git
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"

# Not inside the repo (e.g. piped from curl): clone it and continue there.
if [ ! -f "$HERE/config.py" ]; then
    [ -d SemiSupMDE ] || git clone "$REPO_URL" SemiSupMDE
    exec bash SemiSupMDE/run.sh "$@"
fi
cd "$HERE"

# PyTorch wheels for the installed NVIDIA driver (override with TORCH_INDEX_URL)
if [ -z "${TORCH_INDEX_URL:-}" ]; then
    command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found: a CUDA GPU with the NVIDIA driver is required"; exit 1; }
    cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
    ge() { [ "$(printf '%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }   # version $1 >= $2
    if ge "$cuda" 13.0; then tag=cu130; elif ge "$cuda" 12.6; then tag=cu126; else tag=cu118; fi
    TORCH_INDEX_URL=https://download.pytorch.org/whl/$tag
fi

# Python environment
PY=.venv/bin/python
if [ ! -x "$PY" ] || [ ! -f .venv/.installed ]; then
    echo "== creating the Python environment in .venv (PyTorch from $TORCH_INDEX_URL)"
    if command -v uv >/dev/null 2>&1; then
        uv venv --allow-existing --python 3.11 .venv
        uv pip install --python "$PY" torch torchvision --index-url "$TORCH_INDEX_URL"
        uv pip install --python "$PY" -r requirements.txt
    else
        command -v python3 >/dev/null 2>&1 || { echo "python3 (>= 3.8) is required"; exit 1; }
        python3 -m venv .venv || { echo "python3 -m venv failed: install python3-venv, or install uv (https://docs.astral.sh/uv/)"; exit 1; }
        "$PY" -m pip install --upgrade pip
        "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
        "$PY" -m pip install -r requirements.txt
    fi
    touch .venv/.installed
fi
"$PY" -c "import torch, sys; torch.cuda.is_available() or sys.exit('no CUDA GPU visible to PyTorch')"

cmd="${1:-eval}"
[ $# -gt 0 ] && shift
echo "== downloading the models"
"$PY" download.py models

case "$cmd" in
    setup)
        ;;
    eval)
        echo "== downloading the KITTI Eigen test split"
        "$PY" download.py kitti test
        echo "== evaluating"
        "$PY" evaluate.py --model_path "${1:-checkpoints/finetuned_KITTI.pth}"
        ;;
    finetune)
        echo "== downloading KITTI (test, val, train) and Virtual KITTI"
        "$PY" download.py kitti test val train
        "$PY" download.py vkitti
        echo "== finetuning"
        "$PY" finetune.py "$@"
        ;;
    *)
        echo "usage: $0 [eval [model.pth] | finetune [args] | setup]"; exit 1
        ;;
esac
