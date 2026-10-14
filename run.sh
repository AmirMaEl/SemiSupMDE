#!/usr/bin/env bash
# One command to install, download and run:
#
#   ./run.sh                      evaluate the finetuned model on the KITTI Eigen test split (downloads ~2 GB)
#   ./run.sh eval [model]         evaluate finetuned (default), pretrained or a checkpoint path
#   ./run.sh finetune [args]      vKITTI -> KITTI finetuning (downloads ~40 GB, needs ~35 GB of GPU memory)
#   ./run.sh predict img.png ...  predict depth for your own images (writes <name>_depth.png / .npy)
#   ./run.sh setup                only create the environment and download the models
#
# Any other arguments go to the semisupmde command line tool, e.g. ./run.sh download kitti val.
# Also works without cloning first:  curl -sSL https://raw.githubusercontent.com/AmirMaEl/SemiSupMDE/main/run.sh | bash
# The Python environment goes to .venv/, models to checkpoints/ and data to data/ (set SEMISUPMDE_HOME,
# KITTI_ROOT or VKITTI_ROOT to use other locations).
set -euo pipefail

REPO_URL=https://github.com/AmirMaEl/SemiSupMDE.git
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"

# Not inside the repo (e.g. piped from curl): clone it and continue there.
if [ ! -f "$HERE/pyproject.toml" ] || [ ! -d "$HERE/semisupmde" ]; then
    [ -d SemiSupMDE ] || git clone "$REPO_URL" SemiSupMDE
    exec bash SemiSupMDE/run.sh "$@"
fi
export SEMISUPMDE_HOME="${SEMISUPMDE_HOME:-$HERE}"

# PyTorch wheels for the installed NVIDIA driver (override with TORCH_INDEX_URL)
if [ -z "${TORCH_INDEX_URL:-}" ]; then
    command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found: a CUDA GPU with the NVIDIA driver is required"; exit 1; }
    cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
    ge() { [ "$(printf '%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }   # version $1 >= $2
    if ge "$cuda" 13.0; then tag=cu130; elif ge "$cuda" 12.6; then tag=cu126; else tag=cu118; fi
    TORCH_INDEX_URL=https://download.pytorch.org/whl/$tag
fi

# Python environment
VENV="$HERE/.venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ] || [ ! -f "$VENV/.installed" ] || [ "$HERE/pyproject.toml" -nt "$VENV/.installed" ]; then
    echo "== creating the Python environment in .venv (PyTorch from $TORCH_INDEX_URL)"
    if command -v uv >/dev/null 2>&1; then
        uv venv --allow-existing --python 3.11 "$VENV"
        uv pip install --python "$PY" torch torchvision --index-url "$TORCH_INDEX_URL"
        uv pip install --python "$PY" -e "$HERE"
    else
        command -v python3 >/dev/null 2>&1 || { echo "python3 (>= 3.8) is required"; exit 1; }
        python3 -m venv "$VENV" || { echo "python3 -m venv failed: install python3-venv, or install uv (https://docs.astral.sh/uv/)"; exit 1; }
        "$PY" -m pip install --upgrade pip
        "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
        "$PY" -m pip install -e "$HERE"
    fi
    touch "$VENV/.installed"
fi
"$PY" -c "import torch; torch.cuda.is_available() or print('warning: no CUDA GPU visible to PyTorch, running on the CPU')"

CLI="$VENV/bin/semisupmde"
cmd="${1:-eval}"
[ $# -gt 0 ] && shift
case "$cmd" in
    setup)
        "$CLI" download models ;;
    eval)
        if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then model="$1"; shift; else model=finetuned; fi
        "$CLI" eval --model "$model" "$@" ;;
    *)
        "$CLI" "$cmd" "$@" ;;
esac
