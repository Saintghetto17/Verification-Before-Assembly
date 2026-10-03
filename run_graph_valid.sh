#!/usr/bin/env bash
# Run SmolLM graph validation locally with PyTorch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${ENV_BACKEND:-}" ] && [ -x "$ENV_BACKEND/bin/python" ]; then
    PYTHON_BIN="$ENV_BACKEND/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Error: no Python interpreter found. Create .venv or set ENV_BACKEND."
    exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

mkdir -p "$SCRIPT_DIR/outputs/agent_v6"

CMD=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/run_validation_graph.py"
    --input "$SCRIPT_DIR/graph_creator_model_dataset/val.jsonl"
    --checkpoint "$SCRIPT_DIR/checkpoints/smollm_graph"
    --output "$SCRIPT_DIR/outputs/agent_v6/validation_results.json"
)
CMD+=("$@")

echo "### GRAPH VALIDATION"
echo "### OUTPUT: $SCRIPT_DIR/outputs/agent_v6/validation_results.json"
echo "### CMD:    ${CMD[*]}"

exec "${CMD[@]}"
