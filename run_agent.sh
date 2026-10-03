#!/usr/bin/env bash
# Verification Agent
# Usage:
#   ./run_agent.sh --mode train_arbiter
#   ./run_agent.sh --mode train_sem --exp-name agent_v3
#   ./run_agent.sh --mode eval --exp-name agent_v3 --limit 20
#
# Layout:
#   checkpoints/          — load weights
#   train_ckpt/<exp>/     — save weights for this exp (then promote)
#   outputs/<exp>/        — logs, features.csv, metrics, predictions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    while IFS= read -r _env_line || [ -n "$_env_line" ]; do
        case "$_env_line" in ''|'#'*) continue ;; esac
        [ "$_env_line" = "${_env_line#*=}" ] && continue
        _env_key=${_env_line%%=*}
        [ -z "${!_env_key+x}" ] && export "$_env_line"
    done < "$SCRIPT_DIR/.env"
fi

MODE="eval"
CONFIG_NAME="agent_v3"
EXP_NAME="agent_v3"
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        -c|--config-name) CONFIG_NAME="$2"; shift 2 ;;
        -e|--exp-name) EXP_NAME="$2"; shift 2 ;;
        # backwards compat: --dump-dir NAME → exp name
        -d|--dump-dir) EXP_NAME="$(basename "$2")"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; shift ;;
    esac
done

CONFIG_NAME="${CONFIG_NAME%.yaml}"
EXP_NAME="$(basename "${EXP_NAME}")"
CONFIG="$SCRIPT_DIR/configs/${CONFIG_NAME}.yaml"
OUT_PATH="$SCRIPT_DIR/outputs/$EXP_NAME"
CKPT_TRAIN="$SCRIPT_DIR/train_ckpt/$EXP_NAME"
mkdir -p "$OUT_PATH" "$CKPT_TRAIN" "$SCRIPT_DIR/checkpoints"

# Prefer ENV_BACKEND, then project .venv, then current PATH python.
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

CMD=(
    "$PYTHON_BIN" -m agent_backend
    --config "$CONFIG"
    --mode "$MODE"
    --exp-name "$EXP_NAME"
    --project-root "$SCRIPT_DIR"
)
[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")

echo "### MODE:      $MODE"
echo "### CONFIG:    $CONFIG"
echo "### EXP:       $EXP_NAME"
echo "### OUTPUTS:   $OUT_PATH"
echo "### TRAIN_CKPT:$CKPT_TRAIN"
echo "### LOAD_CKPT: $SCRIPT_DIR/checkpoints"
echo "### CMD:       ${CMD[*]}"

exec "${CMD[@]}"
