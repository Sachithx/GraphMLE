#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?usage: run_scored.sh RUN_ID [CONFIG] [GPU_INDEX]}"
CONFIG="${2:-configs/run.yaml}"
GPU_INDEX="${3:-1}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"
for env_file in .env keys.env; do
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

if [[ -n "${TECHJAM_PYTHON:-}" ]]; then
  PYTHON="$TECHJAM_PYTHON"
elif [[ -x .venv-phase5/bin/python ]]; then
  PYTHON=.venv-phase5/bin/python
elif [[ -x .venv-linux/bin/python ]]; then
  PYTHON=.venv-linux/bin/python
elif [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  echo "no usable project Python environment found" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
"$PYTHON" -m scripts.phase5_preflight \
  --config "$CONFIG" \
  --require-gpus 1 \
  --output "runs/${RUN_ID}_preflight.json"

mkdir -p "runs/$RUN_ID"
set -o pipefail
"$PYTHON" -m agent.run --config "$CONFIG" --run-id "$RUN_ID" \
  2>&1 | tee "runs/$RUN_ID/console.log"
