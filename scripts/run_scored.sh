#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?usage: run_scored.sh RUN_ID [CONFIG] [GPU_INDEX]}"
CONFIG="${2:-configs/run.yaml}"
GPU_INDEX="${3:-1}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
.venv/bin/python -m scripts.phase5_preflight \
  --config "$CONFIG" \
  --require-gpus 1 \
  --output "runs/${RUN_ID}_preflight.json"

mkdir -p "runs/$RUN_ID"
set -o pipefail
.venv/bin/python -m agent.run --config "$CONFIG" --run-id "$RUN_ID" \
  2>&1 | tee "runs/$RUN_ID/console.log"

