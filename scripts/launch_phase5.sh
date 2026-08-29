#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-final_01}"
CONFIG="${2:-configs/run.yaml}"
GPU_INDEX="${3:-1}"
SESSION="techjam_${RUN_ID}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
if [[ -d "runs/$RUN_ID" ]] && find "runs/$RUN_ID" -mindepth 1 ! -name console.log -print -quit | grep -q .; then
  echo "run directory already contains artifacts: runs/$RUN_ID" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_DIR' && bash scripts/run_scored.sh '$RUN_ID' '$CONFIG' '$GPU_INDEX'"
echo "started session=$SESSION run_id=$RUN_ID gpu_index=$GPU_INDEX"
echo "attach: tmux attach -t $SESSION"

