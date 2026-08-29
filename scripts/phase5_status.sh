#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-final_01}"
SESSION="techjam_${RUN_ID}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

tmux has-session -t "$SESSION" 2>/dev/null && echo "session=running" || echo "session=stopped"
if [[ -f "runs/$RUN_ID/run_summary.json" ]]; then
  .venv/bin/python -m json.tool "runs/$RUN_ID/run_summary.json"
elif [[ -f "runs/$RUN_ID/run_log.jsonl" ]]; then
  COMPLETED_ITERATIONS="$(wc -l < "runs/$RUN_ID/run_log.jsonl" | tr -d ' ')"
  echo "completed_iterations=$COMPLETED_ITERATIONS"
  tail -n 1 "runs/$RUN_ID/run_log.jsonl"
elif [[ -f "runs/$RUN_ID/console.log" ]]; then
  tail -n 40 "runs/$RUN_ID/console.log"
else
  echo "run_artifacts=absent"
fi
