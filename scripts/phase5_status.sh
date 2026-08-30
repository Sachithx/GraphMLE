#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-final_01}"
SESSION="techjam_${RUN_ID}"
SERVICE="techjam-${RUN_ID//_/-}.service"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Prefer the project environment over the system interpreter, so the status
# helper works on the Linux run host and on a developer machine alike.
if [[ -n "${TECHJAM_PYTHON:-}" ]]; then
  PYTHON_BIN="$TECHJAM_PYTHON"
elif [[ -x .venv-phase5/bin/python ]]; then
  PYTHON_BIN=".venv-phase5/bin/python"
elif [[ -x .venv-linux/bin/python ]]; then
  PYTHON_BIN=".venv-linux/bin/python"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# A scored run may be supervised either by a user systemd unit or by tmux.
if systemctl --user is-active --quiet "$SERVICE" 2>/dev/null; then
  echo "runner=systemd"
  echo "service=running"
elif tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "runner=tmux"
  echo "session=running"
else
  echo "runner=none"
  echo "service=stopped"
fi

if [[ -f "runs/$RUN_ID/run_summary.json" ]]; then
  "$PYTHON_BIN" -m json.tool "runs/$RUN_ID/run_summary.json"
elif [[ -f "runs/$RUN_ID/run_log.jsonl" ]]; then
  COMPLETED_ITERATIONS="$(wc -l < "runs/$RUN_ID/run_log.jsonl" | tr -d ' ')"
  echo "completed_iterations=$COMPLETED_ITERATIONS"
  tail -n 1 "runs/$RUN_ID/run_log.jsonl"
elif [[ -f "runs/$RUN_ID/console.log" ]]; then
  tail -n 40 "runs/$RUN_ID/console.log"
else
  echo "run_artifacts=absent"
fi
