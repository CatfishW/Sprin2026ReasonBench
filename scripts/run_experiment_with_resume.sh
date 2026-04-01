#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_path> <session_name>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$1"
SESSION_NAME="$2"
LOG_DIR="$ROOT_DIR/logs/$SESSION_NAME"
mkdir -p "$LOG_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

ATTEMPT=1
while true; do
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_LOG="$LOG_DIR/run_${TS}.log"

  {
    echo "============================================================"
    echo "session_name: $SESSION_NAME"
    echo "attempt: $ATTEMPT"
    echo "started_at: $(date -Iseconds)"
    echo "config_path: $CONFIG_PATH"
    echo "cwd: $ROOT_DIR"
    echo "command: PYTHONUNBUFFERED=1 PYTHONPATH=src /usr/bin/python3 -u -m reasonbench.cli.run --config $CONFIG_PATH"
    echo "============================================================"
  } | tee -a "$RUN_LOG"

  set +e
  (
    cd "$ROOT_DIR"
    PYTHONUNBUFFERED=1 PYTHONPATH=src /usr/bin/python3 -u -m reasonbench.cli.run --config "$CONFIG_PATH"
  ) 2>&1 | tee -a "$RUN_LOG"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  {
    echo "finished_at: $(date -Iseconds)"
    echo "exit_code: $EXIT_CODE"
  } | tee -a "$RUN_LOG"

  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Run finished successfully. Exiting launcher loop." | tee -a "$RUN_LOG"
    exit 0
  fi

  echo "Run failed (exit $EXIT_CODE). Will resume from checkpoint after 20s..." | tee -a "$RUN_LOG"
  ATTEMPT=$((ATTEMPT + 1))
  sleep 20

done
