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

CHECKPOINT_PATH="$(
  cd "$ROOT_DIR"
  PYTHONPATH=src /usr/bin/python3 - <<PY
from reasonbench.config import load_experiment_config

cfg = load_experiment_config("$CONFIG_PATH")
print(cfg.output.checkpoint_path)
PY
)"
CHECKPOINT_FILE="$ROOT_DIR/$CHECKPOINT_PATH"

MAX_ATTEMPTS="${RB_MAX_ATTEMPTS:-0}"
MAX_STALE_ATTEMPTS="${RB_MAX_STALE_ATTEMPTS:-8}"
STALE_ATTEMPTS=0

line_count() {
  local target="$1"
  if [[ ! -f "$target" ]]; then
    echo 0
    return 0
  fi
  wc -l < "$target" | tr -d '[:space:]'
}

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
    echo "checkpoint_path: $CHECKPOINT_PATH"
    echo "cwd: $ROOT_DIR"
    echo "command: PYTHONUNBUFFERED=1 PYTHONPATH=src /usr/bin/python3 -u -m reasonbench.cli.run --config $CONFIG_PATH"
    echo "============================================================"
  } | tee -a "$RUN_LOG"

  BEFORE_COUNT="$(line_count "$CHECKPOINT_FILE")"

  set +e
  (
    cd "$ROOT_DIR"
    PYTHONUNBUFFERED=1 PYTHONPATH=src /usr/bin/python3 -u -m reasonbench.cli.run --config "$CONFIG_PATH"
  ) 2>&1 | tee -a "$RUN_LOG"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  AFTER_COUNT="$(line_count "$CHECKPOINT_FILE")"
  NEW_RECORDS=$((AFTER_COUNT - BEFORE_COUNT))
  if [[ $NEW_RECORDS -lt 0 ]]; then
    NEW_RECORDS=0
  fi

  {
    echo "finished_at: $(date -Iseconds)"
    echo "exit_code: $EXIT_CODE"
    echo "checkpoint_records_before: $BEFORE_COUNT"
    echo "checkpoint_records_after: $AFTER_COUNT"
    echo "checkpoint_records_added: $NEW_RECORDS"
  } | tee -a "$RUN_LOG"

  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Run finished successfully. Exiting launcher loop." | tee -a "$RUN_LOG"
    exit 0
  fi

  if [[ $NEW_RECORDS -eq 0 ]]; then
    STALE_ATTEMPTS=$((STALE_ATTEMPTS + 1))
  else
    STALE_ATTEMPTS=0
  fi

  if [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] && [[ "$MAX_ATTEMPTS" -gt 0 ]] && [[ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]]; then
    echo "Reached RB_MAX_ATTEMPTS=$MAX_ATTEMPTS without successful completion. Exiting." | tee -a "$RUN_LOG"
    exit 2
  fi

  if [[ "$MAX_STALE_ATTEMPTS" =~ ^[0-9]+$ ]] && [[ "$MAX_STALE_ATTEMPTS" -gt 0 ]] && [[ "$STALE_ATTEMPTS" -ge "$MAX_STALE_ATTEMPTS" ]]; then
    echo "Detected $STALE_ATTEMPTS consecutive stale attempts (no checkpoint progress). Exiting." | tee -a "$RUN_LOG"
    exit 3
  fi

  echo "Run failed (exit $EXIT_CODE). stale_attempts=$STALE_ATTEMPTS. Will resume from checkpoint after 20s..." | tee -a "$RUN_LOG"
  ATTEMPT=$((ATTEMPT + 1))
  sleep 20

done
