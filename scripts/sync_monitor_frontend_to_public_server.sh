#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$ROOT_DIR/logs/web_sync.log"
REMOTE_DEFAULT="public-server:/www/wwwroot/ai.agaii.org/spring2026/"
REMOTE="${RB_SYNC_REMOTE:-$REMOTE_DEFAULT}"
SYNC_INTERVAL_S="${RB_SYNC_INTERVAL_S:-15}"
RSYNC_PATH="${RB_SYNC_RSYNC_PATH:-}"

mkdir -p "$ROOT_DIR/logs"

echo "[$(date -Iseconds)] Sync worker start remote=$REMOTE interval_s=$SYNC_INTERVAL_S rsync_path=${RSYNC_PATH:-default}" >> "$LOG_FILE"

while true; do
  TS="$(date -Iseconds)"
  echo "[$TS] Sync start" >> "$LOG_FILE"

  RSYNC_ARGS=(-az --delete)
  if [[ -n "$RSYNC_PATH" ]]; then
    RSYNC_ARGS+=(--rsync-path "$RSYNC_PATH")
  fi

  if rsync "${RSYNC_ARGS[@]}" "$ROOT_DIR/web/session-monitor/" "$REMOTE" >> "$LOG_FILE" 2>&1; then
    echo "[$TS] Sync done" >> "$LOG_FILE"
  else
    EXIT_CODE=$?
    echo "[$TS] Sync failed (exit_code=$EXIT_CODE)" >> "$LOG_FILE"
  fi

  sleep "$SYNC_INTERVAL_S"
done
