#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$ROOT_DIR/logs/web_sync.log"
REMOTE="public-server:/www/wwwroot/ai.agaii.org/spring2026/"

mkdir -p "$ROOT_DIR/logs"

while true; do
  TS="$(date -Iseconds)"
  {
    echo "[$TS] Sync start"
    rsync -az --delete "$ROOT_DIR/web/session-monitor/" "$REMOTE"
    echo "[$TS] Sync done"
  } >> "$LOG_FILE" 2>&1

  sleep 15
done
