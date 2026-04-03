#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_RUN_MANIFEST="$ROOT_DIR/web/session-monitor/active_run.json"

KEEP_EXISTING=0
RUN_TAG=""

for arg in "$@"; do
  case "$arg" in
    --keep-existing)
      KEEP_EXISTING=1
      ;;
    --run-tag=*)
      RUN_TAG="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--keep-existing] [--run-tag=<custom_tag>]"
      exit 1
      ;;
  esac
done

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed. Install it first: sudo apt-get install -y screen"
  exit 1
fi

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/web/session-monitor"

PREP_ARGS=("--repo-root" "$ROOT_DIR" "--manifest" "web/session-monitor/active_run.json")
if [[ "$KEEP_EXISTING" -eq 1 ]]; then
  PREP_ARGS+=("--skip-archive")
fi

if [[ "$KEEP_EXISTING" -eq 0 ]]; then
  while IFS= read -r session_name; do
    [[ -z "$session_name" ]] && continue
    if [[ "$session_name" == rb_* ]]; then
      echo "Stopping session: $session_name"
      screen -S "$session_name" -X quit || true
    fi
  done < <(screen -list | awk '/[0-9]+\.[^[:space:]]+[[:space:]]+\((Detached|Attached)\)/ { split($1, parts, "."); print parts[2] }')
fi

echo "Preparing timestamped rerun manifest..."
if [[ -n "$RUN_TAG" ]]; then
  (
    cd "$ROOT_DIR"
    PYTHONPATH=src /usr/bin/python3 scripts/prepare_timestamped_runs.py "${PREP_ARGS[@]}" --run-tag "$RUN_TAG"
  )
else
  (
    cd "$ROOT_DIR"
    PYTHONPATH=src /usr/bin/python3 scripts/prepare_timestamped_runs.py "${PREP_ARGS[@]}"
  )
fi

if [[ ! -f "$ACTIVE_RUN_MANIFEST" ]]; then
  echo "Missing active run manifest: $ACTIVE_RUN_MANIFEST"
  exit 1
fi

launch_if_missing() {
  local session_name="$1"
  local launch_cmd="$2"

  if screen -list | grep -q "[.]${session_name}[[:space:]]"; then
    echo "Session already running: $session_name"
    return 0
  fi

  echo "Starting session: $session_name"
  screen -S "$session_name" -dm bash -lc "$launch_cmd"
}

echo "Launching timestamped experiment sessions..."
while IFS='|' read -r session_name config_path; do
  [[ -z "$session_name" ]] && continue
  launch_if_missing "$session_name" "cd '$ROOT_DIR' && bash scripts/run_experiment_with_resume.sh '$config_path' '$session_name'"
done < <(
  /usr/bin/python3 - <<PY
import json
from pathlib import Path

manifest = json.loads(Path("$ACTIVE_RUN_MANIFEST").read_text(encoding="utf-8"))
for item in manifest.get("sessions", []):
    print(f"{item['session_name']}|{item['config_path']}")
PY
)

launch_if_missing "rb_status_updater" "cd '$ROOT_DIR' && while true; do PYTHONPATH=src /usr/bin/python3 scripts/update_status_snapshot.py --output web/session-monitor/status.json >> logs/status_updater.log 2>&1; sleep 10; done"

echo ""
echo "Active run metadata:"
/usr/bin/python3 - <<PY
import json
from pathlib import Path

manifest = json.loads(Path("$ACTIVE_RUN_MANIFEST").read_text(encoding="utf-8"))
print(f"run_tag: {manifest.get('run_tag', '--')}")
print(f"run_title: {manifest.get('run_title', '--')}")
print(f"sessions: {len(manifest.get('sessions', []))}")
PY

echo ""
echo "Active screen sessions:"
screen -list || true
