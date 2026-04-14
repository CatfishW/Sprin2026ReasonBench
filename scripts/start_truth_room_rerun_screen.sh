#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_RUN_MANIFEST="$ROOT_DIR/web/session-monitor/active_run.json"
RUN_TAG=""

for arg in "$@"; do
  case "$arg" in
    --run-tag=*)
      RUN_TAG="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--run-tag=<custom_tag>]"
      exit 1
      ;;
  esac
done

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed. Install it first: sudo apt-get install -y screen"
  exit 1
fi

PYTHON_BIN="/usr/bin/python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

echo "Using python: $PYTHON_BIN"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/web/session-monitor" "$ROOT_DIR/external"

echo "Stopping all existing screen sessions..."
while IFS= read -r session_name; do
  [[ -z "$session_name" ]] && continue
  echo "Stopping session: $session_name"
  screen -S "$session_name" -X quit || true
done < <(screen -list | sed -nE 's/^[[:space:]]*[0-9]+\.([^[:space:]]+)[[:space:]].*\((Detached|Attached)\).*/\1/p')

echo "Verifying no sessions remain..."
screen -list || true

OFFICIAL_REPO="$ROOT_DIR/external/TruthfulQA"
if [[ ! -f "$OFFICIAL_REPO/truthfulqa/evaluate.py" ]]; then
  echo "Cloning TruthfulQA official repo into $OFFICIAL_REPO"
  rm -rf "$OFFICIAL_REPO"
  git clone --depth 1 https://github.com/sylinrl/TruthfulQA.git "$OFFICIAL_REPO"
fi

if [[ "$PYTHON_BIN" == "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Installing minimal TruthfulQA official scoring dependencies into .venv"
  "$PYTHON_BIN" -m pip install -q pandas numpy openai sacrebleu rouge-score || true
fi

PREP_ARGS=(
  "--repo-root" "$ROOT_DIR"
  "--manifest" "web/session-monitor/active_run.json"
  "--include-base-sessions"
  "rb_room_assignment"
  "rb_truthfulqa"
  "rb_room_assignment_27b"
  "rb_truthfulqa_27b"
)

echo "Preparing timestamped configs + manifest for room_assignment/truthfulqa only..."
if [[ -n "$RUN_TAG" ]]; then
  (
    cd "$ROOT_DIR"
    PYTHONPATH=src "$PYTHON_BIN" scripts/prepare_timestamped_runs.py "${PREP_ARGS[@]}" --run-tag "$RUN_TAG"
  )
else
  (
    cd "$ROOT_DIR"
    PYTHONPATH=src "$PYTHON_BIN" scripts/prepare_timestamped_runs.py "${PREP_ARGS[@]}"
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

echo "Launching selected experiment sessions..."
while IFS='|' read -r session_name config_path; do
  [[ -z "$session_name" ]] && continue
  launch_if_missing "$session_name" "cd '$ROOT_DIR' && RB_PYTHON_BIN='$PYTHON_BIN' bash scripts/run_experiment_with_resume.sh '$config_path' '$session_name'"
done < <(
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

manifest = json.loads(Path("$ACTIVE_RUN_MANIFEST").read_text(encoding="utf-8"))
for item in manifest.get("sessions", []):
    print(f"{item['session_name']}|{item['config_path']}")
PY
)

launch_if_missing "rb_status_updater" "cd '$ROOT_DIR' && while true; do TRUTHFULQA_OFFICIAL_REPO='$OFFICIAL_REPO' RB_TRUTHFULQA_OFFICIAL_MIN_INTERVAL_S=120 PYTHONPATH=src '$PYTHON_BIN' scripts/update_status_snapshot.py --output web/session-monitor/status.json >> logs/status_updater.log 2>&1; sleep 10; done"

echo ""
echo "Active run metadata:"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

manifest = json.loads(Path("$ACTIVE_RUN_MANIFEST").read_text(encoding="utf-8"))
print(f"run_tag: {manifest.get('run_tag', '--')}")
print(f"run_title: {manifest.get('run_title', '--')}")
print(f"sessions: {len(manifest.get('sessions', []))}")
for item in manifest.get("sessions", []):
    print(f"  - {item.get('display_name')} [{item.get('session_name')}] -> {item.get('config_path')}")
PY

echo ""
echo "Active screen sessions:"
screen -list || true
