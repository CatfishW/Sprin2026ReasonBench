#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed. Install it first: sudo apt-get install -y screen"
  exit 1
fi

mkdir -p "$ROOT_DIR/logs"

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

launch_if_missing "rb_room_assignment" "cd '$ROOT_DIR' && bash scripts/run_experiment_with_resume.sh configs/experiments/full_room_assignment_all_strategies.toml rb_room_assignment"
launch_if_missing "rb_truthfulqa" "cd '$ROOT_DIR' && bash scripts/run_experiment_with_resume.sh configs/experiments/full_truthfulqa_all_strategies.toml rb_truthfulqa"
launch_if_missing "rb_livebench" "cd '$ROOT_DIR' && bash scripts/run_experiment_with_resume.sh configs/experiments/full_livebench_all_strategies.toml rb_livebench"

launch_if_missing "rb_status_updater" "cd '$ROOT_DIR' && while true; do PYTHONPATH=src /usr/bin/python3 scripts/update_status_snapshot.py --output web/session-monitor/status.json >> logs/status_updater.log 2>&1; sleep 10; done"

echo ""
echo "Active screen sessions:"
screen -list || true
