from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from reasonbench.config import load_experiment_config


@dataclass(frozen=True)
class SessionSpec:
    session_name: str
    config_path: str
    checkpoint_path: str
    log_dir: str


SESSION_SPECS = [
    SessionSpec(
        session_name="rb_room_assignment",
        config_path="configs/experiments/full_room_assignment_all_strategies.toml",
        checkpoint_path="outputs/full_room_assignment/results.jsonl",
        log_dir="logs/rb_room_assignment",
    ),
    SessionSpec(
        session_name="rb_truthfulqa",
        config_path="configs/experiments/full_truthfulqa_all_strategies.toml",
        checkpoint_path="outputs/full_truthfulqa/results.jsonl",
        log_dir="logs/rb_truthfulqa",
    ),
    SessionSpec(
        session_name="rb_livebench",
        config_path="configs/experiments/full_livebench_all_strategies.toml",
        checkpoint_path="outputs/full_livebench/results.jsonl",
        log_dir="logs/rb_livebench",
    ),
]


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in handle:
            count += 1
    return count


def tail_lines(path: Path, max_lines: int = 8) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def livebench_example_count(root: Path) -> int:
    if root.is_file():
        return line_count(root)
    total = 0
    for question_file in root.rglob("question.jsonl"):
        total += line_count(question_file)
    return total


def dataset_example_count(config_path: Path) -> tuple[int, int]:
    cfg = load_experiment_config(config_path)
    local_path = Path(cfg.dataset.local_path or "")

    if cfg.dataset.kind == "room_assignment":
        examples = line_count(local_path)
    elif cfg.dataset.kind == "truthfulqa":
        examples = max(line_count(local_path) - 1, 0)
    elif cfg.dataset.kind == "livebench":
        examples = livebench_example_count(local_path)
    else:
        examples = 0

    return examples, len(cfg.strategies)


def running_screen_sessions() -> set[str]:
    proc = subprocess.run(["screen", "-list"], capture_output=True, text=True, check=False)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    names = set()
    for line in text.splitlines():
        if "Detached" not in line and "Attached" not in line:
            continue
        match = re.search(r"\d+\.([^\s]+)", line)
        if match:
            names.add(match.group(1))
    return names


def build_snapshot(repo_root: Path) -> dict:
    active_sessions = running_screen_sessions()
    sessions = []
    sum_completed = 0
    sum_expected = 0

    for spec in SESSION_SPECS:
        cfg_path = repo_root / spec.config_path
        checkpoint_path = repo_root / spec.checkpoint_path
        log_dir = repo_root / spec.log_dir

        examples, strategy_count = dataset_example_count(cfg_path)
        expected_records = examples * strategy_count
        completed_records = line_count(checkpoint_path)

        latest_log = None
        last_lines: list[str] = []
        if log_dir.exists():
            logs = sorted(log_dir.glob("run_*.log"))
            if logs:
                latest_log = str(logs[-1].relative_to(repo_root))
                last_lines = tail_lines(logs[-1], max_lines=8)

        running = spec.session_name in active_sessions
        if completed_records >= expected_records and expected_records > 0:
            state = "completed"
        elif running:
            state = "running"
        elif completed_records > 0:
            state = "stopped_resumable"
        else:
            state = "not_started"

        progress = 0.0
        if expected_records > 0:
            progress = min(100.0, (completed_records / expected_records) * 100.0)

        sessions.append(
            {
                "name": spec.session_name,
                "state": state,
                "running": running,
                "config_path": spec.config_path,
                "checkpoint_path": spec.checkpoint_path,
                "log_dir": spec.log_dir,
                "latest_log": latest_log,
                "completed_records": completed_records,
                "expected_records": expected_records,
                "progress_pct": round(progress, 2),
                "last_log_lines": last_lines,
            }
        )

        sum_completed += completed_records
        sum_expected += expected_records

    overall_progress = 0.0
    if sum_expected > 0:
        overall_progress = min(100.0, (sum_completed / sum_expected) * 100.0)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "completed_records": sum_completed,
            "expected_records": sum_expected,
            "progress_pct": round(overall_progress, 2),
        },
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate session status JSON for the frontend monitor.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(repo_root)
    output_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote status snapshot to {output_path}")


if __name__ == "__main__":
    main()
