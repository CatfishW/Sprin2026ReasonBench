from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reasonbench.config import ExperimentConfig, load_experiment_config


@dataclass(frozen=True)
class SessionSpec:
    session_name: str
    display_name: str
    config_path: str
    checkpoint_path: str
    log_dir: str
    output_dir: str | None = None
    run_tag: str | None = None


ACTIVE_RUN_MANIFEST = "web/session-monitor/active_run.json"

DEFAULT_SESSION_SPECS = [
    SessionSpec(
        session_name="rb_room_assignment",
        display_name="Room Assignment (4B)",
        config_path="configs/experiments/full_room_assignment_all_strategies.toml",
        checkpoint_path="outputs/full_room_assignment/results.jsonl",
        log_dir="logs/rb_room_assignment",
        output_dir="outputs/full_room_assignment",
    ),
    SessionSpec(
        session_name="rb_truthfulqa",
        display_name="TruthfulQA (4B)",
        config_path="configs/experiments/full_truthfulqa_all_strategies.toml",
        checkpoint_path="outputs/full_truthfulqa/results.jsonl",
        log_dir="logs/rb_truthfulqa",
        output_dir="outputs/full_truthfulqa",
    ),
    SessionSpec(
        session_name="rb_livebench",
        display_name="LiveBench (27B)",
        config_path="configs/experiments/full_livebench_all_strategies.toml",
        checkpoint_path="outputs/full_livebench/results.jsonl",
        log_dir="logs/rb_livebench",
        output_dir="outputs/full_livebench",
    ),
    SessionSpec(
        session_name="rb_ai2_arc",
        display_name="AI2-ARC (4B)",
        config_path="configs/experiments/full_ai2_arc_all_strategies.toml",
        checkpoint_path="outputs/full_ai2_arc/results.jsonl",
        log_dir="logs/rb_ai2_arc",
        output_dir="outputs/full_ai2_arc",
    ),
    SessionSpec(
        session_name="rb_room_assignment_27b",
        display_name="Room Assignment (27B)",
        config_path="configs/experiments/full_room_assignment_all_strategies_27b.toml",
        checkpoint_path="outputs/full_room_assignment_27b/results.jsonl",
        log_dir="logs/rb_room_assignment_27b",
        output_dir="outputs/full_room_assignment_27b",
    ),
    SessionSpec(
        session_name="rb_truthfulqa_27b",
        display_name="TruthfulQA (27B)",
        config_path="configs/experiments/full_truthfulqa_all_strategies_27b.toml",
        checkpoint_path="outputs/full_truthfulqa_27b/results.jsonl",
        log_dir="logs/rb_truthfulqa_27b",
        output_dir="outputs/full_truthfulqa_27b",
    ),
    SessionSpec(
        session_name="rb_ai2_arc_27b",
        display_name="AI2-ARC (27B)",
        config_path="configs/experiments/full_ai2_arc_all_strategies_27b.toml",
        checkpoint_path="outputs/full_ai2_arc_27b/results.jsonl",
        log_dir="logs/rb_ai2_arc_27b",
        output_dir="outputs/full_ai2_arc_27b",
    ),
]

METRIC_PRIORITY = [
    "accuracy",
    "room_exact_accuracy",
    "entity_room_accuracy",
    "all_rooms_exact",
    "format_valid",
    "truth_delta",
    "proxy_exact_match",
    "proxy_contains_match",
    "is_unscorable",
    "text_match_score",
]

STUCK_STALE_ATTEMPT_THRESHOLD = 3


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


def _resolve_local_path(repo_root: Path, local_path: str | None) -> Path:
    raw = Path(local_path or "")
    if raw.is_absolute():
        return raw
    return repo_root / raw


def livebench_example_count(root: Path) -> int:
    if root.is_file():
        return line_count(root)
    total = 0
    for question_file in root.rglob("question.jsonl"):
        total += line_count(question_file)
    return total


def jsonl_example_count(root: Path) -> int:
    if root.is_file():
        return line_count(root)
    total = 0
    for file_path in root.rglob("*.jsonl"):
        total += line_count(file_path)
    return total


def dataset_example_count(config: ExperimentConfig, repo_root: Path) -> tuple[int, int]:
    local_path = _resolve_local_path(repo_root, config.dataset.local_path)

    if config.dataset.kind == "room_assignment":
        examples = line_count(local_path)
    elif config.dataset.kind == "truthfulqa":
        examples = max(line_count(local_path) - 1, 0)
    elif config.dataset.kind == "livebench":
        examples = livebench_example_count(local_path)
    elif config.dataset.kind == "ai2_arc":
        examples = jsonl_example_count(local_path)
    else:
        examples = 0

    return examples, len(config.strategies)


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


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_log_signals(lines: list[str]) -> tuple[str, str]:
    last_progress = ""
    last_error = ""
    for line in reversed(lines):
        if not last_progress and "progress total=" in line:
            last_progress = line
        if not last_error and "error example_id=" in line:
            last_error = line
        if last_progress and last_error:
            break
    return last_progress, last_error


def _parse_log_attempt_info(log_path: Path) -> dict[str, Any]:
    exit_code: int | None = None
    checkpoint_added: int | None = None
    stale_exit = False
    stale_line = ""

    for line in tail_lines(log_path, max_lines=200):
        stripped = line.strip()
        if stripped.startswith("exit_code:"):
            try:
                exit_code = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                exit_code = None
        elif stripped.startswith("checkpoint_records_added:"):
            try:
                checkpoint_added = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                checkpoint_added = None
        elif "consecutive stale attempts" in stripped:
            stale_exit = True
            stale_line = stripped

    return {
        "exit_code": exit_code,
        "checkpoint_records_added": checkpoint_added,
        "stale_exit": stale_exit,
        "stale_line": stale_line,
    }


def _stale_attempt_state(log_dir: Path) -> dict[str, Any]:
    if not log_dir.exists():
        return {
            "stale_retry_streak": 0,
            "stuck": False,
            "stuck_reason": "",
            "last_exit_code": "",
        }

    logs = sorted(log_dir.glob("run_*.log"))
    if not logs:
        return {
            "stale_retry_streak": 0,
            "stuck": False,
            "stuck_reason": "",
            "last_exit_code": "",
        }

    stale_retry_streak = 0
    stuck_reason = ""
    last_exit_code = ""

    for log_path in reversed(logs[-12:]):
        info = _parse_log_attempt_info(log_path)
        exit_code = info["exit_code"]
        checkpoint_added = info["checkpoint_records_added"]

        if exit_code is None:
            continue

        last_exit_code = str(exit_code)
        if exit_code != 0 and checkpoint_added == 0:
            stale_retry_streak += 1
            if info["stale_exit"] and not stuck_reason:
                stuck_reason = info["stale_line"]
            continue

        break

    stuck = stale_retry_streak >= STUCK_STALE_ATTEMPT_THRESHOLD
    if stuck and not stuck_reason:
        stuck_reason = f"stale_retry_streak={stale_retry_streak}"

    return {
        "stale_retry_streak": stale_retry_streak,
        "stuck": stuck,
        "stuck_reason": stuck_reason,
        "last_exit_code": last_exit_code,
    }


def _checkpoint_live_summary(checkpoint_path: Path, max_leaderboard_rows: int = 5) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "records_scanned": 0,
            "best_strategy": "",
            "leaderboard": [],
            "metric_rollup": {},
        }

    strategy_acc: dict[str, dict[str, Any]] = {}
    metric_acc: dict[str, list[float]] = {}
    records_scanned = 0

    with checkpoint_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            records_scanned += 1
            strategy = str(row.get("strategy_name") or "unknown")
            state = strategy_acc.setdefault(
                strategy,
                {
                    "records": 0,
                    "primary_sum": 0.0,
                    "api_sum": 0.0,
                    "wall_sum": 0.0,
                    "cache_sum": 0.0,
                    "metric_acc": {},
                },
            )

            state["records"] += 1
            state["primary_sum"] += float(row.get("primary_score", 0.0))
            state["api_sum"] += float(row.get("api_calls", 0.0))
            state["wall_sum"] += float(row.get("wall_time_s", 0.0))
            state["cache_sum"] += 1.0 if row.get("from_cache") else 0.0

            for metric_name, metric_value in (row.get("metrics") or {}).items():
                numeric = _as_number(metric_value)
                if numeric is None:
                    continue

                group = metric_acc.setdefault(metric_name, [0.0, 0.0])
                group[0] += numeric
                group[1] += 1.0

                strategy_metric_acc = state["metric_acc"].setdefault(metric_name, [0.0, 0.0])
                strategy_metric_acc[0] += numeric
                strategy_metric_acc[1] += 1.0

    leaderboard: list[dict[str, Any]] = []
    for strategy_name, state in strategy_acc.items():
        records = state["records"]
        if records <= 0:
            continue

        means = {
            metric_name: round(acc[0] / acc[1], 4)
            for metric_name, acc in state["metric_acc"].items()
            if acc[1] > 0 and metric_name in METRIC_PRIORITY
        }
        leaderboard.append(
            {
                "strategy": strategy_name,
                "records": records,
                "mean_primary_score": round(state["primary_sum"] / records, 4),
                "mean_api_calls": round(state["api_sum"] / records, 3),
                "mean_wall_time_s": round(state["wall_sum"] / records, 3),
                "cache_hit_rate": round(state["cache_sum"] / records, 4),
                "metric_means": means,
            }
        )

    leaderboard.sort(key=lambda item: item["mean_primary_score"], reverse=True)
    leaderboard = leaderboard[:max_leaderboard_rows]

    metric_rollup: dict[str, float] = {}
    for metric_name in METRIC_PRIORITY:
        acc = metric_acc.get(metric_name)
        if not acc or acc[1] <= 0:
            continue
        metric_rollup[metric_name] = round(acc[0] / acc[1], 4)

    best_strategy = leaderboard[0]["strategy"] if leaderboard else ""
    return {
        "records_scanned": records_scanned,
        "best_strategy": best_strategy,
        "leaderboard": leaderboard,
        "metric_rollup": metric_rollup,
    }


def _load_session_specs(repo_root: Path) -> tuple[list[SessionSpec], dict[str, Any]]:
    manifest_path = repo_root / ACTIVE_RUN_MANIFEST
    runtime_meta: dict[str, Any] = {
        "run_tag": "",
        "run_title": "Spring 2026 Live Monitor",
        "launched_at": "",
        "run_manifest": "",
    }

    if not manifest_path.exists():
        return list(DEFAULT_SESSION_SPECS), runtime_meta

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return list(DEFAULT_SESSION_SPECS), runtime_meta

    specs: list[SessionSpec] = []
    run_tag = str(manifest.get("run_tag") or "")
    for item in manifest.get("sessions") or []:
        session_name = str(item.get("session_name") or "").strip()
        config_path = str(item.get("config_path") or "").strip()
        checkpoint_path = str(item.get("checkpoint_path") or "").strip()
        log_dir = str(item.get("log_dir") or "").strip()
        if not session_name or not config_path or not checkpoint_path or not log_dir:
            continue
        specs.append(
            SessionSpec(
                session_name=session_name,
                display_name=str(item.get("display_name") or session_name),
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                log_dir=log_dir,
                output_dir=str(item.get("output_dir") or "").strip() or None,
                run_tag=str(item.get("run_tag") or run_tag or "").strip() or None,
            )
        )

    if specs:
        runtime_meta = {
            "run_tag": run_tag,
            "run_title": str(manifest.get("run_title") or f"Spring 2026 • {run_tag}"),
            "launched_at": str(manifest.get("launched_at") or ""),
            "run_manifest": str(manifest_path.relative_to(repo_root)),
        }
        return specs, runtime_meta

    return list(DEFAULT_SESSION_SPECS), runtime_meta


def build_snapshot(repo_root: Path) -> dict[str, Any]:
    active_sessions = running_screen_sessions()
    session_specs, run_meta = _load_session_specs(repo_root)
    sessions: list[dict[str, Any]] = []
    sum_completed = 0
    sum_expected = 0

    for spec in session_specs:
        cfg_path = repo_root / spec.config_path
        checkpoint_path = repo_root / spec.checkpoint_path
        log_dir = repo_root / spec.log_dir

        examples = 0
        strategy_count = 0
        dataset_kind = ""
        model = ""
        if cfg_path.exists():
            cfg = load_experiment_config(cfg_path)
            examples, strategy_count = dataset_example_count(cfg, repo_root)
            dataset_kind = cfg.dataset.kind
            model = cfg.client.model

        expected_records = examples * strategy_count
        completed_records = line_count(checkpoint_path)

        latest_log = None
        last_lines: list[str] = []
        stale_state = {
            "stale_retry_streak": 0,
            "stuck": False,
            "stuck_reason": "",
            "last_exit_code": "",
        }
        if log_dir.exists():
            logs = sorted(log_dir.glob("run_*.log"))
            if logs:
                latest_log = str(logs[-1].relative_to(repo_root))
                last_lines = tail_lines(logs[-1], max_lines=12)
            stale_state = _stale_attempt_state(log_dir)

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

        last_progress_line, last_error_line = _extract_log_signals(last_lines)
        live_summary = _checkpoint_live_summary(checkpoint_path)

        sessions.append(
            {
                "name": spec.session_name,
                "display_name": spec.display_name,
                "state": state,
                "running": running,
                "run_tag": spec.run_tag or run_meta.get("run_tag") or "",
                "config_path": spec.config_path,
                "output_dir": spec.output_dir or str(Path(spec.checkpoint_path).parent),
                "checkpoint_path": spec.checkpoint_path,
                "log_dir": spec.log_dir,
                "latest_log": latest_log,
                "dataset_kind": dataset_kind,
                "model": model,
                "strategy_count": strategy_count,
                "example_count": examples,
                "completed_records": completed_records,
                "expected_records": expected_records,
                "remaining_records": max(expected_records - completed_records, 0),
                "progress_pct": round(progress, 2),
                "last_progress_line": last_progress_line,
                "last_error_line": last_error_line,
                "stale_retry_streak": stale_state["stale_retry_streak"],
                "stuck": stale_state["stuck"],
                "stuck_reason": stale_state["stuck_reason"],
                "last_exit_code": stale_state["last_exit_code"],
                "last_log_lines": last_lines,
                "live_summary": live_summary,
            }
        )

        sum_completed += completed_records
        sum_expected += expected_records

    overall_progress = 0.0
    if sum_expected > 0:
        overall_progress = min(100.0, (sum_completed / sum_expected) * 100.0)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_meta.get("run_tag") or "",
        "run_title": run_meta.get("run_title") or "Spring 2026 Live Monitor",
        "launched_at": run_meta.get("launched_at") or "",
        "run_manifest": run_meta.get("run_manifest") or "",
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
