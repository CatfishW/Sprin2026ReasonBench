from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reasonbench.config import ExperimentConfig, load_experiment_config
from reasonbench.integrations.truthfulqa_official import run_official_truthfulqa


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
TRUTHFULQA_DEFAULT_OFFICIAL_REPO = "external/TruthfulQA"
TRUTHFULQA_OFFICIAL_DIRNAME = "official_truthfulqa"
RUN_HISTORY_ROOT = "web/session-monitor/history"
RUN_HISTORY_INDEX = "web/session-monitor/run_history.json"
RUN_HISTORY_INDEX_V2 = "web/session-monitor/run_history_v2.json"
RUN_HISTORY_MAX_SNAPSHOTS_PER_RUN = 240


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


def _checkpoint_live_summary(checkpoint_path: Path, strategy_names: list[str] | None = None) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "records_scanned": 0,
            "best_strategy": "",
            "leaderboard": [],
            "metric_rollup": {},
        }

    strategy_acc: dict[str, dict[str, Any]] = {}
    for strategy_name in strategy_names or []:
        strategy_acc[str(strategy_name)] = {
            "records": 0,
            "primary_sum": 0.0,
            "api_sum": 0.0,
            "wall_sum": 0.0,
            "cache_sum": 0.0,
            "metric_acc": {},
        }
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

            metric_payload = row.get("metrics")
            if not isinstance(metric_payload, dict):
                metric_payload = {}

            for metric_name, metric_value in metric_payload.items():
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
        means = {}
        if records > 0:
            means = {
                metric_name: round(acc[0] / acc[1], 4)
                for metric_name, acc in state["metric_acc"].items()
                if acc[1] > 0 and metric_name in METRIC_PRIORITY
            }

        mean_primary_score = round(state["primary_sum"] / records, 4) if records > 0 else 0.0
        mean_api_calls = round(state["api_sum"] / records, 3) if records > 0 else 0.0
        mean_wall_time_s = round(state["wall_sum"] / records, 3) if records > 0 else 0.0
        cache_hit_rate = round(state["cache_sum"] / records, 4) if records > 0 else 0.0

        leaderboard.append(
            {
                "strategy": strategy_name,
                "records": records,
                "mean_primary_score": mean_primary_score,
                "mean_api_calls": mean_api_calls,
                "mean_wall_time_s": mean_wall_time_s,
                "cache_hit_rate": cache_hit_rate,
                "metric_means": means,
            }
        )

    leaderboard.sort(key=lambda item: (item["records"] > 0, item["mean_primary_score"]), reverse=True)

    metric_rollup: dict[str, float] = {}
    for metric_name in METRIC_PRIORITY:
        acc = metric_acc.get(metric_name)
        if not acc or acc[1] <= 0:
            continue
        metric_rollup[metric_name] = round(acc[0] / acc[1], 4)

    scored_rows = [item for item in leaderboard if item.get("records", 0) > 0]
    best_strategy = scored_rows[0]["strategy"] if scored_rows else ""
    return {
        "records_scanned": records_scanned,
        "best_strategy": best_strategy,
        "leaderboard": leaderboard,
        "metric_rollup": metric_rollup,
    }


def _normalize_metric_name(metric_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (metric_name or "").strip().lower())
    return normalized.strip("_")


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _truthfulqa_official_repo(repo_root: Path) -> Path | None:
    env_path = os.getenv("TRUTHFULQA_OFFICIAL_REPO", "").strip()
    if env_path:
        candidate = Path(env_path)
    else:
        candidate = repo_root / TRUTHFULQA_DEFAULT_OFFICIAL_REPO
    if not candidate.exists():
        return None
    if not (candidate / "truthfulqa" / "evaluate.py").exists():
        return None
    return candidate


def _truthfulqa_official_paths(session_output_dir: Path) -> dict[str, Path]:
    official_dir = session_output_dir / TRUTHFULQA_OFFICIAL_DIRNAME
    return {
        "dir": official_dir,
        "state": official_dir / "state.json",
        "input": official_dir / "official_input.csv",
        "answers": official_dir / "official_answers.csv",
        "summary": official_dir / "summary.csv",
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _collect_truthfulqa_answers(checkpoint_path: Path) -> dict[str, dict[int, str]]:
    answers_by_strategy: dict[str, dict[int, str]] = {}
    if not checkpoint_path.exists():
        return answers_by_strategy

    with checkpoint_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            strategy = str(row.get("strategy_name") or "").strip()
            example_id = str(row.get("example_id") or "").strip()
            if not strategy or not example_id.isdigit():
                continue

            idx = int(example_id)
            final_text = str(row.get("final_text") or "")
            answers_by_strategy.setdefault(strategy, {})[idx] = final_text

    return answers_by_strategy


def _write_truthfulqa_official_input(
    dataset_csv_path: Path,
    output_csv_path: Path,
    answers_by_strategy: dict[str, dict[int, str]],
    strategy_names: list[str],
) -> list[str]:
    if not dataset_csv_path.exists():
        return []

    with dataset_csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        base_fields = list(reader.fieldnames or [])

    model_columns: list[str] = []
    for strategy_name in strategy_names:
        answers = answers_by_strategy.get(strategy_name, {})
        has_any_answer = any(str(value).strip() for value in answers.values())
        if has_any_answer and strategy_name not in model_columns:
            model_columns.append(strategy_name)

    fieldnames = base_fields + [col for col in model_columns if col not in base_fields]
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(rows):
            out_row = dict(row)
            for strategy_name in model_columns:
                out_row[strategy_name] = answers_by_strategy.get(strategy_name, {}).get(row_index, "")
            writer.writerow(out_row)

    return model_columns


def _read_truthfulqa_official_summary(summary_path: Path) -> dict[str, dict[str, float]]:
    if not summary_path.exists():
        return {}

    metrics_by_strategy: dict[str, dict[str, float]] = {}
    with summary_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strategy = str(row.get("Model") or row.get("model") or "").strip()
            if not strategy:
                continue
            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key is None:
                    continue
                normalized_key = _normalize_metric_name(key)
                if normalized_key in {"model", ""}:
                    continue
                try:
                    metrics[normalized_key] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
            metrics_by_strategy[strategy] = metrics
    return metrics_by_strategy


def _truthfulqa_official_summary(
    repo_root: Path,
    session_output_dir: Path,
    dataset_csv_path: Path | None,
    checkpoint_path: Path,
    strategy_names: list[str],
    records_scanned: int,
) -> dict[str, Any]:
    paths = _truthfulqa_official_paths(session_output_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    state = _load_json(paths["state"])
    summary = _read_truthfulqa_official_summary(paths["summary"])
    primary_metric = _normalize_metric_name(os.getenv("RB_TRUTHFULQA_OFFICIAL_PRIMARY_METRIC", "ROUGE1 acc"))

    repo_path = _truthfulqa_official_repo(repo_root)
    if repo_path is None:
        return {
            "available": bool(summary),
            "status": "missing_repo",
            "message": "TruthfulQA official repository not found; set TRUTHFULQA_OFFICIAL_REPO or clone external/TruthfulQA.",
            "primary_metric": primary_metric,
            "metrics_by_strategy": summary,
            "summary_path": _to_repo_relative(paths["summary"], repo_root),
        }

    if dataset_csv_path is None or not dataset_csv_path.exists():
        return {
            "available": bool(summary),
            "status": "missing_dataset",
            "message": f"Missing TruthfulQA dataset CSV: {dataset_csv_path}",
            "primary_metric": primary_metric,
            "metrics_by_strategy": summary,
            "summary_path": _to_repo_relative(paths["summary"], repo_root),
        }

    if records_scanned <= 0:
        return {
            "available": bool(summary),
            "status": "waiting_records",
            "message": "Waiting for generated TruthfulQA records before official scoring.",
            "primary_metric": primary_metric,
            "metrics_by_strategy": summary,
            "summary_path": _to_repo_relative(paths["summary"], repo_root),
        }

    previous_records = int(state.get("records_scanned", -1)) if isinstance(state.get("records_scanned"), int) else -1
    now_ts = datetime.now(timezone.utc).timestamp()
    last_run_ts = float(state.get("last_run_ts", 0.0)) if state else 0.0
    min_interval_s = int(os.getenv("RB_TRUTHFULQA_OFFICIAL_MIN_INTERVAL_S", "600"))
    should_run = not summary or records_scanned != previous_records
    throttled = should_run and (now_ts - last_run_ts) < min_interval_s

    if should_run and not throttled:
        metrics_raw = os.getenv("RB_TRUTHFULQA_OFFICIAL_METRICS", "bleu rouge").strip()
        metrics = tuple(item for item in metrics_raw.split() if item)
        if not metrics:
            metrics = ("bleu", "rouge")

        answers_by_strategy = _collect_truthfulqa_answers(checkpoint_path)
        model_columns = _write_truthfulqa_official_input(dataset_csv_path, paths["input"], answers_by_strategy, strategy_names)

        run_state: dict[str, Any] = {
            "records_scanned": records_scanned,
            "last_run_ts": now_ts,
            "status": "ok",
            "message": "",
        }

        if model_columns:
            try:
                run_official_truthfulqa(
                    repo_path=str(repo_path),
                    input_path=str(paths["input"]),
                    models=model_columns,
                    metrics=metrics,
                    output_path=str(paths["answers"]),
                )
                repo_summary = repo_path / "summary.csv"
                if repo_summary.exists():
                    paths["summary"].write_text(repo_summary.read_text(encoding="utf-8"), encoding="utf-8")
                summary = _read_truthfulqa_official_summary(paths["summary"])
            except Exception as exc:  # noqa: BLE001
                run_state["status"] = "error"
                detail = str(exc)
                if isinstance(exc, subprocess.CalledProcessError):
                    stderr_text = (exc.stderr or "").strip()
                    stdout_text = (exc.stdout or "").strip()
                    if stderr_text:
                        detail += f" | stderr: {stderr_text[-600:]}"
                    elif stdout_text:
                        detail += f" | stdout: {stdout_text[-600:]}"
                run_state["message"] = detail
        else:
            run_state["status"] = "no_models"
            run_state["message"] = "No strategy columns were available for official TruthfulQA scoring."

        _save_json(paths["state"], run_state)
        state = run_state

    return {
        "available": bool(summary),
        "status": str(state.get("status") or ("throttled" if throttled else "ok")),
        "message": str(state.get("message") or ("Official scoring throttled." if throttled else "")),
        "primary_metric": primary_metric,
        "metrics_by_strategy": summary,
        "summary_path": _to_repo_relative(paths["summary"], repo_root),
    }


def _merge_truthfulqa_official_live_summary(live_summary: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    merged = dict(live_summary)
    merged["evaluation_mode"] = "proxy_truth_delta"
    merged["official_status"] = official.get("status", "unavailable")
    merged["official_message"] = official.get("message", "")
    merged["official_summary_path"] = official.get("summary_path", "")

    metrics_by_strategy = official.get("metrics_by_strategy") or {}
    if not isinstance(metrics_by_strategy, dict) or not metrics_by_strategy:
        return merged

    base_rows = {
        str(row.get("strategy") or ""): row
        for row in (live_summary.get("leaderboard") or [])
        if isinstance(row, dict)
    }
    primary_metric = str(official.get("primary_metric") or "rouge1_acc")

    leaderboard: list[dict[str, Any]] = []
    for strategy_name, metrics in metrics_by_strategy.items():
        if not isinstance(metrics, dict):
            continue
        normalized_metrics = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(key, str) and isinstance(value, (int, float))
        }
        base = base_rows.get(strategy_name, {})
        primary = normalized_metrics.get(primary_metric)
        if primary is None and normalized_metrics:
            primary = next(iter(normalized_metrics.values()))
        if primary is None:
            primary = 0.0

        leaderboard.append(
            {
                "strategy": strategy_name,
                "records": int(base.get("records", 0)),
                "mean_primary_score": round(float(primary), 4),
                "mean_api_calls": float(base.get("mean_api_calls", 0.0)),
                "mean_wall_time_s": float(base.get("mean_wall_time_s", 0.0)),
                "cache_hit_rate": float(base.get("cache_hit_rate", 0.0)),
                "metric_means": normalized_metrics,
                "primary_metric": primary_metric,
            }
        )

    leaderboard.sort(key=lambda item: (item.get("mean_primary_score", 0.0), item.get("records", 0)), reverse=True)

    metric_rollup: dict[str, float] = {}
    if leaderboard:
        all_metric_names = sorted(
            {
                metric_name
                for row in leaderboard
                for metric_name in (row.get("metric_means") or {}).keys()
            }
        )
        for metric_name in all_metric_names:
            values = [
                row.get("metric_means", {}).get(metric_name)
                for row in leaderboard
                if metric_name in row.get("metric_means", {})
            ]
            if values:
                metric_rollup[metric_name] = round(sum(values) / len(values), 4)

    merged["evaluation_mode"] = "official_truthfulqa"
    merged["primary_metric"] = primary_metric
    merged["best_strategy"] = leaderboard[0]["strategy"] if leaderboard else ""
    merged["leaderboard"] = leaderboard
    merged["proxy_metric_rollup"] = live_summary.get("metric_rollup", {})
    merged["metric_rollup"] = metric_rollup
    return merged


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


def _build_snapshot_for_specs(
    repo_root: Path,
    session_specs: list[SessionSpec],
    run_meta: dict[str, Any],
    active_sessions: set[str],
    include_truthfulqa_official: bool,
) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    sum_completed = 0
    sum_expected = 0

    for spec in session_specs:
        cfg_path = repo_root / spec.config_path
        checkpoint_path = repo_root / spec.checkpoint_path
        log_dir = repo_root / spec.log_dir
        session_output_dir = repo_root / (spec.output_dir or str(Path(spec.checkpoint_path).parent))

        examples = 0
        strategy_count = 0
        dataset_kind = ""
        model = ""
        strategy_names: list[str] = []
        dataset_local_path: str | None = None
        if cfg_path.exists():
            cfg = load_experiment_config(cfg_path)
            examples, strategy_count = dataset_example_count(cfg, repo_root)
            dataset_kind = cfg.dataset.kind
            model = cfg.client.model
            strategy_names = [item.name for item in cfg.strategies]
            dataset_local_path = cfg.dataset.local_path

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
        live_summary = _checkpoint_live_summary(checkpoint_path, strategy_names=strategy_names)
        if include_truthfulqa_official and dataset_kind == "truthfulqa":
            dataset_csv = _resolve_local_path(repo_root, dataset_local_path) if dataset_local_path else None
            official = _truthfulqa_official_summary(
                repo_root=repo_root,
                session_output_dir=session_output_dir,
                dataset_csv_path=dataset_csv,
                checkpoint_path=checkpoint_path,
                strategy_names=strategy_names,
                records_scanned=int(live_summary.get("records_scanned", 0)),
            )
            live_summary = _merge_truthfulqa_official_live_summary(live_summary, official)

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


def build_snapshot(repo_root: Path) -> dict[str, Any]:
    active_sessions = running_screen_sessions()
    session_specs, run_meta = _load_session_specs(repo_root)
    return _build_snapshot_for_specs(
        repo_root=repo_root,
        session_specs=session_specs,
        run_meta=run_meta,
        active_sessions=active_sessions,
        include_truthfulqa_official=True,
    )


def _default_specs_by_output_key() -> dict[str, SessionSpec]:
    by_output: dict[str, SessionSpec] = {}
    for spec in DEFAULT_SESSION_SPECS:
        output_key = Path(spec.output_dir or Path(spec.checkpoint_path).parent).name
        by_output[output_key] = spec
    return by_output


def _parse_run_tag_datetime(run_tag: str) -> datetime | None:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})", run_tag)
    if not match:
        return None
    try:
        year, month, day, hour, minute, second = [int(item) for item in match.groups()]
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None


def _run_tag_log_suffix(run_tag: str) -> str:
    parsed = _parse_run_tag_datetime(run_tag)
    if parsed is None:
        return ""
    return parsed.strftime("%y%m%d_%H%M%S")


def _discover_archived_run_specs(repo_root: Path) -> list[dict[str, Any]]:
    previous_root = repo_root / "previous_configs_results"

    by_output = _default_specs_by_output_key()
    grouped: dict[str, dict[str, Any]] = {}

    scan_roots: list[tuple[Path, Path]] = []
    if previous_root.exists():
        for archive_root in sorted(path for path in previous_root.iterdir() if path.is_dir()):
            scan_roots.append((archive_root, archive_root / "outputs"))

    # Also scan current outputs so historical mixed-source runs (e.g., 4/8) include all datasets.
    scan_roots.append((repo_root, repo_root / "outputs"))

    for archive_root, outputs_root in scan_roots:
        if not outputs_root.exists():
            continue

        for output_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
            if "__" not in output_dir.name:
                continue
            output_key, run_tag = output_dir.name.split("__", 1)
            template = by_output.get(output_key)
            if template is None:
                continue

            entry = grouped.setdefault(
                run_tag,
                {
                    "run_tag": run_tag,
                    "archive_root": archive_root,
                    "sessions": {},
                },
            )
            session_name = template.session_name
            # Prefer the first archive root encountered for a run_tag/session pair.
            entry["sessions"].setdefault(
                session_name,
                {
                    "template": template,
                    "output_dir": output_dir,
                },
            )

    archived_runs: list[dict[str, Any]] = []
    for run_tag, entry in grouped.items():
        archive_root = entry["archive_root"]
        generated_config_root = archive_root / "configs/experiments/generated" / run_tag
        session_specs: list[SessionSpec] = []

        for item in entry["sessions"].values():
            template: SessionSpec = item["template"]
            output_dir: Path = item["output_dir"]
            checkpoint_path = output_dir / "results.jsonl"
            if not checkpoint_path.exists():
                continue

            generated_cfg = generated_config_root / Path(template.config_path).name
            if generated_cfg.exists():
                config_path_rel = _to_repo_relative(generated_cfg, repo_root)
            else:
                config_path_rel = template.config_path

            log_dir_rel = template.log_dir
            suffix = _run_tag_log_suffix(run_tag)
            if suffix:
                log_candidate = repo_root / "logs" / f"{template.session_name}__{suffix}"
                if log_candidate.exists():
                    log_dir_rel = _to_repo_relative(log_candidate, repo_root)

            output_dir_rel = _to_repo_relative(output_dir, repo_root)
            checkpoint_rel = _to_repo_relative(checkpoint_path, repo_root)
            session_specs.append(
                SessionSpec(
                    session_name=template.session_name,
                    display_name=template.display_name,
                    config_path=config_path_rel,
                    checkpoint_path=checkpoint_rel,
                    log_dir=log_dir_rel,
                    output_dir=output_dir_rel,
                    run_tag=run_tag,
                )
            )

        if not session_specs:
            continue

        parsed_dt = _parse_run_tag_datetime(run_tag)
        launched_at = parsed_dt.isoformat() if parsed_dt else ""
        archived_runs.append(
            {
                "run_tag": run_tag,
                "run_title": f"Spring 2026 • {run_tag}",
                "launched_at": launched_at,
                "run_manifest": _to_repo_relative(archive_root, repo_root),
                "session_specs": session_specs,
            }
        )

    archived_runs.sort(key=lambda item: item.get("launched_at") or item.get("run_tag") or "", reverse=True)
    return archived_runs


def _seed_archived_history(repo_root: Path, active_run_tag: str) -> None:
    history_root = repo_root / RUN_HISTORY_ROOT
    history_root.mkdir(parents=True, exist_ok=True)

    for archived in _discover_archived_run_specs(repo_root):
        run_tag = str(archived.get("run_tag") or "").strip()
        if not run_tag or run_tag == active_run_tag:
            continue

        run_meta = {
            "run_tag": run_tag,
            "run_title": str(archived.get("run_title") or run_tag),
            "launched_at": str(archived.get("launched_at") or ""),
            "run_manifest": str(archived.get("run_manifest") or ""),
        }
        session_specs = list(archived.get("session_specs") or [])
        snapshot = _build_snapshot_for_specs(
            repo_root=repo_root,
            session_specs=session_specs,
            run_meta=run_meta,
            active_sessions=set(),
            include_truthfulqa_official=False,
        )
        if not snapshot.get("sessions"):
            continue

        parsed_dt = _parse_run_tag_datetime(run_tag)
        if parsed_dt is not None:
            snapshot["generated_at"] = parsed_dt.isoformat()
            if not snapshot.get("launched_at"):
                snapshot["launched_at"] = parsed_dt.isoformat()

        _write_run_history_snapshot(repo_root, snapshot)


def _write_run_history_snapshot(repo_root: Path, snapshot: dict[str, Any]) -> None:
    run_tag = str(snapshot.get("run_tag") or "").strip()
    if not run_tag:
        return

    history_dir = repo_root / RUN_HISTORY_ROOT / run_tag
    history_dir.mkdir(parents=True, exist_ok=True)
    history_status_path = history_dir / "status.json"
    history_status_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    snapshot_generated_at = str(snapshot.get("generated_at") or datetime.now(timezone.utc).isoformat())
    safe_stamp = re.sub(r"[^0-9A-Za-z]+", "_", snapshot_generated_at).strip("_")
    snapshots_dir = history_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = snapshots_dir / f"status_{safe_stamp}.json"
    snapshots_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    snapshot_files = sorted(snapshots_dir.glob("status_*.json"))
    excess = len(snapshot_files) - RUN_HISTORY_MAX_SNAPSHOTS_PER_RUN
    if excess > 0:
        for old_file in snapshot_files[:excess]:
            old_file.unlink(missing_ok=True)


def _build_run_history_index(repo_root: Path, active_snapshot: dict[str, Any]) -> dict[str, Any]:
    history_root = repo_root / RUN_HISTORY_ROOT
    history_root.mkdir(parents=True, exist_ok=True)
    monitor_root = repo_root / "web/session-monitor"

    runs: list[dict[str, Any]] = []
    active_run_tag = str(active_snapshot.get("run_tag") or "").strip()

    for run_dir in sorted(path for path in history_root.iterdir() if path.is_dir()):
        latest_status_path = run_dir / "status.json"
        latest_payload = _load_json(latest_status_path)
        run_tag = str(latest_payload.get("run_tag") or run_dir.name)
        run_title = str(latest_payload.get("run_title") or run_tag)
        launched_at = str(latest_payload.get("launched_at") or "")
        latest_generated_at = str(latest_payload.get("generated_at") or "")

        if latest_status_path.exists():
            runs.append(
                {
                    "run_tag": run_tag,
                    "run_title": run_title,
                    "launched_at": launched_at,
                    "generated_at": latest_generated_at,
                    "status_path": _to_repo_relative(latest_status_path, monitor_root),
                    "is_active": bool(active_run_tag and run_tag == active_run_tag),
                    "view_kind": "latest",
                }
            )

        snapshots_dir = run_dir / "snapshots"
        if snapshots_dir.exists():
            snapshot_files = sorted(snapshots_dir.glob("status_*.json"), reverse=True)
            for snapshot_path in snapshot_files[:60]:
                snapshot_payload = _load_json(snapshot_path)
                snapshot_generated_at = str(snapshot_payload.get("generated_at") or "")
                runs.append(
                    {
                        "run_tag": run_tag,
                        "run_title": run_title,
                        "launched_at": launched_at,
                        "generated_at": snapshot_generated_at,
                        "status_path": _to_repo_relative(snapshot_path, monitor_root),
                        "is_active": False,
                        "view_kind": "snapshot",
                    }
                )

    runs.sort(key=lambda item: (item.get("launched_at") or item.get("generated_at") or ""), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_status_path": "status.json",
        "active_run_tag": active_run_tag,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate session status JSON for the frontend monitor.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--skip-archived-history",
        action="store_true",
        help="Skip rebuilding archived run history snapshots for faster live updates",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(repo_root)
    output_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_run_history_snapshot(repo_root, snapshot)

    skip_archived_history = args.skip_archived_history or os.getenv("RB_SKIP_ARCHIVED_HISTORY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not skip_archived_history:
        try:
            _seed_archived_history(repo_root, str(snapshot.get("run_tag") or ""))
        except Exception as exc:  # noqa: BLE001
            print(f"Archived history seed failed: {exc}")

    run_history = _build_run_history_index(repo_root, snapshot)
    run_history_path = repo_root / RUN_HISTORY_INDEX
    run_history_path.parent.mkdir(parents=True, exist_ok=True)
    run_history_path.write_text(json.dumps(run_history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_history_v2_path = repo_root / RUN_HISTORY_INDEX_V2
    run_history_v2_path.parent.mkdir(parents=True, exist_ok=True)
    run_history_v2_path.write_text(json.dumps(run_history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote status snapshot to {output_path}")


if __name__ == "__main__":
    main()
