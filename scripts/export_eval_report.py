from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reasonbench.config import load_experiment_config


NUMERIC_TYPES = (int, float)

STRATEGY_ORDER = [
    "direct",
    "concise_cot",
    "least_to_most",
    "constraint_decompose",
    "self_verify",
    "critique_refine",
    "self_consistency",
    "selective_self_consistency",
    "budgeted_cascade",
    "few_shot_exemplar",
]

STRATEGY_DISPLAY = {
    "direct": "Direct",
    "concise_cot": "Concise-CoT",
    "least_to_most": "Least-to-Most",
    "constraint_decompose": "Constraint-Decompose",
    "self_verify": "Self-Verify",
    "critique_refine": "Critique-Refine",
    "self_consistency": "Self-Consistency",
    "selective_self_consistency": "Selective-SC",
    "budgeted_cascade": "Budgeted-Cascade",
    "few_shot_exemplar": "Few-Shot Exemplar",
}

METRIC_ORDER = [
    "accuracy",
    "room_exact_accuracy",
    "entity_room_accuracy",
    "all_rooms_exact",
    "format_valid",
    "predicted_room_count",
    "truth_delta",
    "max_correct_similarity",
    "max_incorrect_similarity",
    "best_answer_similarity",
    "informativeness_proxy",
    "looks_uninformative",
    "proxy_exact_match",
    "proxy_contains_match",
    "is_unscorable",
    "official_scorer_recommended",
    "choice_count",
    "text_match_score",
]

METRIC_ALIAS = {
    "accuracy": "acc",
    "room_exact_accuracy": "room-exact",
    "entity_room_accuracy": "entity-room",
    "all_rooms_exact": "all-rooms",
    "format_valid": "fmt-valid",
    "predicted_room_count": "pred-rooms",
    "truth_delta": "truth-delta",
    "max_correct_similarity": "correct-sim",
    "max_incorrect_similarity": "incorrect-sim",
    "best_answer_similarity": "best-sim",
    "informativeness_proxy": "info-proxy",
    "looks_uninformative": "uninformative",
    "proxy_exact_match": "proxy-exact",
    "proxy_contains_match": "proxy-contains",
    "is_unscorable": "unscorable",
    "official_scorer_recommended": "official-flag",
    "choice_count": "choice-count",
    "text_match_score": "text-match",
}

METRIC_DESCRIPTIONS = {
    "accuracy": (
        "AI2-ARC multiple-choice accuracy.",
        "Per record: 1 if extracted label equals gold answer key, else 0. Table value is the mean across records.",
        "[0, 1]",
    ),
    "room_exact_accuracy": (
        "Room-level exact placement accuracy.",
        "For each record, proportion of rooms whose full occupant set exactly matches gold; table shows mean.",
        "[0, 1]",
    ),
    "entity_room_accuracy": (
        "Entity-to-room assignment accuracy.",
        "For each record, proportion of entities assigned to the correct room; table shows mean.",
        "[0, 1]",
    ),
    "all_rooms_exact": (
        "Perfect room assignment rate.",
        "Boolean metric per record (all rooms exactly correct); converted to rate by averaging 0/1.",
        "[0, 1]",
    ),
    "format_valid": (
        "Output format validity rate.",
        "Boolean per record (parser accepted required format); converted to rate by averaging 0/1.",
        "[0, 1]",
    ),
    "predicted_room_count": (
        "Predicted room count.",
        "Number of room entries parsed from prediction per record; table shows mean.",
        "Non-negative",
    ),
    "truth_delta": (
        "Truthfulness delta score.",
        "Per record: max similarity to correct answers minus max similarity to incorrect answers; table shows mean.",
        "Approximately [-1, 1]",
    ),
    "max_correct_similarity": (
        "Best correct-answer similarity.",
        "Maximum soft similarity between prediction and any correct reference answer; table shows mean.",
        "[0, 1]",
    ),
    "max_incorrect_similarity": (
        "Best incorrect-answer similarity.",
        "Maximum soft similarity between prediction and any incorrect reference answer; table shows mean.",
        "[0, 1]",
    ),
    "best_answer_similarity": (
        "Best-answer similarity.",
        "Soft similarity between prediction and canonical best answer; table shows mean.",
        "[0, 1]",
    ),
    "informativeness_proxy": (
        "Informativeness proxy.",
        "Length/verbosity-based informativeness score from evaluator; table shows mean.",
        "[0, 1]",
    ),
    "looks_uninformative": (
        "Uninformative response rate.",
        "Boolean heuristic per record for non-informative answers; converted to rate by averaging 0/1.",
        "[0, 1]",
    ),
    "proxy_exact_match": (
        "LiveBench proxy exact-match rate.",
        "Per record: exact normalized match against provided ground truth (proxy evaluator); table shows mean.",
        "[0, 1]",
    ),
    "proxy_contains_match": (
        "LiveBench proxy contains-match rate.",
        "Per record: normalized ground truth appears in prediction (proxy evaluator); table shows mean.",
        "[0, 1]",
    ),
    "is_unscorable": (
        "Unscorable record flag.",
        "Boolean marker set to 1 when a record cannot be validly scored by the current evaluator path.",
        "[0, 1]",
    ),
    "official_scorer_recommended": (
        "Official scorer recommendation flag.",
        "Boolean marker emitted by evaluator; converted to rate by averaging 0/1.",
        "[0, 1]",
    ),
    "choice_count": (
        "Average number of answer options.",
        "Number of choices available in the ARC item; table shows mean.",
        "Positive",
    ),
    "text_match_score": (
        "Text-fallback similarity score.",
        "When label extraction fails, fallback best text similarity to options; table reports mean score.",
        "[0, 1]",
    ),
}


STRATEGY_WRITEUP = {
    "direct": {
        "type": "Single-shot prompting",
        "instruction": "Solve the task directly. Keep the answer concise and avoid unnecessary explanation.",
        "summary": (
            "Baseline strategy: a single forward pass that prioritizes concise, format-compliant answers. "
            "This is the cheapest option and is used as the fast path for compute-adaptive strategies."
        ),
        "details": (
            "Implementation: inherits SingleShotStrategy. The strategy instruction is appended to the system prompt, "
            "then the example is executed over T turns (usually T=1). On the final turn, any dataset-specific "
            "format hint is appended to the user message."
        ),
        "calls": r"$T$",
        "params": "(none)",
    },
    "concise_cot": {
        "type": "Single-shot prompting",
        "instruction": "Reason step by step briefly, then provide the final answer in the requested format.",
        "summary": (
            "Adds a light chain-of-thought style scaffold while still emphasizing a final answer in the required format. "
            "Designed to improve multi-step correctness with minimal added verbosity."
        ),
        "details": (
            "Implementation: identical call pattern to Direct (SingleShotStrategy over turns), but with an instruction that "
            "encourages brief intermediate reasoning. In strict benchmark mode, the leakage guard remains active."
        ),
        "calls": r"$T$",
        "params": "(none)",
    },
    "least_to_most": {
        "type": "Single-shot prompting",
        "instruction": "Break the problem into smaller subproblems, solve them in order, and then provide the final answer. Keep the decomposition compact.",
        "summary": (
            "Encourages explicit subproblem decomposition (least-to-most) to reduce reasoning jumps. "
            "Useful for tasks where one missing step can derail the final output."
        ),
        "details": (
            "Implementation: SingleShotStrategy. The model is asked to generate a compact decomposition and solve sequentially "
            "within the same completion. The evaluator only consumes the final answer, but the decomposition can help the model stay consistent."
        ),
        "calls": r"$T$",
        "params": "(none)",
    },
    "constraint_decompose": {
        "type": "Single-shot prompting",
        "instruction": "First extract the critical constraints or facts in a compact list, then solve using those constraints, then give the final answer.",
        "summary": (
            "A constraint-first variant of decomposition: it makes the model externalize key facts and constraints before solving. "
            "This often helps structured tasks (e.g., assignments, consistency constraints) where violating a single constraint breaks correctness."
        ),
        "details": (
            "Implementation: SingleShotStrategy. The constraint list is produced inside the same completion; no extra tool calls are used. "
            "Because the formatting hint is appended to the final user turn, the strategy still has an explicit reminder to output in the expected schema."
        ),
        "calls": r"$T$",
        "params": "(none)",
    },
    "few_shot_exemplar": {
        "type": "Few-shot prompting (external demos)",
        "instruction": "Use the external exemplars as style guidance only, then solve the new problem.",
        "summary": (
            "Prepends a fixed set of exemplar user and assistant messages from a JSONL demo file. "
            "This can improve formatting consistency and provide implicit task priors without changing the number of calls."
        ),
        "details": (
            "Implementation: FewShotExemplarStrategy loads a demo JSONL containing {user, assistant} pairs and injects them as prior messages "
            "before the task. This increases context length (token cost) but typically keeps the number of API calls at T. "
            "The instruction explicitly frames exemplars as style guidance to reduce overfitting to demo content."
        ),
        "calls": r"$T$",
        "params": r"demo_path (required)",
    },
    "self_verify": {
        "type": "Two-pass reflection",
        "instruction": "Produce a draft answer, mentally verify it against the task requirements, and then return a corrected final answer.",
        "summary": (
            "Runs a draft pass and then a dedicated verification pass that checks the draft for common failure modes and fixes mistakes. "
            "Often improves format validity and reduces simple logical errors at roughly 2x the call budget."
        ),
        "details": (
            "Implementation: first calls SingleShotStrategy._run_turns to obtain a draft (T calls), then issues one extra call with a user prompt that includes "
            "the task text and the draft answer. The verification instruction requests a corrected final answer only. "
            "Trace includes an extra step with turn_index=verify."
        ),
        "calls": r"$T + 1$",
        "params": "(none)",
    },
    "critique_refine": {
        "type": "Three-pass reflection",
        "instruction": "Draft an answer, critique it for correctness and format, then refine it into the final answer.",
        "summary": (
            "Adds an explicit critique stage with a stricter reviewer persona, then a refinement stage conditioned on that critique. "
            "This can catch subtle formatting and reasoning defects but costs ~3x calls in single-turn tasks."
        ),
        "details": (
            "Implementation: (1) draft via _run_turns (T calls); (2) critique with a dedicated system message 'You are a strict benchmark reviewer...' at temperature 0.0; "
            "(3) refine with the standard system prompt, conditioned on Task + Draft + Critique, returning only the improved final answer. "
            "Trace includes turn_index=critique and turn_index=refine."
        ),
        "calls": r"$T + 2$",
        "params": "(none)",
    },
    "self_consistency": {
        "type": "Sampling + majority vote",
        "instruction": "Answer carefully.",
        "summary": (
            "Samples multiple independent solutions at a higher temperature and returns the most common answer after canonicalization. "
            "This reduces variance and can improve reliability when the model is unstable, but increases compute linearly with the number of samples."
        ),
        "details": (
            "Implementation: runs _run_turns S times (default S=5) with temperature=max(default_temperature, 0.6) unless overridden. "
            "Each sample is mapped to a vote key via canonical_vote_key: for room-assignment style outputs it parses rooms/occupants; otherwise it normalizes text "
            "and prefers a 'final answer:' segment if present. The winner is the most frequent vote key; the returned text is the first sample that produced that key."
        ),
        "calls": r"$S \cdot T$",
        "params": r"num_samples (default 5), temperature (default max(t, 0.6))",
    },
    "selective_self_consistency": {
        "type": "Adaptive sampling",
        "instruction": "Answer carefully.",
        "summary": (
            "Runs a cheap first pass and only escalates to a smaller self-consistency vote when the output looks uncertain or malformed. "
            "This targets most of the benefit of self-consistency with a significantly lower average call budget."
        ),
        "details": (
            "Implementation: first executes _run_turns once (T calls). It escalates if the text is empty, contains uncertainty markers (maybe/perhaps/not sure), "
            "or fails dataset-specific heuristics (e.g., missing a room marker for room_assignment; too short for truthfulqa). "
            "If escalated, it runs S-1 additional samples (default S=4 total, seeding the vote with the first pass) and majority-votes using canonical_vote_key. "
            "Metadata includes escalated=true/false and a vote histogram."
        ),
        "calls": r"$T$ (no escalation) or $S \cdot T$ (escalated)",
        "params": r"num_samples (default 4), temperature (default max(t, 0.6))",
    },
    "budgeted_cascade": {
        "type": "Fast/slow cascade",
        "instruction": "Fast path runs Direct; if it fails a quality gate, fall back to Self-Verify.",
        "summary": (
            "A compute-budgeted reliability strategy: try a fast Direct answer first, and only if it fails a lightweight quality gate, fall back to Self-Verify. "
            "This often improves format correctness while keeping average cost low."
        ),
        "details": (
            "Implementation: fast path runs DirectStrategy. If the output 'looks good enough' (dataset-specific heuristics; e.g., room marker present; not obviously uncertain), "
            "it returns immediately with cascade_path=fast_only. Otherwise it runs SelfVerifyStrategy and returns the verified answer with cascade_path=fast_then_verify. "
            "When escalated, total calls are T (fast) + (T+1) (self-verify) = 2T+1 (3 calls when T=1). The trace is prefixed with a cascade_stage=fast entry."
        ),
        "calls": r"$T$ (fast-only) or $2T + 1$ (fast+verify)",
        "params": "(none)",
    },
}


@dataclass
class RunConfigRef:
    config_path: Path
    dataset_kind: str
    local_path: str | None
    model: str
    base_url: str
    strategy_names: list[str]
    checkpoint_path: str


@dataclass
class LogSummary:
    session_name: str
    attempts: int
    finished_attempts: int
    success_attempts: int
    failed_attempts: int
    last_exit_code: str
    latest_log_file: str


@dataclass
class RunAggregate:
    run_name: str
    result_file: str
    completed_records: int
    completion_pct: float
    meta: dict[str, Any]
    numeric_metrics: list[str]
    per_strategy: list[dict[str, Any]]
    best_strategy: str


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def is_numeric_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return isinstance(value, NUMERIC_TYPES) and not isinstance(value, bool)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in handle:
            count += 1
    return count


def count_examples(dataset_kind: str, local_path: Path) -> int:
    if dataset_kind in {"room_assignment", "ai2_arc"}:
        return count_lines(local_path)
    if dataset_kind == "truthfulqa":
        return max(count_lines(local_path) - 1, 0)
    if dataset_kind == "livebench":
        if local_path.is_file():
            return count_lines(local_path)
        total = 0
        for qf in local_path.rglob("question.jsonl"):
            total += count_lines(qf)
        return total
    return 0


def sanitize_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def short_model_name(model: str) -> str:
    if "Qwen3.5-27B" in model:
        return "Qwen3.5-27B-FP8"
    if "Qwen3.5-4B" in model:
        return "Qwen3.5-4B"
    return model.rsplit("/", 1)[-1] if "/" in model else model


def metric_alias(metric_name: str) -> str:
    return METRIC_ALIAS.get(metric_name, metric_name)


def sort_strategies(strategies: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(STRATEGY_ORDER)}
    return sorted(strategies, key=lambda x: order.get(x, 9999))


def sort_metrics(metrics: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(METRIC_ORDER)}
    return sorted(metrics, key=lambda x: order.get(x, 9999))


def discover_configs(config_dir: Path) -> dict[str, RunConfigRef]:
    refs: dict[str, RunConfigRef] = {}
    for cfg_file in sorted(config_dir.glob("*.toml")):
        cfg = load_experiment_config(cfg_file)
        ref = RunConfigRef(
            config_path=cfg_file,
            dataset_kind=cfg.dataset.kind,
            local_path=cfg.dataset.local_path,
            model=cfg.client.model,
            base_url=cfg.client.base_url,
            strategy_names=[s.name for s in cfg.strategies],
            checkpoint_path=cfg.output.checkpoint_path,
        )
        refs[cfg.output.checkpoint_path] = ref
    return refs


def collect_log_summaries(logs_dir: Path) -> list[LogSummary]:
    summaries: list[LogSummary] = []
    if not logs_dir.exists():
        return summaries

    for session_dir in sorted(logs_dir.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith("rb_"):
            continue

        attempts = 0
        finished = 0
        success = 0
        failed = 0
        last_exit_code = "--"
        latest_log = ""

        log_files = sorted(session_dir.glob("run_*.log"))
        attempts = len(log_files)
        if log_files:
            latest_log = log_files[-1].name

        for log_file in log_files:
            exit_code = None
            with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith("exit_code:"):
                        try:
                            exit_code = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            exit_code = None
            if exit_code is not None:
                finished += 1
                last_exit_code = str(exit_code)
                if exit_code == 0:
                    success += 1
                else:
                    failed += 1

        summaries.append(
            LogSummary(
                session_name=session_dir.name,
                attempts=attempts,
                finished_attempts=finished,
                success_attempts=success,
                failed_attempts=failed,
                last_exit_code=last_exit_code,
                latest_log_file=latest_log,
            )
        )

    return summaries


def aggregate_run(result_file: Path, cfg_ref: RunConfigRef | None, repo_root: Path) -> RunAggregate:
    records = read_jsonl(result_file)
    dataset_name = records[0].get("dataset_name", "unknown") if records else "unknown"

    strategy_groups: dict[str, list[dict[str, Any]]] = {}
    numeric_metric_names: set[str] = set()

    for row in records:
        strat = str(row.get("strategy_name") or "unknown")
        strategy_groups.setdefault(strat, []).append(row)
        for key, value in (row.get("metrics") or {}).items():
            if is_numeric_metric(value):
                numeric_metric_names.add(key)

    ordered_metrics = sort_metrics(list(numeric_metric_names))

    run_meta = {
        "dataset_kind": dataset_name,
        "model": "--",
        "base_url": "--",
        "config_path": "--",
        "strategy_count": len(strategy_groups),
        "expected_examples": 0,
        "expected_records": 0,
    }

    if cfg_ref is not None and cfg_ref.local_path:
        expected_examples = count_examples(cfg_ref.dataset_kind, repo_root / cfg_ref.local_path)
        expected_records = expected_examples * len(cfg_ref.strategy_names)
        run_meta.update(
            {
                "dataset_kind": cfg_ref.dataset_kind,
                "model": cfg_ref.model,
                "base_url": cfg_ref.base_url,
                "config_path": str(cfg_ref.config_path.relative_to(repo_root)),
                "strategy_count": len(cfg_ref.strategy_names),
                "expected_examples": expected_examples,
                "expected_records": expected_records,
            }
        )

    per_strategy: list[dict[str, Any]] = []
    for strat in sort_strategies(list(strategy_groups)):
        rows = strategy_groups[strat]
        scorable_rows = [row for row in rows if not bool((row.get("metrics") or {}).get("is_unscorable", False))]
        metrics_by_name: dict[str, list[float]] = {m: [] for m in ordered_metrics}

        for row in rows:
            metrics = row.get("metrics") or {}
            for m in ordered_metrics:
                if m not in metrics:
                    continue
                value = metrics[m]
                if isinstance(value, bool):
                    metrics_by_name[m].append(1.0 if value else 0.0)
                elif isinstance(value, NUMERIC_TYPES):
                    metrics_by_name[m].append(float(value))

        per_strategy.append(
            {
                "strategy": strat,
                "strategy_display": STRATEGY_DISPLAY.get(strat, strat),
                "n": len(rows),
                "scorable_n": len(scorable_rows),
                "unscorable_n": len(rows) - len(scorable_rows),
                "unscorable_rate": safe_mean(
                    [1.0 if (r.get("metrics") or {}).get("is_unscorable", False) else 0.0 for r in rows]
                ),
                "coverage_pct": (100.0 * len(rows) / run_meta["expected_examples"])
                if run_meta["expected_examples"]
                else float("nan"),
                "primary_score_mean": safe_mean([float(r.get("primary_score", 0.0)) for r in scorable_rows]),
                "api_calls_mean": safe_mean([float(r.get("api_calls", 0.0)) for r in rows]),
                "wall_time_mean": safe_mean([float(r.get("wall_time_s", 0.0)) for r in rows]),
                "cache_hit_rate": safe_mean([1.0 if r.get("from_cache") else 0.0 for r in rows]),
                "metrics": {k: safe_mean(v) for k, v in metrics_by_name.items()},
            }
        )

    expected_records = run_meta["expected_records"]
    completion_pct = (100.0 * len(records) / expected_records) if expected_records else float("nan")

    best_strategy = "--"
    if per_strategy:
        ranked = sorted(
            per_strategy,
            key=lambda x: x["primary_score_mean"] if not math.isnan(x["primary_score_mean"]) else float("-inf"),
            reverse=True,
        )
        best_strategy = ranked[0]["strategy_display"]

    return RunAggregate(
        run_name=result_file.parent.name,
        result_file=str(result_file),
        completed_records=len(records),
        completion_pct=completion_pct,
        meta=run_meta,
        numeric_metrics=ordered_metrics,
        per_strategy=per_strategy,
        best_strategy=best_strategy,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latex_core_table_for_run(run: RunAggregate) -> str:
    label_slug = re.sub(r"[^a-zA-Z0-9]+", "_", run.run_name).strip("_")

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Core results for {sanitize_tex(run.run_name)}.}}")
    lines.append(rf"\label{{tab:{label_slug}_core}}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Strategy & $n$ & cov\% & primary & api & time(s) & cache\% \\")
    lines.append(r"\midrule")

    for row in run.per_strategy:
        lines.append(
            f"{sanitize_tex(row['strategy_display'])} & {row['n']} & {fmt(row['coverage_pct'], 2)} & "
            f"{fmt(row['primary_score_mean'], 4)} & {fmt(row['api_calls_mean'], 3)} & "
            f"{fmt(row['wall_time_mean'], 3)} & {fmt(100.0 * row['cache_hit_rate'], 2)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(
        rf"\vspace{{2pt}}\parbox{{0.98\textwidth}}{{\footnotesize "
        rf"Completed records: {run.completed_records} / {run.meta['expected_records'] if run.meta['expected_records'] else '--'} "
        rf"({fmt(run.completion_pct, 2)}\%). "
        rf"Expected examples per strategy: {run.meta['expected_examples'] if run.meta['expected_examples'] else '--'}. "
        rf"Primary excludes records flagged unscorable. "
        rf"Model: {sanitize_tex(short_model_name(str(run.meta['model'])))}. "
        rf"Best observed strategy so far (by primary): {sanitize_tex(run.best_strategy)}.}}"
    )
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def latex_dataset_metric_table_for_run(run: RunAggregate) -> str:
    if not run.numeric_metrics:
        return ""

    label_slug = re.sub(r"[^a-zA-Z0-9]+", "_", run.run_name).strip("_")
    headers = [metric_alias(m) for m in run.numeric_metrics]

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Dataset-specific metric means for {sanitize_tex(run.run_name)}.}}")
    lines.append(rf"\label{{tab:{label_slug}_dataset}}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{l" + ("r" * len(headers)) + "}")
    lines.append(r"\toprule")
    lines.append("Strategy & " + " & ".join(sanitize_tex(h) for h in headers) + r" \\")
    lines.append(r"\midrule")

    for row in run.per_strategy:
        vals: list[str] = []
        for m in run.numeric_metrics:
            vals.append(fmt(row["metrics"].get(m, float("nan")), 4))
        lines.append(sanitize_tex(row["strategy_display"]) + " & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\vspace{2pt}\parbox{0.98\textwidth}{\footnotesize Metric aliases are expanded in the Metric Glossary section.}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def latex_log_table(log_summaries: list[LogSummary]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Execution log summary by session (derived from run log files).}")
    lines.append(r"\label{tab:log_summary}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{lrrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Session & Attempts & Finished & Success & Failed & LastExit & LatestLog \\")
    lines.append(r"\midrule")

    for row in log_summaries:
        lines.append(
            f"{sanitize_tex(row.session_name)} & {row.attempts} & {row.finished_attempts} & "
            f"{row.success_attempts} & {row.failed_attempts} & {sanitize_tex(row.last_exit_code)} & "
            f"{sanitize_tex(row.latest_log_file)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def latex_variable_definition_table() -> str:
    rows = [
        {
            "var": "$n$",
            "meaning": "Completed examples",
            "measured": "Number of records aggregated for a strategy in this run.",
            "measured_is_latex": False,
            "unit": "count",
        },
        {
            "var": "cov\\%",
            "meaning": "Coverage",
            "measured": r"$100 \times n / N_{\text{expected examples}}$",
            "measured_is_latex": True,
            "unit": "percentage",
        },
        {
            "var": "primary",
            "meaning": "Primary score",
            "measured": "Mean of evaluator-provided primary_score over completed records.",
            "measured_is_latex": False,
            "unit": "dataset-dependent",
        },
        {
            "var": "api",
            "meaning": "API calls",
            "measured": "Mean number of model API calls used per completed record.",
            "measured_is_latex": False,
            "unit": "calls/record",
        },
        {
            "var": "time(s)",
            "meaning": "Latency",
            "measured": "Mean wall_time_s measured per completed record.",
            "measured_is_latex": False,
            "unit": "seconds/record",
        },
        {
            "var": "cache\\%",
            "meaning": "Cache hit rate",
            "measured": r"$100 \times \frac{1}{n}\sum_i c_i$, where $c_i \in \{0,1\}$",
            "measured_is_latex": True,
            "unit": "percentage",
        },
    ]

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Core table variables and measurement protocol.}")
    lines.append(r"\label{tab:variables}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(r"Variable & Meaning & How measured & Unit/Range \\")
    lines.append(r"\midrule")
    for row in rows:
        measured_cell = row["measured"] if row["measured_is_latex"] else sanitize_tex(row["measured"])
        lines.append(
            f"{row['var']} & {sanitize_tex(row['meaning'])} & {measured_cell} & {sanitize_tex(row['unit'])} \\\\" 
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def latex_metric_glossary(metrics_present: list[str]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Metric glossary: definitions and measurement details for every reported metric.}")
    lines.append(r"\label{tab:metric_glossary}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(r"Alias & Original metric name & How measured & Range \\")
    lines.append(r"\midrule")

    for metric in metrics_present:
        alias = metric_alias(metric)
        desc = METRIC_DESCRIPTIONS.get(
            metric,
            (
                "Metric from evaluator.",
                "Aggregated as arithmetic mean over completed records.",
                "dataset-dependent",
            ),
        )
        lines.append(
            f"{sanitize_tex(alias)} & {sanitize_tex(metric)} & {sanitize_tex(desc[1])} & {sanitize_tex(desc[2])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def latex_strategy_definitions_section() -> str:
    lines: list[str] = []

    def tex_tt_breakable(text: str) -> str:
        return sanitize_tex(text).replace(r"\_", r"\_\hspace{0pt}")

    lines.append(
        "All strategies in this report share a common execution wrapper. "
        "For an example with $T$ turns (the length of \\texttt{example.turns}), SingleShot-style strategies "
        "issue one model call per turn and carry forward the chat history. "
        "On the final turn, any dataset-provided formatting hint is appended to the user message. "
        "In strict benchmark mode, the system prompt includes an explicit leakage guard."
    )
    lines.append("")

    lines.append(r"\begin{quote}\footnotesize")
    lines.append(
        sanitize_tex(
            "Base system instruction (strict mode): You are a careful reasoning assistant. Follow the task instructions exactly, "
            "avoid hallucinations, and prefer explicit uncertainty over fabricated claims. Do not use hidden benchmark examples or benchmark-specific leakage."
        )
    )
    lines.append(r"\end{quote}")
    lines.append("")

    # Summary table
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Reasoning strategies implemented in this evaluation suite.}")
    lines.append(r"\label{tab:strategy_summary}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(
        r"\begin{tabularx}{\textwidth}{>{\RaggedRight\arraybackslash}p{0.14\textwidth} >{\RaggedRight\arraybackslash}p{0.18\textwidth} >{\RaggedRight\arraybackslash}p{0.16\textwidth} >{\Centering\arraybackslash}p{0.10\textwidth} >{\RaggedRight\arraybackslash}X}"
    )
    lines.append(r"\toprule")
    lines.append(r"Display & Name & Type & API calls & What it changes \\")
    lines.append(r"\midrule")
    for strategy_name in STRATEGY_ORDER:
        if strategy_name not in STRATEGY_WRITEUP:
            continue
        spec = STRATEGY_WRITEUP[strategy_name]
        display = STRATEGY_DISPLAY.get(strategy_name, strategy_name)
        lines.append(
            f"{sanitize_tex(display)} & \\texttt{{{tex_tt_breakable(strategy_name)}}} & {sanitize_tex(spec['type'])} & "
            f"{spec['calls']} & {sanitize_tex(spec['summary'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\normalsize")
    lines.append(r"\end{table*}")
    lines.append("")

    # Per-strategy detailed writeup
    for strategy_name in STRATEGY_ORDER:
        if strategy_name not in STRATEGY_WRITEUP:
            continue
        spec = STRATEGY_WRITEUP[strategy_name]
        display = STRATEGY_DISPLAY.get(strategy_name, strategy_name)

        lines.append(rf"\subsection*{{{sanitize_tex(display)}}}")
        lines.append(rf"\noindent\textbf{{Type.}} {sanitize_tex(spec['type'])}\\")
        lines.append(rf"\textbf{{Name.}} \texttt{{{tex_tt_breakable(strategy_name)}}}\\")
        lines.append(rf"\textbf{{API calls.}} {spec['calls']}\\")
        lines.append(rf"\textbf{{Config parameters.}} {sanitize_tex(spec['params'])}\\")
        lines.append("")

        if strategy_name == "budgeted_cascade":
            lines.append(r"\noindent\textbf{System-prompt additions / instructions.}")
            lines.append(r"\begin{quote}\footnotesize")
            lines.append(r"Fast path (Direct): " + sanitize_tex(STRATEGY_WRITEUP["direct"]["instruction"]))
            lines.append(r"\\")
            lines.append(r"Slow path (Self-Verify): " + sanitize_tex(STRATEGY_WRITEUP["self_verify"]["instruction"]))
            lines.append(r"\end{quote}")
        else:
            lines.append(r"\noindent\textbf{System-prompt addition / instruction.}")
            lines.append(r"\begin{quote}\footnotesize")
            lines.append(sanitize_tex(spec["instruction"]))
            lines.append(r"\end{quote}")

        lines.append(r"\noindent\textbf{What it does.} " + sanitize_tex(spec["summary"]))
        lines.append("")
        lines.append(r"\noindent\textbf{Implementation details.} " + sanitize_tex(spec["details"]))
        lines.append("")

        if strategy_name == "self_consistency":
            lines.append(r"\noindent\textbf{Voting rule.} We compute a canonical vote key $v=\mathrm{canonical\_vote\_key}(y)$ for each sample output $y$. "
                         r"For room-assignment outputs, $v$ is a normalized, sorted room-to-occupants representation; otherwise $v$ is normalized text (preferring a 'final answer' segment if present). "
                         r"The selected answer is the first output among the samples whose vote key attains the maximum count.")
            lines.append("")

        if strategy_name == "selective_self_consistency":
            lines.append(r"\noindent\textbf{Escalation heuristic.} Escalate when the first-pass output is empty, contains uncertainty markers (e.g., \emph{maybe}, \emph{not sure}), "
                         r"or appears malformed for a dataset (e.g., missing room markers for room assignment; too-short responses for TruthfulQA). "
                         r"If no escalation, the returned output is the first pass (metadata: escalated=false).")
            lines.append("")

        if strategy_name == "budgeted_cascade":
            lines.append(r"\noindent\textbf{Quality gate.} Accept the fast Direct output if it is non-empty and does not look uncertain. "
                         r"For some datasets, additional schema/marker checks are applied (e.g., room markers for room assignment; minimum length for TruthfulQA). "
                         r"If rejected, run Self-Verify and return the verified answer (metadata: cascade\_path=fast\_then\_verify).")
            lines.append("")

    return "\n".join(lines)


def latex_illustrations_block() -> str:
    lines: list[str] = []

    lines.append(r"\begin{figure*}[t]")
    lines.append(r"\centering")
    lines.append(r"\fbox{\parbox{0.96\textwidth}{")
    lines.append(r"\textbf{Illustration 1: Aggregation pipeline used in this report.}\\")
    lines.append(r"(a) Parse each \texttt{results.jsonl} record into \{strategy, primary score, api calls, wall time, metrics\}.\\")
    lines.append(r"(b) Convert boolean metrics to 0/1, keep numeric metrics as floats, and compute per-strategy means.\\")
    lines.append(r"(c) Compute coverage against expected examples from the corresponding TOML config.\\")
    lines.append(r"(d) Emit two tables per run: core metrics + dataset-specific metrics, then merge all runs into one PDF.")
    lines.append(r"}}")
    lines.append(r"\caption{End-to-end measurement and reporting pipeline.}")
    lines.append(r"\label{fig:pipeline}")
    lines.append(r"\end{figure*}")
    lines.append("")

    lines.append(r"\begin{figure*}[t]")
    lines.append(r"\centering")
    lines.append(r"\fbox{\parbox{0.96\textwidth}{")
    lines.append(r"\textbf{Illustration 2: How partial-progress runs are interpreted.}\\")
    lines.append(r"For each run, we report only completed records. Let $R$ be completed records and $E$ be expected records.\\")
    lines.append(r"Completion rate is $100 \times R/E$. Strategy-level coverage is $100 \times n/N_{\text{expected examples}}$.\\")
    lines.append(r"This avoids extrapolating unfinished evaluations while keeping all measured evidence visible.")
    lines.append(r"}}")
    lines.append(r"\caption{Partial-run accounting policy (strictly completed data only).}")
    lines.append(r"\label{fig:partial_policy}")
    lines.append(r"\end{figure*}")
    lines.append("")

    return "\n".join(lines)


def build_results_narrative(runs: list[RunAggregate]) -> str:
    parts: list[str] = []
    for run in runs:
        model = short_model_name(str(run.meta["model"]))
        expected = run.meta["expected_records"] if run.meta["expected_records"] else "--"
        parts.append(
            f"\\textbf{{{sanitize_tex(run.run_name)}}} uses {sanitize_tex(model)} and currently has "
            f"{run.completed_records} completed records out of {expected} "
            f"({fmt(run.completion_pct, 2)}\\%). "
            f"The best observed strategy at this checkpoint is {sanitize_tex(run.best_strategy)} by mean primary score."
        )
    return "\n\n".join(parts)


def build_latex_document(runs: list[RunAggregate], log_summaries: list[LogSummary]) -> str:
    metrics_present: list[str] = []
    seen = set()
    for run in runs:
        for metric in run.numeric_metrics:
            if metric not in seen:
                seen.add(metric)
                metrics_present.append(metric)
    metrics_present = sort_metrics(metrics_present)

    doc: list[str] = []
    doc.append(r"\documentclass[10pt,twocolumn]{article}")
    doc.append(r"\usepackage[margin=0.75in]{geometry}")
    doc.append(r"\usepackage{booktabs}")
    doc.append(r"\usepackage{array}")
    doc.append(r"\usepackage{tabularx}")
    doc.append(r"\usepackage{ragged2e}")
    doc.append(r"\usepackage{caption}")
    doc.append(r"\usepackage{float}")
    doc.append(r"\usepackage{amsmath}")
    doc.append(r"\usepackage{adjustbox}")
    doc.append(r"\usepackage{microtype}")
    doc.append(r"\captionsetup{font=small,labelfont=bf}")
    doc.append(r"\setlength{\emergencystretch}{2em}")
    doc.append(r"\title{ReasonBench Experimental Results (Partial Progress Snapshot)}")
    doc.append(r"\author{Automated Log/JSONL Extraction Pipeline}")
    doc.append(r"\date{\today}")

    doc.append(r"\begin{document}")
    doc.append(r"\maketitle")

    doc.append(r"\begin{abstract}")
    doc.append(
        "This document is written as an experiment-results section for a top AI conference style report. "
        "It aggregates every currently completed evaluation record from active ReasonBench runs, with no extrapolation for unfinished jobs. "
        "For each run, we present strategy-level core metrics, dataset-specific metric means, completion coverage, and log-derived execution diagnostics."
    )
    doc.append(r"\end{abstract}")

    doc.append(r"\section*{Experimental Setup and Reporting Protocol}")
    doc.append(
        "We evaluate multiple reasoning strategies across several benchmark datasets and model variants. "
        "The reporting policy is intentionally conservative: we only aggregate records that are fully completed and written to checkpoint/result JSONL files. "
        "Ongoing sessions are treated as partial snapshots rather than final claims."
    )
    doc.append("")
    doc.append(r"\textbf{Measurement equations.} "
               r"Core columns are computed as follows: "
               r"$\mathrm{cov\%}=100\times n/N_{\text{expected examples}}$, "
               r"$\mathrm{primary}=\frac{1}{n}\sum_i s_i$, "
               r"$\mathrm{api}=\frac{1}{n}\sum_i a_i$, "
               r"$\mathrm{time}=\frac{1}{n}\sum_i t_i$, "
               r"$\mathrm{cache\%}=100\times\frac{1}{n}\sum_i c_i$."
    )

    doc.append(r"\section*{Strategy Definitions}")
    doc.append(latex_strategy_definitions_section())

    doc.append(r"\section*{Illustrations}")
    doc.append(latex_illustrations_block())

    doc.append(r"\section*{Run Inventory}")
    doc.append(r"\begin{table*}[t]")
    doc.append(r"\centering")
    doc.append(r"\caption{Inventory of all discovered runs and current completion state.}")
    doc.append(r"\label{tab:run_inventory}")
    doc.append(r"\begin{adjustbox}{max width=\textwidth}")
    doc.append(r"\begin{tabular}{lrrrrl}")
    doc.append(r"\toprule")
    doc.append(r"Run & CompletedRecords & ExpectedRecords & Completion\% & StrategyCount & Model \\")
    doc.append(r"\midrule")
    for run in runs:
        m = run.meta
        doc.append(
            f"{sanitize_tex(run.run_name)} & {run.completed_records} & "
            f"{m['expected_records'] if m['expected_records'] else '--'} & "
            f"{fmt(run.completion_pct, 2)} & {m['strategy_count']} & "
            f"{sanitize_tex(short_model_name(str(m['model'])))} \\\\"
        )
    doc.append(r"\bottomrule")
    doc.append(r"\end{tabular}")
    doc.append(r"\end{adjustbox}")
    doc.append(r"\end{table*}")

    doc.append(r"\section*{Results Narrative}")
    doc.append(build_results_narrative(runs))

    doc.append(r"\section*{Core Variables and Metric Definitions}")
    doc.append(latex_variable_definition_table())
    doc.append(latex_metric_glossary(metrics_present))

    doc.append(r"\section*{Per-Run Strategy Results}")
    for run in runs:
        doc.append(latex_core_table_for_run(run))
        metric_table = latex_dataset_metric_table_for_run(run)
        if metric_table:
            doc.append(metric_table)

    doc.append(r"\section*{Log-Derived Execution Summary}")
    doc.append(latex_log_table(log_summaries))

    doc.append(r"\end{document}")
    return "\n".join(doc) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export conference-style evaluation report from JSONL + logs.")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--out-dir", default="reports/eval_report", help="Output directory")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_refs = discover_configs(repo_root / "configs" / "experiments")
    result_files = sorted((repo_root / "outputs").glob("*/results.jsonl"))

    runs: list[RunAggregate] = []
    for rf in result_files:
        rel_checkpoint = str(rf.relative_to(repo_root))
        cfg_ref = config_refs.get(rel_checkpoint)
        runs.append(aggregate_run(rf, cfg_ref, repo_root))

    log_summaries = collect_log_summaries(repo_root / "logs")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {
                "run_name": run.run_name,
                "result_file": run.result_file,
                "completed_records": run.completed_records,
                "completion_pct": run.completion_pct,
                "meta": run.meta,
                "numeric_metrics": run.numeric_metrics,
                "per_strategy": run.per_strategy,
                "best_strategy": run.best_strategy,
            }
            for run in runs
        ],
        "log_summaries": [
            {
                "session_name": item.session_name,
                "attempts": item.attempts,
                "finished_attempts": item.finished_attempts,
                "success_attempts": item.success_attempts,
                "failed_attempts": item.failed_attempts,
                "last_exit_code": item.last_exit_code,
                "latest_log_file": item.latest_log_file,
            }
            for item in log_summaries
        ],
    }

    json_path = out_dir / "evaluation_summary.json"
    tex_path = out_dir / "evaluation_tables.tex"

    write_json(json_path, payload)
    tex_doc = build_latex_document(runs, log_summaries)
    tex_path.write_text(tex_doc, encoding="utf-8")

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote LaTeX report: {tex_path}")


if __name__ == "__main__":
    main()
