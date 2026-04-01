from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset


OUT_DIR = Path("data")
LIVEBENCH_CATEGORIES = [
    "reasoning",
    "math",
    "coding",
    "language",
    "data_analysis",
    "instruction_following",
]


def normalize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    return str(value)


def write_room_assignment() -> int:
    dataset = load_dataset("emunah/deductive_logical_reasoning-room_assignment", split="train")
    out_path = OUT_DIR / "room_assignment" / "room_assignment_train_full.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            row = {k: normalize_value(v) for k, v in row.items()}
            row["id"] = str(idx)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_truthfulqa() -> int:
    dataset = load_dataset("domenicrosati/TruthfulQA", split="train")
    out_path = OUT_DIR / "truthfulqa" / "truthfulqa_train_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [{k: normalize_value(v) for k, v in row.items()} for row in dataset]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset.column_names))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_livebench() -> dict[str, int]:
    counts: dict[str, int] = {}

    for category in LIVEBENCH_CATEGORIES:
        dataset = load_dataset(f"livebench/{category}", split="test")
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in dataset:
            normalized = {k: normalize_value(v) for k, v in row.items()}
            task = str(normalized.get("task") or "unknown_task")
            by_task[task].append(normalized)

        total = 0
        for task, rows in by_task.items():
            out_path = OUT_DIR / "livebench" / category / task / "question.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)

        counts[category] = total

    return counts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    room_count = write_room_assignment()
    truthfulqa_count = write_truthfulqa()
    livebench_counts = write_livebench()

    summary = {
        "room_assignment_train": room_count,
        "truthfulqa_train": truthfulqa_count,
        "livebench": livebench_counts,
        "livebench_total": sum(livebench_counts.values()),
    }

    summary_path = OUT_DIR / "dataset_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
