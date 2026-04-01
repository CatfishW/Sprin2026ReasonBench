from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from reasonbench.types import ExperimentRecord


def write_manifest(path: str, manifest: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_summary(records: list[ExperimentRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.strategy_name].append(record)

    rows: list[dict[str, Any]] = []
    for strategy_name, items in grouped.items():
        rows.append(
            {
                "strategy": strategy_name,
                "examples": len(items),
                "mean_primary_score": round(mean(item.primary_score for item in items), 4),
                "mean_wall_time_s": round(mean(item.wall_time_s for item in items), 4),
                "mean_api_calls": round(mean(item.api_calls for item in items), 3),
                "cache_hit_rate": round(mean(1.0 if item.from_cache else 0.0 for item in items), 4),
                "format_valid_rate": round(mean(1.0 if item.metrics.get("format_valid", True) else 0.0 for item in items), 4),
            }
        )
    rows.sort(key=lambda row: row["mean_primary_score"], reverse=True)
    return rows


def write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
