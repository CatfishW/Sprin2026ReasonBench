from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import csv
import json
from typing import Any, Iterable

from reasonbench.config import DatasetConfig
from reasonbench.types import Example


class DatasetAdapter(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def load(self) -> list[Example]:
        raise NotImplementedError


class DatasetLoadError(RuntimeError):
    pass


def _maybe_limit(examples: list[Example], limit: int | None) -> list[Example]:
    return examples[:limit] if limit is not None else examples


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_json_per_line_or_text(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as handle:
        text = handle.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip().rstrip(',')
        if not line or line in {'[', ']', '{', '}'}:
            continue
        if line.startswith('{') and line.endswith('}'):
            rows.append(json.loads(line))
    return rows


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    with open(path, 'r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def load_hf_dataset(config: DatasetConfig) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetLoadError(
            'The `datasets` package is required for Hugging Face loading. Install with `pip install -e .[hf]`.') from exc

    dataset_name = config.hf_dataset
    if not dataset_name:
        raise DatasetLoadError('hf_dataset is required for Hugging Face loading.')
    dataset = load_dataset(dataset_name, config.hf_subset, split=config.split)
    return list(dataset)


def make_dataset_adapter(config: DatasetConfig) -> DatasetAdapter:
    if config.kind == 'room_assignment':
        from reasonbench.datasets.room_assignment import RoomAssignmentAdapter
        return RoomAssignmentAdapter(config)
    if config.kind == 'truthfulqa':
        from reasonbench.datasets.truthfulqa import TruthfulQAAdapter
        return TruthfulQAAdapter(config)
    if config.kind == 'livebench':
        from reasonbench.datasets.livebench import LiveBenchAdapter
        return LiveBenchAdapter(config)
    raise ValueError(f'Unsupported dataset kind: {config.kind}')
