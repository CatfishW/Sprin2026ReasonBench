from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


@dataclass
class DatasetConfig:
    kind: str
    split: str = "train"
    local_path: str | None = None
    hf_dataset: str | None = None
    hf_subset: str | None = None
    limit: int | None = None
    shuffle: bool = False
    seed: int = 0
    task_filters: list[str] = field(default_factory=list)


@dataclass
class ClientConfig:
    kind: str = "openai_compatible"
    base_url: str = ""
    model: str = ""
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: int = 180
    max_retries: int = 4
    min_request_interval_s: float = 0.0
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    supports_batch: bool = False
    completions_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_payload: dict[str, Any] = field(default_factory=dict)
    loop_retries: int = 2
    loop_length_ceiling: int = 50000


@dataclass
class RunConfig:
    experiment_name: str = "reasonbench_run"
    max_workers: int = 4
    seed: int = 0
    strict_benchmark_mode: bool = True
    enable_cache: bool = True
    enable_checkpoint: bool = True
    enable_leakage_check: bool = True
    continue_on_error: bool = True
    max_error_records: int = 1000


@dataclass
class OutputConfig:
    output_dir: str = "outputs/default"
    cache_path: str = "outputs/default/cache.sqlite"
    checkpoint_path: str = "outputs/default/results.jsonl"
    summary_path: str = "outputs/default/summary.csv"
    manifest_path: str = "outputs/default/manifest.json"


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    client: ClientConfig
    run: RunConfig
    output: OutputConfig
    strategies: list[StrategyConfig]
    raw: dict[str, Any] = field(default_factory=dict)


def _read_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _parse_strategies(raw: dict[str, Any]) -> list[StrategyConfig]:
    section = raw.get("strategies", {})
    strategies: list[StrategyConfig] = []
    for name, params in section.items():
        if isinstance(params, dict):
            params_copy = dict(params)
            enabled = params_copy.pop("enabled", True)
            if enabled:
                strategies.append(StrategyConfig(name=name, params=params_copy))
        else:
            strategies.append(StrategyConfig(name=name, params={}))
    if not strategies:
        raise ValueError("No strategies enabled in config.")
    return strategies


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _read_toml(path)
    dataset = DatasetConfig(**raw.get("dataset", {}))
    client = ClientConfig(**raw.get("client", {}))
    run = RunConfig(**raw.get("run", {}))
    output = OutputConfig(**raw.get("output", {}))
    strategies = _parse_strategies(raw)
    return ExperimentConfig(dataset=dataset, client=client, run=run, output=output, strategies=strategies, raw=raw)
