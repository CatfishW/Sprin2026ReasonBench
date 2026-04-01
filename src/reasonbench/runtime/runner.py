from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import time
from typing import Any

from reasonbench.clients.openai_compatible import OpenAICompatibleClient
from reasonbench.config import ExperimentConfig
from reasonbench.datasets.base import make_dataset_adapter
from reasonbench.evaluators.base import make_evaluator
from reasonbench.runtime.cache import SQLiteCache
from reasonbench.runtime.checkpoint import JSONLCheckpointStore
from reasonbench.runtime.leakage_guard import check_demo_leakage
from reasonbench.runtime.reports import build_summary, write_manifest, write_summary_csv, write_summary_markdown
from reasonbench.strategies.base import StrategyRuntimeContext
from reasonbench.strategies.registry import build_strategy
from reasonbench.types import Example, ExperimentRecord


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, client: OpenAICompatibleClient | None = None):
        self.config = config
        self.dataset_adapter = make_dataset_adapter(config.dataset)
        self.examples = self.dataset_adapter.load()
        self.evaluator = make_evaluator(config.dataset.kind)
        self.cache = SQLiteCache(config.output.cache_path) if config.run.enable_cache else None
        self.client = client or OpenAICompatibleClient(config.client, cache=self.cache)
        self.checkpoint = JSONLCheckpointStore(config.output.checkpoint_path) if config.run.enable_checkpoint else None
        self.strategy_context = StrategyRuntimeContext(
            strict_benchmark_mode=config.run.strict_benchmark_mode,
            default_temperature=config.client.default_temperature,
            default_max_tokens=config.client.default_max_tokens,
        )
        self.strategies = [build_strategy(item.name, **item.params) for item in config.strategies]

    def _preflight(self) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        if not self.config.run.enable_leakage_check:
            return warnings
        for strategy in self.strategies:
            demo_texts = strategy.demo_texts()
            if demo_texts:
                warnings.extend(check_demo_leakage(self.examples, demo_texts))
        return warnings

    def _run_one(self, example: Example, strategy) -> ExperimentRecord:
        strategy_result = strategy.run(example, self.client, self.strategy_context)
        evaluation = self.evaluator.evaluate(example, strategy_result.final_text)
        from_cache = bool(strategy_result.metadata.get('from_cache', False))
        return ExperimentRecord(
            example_id=example.example_id,
            dataset_name=example.dataset_name,
            strategy_name=strategy_result.strategy_name,
            final_text=strategy_result.final_text,
            primary_score=evaluation.primary_score,
            metrics=evaluation.metrics,
            api_calls=strategy_result.api_calls,
            wall_time_s=strategy_result.wall_time_s,
            from_cache=from_cache,
            metadata={'example_metadata': example.metadata, 'trace': strategy_result.trace},
        )

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        warnings = self._preflight()
        if warnings and self.config.run.strict_benchmark_mode:
            raise RuntimeError(f'Potential demo leakage detected: {warnings[:3]}')

        pending: list[tuple[Example, Any]] = []
        for example in self.examples:
            for strategy in self.strategies:
                if self.checkpoint and self.checkpoint.has(example.example_id, strategy.name):
                    continue
                pending.append((example, strategy))

        new_records: list[ExperimentRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.run.max_workers) as executor:
            futures = {executor.submit(self._run_one, ex, strat): (ex, strat) for ex, strat in pending}
            for future in as_completed(futures):
                record = future.result()
                new_records.append(record)
                if self.checkpoint:
                    self.checkpoint.append(record)

        records = self.checkpoint.load_all() if self.checkpoint else new_records
        summary_rows = build_summary(records)
        write_summary_csv(self.config.output.summary_path, summary_rows)
        write_summary_markdown(self.config.output.summary_path.replace('.csv', '.md'), summary_rows)
        manifest = {
            'experiment_name': self.config.run.experiment_name,
            'dataset': asdict(self.config.dataset),
            'client': {k: v for k, v in asdict(self.config.client).items() if k != 'api_key'},
            'run': asdict(self.config.run),
            'output': asdict(self.config.output),
            'strategies': [asdict(item) for item in self.config.strategies],
            'example_count': len(self.examples),
            'completed_records': len(records),
            'new_records': len(new_records),
            'warnings': warnings,
            'elapsed_s': round(time.perf_counter() - started, 4),
        }
        write_manifest(self.config.output.manifest_path, manifest)
        return {'records': records, 'summary': summary_rows, 'manifest': manifest}
