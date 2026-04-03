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
        evaluation = self.evaluator.evaluate(
            example,
            strategy_result.final_text,
            reasoning_content=strategy_result.reasoning_content,
        )
        from_cache = bool(strategy_result.metadata.get('from_cache', False))
        return ExperimentRecord(
            example_id=example.example_id,
            dataset_name=example.dataset_name,
            strategy_name=strategy_result.strategy_name,
            final_text=strategy_result.final_text,
            reasoning_content=strategy_result.reasoning_content,
            primary_score=evaluation.primary_score,
            metrics=evaluation.metrics,
            api_calls=strategy_result.api_calls,
            wall_time_s=strategy_result.wall_time_s,
            from_cache=from_cache,
            metadata={'example_metadata': example.metadata, 'trace': strategy_result.trace},
        )

    def _build_error_record(self, example: Example, strategy_name: str, exc: Exception) -> ExperimentRecord:
        return ExperimentRecord(
            example_id=example.example_id,
            dataset_name=example.dataset_name,
            strategy_name=strategy_name,
            final_text="",
            primary_score=0.0,
            metrics={
                "is_unscorable": True,
                "unscorable_reason": "generation_error",
                "format_valid": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            api_calls=0,
            wall_time_s=0.0,
            reasoning_content=None,
            from_cache=False,
            metadata={"example_metadata": example.metadata, "trace": []},
        )

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        warnings = self._preflight()
        if warnings and self.config.run.strict_benchmark_mode:
            raise RuntimeError(f'Potential demo leakage detected: {warnings[:3]}')

        total_expected_records = len(self.examples) * len(self.strategies)
        existing_records = len(self.checkpoint.load_all()) if self.checkpoint else 0
        print(
            f"[ReasonBench] experiment={self.config.run.experiment_name} "
            f"examples={len(self.examples)} strategies={len(self.strategies)} "
            f"expected_total={total_expected_records} existing_checkpoint_records={existing_records}"
        )

        pending: list[tuple[Example, Any]] = []
        for example in self.examples:
            for strategy in self.strategies:
                if self.checkpoint and self.checkpoint.has(example.example_id, strategy.name):
                    continue
                pending.append((example, strategy))

        print(f"[ReasonBench] pending_records={len(pending)} max_workers={self.config.run.max_workers}")

        new_records: list[ExperimentRecord] = []
        error_records = 0
        max_error_records = max(self.config.run.max_error_records, 0)
        with ThreadPoolExecutor(max_workers=self.config.run.max_workers) as executor:
            futures = {executor.submit(self._run_one, ex, strat): (ex, strat) for ex, strat in pending}
            progress_every = max(25, len(pending) // 200) if pending else 1
            completed_new = 0
            for future in as_completed(futures):
                ex, strat = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    print(
                        f"[ReasonBench] error example_id={ex.example_id} "
                        f"strategy={strat.name} error={exc!r}"
                    )
                    if not self.config.run.continue_on_error:
                        raise
                    error_records += 1
                    record = self._build_error_record(ex, strat.name, exc)
                    print(
                        f"[ReasonBench] marked_unscorable example_id={ex.example_id} "
                        f"strategy={strat.name} error_records={error_records}"
                    )
                new_records.append(record)
                if self.checkpoint:
                    self.checkpoint.append(record)
                completed_new += 1

                if max_error_records and error_records >= max_error_records:
                    raise RuntimeError(
                        f"Exceeded max_error_records={max_error_records}; aborting run after repeated generation failures."
                    )

                if completed_new == 1 or completed_new % progress_every == 0 or completed_new == len(pending):
                    total_done = existing_records + completed_new
                    pct = (100.0 * total_done / total_expected_records) if total_expected_records else 100.0
                    elapsed = time.perf_counter() - started
                    print(
                        f"[ReasonBench] progress total={total_done}/{total_expected_records} "
                        f"({pct:.2f}%) new={completed_new}/{len(pending)} elapsed_s={elapsed:.1f}"
                    )

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
            'error_records': error_records,
            'warnings': warnings,
            'elapsed_s': round(time.perf_counter() - started, 4),
        }
        write_manifest(self.config.output.manifest_path, manifest)
        return {'records': records, 'summary': summary_rows, 'manifest': manifest}
