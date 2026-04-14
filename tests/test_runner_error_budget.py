from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.runtime.runner import ExperimentRunner
from reasonbench.types import GenerationRequest


class RetryError(Exception):
    pass


class RetryFailClient:
    def generate(self, request: GenerationRequest):  # noqa: ARG002
        raise RetryError(
            "RetryError(MaxRetryError(\"HTTPSConnectionPool(host='llm.agaii.org', port=443): "
            "Max retries exceeded with url: /v1/chat/completions (Caused by ResponseError('too many 504 error responses'))\"))"
        )


def test_retry_error_does_not_consume_blocking_budget(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project_root / "configs/experiments/room_assignment.toml")
    config.output.output_dir = str(tmp_path / "out")
    config.output.cache_path = str(tmp_path / "out/cache.sqlite")
    config.output.checkpoint_path = str(tmp_path / "out/results.jsonl")
    config.output.summary_path = str(tmp_path / "out/summary.csv")
    config.output.manifest_path = str(tmp_path / "out/manifest.json")
    config.dataset.limit = 1
    config.run.max_workers = 1
    config.run.continue_on_error = True
    config.run.max_error_records = 1
    config.strategies = config.strategies[:2]

    runner = ExperimentRunner(config, client=RetryFailClient())
    result = runner.run()

    manifest = result["manifest"]
    assert manifest["error_records"] == 2
    assert manifest["blocking_error_records"] == 0
