from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.runtime.runner import ExperimentRunner
from reasonbench.types import GenerationRequest, GenerationResult


class FakeClient:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(text="room 1: Alice\nroom 2: Charlie\nroom 3: Bob")


def test_checkpoint_resume_keeps_full_summary(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project_root / "configs/experiments/room_assignment.toml")
    config.output.output_dir = str(tmp_path / "out")
    config.output.cache_path = str(tmp_path / "out/cache.sqlite")
    config.output.checkpoint_path = str(tmp_path / "out/results.jsonl")
    config.output.summary_path = str(tmp_path / "out/summary.csv")
    config.output.manifest_path = str(tmp_path / "out/manifest.json")
    config.dataset.limit = 1

    first = ExperimentRunner(config, client=FakeClient()).run()
    second = ExperimentRunner(config, client=FakeClient()).run()

    assert first["manifest"]["completed_records"] == second["manifest"]["completed_records"]
    assert second["manifest"]["new_records"] == 0
    assert second["summary"]
