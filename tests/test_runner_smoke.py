from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.runtime.runner import ExperimentRunner
from reasonbench.types import GenerationRequest, GenerationResult


class FakeClient:
    def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = request.messages[-1].content.lower()
        if "what color is the sun" in prompt:
            return GenerationResult(text="The sun looks white from space.")
        if "what are the tenants of the rooms" in prompt:
            return GenerationResult(text="room 1: Alice\nroom 2: Charlie\nroom 3: Bob")
        if "2+2" in prompt:
            return GenerationResult(text="4")
        return GenerationResult(text="room 1: Alice\nroom 2: Bob")


def test_runner_smoke(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project_root / "configs/experiments/room_assignment.toml")
    config.output.output_dir = str(tmp_path / "out")
    config.output.cache_path = str(tmp_path / "out/cache.sqlite")
    config.output.checkpoint_path = str(tmp_path / "out/results.jsonl")
    config.output.summary_path = str(tmp_path / "out/summary.csv")
    config.output.manifest_path = str(tmp_path / "out/manifest.json")
    config.dataset.limit = 1
    runner = ExperimentRunner(config, client=FakeClient())
    result = runner.run()
    assert result["summary"]
