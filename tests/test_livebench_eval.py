from reasonbench.evaluators.livebench import LiveBenchProxyEvaluator
from reasonbench.types import Example


def _example(ground_truth, task: str = "LCB_generation") -> Example:
    return Example(
        example_id="livebench-test-1",
        dataset_name="livebench",
        split="test",
        turns=["dummy"],
        reference={"ground_truth": ground_truth, "task": task, "category": "coding"},
        metadata={"task": task, "category": "coding"},
    )


def test_livebench_marks_missing_ground_truth_unscorable() -> None:
    result = LiveBenchProxyEvaluator().evaluate(_example(None), "print('hello')")
    assert result.primary_score == 0.0
    assert result.metrics["is_unscorable"] is True
    assert result.metrics["unscorable_reason"] == "missing_ground_truth"


def test_livebench_scores_when_ground_truth_present() -> None:
    result = LiveBenchProxyEvaluator().evaluate(_example("final answer"), "final answer")
    assert result.primary_score == 1.0
    assert result.metrics["is_unscorable"] is False


def test_livebench_strips_reasoning_prefix_when_provided() -> None:
    reasoning = "Reasoning details go here."
    prediction = reasoning + "\n\nfinal answer"
    result = LiveBenchProxyEvaluator().evaluate(_example("final answer", task="language"), prediction, reasoning_content=reasoning)
    assert result.primary_score == 1.0
    assert result.metrics["is_unscorable"] is False
