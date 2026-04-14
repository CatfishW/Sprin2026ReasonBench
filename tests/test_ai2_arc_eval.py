from reasonbench.evaluators.ai2_arc import AI2ARCEvaluator
from reasonbench.types import Example


def _example(answer_key: str = "B") -> Example:
    return Example(
        example_id="arc-test-1",
        dataset_name="ai2_arc",
        split="test",
        turns=["dummy"],
        reference={
            "answer_key": answer_key,
            "choices": {
                "A": "alpha option",
                "B": "beta option",
                "C": "gamma option",
                "D": "delta option",
            },
            "subset": "ARC-Challenge",
        },
    )


def test_arc_prefers_final_answer_over_early_mentions() -> None:
    prediction = "I first considered A, but that was wrong. FINAL ANSWER: B"
    result = AI2ARCEvaluator().evaluate(_example(answer_key="B"), prediction)
    assert result.primary_score == 1.0
    assert result.metrics["selected_label"] == "B"


def test_arc_uses_latest_explicit_answer() -> None:
    prediction = "Answer: A. After checking again, answer: B"
    result = AI2ARCEvaluator().evaluate(_example(answer_key="B"), prediction)
    assert result.primary_score == 1.0
    assert result.metrics["selected_label"] == "B"


def test_arc_strips_reasoning_prefix_when_provided() -> None:
    reasoning = "Let's think: maybe A due to pattern."
    prediction = reasoning + "\n\nB"
    result = AI2ARCEvaluator().evaluate(_example(answer_key="B"), prediction, reasoning_content=reasoning)
    assert result.primary_score == 1.0
    assert result.metrics["selected_label"] == "B"


def test_arc_text_fallback_requires_strong_margin() -> None:
    example = _example(answer_key="B")
    result = AI2ARCEvaluator().evaluate(example, "beta option")
    assert result.metrics["selected_label"] == "B"
    assert result.metrics["text_match_score"] >= 0.9


def test_arc_does_not_guess_from_ambiguous_mentions() -> None:
    example = _example(answer_key="B")
    prediction = "I considered A and C while reasoning, but I am not confident."
    result = AI2ARCEvaluator().evaluate(example, prediction)
    assert result.metrics["selected_label"] == ""
    assert result.metrics["format_valid"] is False
