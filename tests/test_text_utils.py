from reasonbench.utils.text import strip_reasoning_prefix


def test_strip_reasoning_prefix_exact_match() -> None:
    reasoning = "analysis content"
    prediction = reasoning + "\n\nfinal answer"
    assert strip_reasoning_prefix(prediction, reasoning) == "final answer"


def test_strip_reasoning_prefix_does_not_strip_non_matching_prefix() -> None:
    reasoning = "other reasoning"
    prediction = "final answer"
    assert strip_reasoning_prefix(prediction, reasoning) == "final answer"


def test_strip_reasoning_prefix_preserves_colon_without_space() -> None:
    reasoning = "analysis"
    prediction = reasoning + "\n\n:final"
    assert strip_reasoning_prefix(prediction, reasoning) == ":final"


def test_strip_reasoning_prefix_removes_delimiter_with_space() -> None:
    reasoning = "analysis"
    prediction = reasoning + "\n\n: final"
    assert strip_reasoning_prefix(prediction, reasoning) == "final"
