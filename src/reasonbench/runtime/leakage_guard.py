from __future__ import annotations

from reasonbench.types import Example
from reasonbench.utils.text import tokenize


def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def check_demo_leakage(examples: list[Example], demo_texts: list[str], threshold: float = 0.85) -> list[dict[str, str | float]]:
    warnings: list[dict[str, str | float]] = []
    for demo in demo_texts:
        for example in examples:
            score = jaccard_similarity(demo, example.prompt_text)
            if score >= threshold:
                warnings.append({'example_id': example.example_id, 'score': score, 'demo_text': demo[:160]})
    return warnings
