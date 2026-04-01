from __future__ import annotations

from reasonbench.evaluators.base import Evaluator
from reasonbench.types import EvaluationResult, Example
from reasonbench.utils.text import normalize_text


class LiveBenchProxyEvaluator(Evaluator):
    def evaluate(self, example: Example, prediction: str) -> EvaluationResult:
        ground_truth = str(example.reference.get('ground_truth') or '')
        gt_norm = normalize_text(ground_truth)
        pred_norm = normalize_text(prediction)
        exact = float(bool(gt_norm) and pred_norm == gt_norm)
        contains = float(bool(gt_norm) and gt_norm in pred_norm)
        score = exact if exact else contains * 0.5
        return EvaluationResult(
            primary_score=score,
            metrics={
                'proxy_exact_match': exact,
                'proxy_contains_match': contains,
                'official_scorer_recommended': True,
                'task': example.metadata.get('task', ''),
                'category': example.metadata.get('category', ''),
            },
        )
