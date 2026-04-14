from __future__ import annotations

from reasonbench.evaluators.base import Evaluator
from reasonbench.types import EvaluationResult, Example
from reasonbench.utils.text import normalize_text, strip_reasoning_prefix


class LiveBenchProxyEvaluator(Evaluator):
    def evaluate(self, example: Example, prediction: str, reasoning_content: str | None = None) -> EvaluationResult:
        cleaned_prediction = strip_reasoning_prefix(prediction, reasoning_content)
        ground_truth_raw = example.reference.get('ground_truth')
        has_ground_truth = ground_truth_raw is not None and bool(str(ground_truth_raw).strip())

        if not has_ground_truth:
            return EvaluationResult(
                primary_score=0.0,
                metrics={
                    'proxy_exact_match': 0.0,
                    'proxy_contains_match': 0.0,
                    'official_scorer_recommended': True,
                    'is_unscorable': True,
                    'unscorable_reason': 'missing_ground_truth',
                    'task': example.metadata.get('task', ''),
                    'category': example.metadata.get('category', ''),
                },
            )

        ground_truth = str(ground_truth_raw)
        gt_norm = normalize_text(ground_truth)
        pred_norm = normalize_text(cleaned_prediction)
        exact = float(bool(gt_norm) and pred_norm == gt_norm)
        contains = float(bool(gt_norm) and gt_norm in pred_norm)
        score = exact if exact else contains * 0.5
        return EvaluationResult(
            primary_score=score,
            metrics={
                'proxy_exact_match': exact,
                'proxy_contains_match': contains,
                'official_scorer_recommended': True,
                'is_unscorable': False,
                'unscorable_reason': '',
                'task': example.metadata.get('task', ''),
                'category': example.metadata.get('category', ''),
            },
        )
