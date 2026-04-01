from __future__ import annotations

from abc import ABC, abstractmethod

from reasonbench.types import EvaluationResult, Example


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, example: Example, prediction: str) -> EvaluationResult:
        raise NotImplementedError


def make_evaluator(dataset_name: str) -> Evaluator:
    if dataset_name == 'room_assignment':
        from reasonbench.evaluators.room_assignment import RoomAssignmentEvaluator
        return RoomAssignmentEvaluator()
    if dataset_name == 'truthfulqa':
        from reasonbench.evaluators.truthfulqa import TruthfulQAEvaluator
        return TruthfulQAEvaluator()
    if dataset_name == 'livebench':
        from reasonbench.evaluators.livebench import LiveBenchProxyEvaluator
        return LiveBenchProxyEvaluator()
    raise ValueError(f'Unsupported dataset for evaluation: {dataset_name}')
