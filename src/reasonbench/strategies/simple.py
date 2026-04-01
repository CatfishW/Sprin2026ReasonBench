from __future__ import annotations

from reasonbench.strategies.base import FewShotExemplarStrategy, SingleShotStrategy
from reasonbench.types import Example


class DirectStrategy(SingleShotStrategy):
    name = 'direct'

    def strategy_instruction(self, example: Example) -> str:
        return 'Solve the task directly. Keep the answer concise and avoid unnecessary explanation.'


class ConciseCoTStrategy(SingleShotStrategy):
    name = 'concise_cot'

    def strategy_instruction(self, example: Example) -> str:
        return 'Reason step by step briefly, then provide the final answer in the requested format.'


class LeastToMostStrategy(SingleShotStrategy):
    name = 'least_to_most'

    def strategy_instruction(self, example: Example) -> str:
        return (
            'Break the problem into smaller subproblems, solve them in order, and then provide the final answer. '
            'Keep the decomposition compact.'
        )


class ConstraintDecomposeStrategy(SingleShotStrategy):
    name = 'constraint_decompose'

    def strategy_instruction(self, example: Example) -> str:
        return (
            'First extract the critical constraints or facts in a compact list, then solve using those constraints, '
            'then give the final answer.'
        )


class ExternalDemoStrategy(FewShotExemplarStrategy):
    name = 'few_shot_exemplar'

    def strategy_instruction(self, example: Example) -> str:
        return 'Use the external exemplars as style guidance only, then solve the new problem.'
