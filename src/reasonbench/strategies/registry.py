from __future__ import annotations

from typing import Callable

from reasonbench.strategies.base import Strategy
from reasonbench.strategies.reflection import (
    BudgetedCascadeStrategy,
    CritiqueRefineStrategy,
    SelectiveSelfConsistencyStrategy,
    SelfConsistencyStrategy,
    SelfVerifyStrategy,
)
from reasonbench.strategies.simple import ConciseCoTStrategy, ConstraintDecomposeStrategy, DirectStrategy, ExternalDemoStrategy, LeastToMostStrategy


_STRATEGY_BUILDERS: dict[str, Callable[..., Strategy]] = {
    'direct': DirectStrategy,
    'concise_cot': ConciseCoTStrategy,
    'least_to_most': LeastToMostStrategy,
    'constraint_decompose': ConstraintDecomposeStrategy,
    'self_verify': SelfVerifyStrategy,
    'critique_refine': CritiqueRefineStrategy,
    'self_consistency': SelfConsistencyStrategy,
    'selective_self_consistency': SelectiveSelfConsistencyStrategy,
    'budgeted_cascade': BudgetedCascadeStrategy,
    'few_shot_exemplar': ExternalDemoStrategy,
}


def available_strategies() -> list[str]:
    return sorted(_STRATEGY_BUILDERS)


def build_strategy(name: str, **params) -> Strategy:
    if name not in _STRATEGY_BUILDERS:
        raise KeyError(f'Unknown strategy: {name}')
    return _STRATEGY_BUILDERS[name](**params)
