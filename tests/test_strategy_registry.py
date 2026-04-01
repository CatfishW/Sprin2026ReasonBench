from reasonbench.strategies.registry import available_strategies


def test_registry_contains_expected_strategies():
    names = set(available_strategies())
    assert {'direct', 'concise_cot', 'self_verify', 'self_consistency', 'selective_self_consistency', 'budgeted_cascade'} <= names
