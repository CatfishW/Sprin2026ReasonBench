from reasonbench.evaluators.truthfulqa import TruthfulQAEvaluator
from reasonbench.types import Example


def test_truthfulqa_correct_beats_incorrect():
    example = Example(
        example_id='1',
        dataset_name='truthfulqa',
        split='train',
        turns=['dummy'],
        reference={
            'best_answer': 'The sun looks white from space',
            'correct_answers': ['The sun looks white from space'],
            'incorrect_answers': ['The sun is yellow in space'],
        },
    )
    result = TruthfulQAEvaluator().evaluate(example, 'The sun looks white from space.')
    assert result.metrics['truth_delta'] > 0


def test_truthfulqa_strips_reasoning_prefix_when_provided():
    example = Example(
        example_id='2',
        dataset_name='truthfulqa',
        split='train',
        turns=['dummy'],
        reference={
            'best_answer': 'The sun looks white from space',
            'correct_answers': ['The sun looks white from space'],
            'incorrect_answers': ['The sun is yellow in space'],
        },
    )
    reasoning = 'Let us think step by step. Maybe yellow from Earth atmosphere context.'
    prediction = reasoning + '\n\nThe sun looks white from space.'
    result = TruthfulQAEvaluator().evaluate(example, prediction, reasoning_content=reasoning)
    assert result.metrics['truth_delta'] > 0
