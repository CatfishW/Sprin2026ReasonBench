# Research playbook

## Recommended ablations

1. Direct vs concise CoT vs least-to-most
2. Self-verify vs critique-refine
3. Full self-consistency vs selective self-consistency (new built-in strategy)
4. Budgeted cascade vs always-expensive reasoning
5. Benchmark-safe zero-shot vs external-demo few-shot

## Suggested primary metrics

- Room Assignment: room exact accuracy + entity room accuracy
- TruthfulQA: truth delta proxy and informativeness proxy; use official evaluation when possible
- LiveBench: official scorer whenever reporting headline numbers

## Suggested efficiency metrics

- wall-clock time per example
- API calls per example
- cache hit rate
- invalid format rate

## Paper-facing hypotheses

- symbolic reasoning tasks benefit more from constraint decomposition and verification
- truthfulness tasks benefit more from concise answers and refusal-to-speculate policies than from long chains of thought
- selective escalation recovers much of the quality of expensive reasoning while preserving latency
