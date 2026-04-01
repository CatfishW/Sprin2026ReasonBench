# Architecture notes

## Design goals

1. High cohesion: each package owns one responsibility.
2. Low coupling: strategies do not know transport internals; evaluators do not know prompt templates; datasets only expose normalized `Example` objects.
3. Benchmark safety: contamination-prone features are opt-in.
4. Reproducibility: config + manifest + JSONL records + summary reports.

## Why the old naming was changed

The uploaded code used names like `PURE_RL`, `SFT`, and `DISTILLATION` for mechanisms that were actually inference-time prompting patterns, exemplar injection, or multi-call teacher/student prompting. Those names are misleading in a paper. ReasonBench replaces them with accurate names such as:

- `few_shot_exemplar`
- `self_consistency`
- `self_verify`
- `critique_refine`
- `budgeted_cascade`
- `selective_self_consistency`
- `constraint_decompose`

## Speed / latency principles

- thread-safe retries and rate limiting
- optional SQLite prompt cache
- checkpoint append-only JSONL for resume
- minimal-overhead single-shot strategies for fast baselines
- selective escalation for harder examples instead of always paying for expensive reasoning

## Benchmark integration policy

- Room Assignment: native adapter + native evaluator
- TruthfulQA: native adapter + proxy evaluator + optional official repo bridge
- LiveBench: native question loader + proxy evaluator + optional official repo bridge for publication-grade numbers
