# ReasonBench

ReasonBench is a research-oriented prompt evaluation harness that refactors and unifies the two uploaded codebases into a high-cohesion, low-coupling project.

## What changed

The original projects had useful ideas—reasoning modes, self-consistency-style retries, checkpointing, pooled HTTP clients, and dataset-specific grading—but they also mixed transport logic, prompting, evaluation, hard-coded secrets, Windows-only paths, and benchmark orchestration in the same files.

ReasonBench separates those concerns into:

- `datasets/`: dataset adapters and benchmark-safe formatting hints
- `strategies/`: prompt/orchestration strategies
- `clients/`: model API clients
- `evaluators/`: dataset-specific scoring
- `runtime/`: caching, checkpointing, rate limiting, leakage checks, reporting
- `integrations/`: optional bridges to official LiveBench / TruthfulQA scorers
- `cli/`: config-driven entrypoints

## Supported datasets

- `emunah/deductive_logical_reasoning-room_assignment`
- `domenicrosati/TruthfulQA`
- LiveBench `question.jsonl` files and optional official LiveBench repo integration

## Core ideas retained and improved

- prompt strategies from both repos are retained, but renamed honestly and implemented modularly
- pooled HTTP + retries + loop detection + caching + checkpoint resume
- strategy sweeps and latency accounting
- selective self-consistency for better quality/latency trade-offs
- benchmark-safe defaults to reduce contamination risk
- optional official benchmark bridges rather than fragile re-implementation where task-specific scorers already exist

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run a local sample room-assignment experiment:

```bash
reasonbench-run --config configs/experiments/room_assignment.toml
```

Run a TruthfulQA sweep:

```bash
reasonbench-run --config configs/experiments/truthfulqa.toml
```

Run a LiveBench-compatible local JSONL experiment:

```bash
reasonbench-run --config configs/experiments/livebench_reasoning.toml
```

## Configuration philosophy

ReasonBench uses TOML so experiments are portable, explicit, and diff-friendly. Secrets are resolved from environment variables, never from source code.

## Important benchmark note

Few-shot prompting is supported, but disabled by default for benchmarked runs. Use only external demos, not benchmark questions, unless you intentionally want an ablation that studies contamination or in-context adaptation.

See `docs/architecture.md` and `docs/research_playbook.md` for the full design rationale.
