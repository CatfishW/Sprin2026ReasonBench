from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Example:
    example_id: str
    dataset_name: str
    split: str
    turns: list[str]
    reference: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return "\n\n".join(self.turns)


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int | None = None
    extra_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    raw_response: dict[str, Any] | None = None
    latency_s: float = 0.0
    attempts: int = 1
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    strategy_name: str
    final_text: str
    api_calls: int
    wall_time_s: float
    trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    primary_score: float
    metrics: dict[str, Any]


@dataclass
class ExperimentRecord:
    example_id: str
    dataset_name: str
    strategy_name: str
    final_text: str
    primary_score: float
    metrics: dict[str, Any]
    api_calls: int
    wall_time_s: float
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
