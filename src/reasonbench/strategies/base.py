from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any

from reasonbench.clients.base import BaseLLMClient
from reasonbench.types import ChatMessage, Example, GenerationRequest, StrategyResult


@dataclass
class StrategyRuntimeContext:
    strict_benchmark_mode: bool = True
    default_temperature: float = 0.0
    default_max_tokens: int = 1024


class Strategy(ABC):
    name: str = "strategy"
    uses_external_demos: bool = False

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        raise NotImplementedError

    def demo_texts(self) -> list[str]:
        return []


class SingleShotStrategy(Strategy):
    def system_instruction(self, example: Example, context: StrategyRuntimeContext) -> str:
        base = (
            "You are a careful reasoning assistant. Follow the task instructions exactly, "
            "avoid hallucinations, and prefer explicit uncertainty over fabricated claims."
        )
        if context.strict_benchmark_mode:
            base += " Do not use hidden benchmark examples or benchmark-specific leakage."
        extra = self.strategy_instruction(example)
        return f"{base}\n\n{extra}".strip()

    def strategy_instruction(self, example: Example) -> str:
        return "Answer carefully."

    def few_shot_messages(self, example: Example) -> list[ChatMessage]:
        return []

    def render_turn(self, turn_text: str, example: Example, turn_index: int, total_turns: int) -> str:
        if turn_index == total_turns - 1 and example.metadata.get("format_hint"):
            return f"{turn_text}\n\nFormatting requirement:\n{example.metadata['format_hint']}"
        return turn_text

    def _run_turns(
        self,
        example: Example,
        client: BaseLLMClient,
        context: StrategyRuntimeContext,
        system_instruction: str | None = None,
        extra_messages: list[ChatMessage] | None = None,
        temperature: float | None = None,
    ) -> StrategyResult:
        started = time.perf_counter()
        system_message = ChatMessage("system", system_instruction or self.system_instruction(example, context))
        history: list[ChatMessage] = []
        trace: list[dict[str, Any]] = []
        final_text = ""
        api_calls = 0
        extra_messages = extra_messages or []
        for turn_index, turn_text in enumerate(example.turns):
            rendered_turn = self.render_turn(turn_text, example, turn_index, len(example.turns))
            request = GenerationRequest(
                messages=[system_message, *extra_messages, *history, ChatMessage("user", rendered_turn)],
                temperature=context.default_temperature if temperature is None else temperature,
                max_tokens=context.default_max_tokens,
            )
            result = client.generate(request)
            api_calls += 1
            final_text = result.text
            history.extend([ChatMessage("user", rendered_turn), ChatMessage("assistant", result.text)])
            trace.append(
                {
                    "turn_index": turn_index,
                    "user": rendered_turn,
                    "assistant": result.text,
                    "latency_s": result.latency_s,
                    "from_cache": result.from_cache,
                    **result.metadata,
                }
            )
        return StrategyResult(
            strategy_name=self.name,
            final_text=final_text,
            api_calls=api_calls,
            wall_time_s=time.perf_counter() - started,
            trace=trace,
            metadata={"from_cache": all(item.get("from_cache", False) for item in trace)} if trace else {},
        )

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        return self._run_turns(
            example=example,
            client=client,
            context=context,
            extra_messages=self.few_shot_messages(example),
        )


class FewShotExemplarStrategy(SingleShotStrategy):
    uses_external_demos = True

    def __init__(self, demo_path: str, **params: Any):
        super().__init__(demo_path=demo_path, **params)
        self.demo_path = demo_path
        self._messages = self._load_demo_messages(demo_path)

    def _load_demo_messages(self, demo_path: str) -> list[ChatMessage]:
        path = Path(demo_path)
        with open(path, "r", encoding="utf-8") as handle:
            data = [json.loads(line) for line in handle if line.strip()]
        messages: list[ChatMessage] = []
        for item in data:
            user = str(item.get("user") or "").strip()
            assistant = str(item.get("assistant") or "").strip()
            if user and assistant:
                messages.append(ChatMessage("user", user))
                messages.append(ChatMessage("assistant", assistant))
        return messages

    def few_shot_messages(self, example: Example) -> list[ChatMessage]:
        return list(self._messages)

    def demo_texts(self) -> list[str]:
        return [msg.content for msg in self._messages if msg.role == "user"]
