from __future__ import annotations

from collections import Counter
import re
import time
from typing import Any

from reasonbench.clients.base import BaseLLMClient
from reasonbench.strategies.base import SingleShotStrategy, Strategy, StrategyRuntimeContext
from reasonbench.types import ChatMessage, Example, GenerationRequest, StrategyResult
from reasonbench.utils.text import canonical_vote_key


_UNCERTAIN_RE = re.compile(r"\b(maybe|perhaps|probably|guess|not sure|uncertain)\b", re.IGNORECASE)


class SelfVerifyStrategy(SingleShotStrategy):
    name = "self_verify"

    def strategy_instruction(self, example: Example) -> str:
        return "Produce a draft answer, mentally verify it against the task requirements, and then return a corrected final answer."

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        started = time.perf_counter()
        draft_result = self._run_turns(example, client, context)
        verify_prompt = (
            f"Task:\n{example.prompt_text}\n\n"
            f"Draft answer:\n{draft_result.final_text}\n\n"
            "Check the draft against the task instructions and common failure modes. "
            "If it is wrong, fix it. Return only the corrected final answer in the required format."
        )
        verify = client.generate(
            GenerationRequest(
                messages=[
                    ChatMessage("system", self.system_instruction(example, context)),
                    ChatMessage("user", verify_prompt),
                ],
                temperature=context.default_temperature,
                max_tokens=context.default_max_tokens,
            )
        )
        trace = list(draft_result.trace)
        trace.append({"turn_index": "verify", "assistant": verify.text, "latency_s": verify.latency_s, "from_cache": verify.from_cache})
        return StrategyResult(
            strategy_name=self.name,
            final_text=verify.text,
            api_calls=draft_result.api_calls + 1,
            wall_time_s=time.perf_counter() - started,
            trace=trace,
            metadata={"from_cache": draft_result.metadata.get("from_cache", False) and verify.from_cache},
        )


class CritiqueRefineStrategy(SingleShotStrategy):
    name = "critique_refine"

    def strategy_instruction(self, example: Example) -> str:
        return "Draft an answer, critique it for correctness and format, then refine it into the final answer."

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        started = time.perf_counter()
        draft_result = self._run_turns(example, client, context)
        critique = client.generate(
            GenerationRequest(
                messages=[
                    ChatMessage("system", "You are a strict benchmark reviewer. Identify correctness, reasoning, and formatting defects."),
                    ChatMessage(
                        "user",
                        f"Task:\n{example.prompt_text}\n\nDraft answer:\n{draft_result.final_text}\n\nProvide a concise critique.",
                    ),
                ],
                temperature=0.0,
                max_tokens=max(256, context.default_max_tokens // 2),
            )
        )
        refined = client.generate(
            GenerationRequest(
                messages=[
                    ChatMessage("system", self.system_instruction(example, context)),
                    ChatMessage(
                        "user",
                        f"Task:\n{example.prompt_text}\n\nDraft answer:\n{draft_result.final_text}\n\nCritique:\n{critique.text}\n\nReturn the improved final answer only.",
                    ),
                ],
                temperature=context.default_temperature,
                max_tokens=context.default_max_tokens,
            )
        )
        trace = list(draft_result.trace)
        trace.extend(
            [
                {"turn_index": "critique", "assistant": critique.text, "latency_s": critique.latency_s, "from_cache": critique.from_cache},
                {"turn_index": "refine", "assistant": refined.text, "latency_s": refined.latency_s, "from_cache": refined.from_cache},
            ]
        )
        return StrategyResult(
            strategy_name=self.name,
            final_text=refined.text,
            api_calls=draft_result.api_calls + 2,
            wall_time_s=time.perf_counter() - started,
            trace=trace,
        )


class SelfConsistencyStrategy(SingleShotStrategy):
    name = "self_consistency"

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        started = time.perf_counter()
        samples = int(self.params.get("num_samples", 5))
        temperature = float(self.params.get("temperature", max(context.default_temperature, 0.6)))
        votes: Counter[str] = Counter()
        winning_texts: dict[str, str] = {}
        trace: list[dict[str, Any]] = []
        api_calls = 0
        for sample_index in range(samples):
            result = self._run_turns(example, client, context, temperature=temperature)
            api_calls += result.api_calls
            vote_key = canonical_vote_key(result.final_text)
            votes[vote_key] += 1
            winning_texts.setdefault(vote_key, result.final_text)
            trace.append({"sample_index": sample_index, "vote_key": vote_key, "final_text": result.final_text, "wall_time_s": result.wall_time_s})
        winner_key, _ = votes.most_common(1)[0]
        final_text = winning_texts[winner_key]
        return StrategyResult(
            strategy_name=self.name,
            final_text=final_text,
            api_calls=api_calls,
            wall_time_s=time.perf_counter() - started,
            trace=trace,
            metadata={"vote_histogram": dict(votes)},
        )


class SelectiveSelfConsistencyStrategy(SelfConsistencyStrategy):
    name = "selective_self_consistency"

    def _needs_escalation(self, example: Example, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return True
        if _UNCERTAIN_RE.search(normalized):
            return True
        if example.dataset_name == "room_assignment":
            return "room 1" not in normalized.lower() and '"rooms"' not in normalized.lower()
        if example.dataset_name == "truthfulqa":
            return len(normalized.split()) < 4
        return False

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        started = time.perf_counter()
        first_pass = self._run_turns(example, client, context)
        if not self._needs_escalation(example, first_pass.final_text):
            first_pass.strategy_name = self.name
            first_pass.wall_time_s = time.perf_counter() - started
            first_pass.metadata["escalated"] = False
            return first_pass

        samples = int(self.params.get("num_samples", 4))
        temperature = float(self.params.get("temperature", max(context.default_temperature, 0.6)))
        votes: Counter[str] = Counter()
        winning_texts: dict[str, str] = {}
        trace: list[dict[str, Any]] = [{
            "stage": "first_pass",
            "final_text": first_pass.final_text,
            "wall_time_s": first_pass.wall_time_s,
        }]
        api_calls = first_pass.api_calls

        seed_key = canonical_vote_key(first_pass.final_text)
        votes[seed_key] += 1
        winning_texts.setdefault(seed_key, first_pass.final_text)

        for sample_index in range(samples - 1):
            result = self._run_turns(example, client, context, temperature=temperature)
            api_calls += result.api_calls
            vote_key = canonical_vote_key(result.final_text)
            votes[vote_key] += 1
            winning_texts.setdefault(vote_key, result.final_text)
            trace.append({
                "stage": "escalated_sample",
                "sample_index": sample_index,
                "vote_key": vote_key,
                "final_text": result.final_text,
                "wall_time_s": result.wall_time_s,
            })

        winner_key, _ = votes.most_common(1)[0]
        return StrategyResult(
            strategy_name=self.name,
            final_text=winning_texts[winner_key],
            api_calls=api_calls,
            wall_time_s=time.perf_counter() - started,
            trace=trace,
            metadata={
                "vote_histogram": dict(votes),
                "escalated": True,
            },
        )


class BudgetedCascadeStrategy(Strategy):
    name = "budgeted_cascade"

    def __init__(self, **params: Any):
        super().__init__(**params)
        from reasonbench.strategies.simple import DirectStrategy

        self.fast = DirectStrategy()
        self.slow = SelfVerifyStrategy()

    def _looks_good_enough(self, example: Example, text: str) -> bool:
        if example.dataset_name == "room_assignment":
            return "room 1" in text.lower() or '"rooms"' in text.lower()
        if example.dataset_name == "truthfulqa":
            return len(text.strip().split()) >= 3 and not _UNCERTAIN_RE.search(text)
        return bool(text.strip()) and not _UNCERTAIN_RE.search(text)

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        started = time.perf_counter()
        fast_result = self.fast.run(example, client, context)
        if self._looks_good_enough(example, fast_result.final_text):
            fast_result.strategy_name = self.name
            fast_result.wall_time_s = time.perf_counter() - started
            fast_result.metadata["cascade_path"] = "fast_only"
            return fast_result
        slow_result = self.slow.run(example, client, context)
        slow_result.strategy_name = self.name
        slow_result.wall_time_s = time.perf_counter() - started
        slow_result.api_calls += fast_result.api_calls
        slow_result.trace = [{"cascade_stage": "fast", "final_text": fast_result.final_text}] + slow_result.trace
        slow_result.metadata["cascade_path"] = "fast_then_verify"
        return slow_result
