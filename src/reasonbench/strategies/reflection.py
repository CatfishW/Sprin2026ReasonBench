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
            reasoning_content=verify.reasoning_content,
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
            reasoning_content=refined.reasoning_content,
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
        winning_reasoning: dict[str, str | None] = {}
        trace: list[dict[str, Any]] = []
        api_calls = 0
        for sample_index in range(samples):
            result = self._run_turns(example, client, context, temperature=temperature)
            api_calls += result.api_calls
            vote_key = canonical_vote_key(result.final_text)
            votes[vote_key] += 1
            winning_texts.setdefault(vote_key, result.final_text)
            winning_reasoning.setdefault(vote_key, result.reasoning_content)
            trace.append(
                {
                    "sample_index": sample_index,
                    "vote_key": vote_key,
                    "final_text": result.final_text,
                    "reasoning_content": result.reasoning_content,
                    "wall_time_s": result.wall_time_s,
                }
            )
        winner_key, _ = votes.most_common(1)[0]
        final_text = winning_texts[winner_key]
        return StrategyResult(
            strategy_name=self.name,
            final_text=final_text,
            api_calls=api_calls,
            wall_time_s=time.perf_counter() - started,
            reasoning_content=winning_reasoning.get(winner_key),
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
        winning_reasoning: dict[str, str | None] = {}
        trace: list[dict[str, Any]] = [{
            "stage": "first_pass",
            "final_text": first_pass.final_text,
            "reasoning_content": first_pass.reasoning_content,
            "wall_time_s": first_pass.wall_time_s,
        }]
        api_calls = first_pass.api_calls

        seed_key = canonical_vote_key(first_pass.final_text)
        votes[seed_key] += 1
        winning_texts.setdefault(seed_key, first_pass.final_text)
        winning_reasoning.setdefault(seed_key, first_pass.reasoning_content)

        for sample_index in range(samples - 1):
            result = self._run_turns(example, client, context, temperature=temperature)
            api_calls += result.api_calls
            vote_key = canonical_vote_key(result.final_text)
            votes[vote_key] += 1
            winning_texts.setdefault(vote_key, result.final_text)
            winning_reasoning.setdefault(vote_key, result.reasoning_content)
            trace.append({
                "stage": "escalated_sample",
                "sample_index": sample_index,
                "vote_key": vote_key,
                "final_text": result.final_text,
                "reasoning_content": result.reasoning_content,
                "wall_time_s": result.wall_time_s,
            })

        winner_key, _ = votes.most_common(1)[0]
        return StrategyResult(
            strategy_name=self.name,
            final_text=winning_texts[winner_key],
            api_calls=api_calls,
            wall_time_s=time.perf_counter() - started,
            reasoning_content=winning_reasoning.get(winner_key),
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


class TreeOfThoughtsStrategy(Strategy):
    name = "tree_of_thoughts"

    _EXPERT_STYLES: list[tuple[str, str]] = [
        (
            "Logic Expert",
            "Focus on strict constraints and eliminate contradictions quickly.",
        ),
        (
            "Scenario Expert",
            "Propose plausible candidate solutions and stress-test edge cases.",
        ),
        (
            "Reviewer",
            "Audit prior candidate reasoning, then recommend the most robust final answer.",
        ),
        (
            "Skeptic",
            "Search for hidden failure modes and ambiguous assumptions before finalizing.",
        ),
    ]

    def _system_instruction(self, context: StrategyRuntimeContext) -> str:
        base = (
            "You are a careful reasoning assistant. Use deliberate branching when needed, "
            "avoid fabricated claims, and keep outputs benchmark-safe."
        )
        if context.strict_benchmark_mode:
            base += " Do not use hidden benchmark examples or benchmark-specific leakage."
        return base

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        started = time.perf_counter()
        branches = max(2, int(self.params.get("branches", 3)))
        branch_temperature = float(self.params.get("temperature", max(context.default_temperature, 0.6)))
        branch_max_tokens = max(256, context.default_max_tokens // 2)

        branch_outputs: list[dict[str, Any]] = []
        api_calls = 0

        for branch_index in range(branches):
            expert_name, expert_style = self._EXPERT_STYLES[branch_index % len(self._EXPERT_STYLES)]
            branch_prompt = (
                f"Task:\n{example.prompt_text}\n\n"
                f"Role: {expert_name}\n"
                f"Style: {expert_style}\n\n"
                "Generate one candidate reasoning path and a candidate final answer. "
                "Keep it concise and avoid verbose scratchpad text."
            )
            branch_result = client.generate(
                GenerationRequest(
                    messages=[
                        ChatMessage("system", self._system_instruction(context)),
                        ChatMessage("user", branch_prompt),
                    ],
                    temperature=branch_temperature,
                    max_tokens=branch_max_tokens,
                )
            )
            api_calls += 1
            branch_outputs.append(
                {
                    "branch_index": branch_index,
                    "expert": expert_name,
                    "assistant": branch_result.text,
                    "reasoning_content": branch_result.reasoning_content,
                    "latency_s": branch_result.latency_s,
                    "from_cache": branch_result.from_cache,
                }
            )

        branch_digest = "\n\n".join(
            f"[{item['expert']} / branch {item['branch_index']}]\n{item['assistant']}" for item in branch_outputs
        )
        synthesis_prompt = (
            f"Task:\n{example.prompt_text}\n\n"
            f"Candidate branches:\n{branch_digest}\n\n"
            "Select the best reasoning path, fix contradictions, and return ONLY the final answer in the required format."
        )
        format_hint = str(example.metadata.get("format_hint") or "").strip()
        if format_hint:
            synthesis_prompt += f"\n\nFormatting requirement:\n{format_hint}"

        final_result = client.generate(
            GenerationRequest(
                messages=[
                    ChatMessage("system", self._system_instruction(context)),
                    ChatMessage("user", synthesis_prompt),
                ],
                temperature=context.default_temperature,
                max_tokens=context.default_max_tokens,
            )
        )
        api_calls += 1

        trace = list(branch_outputs)
        trace.append(
            {
                "stage": "synthesis",
                "assistant": final_result.text,
                "reasoning_content": final_result.reasoning_content,
                "latency_s": final_result.latency_s,
                "from_cache": final_result.from_cache,
            }
        )

        all_cached = all(item.get("from_cache", False) for item in trace) if trace else False
        return StrategyResult(
            strategy_name=self.name,
            final_text=final_result.text,
            api_calls=api_calls,
            wall_time_s=time.perf_counter() - started,
            reasoning_content=final_result.reasoning_content,
            trace=trace,
            metadata={"from_cache": all_cached},
        )


class CoconutStrategy(Strategy):
    name = "coconut"

    _STEP_RE = re.compile(r"^\s*(?:\d+[\).:-]|[-*])\s*(.+?)\s*$")

    def _system_instruction(self, context: StrategyRuntimeContext) -> str:
        base = (
            "You are a consistent multi-turn reasoning assistant. Build on prior facts, "
            "track constraints, and avoid speculative leaps."
        )
        if context.strict_benchmark_mode:
            base += " Do not use hidden benchmark examples or benchmark-specific leakage."
        return base

    def _parse_steps(self, planner_text: str, max_steps: int) -> list[str]:
        steps: list[str] = []
        for line in planner_text.splitlines():
            match = self._STEP_RE.match(line)
            candidate = match.group(1).strip() if match else line.strip()
            if candidate:
                steps.append(candidate)
            if len(steps) >= max_steps:
                break
        if steps:
            return steps
        return [
            "Extract key constraints or facts.",
            "Resolve the core reasoning bottleneck.",
            "Produce a final answer that satisfies all constraints.",
        ][:max_steps]

    def run(self, example: Example, client: BaseLLMClient, context: StrategyRuntimeContext) -> StrategyResult:
        started = time.perf_counter()
        memory_turns = max(2, int(self.params.get("memory_turns", 3)))
        thinking_temperature = float(self.params.get("temperature", max(context.default_temperature, 0.4)))

        planner_prompt = (
            f"Task:\n{example.prompt_text}\n\n"
            f"Create a compact sequence of {memory_turns} reasoning steps for continuous multi-turn solving. "
            "Return one step per line."
        )
        planner = client.generate(
            GenerationRequest(
                messages=[
                    ChatMessage("system", self._system_instruction(context)),
                    ChatMessage("user", planner_prompt),
                ],
                temperature=thinking_temperature,
                max_tokens=max(192, context.default_max_tokens // 3),
            )
        )

        steps = self._parse_steps(planner.text, memory_turns)
        memory_state = ""
        trace: list[dict[str, Any]] = [
            {
                "stage": "plan",
                "assistant": planner.text,
                "reasoning_content": planner.reasoning_content,
                "latency_s": planner.latency_s,
                "from_cache": planner.from_cache,
                "steps": steps,
            }
        ]
        api_calls = 1

        for step_index, step_text in enumerate(steps, start=1):
            memory_prompt = (
                f"Task:\n{example.prompt_text}\n\n"
                f"Current working memory:\n{memory_state or '(empty)'}\n\n"
                f"Step {step_index}/{len(steps)}: {step_text}\n\n"
                "Update the working memory with only essential facts and partial conclusions. "
                "Do not give the final answer yet."
            )
            memory_result = client.generate(
                GenerationRequest(
                    messages=[
                        ChatMessage("system", self._system_instruction(context)),
                        ChatMessage("user", memory_prompt),
                    ],
                    temperature=thinking_temperature,
                    max_tokens=max(256, context.default_max_tokens // 2),
                )
            )
            api_calls += 1
            memory_state = memory_result.text.strip() or memory_state
            trace.append(
                {
                    "stage": "memory_turn",
                    "step_index": step_index,
                    "step": step_text,
                    "assistant": memory_result.text,
                    "reasoning_content": memory_result.reasoning_content,
                    "latency_s": memory_result.latency_s,
                    "from_cache": memory_result.from_cache,
                }
            )

        final_prompt = (
            f"Task:\n{example.prompt_text}\n\n"
            f"Final working memory:\n{memory_state}\n\n"
            "Produce the final answer only."
        )
        format_hint = str(example.metadata.get("format_hint") or "").strip()
        if format_hint:
            final_prompt += f"\n\nFormatting requirement:\n{format_hint}"

        final = client.generate(
            GenerationRequest(
                messages=[
                    ChatMessage("system", self._system_instruction(context)),
                    ChatMessage("user", final_prompt),
                ],
                temperature=context.default_temperature,
                max_tokens=context.default_max_tokens,
            )
        )
        api_calls += 1

        trace.append(
            {
                "stage": "final",
                "assistant": final.text,
                "reasoning_content": final.reasoning_content,
                "latency_s": final.latency_s,
                "from_cache": final.from_cache,
            }
        )
        all_cached = all(item.get("from_cache", False) for item in trace) if trace else False
        return StrategyResult(
            strategy_name=self.name,
            final_text=final.text,
            api_calls=api_calls,
            wall_time_s=time.perf_counter() - started,
            reasoning_content=final.reasoning_content,
            trace=trace,
            metadata={"from_cache": all_cached},
        )
