from __future__ import annotations

import re

from reasonbench.evaluators.base import Evaluator
from reasonbench.types import EvaluationResult, Example
from reasonbench.utils.text import normalize_text, soft_similarity, strip_code_fences, strip_reasoning_prefix


class AI2ARCEvaluator(Evaluator):
    _EXPLICIT_LABEL_PATTERNS = [
        re.compile(r"(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\s*[:=\-]?\s*([A-Z0-9]+)", re.IGNORECASE),
        re.compile(r"(?:SELECTED|CORRECT)\s+(?:ANSWER|CHOICE)\s*[:=\-]?\s*([A-Z0-9]+)", re.IGNORECASE),
        re.compile(r"\(([A-Z0-9]+)\)\s*$", re.IGNORECASE),
    ]

    def _tail_segments(self, text: str) -> list[str]:
        segments: list[str] = []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            segments.append(lines[-1])

        sentences = [chunk.strip() for chunk in re.split(r"[.!?]", text) if chunk.strip()]
        if sentences:
            segments.append(sentences[-1])

        unique_segments: list[str] = []
        for segment in segments:
            if segment not in unique_segments:
                unique_segments.append(segment)
        return unique_segments

    def _single_unambiguous_label(self, text: str, valid_labels: set[str]) -> str | None:
        tokens = re.findall(r"[A-Z0-9]+", text.upper())
        labels_in_order: list[str] = []
        for token in tokens:
            if token in valid_labels and token not in labels_in_order:
                labels_in_order.append(token)

        if len(labels_in_order) == 1:
            return labels_in_order[0]
        return None

    def _extract_choice_label(self, prediction: str, valid_labels: set[str]) -> str | None:
        text = strip_code_fences(prediction).strip()
        if not text:
            return None

        upper = text.upper()
        if upper in valid_labels:
            return upper

        for pattern in self._EXPLICIT_LABEL_PATTERNS:
            matches = list(pattern.finditer(upper))
            for match in reversed(matches):
                label = match.group(1)
                if label in valid_labels:
                    return label

        for segment in self._tail_segments(upper):
            label = self._single_unambiguous_label(segment, valid_labels)
            if label is not None:
                return label

        return None

    def _text_match_scores(self, prediction: str, choices: dict[str, str]) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for label, choice_text in choices.items():
            score = soft_similarity(prediction, choice_text)
            scored.append((label, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def evaluate(self, example: Example, prediction: str, reasoning_content: str | None = None) -> EvaluationResult:
        cleaned_prediction = strip_reasoning_prefix(prediction, reasoning_content)
        answer_key = str(example.reference.get("answer_key") or "").upper()
        choices = {
            str(label).upper(): str(text)
            for label, text in (example.reference.get("choices") or {}).items()
            if str(label).strip()
        }
        valid_labels = set(choices)

        selected_label = self._extract_choice_label(cleaned_prediction, valid_labels)
        text_match_label = None
        text_match_score = 0.0

        if selected_label is None and choices:
            scored = self._text_match_scores(cleaned_prediction, choices)
            if scored:
                text_match_label, text_match_score = scored[0]
                second_score = scored[1][1] if len(scored) > 1 else 0.0
                if text_match_score >= 0.9 and (text_match_score - second_score) >= 0.1:
                    selected_label = text_match_label

        correct = bool(selected_label and selected_label == answer_key)

        return EvaluationResult(
            primary_score=1.0 if correct else 0.0,
            metrics={
                "accuracy": 1.0 if correct else 0.0,
                "selected_label": selected_label or "",
                "answer_key": answer_key,
                "format_valid": selected_label is not None,
                "choice_count": len(choices),
                "pred_norm": normalize_text(cleaned_prediction),
                "text_match_label": text_match_label or "",
                "text_match_score": round(text_match_score, 4),
                "subset": example.reference.get("subset", ""),
            },
        )
