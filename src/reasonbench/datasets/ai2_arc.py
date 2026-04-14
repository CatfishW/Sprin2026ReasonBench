from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

from reasonbench.datasets.base import DatasetAdapter, _maybe_limit, load_hf_dataset, read_jsonl_records
from reasonbench.types import Example


class AI2ARCAdapter(DatasetAdapter):
    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.config.local_path:
            return list(load_hf_dataset(self.config))

        path = Path(self.config.local_path)
        if path.is_file():
            return read_jsonl_records(path)

        rows: list[dict[str, Any]] = []
        for file_path in sorted(path.rglob("*.jsonl")):
            rows.extend(read_jsonl_records(file_path))
        return rows

    def _extract_choices(self, row: dict[str, Any]) -> tuple[list[str], list[str]]:
        raw = row.get("choices")
        if isinstance(raw, dict):
            labels_raw = raw.get("label") or []
            texts_raw = raw.get("text") or []
            labels = [str(item).strip().upper() for item in labels_raw]
            texts = [str(item).strip() for item in texts_raw]
            return labels, texts

        if isinstance(raw, list):
            labels: list[str] = []
            texts: list[str] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                labels.append(str(item.get("label") or "").strip().upper())
                texts.append(str(item.get("text") or "").strip())
            return labels, texts

        return [], []

    def _render_prompt(self, question: str, labels: list[str], texts: list[str]) -> str:
        lines = [
            "Answer the multiple-choice science question.",
            "Return only the choice label (for example: A).",
            "",
            f"Question: {question}",
            "",
            "Choices:",
        ]
        for label, text in zip(labels, texts):
            lines.append(f"{label}. {text}")
        return "\n".join(lines).strip()

    def load(self) -> list[Example]:
        rows = self._load_rows()
        examples: list[Example] = []

        for index, row in enumerate(rows):
            question = str(row.get("question") or "").strip()
            answer_key = str(row.get("answerKey") or row.get("answer_key") or "").strip().upper()
            labels, texts = self._extract_choices(row)

            if not question or not labels or not texts or not answer_key:
                continue

            if len(labels) != len(texts):
                continue

            choice_map = {label: text for label, text in zip(labels, texts) if label and text}
            if answer_key not in choice_map:
                continue

            subset = str(row.get("arc_subset") or row.get("subset") or "unknown")
            row_split = str(row.get("arc_split") or row.get("split") or self.config.split)
            raw_id = str(row.get("id") or index)
            example_id = f"{subset}:{row_split}:{raw_id}"

            examples.append(
                Example(
                    example_id=example_id,
                    dataset_name="ai2_arc",
                    split=row_split,
                    turns=[self._render_prompt(question, labels, texts)],
                    reference={
                        "answer_key": answer_key,
                        "choices": choice_map,
                        "subset": subset,
                    },
                    metadata={
                        "subset": subset,
                        "split": row_split,
                        "choice_labels": labels,
                        "format_hint": "Output exactly one choice label, such as A, B, C, or D.",
                    },
                )
            )

        if self.config.shuffle:
            rnd = Random(self.config.seed)
            rnd.shuffle(examples)
        return _maybe_limit(examples, self.config.limit)
