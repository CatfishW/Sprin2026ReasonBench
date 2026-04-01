from __future__ import annotations

from pathlib import Path
from random import Random

from reasonbench.datasets.base import DatasetAdapter, _maybe_limit, load_hf_dataset, read_json_per_line_or_text, read_jsonl_records
from reasonbench.types import Example


class RoomAssignmentAdapter(DatasetAdapter):
    def load(self) -> list[Example]:
        rows: list[dict]
        if self.config.local_path:
            path = Path(self.config.local_path)
            suffix = path.suffix.lower()
            if suffix == ".jsonl":
                rows = read_jsonl_records(path)
            else:
                rows = read_json_per_line_or_text(path)
        else:
            rows = list(load_hf_dataset(self.config))

        examples: list[Example] = []
        for index, row in enumerate(rows):
            prompt = str(row.get("prompt") or "").strip()
            question = str(row.get("question") or "").strip()
            turns: list[str] = []
            if prompt:
                turns.append(prompt)
            if question and question not in prompt:
                if turns:
                    turns[-1] = f"{turns[-1]}\n\nQuestion:\n{question}"
                else:
                    turns.append(question)
            example = Example(
                example_id=str(row.get("id") or row.get("question_id") or index),
                dataset_name="room_assignment",
                split=self.config.split,
                turns=turns or [question],
                reference={
                    "completion": str(row.get("completion") or ""),
                    "label": row.get("label"),
                },
                metadata={
                    "format_hint": "Return the final arrangement as `room N: occupant1, occupant2` lines or equivalent JSON.",
                    "source_row": index,
                },
            )
            examples.append(example)

        if self.config.shuffle:
            rnd = Random(self.config.seed)
            rnd.shuffle(examples)
        return _maybe_limit(examples, self.config.limit)
