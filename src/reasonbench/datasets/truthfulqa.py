from __future__ import annotations

from pathlib import Path
from random import Random

from reasonbench.datasets.base import DatasetAdapter, _maybe_limit, load_hf_dataset, read_csv_records
from reasonbench.types import Example
from reasonbench.utils.text import split_semicolon_answers


class TruthfulQAAdapter(DatasetAdapter):
    def load(self) -> list[Example]:
        rows: list[dict]
        if self.config.local_path:
            rows = read_csv_records(self.config.local_path)
        else:
            rows = list(load_hf_dataset(self.config))

        examples: list[Example] = []
        for index, row in enumerate(rows):
            question = str(row.get('Question') or row.get('question') or '').strip()
            best_answer = str(row.get('Best Answer') or row.get('best_answer') or '').strip()
            correct_answers = str(row.get('Correct Answers') or row.get('correct_answers') or '').strip()
            incorrect_answers = str(row.get('Incorrect Answers') or row.get('incorrect_answers') or '').strip()
            examples.append(
                Example(
                    example_id=str(index),
                    dataset_name='truthfulqa',
                    split=self.config.split,
                    turns=[question],
                    reference={
                        'best_answer': best_answer,
                        'correct_answers': split_semicolon_answers(correct_answers) or ([best_answer] if best_answer else []),
                        'incorrect_answers': split_semicolon_answers(incorrect_answers),
                        'source': row.get('Source') or row.get('source') or '',
                    },
                    metadata={
                        'category': row.get('Category') or row.get('category') or '',
                        'type': row.get('Type') or row.get('type') or '',
                        'format_hint': 'Answer in 1-2 sentences. Prefer truthfulness over speculation. If uncertain, say you do not know.',
                    },
                )
            )
        if self.config.shuffle:
            rnd = Random(self.config.seed)
            rnd.shuffle(examples)
        return _maybe_limit(examples, self.config.limit)
