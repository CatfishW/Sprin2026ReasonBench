from __future__ import annotations

from pathlib import Path
from random import Random

from reasonbench.datasets.base import DatasetAdapter, _maybe_limit, read_jsonl_records
from reasonbench.types import Example


class LiveBenchAdapter(DatasetAdapter):
    def _discover_files(self) -> list[Path]:
        if not self.config.local_path:
            raise ValueError('LiveBench currently requires local_path pointing to question.jsonl or a directory containing question.jsonl files.')
        root = Path(self.config.local_path)
        if root.is_file():
            return [root]
        files = sorted(root.rglob('question.jsonl'))
        if not files:
            raise FileNotFoundError(f'No question.jsonl files found under {root}')
        return files

    def load(self) -> list[Example]:
        examples: list[Example] = []
        task_filters = set(self.config.task_filters)
        missing_ground_truth_by_task: dict[str, int] = {}
        for file_path in self._discover_files():
            for row in read_jsonl_records(file_path):
                task = str(row.get('task') or '')
                if task_filters and task not in task_filters:
                    continue
                ground_truth = row.get('ground_truth')
                if ground_truth is None or not str(ground_truth).strip():
                    missing_ground_truth_by_task[task] = missing_ground_truth_by_task.get(task, 0) + 1
                turns = row.get('turns') or []
                if isinstance(turns, str):
                    turns = [turns]
                examples.append(
                    Example(
                        example_id=str(row.get('question_id') or row.get('id') or len(examples)),
                        dataset_name='livebench',
                        split=self.config.split,
                        turns=[str(turn).strip() for turn in turns if str(turn).strip()],
                        reference={
                            'ground_truth': ground_truth,
                            'task': task,
                            'category': row.get('category') or '',
                        },
                        metadata={
                            'task': task,
                            'category': row.get('category') or '',
                            'source_file': str(file_path),
                            'format_hint': 'Preserve the benchmark-required answer format exactly. Add no extra preamble unless requested by the task.',
                        },
                    )
                )
        if missing_ground_truth_by_task:
            missing_total = sum(missing_ground_truth_by_task.values())
            print(
                f"[ReasonBench] livebench_warning missing_ground_truth={missing_total}/{len(examples)} "
                "proxy_scoring_may_be_invalid_for_these_tasks"
            )
            for task, count in sorted(missing_ground_truth_by_task.items()):
                print(f"[ReasonBench] livebench_warning task={task or 'unknown'} missing_ground_truth={count}")
        if self.config.shuffle:
            rnd = Random(self.config.seed)
            rnd.shuffle(examples)
        return _maybe_limit(examples, self.config.limit)
