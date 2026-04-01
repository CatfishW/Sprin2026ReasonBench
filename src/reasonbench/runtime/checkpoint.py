from __future__ import annotations

import json
import threading
from pathlib import Path

from reasonbench.types import ExperimentRecord


class JSONLCheckpointStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._completed: set[tuple[str, str]] = set()
        self._records: list[ExperimentRecord] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    record = ExperimentRecord(**row)
                    self._records.append(record)
                    self._completed.add((record.example_id, record.strategy_name))

    def has(self, example_id: str, strategy_name: str) -> bool:
        return (example_id, strategy_name) in self._completed

    def append(self, record: ExperimentRecord) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
            self._records.append(record)
            self._completed.add((record.example_id, record.strategy_name))

    def load_all(self) -> list[ExperimentRecord]:
        return list(self._records)
