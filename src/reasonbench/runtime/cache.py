from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class SQLiteCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    response_text TEXT NOT NULL,
                    response_json TEXT,
                    created_at REAL DEFAULT (strftime('%s','now'))
                )
                """
            )

    def get(self, cache_key: str) -> tuple[str, dict | None] | None:
        with self._lock, sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT response_text, response_json FROM cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        text, raw_json = row
        parsed = json.loads(raw_json) if raw_json else None
        return text, parsed

    def set(self, cache_key: str, response_text: str, response_json: dict | None) -> None:
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache(cache_key, response_text, response_json) VALUES (?, ?, ?)",
                (cache_key, response_text, json.dumps(response_json) if response_json is not None else None),
            )
            conn.commit()
