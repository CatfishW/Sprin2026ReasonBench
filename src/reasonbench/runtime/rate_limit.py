from __future__ import annotations

import threading
import time


class MinIntervalRateLimiter:
    def __init__(self, min_interval_s: float = 0.0):
        self.min_interval_s = max(min_interval_s, 0.0)
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.min_interval_s:
                time.sleep(self.min_interval_s - elapsed)
            self._last_request = time.monotonic()
