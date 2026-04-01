from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopDetectionConfig:
    length_ceiling: int = 50000
    repetition_window: int = 80
    repetition_threshold: float = 0.5
    stalled_phrase_len: int = 30
    stalled_phrase_count: int = 8


def detect_loop(text: str, config: LoopDetectionConfig) -> tuple[bool, str]:
    if not text.strip():
        return False, ""
    if len(text) > config.length_ceiling:
        return True, f"length>{config.length_ceiling}"

    window = config.repetition_window
    if len(text) >= window * 2:
        chunks = [text[i:i + window] for i in range(0, len(text) - window + 1, window)]
        seen: set[str] = set()
        duplicates = 0
        for chunk in chunks:
            if chunk in seen:
                duplicates += 1
            seen.add(chunk)
        if chunks and duplicates / len(chunks) >= config.repetition_threshold:
            return True, "duplicate_windows"

    phrase_len = config.stalled_phrase_len
    if len(text) >= phrase_len * 4:
        phrase = text[len(text) // 2: len(text) // 2 + phrase_len]
        if phrase and text.count(phrase) >= config.stalled_phrase_count:
            return True, "stalled_phrase"

    return False, ""
