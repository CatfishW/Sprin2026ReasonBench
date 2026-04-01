from __future__ import annotations

import json
import re
import string
from difflib import SequenceMatcher

_ARTICLES = {"a", "an", "the"}
_NOISE_RE = re.compile(r"\s+")
_ROOM_LINE_RE = re.compile(r"^\s*(?:room\s*)?(\d+)\s*[:.-]\s*(.+?)\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = strip_code_fences(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("#", "")))
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return _NOISE_RE.sub(" ", " ".join(tokens)).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [tok for tok in normalized.split() if tok]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(tok, 0), ref_counts.get(tok, 0)) for tok in set(pred_counts) | set(ref_counts))
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_similarity(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, normalize_text(prediction), normalize_text(reference)).ratio()


def soft_similarity(prediction: str, reference: str) -> float:
    return max(token_f1(prediction, reference), sequence_similarity(prediction, reference))


def split_semicolon_answers(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def try_parse_json(text: str) -> dict | list | None:
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_room_lines(text: str) -> dict[str, list[str]]:
    data = try_parse_json(text)
    if isinstance(data, dict) and "rooms" in data and isinstance(data["rooms"], list):
        parsed: dict[str, list[str]] = {}
        for room in data["rooms"]:
            if not isinstance(room, dict):
                continue
            room_id = str(room.get("room") or room.get("id") or room.get("number") or "")
            occupants = room.get("occupants") or room.get("tenants") or []
            if room_id and isinstance(occupants, list):
                parsed[room_id] = [normalize_text(str(item)) for item in occupants if str(item).strip()]
        if parsed:
            return parsed

    parsed: dict[str, list[str]] = {}
    for line in strip_code_fences(text).splitlines():
        match = _ROOM_LINE_RE.match(line)
        if not match:
            continue
        room_id = match.group(1)
        occupants_raw = match.group(2)
        occupants = [normalize_text(x) for x in re.split(r",|\band\b", occupants_raw) if normalize_text(x)]
        if occupants:
            parsed[room_id] = occupants
    return parsed


def canonical_vote_key(text: str) -> str:
    rooms = parse_room_lines(text)
    if rooms:
        parts = []
        for room_id in sorted(rooms, key=lambda x: int(x)):
            occupants = ",".join(sorted(rooms[room_id]))
            parts.append(f"room{room_id}:{occupants}")
        return "|".join(parts)
    normalized = normalize_text(text)
    final_only = re.findall(r"final answer\s*[:.-]?\s*(.+)$", normalized, flags=re.IGNORECASE | re.MULTILINE)
    if final_only:
        return final_only[-1].strip()
    return normalized
