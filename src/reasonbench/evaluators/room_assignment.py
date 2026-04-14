from __future__ import annotations

from collections import defaultdict

from reasonbench.evaluators.base import Evaluator
from reasonbench.types import EvaluationResult, Example
from reasonbench.utils.text import normalize_text, parse_room_lines, strip_reasoning_prefix


class RoomAssignmentEvaluator(Evaluator):
    @staticmethod
    def _room_sort_key(room_id: str) -> tuple[int, int | str]:
        if room_id.isdigit():
            return (0, int(room_id))
        return (1, room_id)

    def _entity_room_map(self, assignments: dict[str, list[str]]) -> dict[str, str]:
        entity_room: dict[str, str] = {}
        for room_id, occupants in assignments.items():
            for occupant in occupants:
                entity_room[occupant] = room_id
        return entity_room

    def _entity_occurrences(self, assignments: dict[str, list[str]]) -> dict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for occupants in assignments.values():
            for occupant in occupants:
                counts[occupant] += 1
        return dict(counts)

    def _entity_room_counts(self, assignments: dict[str, list[str]]) -> dict[tuple[str, str], int]:
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for room_id, occupants in assignments.items():
            for occupant in occupants:
                counts[(occupant, room_id)] += 1
        return dict(counts)

    def evaluate(self, example: Example, prediction: str, reasoning_content: str | None = None) -> EvaluationResult:
        cleaned_prediction = strip_reasoning_prefix(prediction, reasoning_content)
        ground_truth = str(example.reference.get('completion') or '')
        gt_rooms = parse_room_lines(ground_truth)
        pred_rooms = parse_room_lines(cleaned_prediction)

        gt_entities = self._entity_room_map(gt_rooms)
        pred_entities = self._entity_room_map(pred_rooms)
        gt_entity_set = set(gt_entities)
        pred_entity_set = set(pred_entities)

        pred_entity_occurrences = self._entity_occurrences(pred_rooms)
        duplicate_entities = sorted(entity for entity, count in pred_entity_occurrences.items() if count > 1)

        gt_pair_counts = self._entity_room_counts(gt_rooms)
        pred_pair_counts = self._entity_room_counts(pred_rooms)
        all_pair_keys = set(gt_pair_counts) | set(pred_pair_counts)
        correct_pair_assignments = sum(min(gt_pair_counts.get(key, 0), pred_pair_counts.get(key, 0)) for key in all_pair_keys)
        gt_total_assignments = sum(gt_pair_counts.values())
        pred_total_assignments = sum(pred_pair_counts.values())

        if gt_total_assignments == 0 and pred_total_assignments == 0:
            entity_room_precision = 1.0
            entity_room_accuracy = 1.0
        else:
            entity_room_precision = correct_pair_assignments / max(pred_total_assignments, 1)
            entity_room_accuracy = correct_pair_assignments / max(gt_total_assignments, 1)

        if entity_room_precision + entity_room_accuracy > 0:
            entity_room_f1 = 2 * entity_room_precision * entity_room_accuracy / (entity_room_precision + entity_room_accuracy)
        else:
            entity_room_f1 = 0.0

        room_ids = sorted(set(gt_rooms) | set(pred_rooms), key=self._room_sort_key)
        exact_matches = 0
        for room_id in room_ids:
            if sorted(gt_rooms.get(room_id, [])) == sorted(pred_rooms.get(room_id, [])):
                exact_matches += 1
        room_exact_accuracy = exact_matches / max(len(room_ids), 1)

        format_valid = bool(pred_rooms) and not duplicate_entities
        primary = 0.5 * room_exact_accuracy + 0.5 * entity_room_f1
        return EvaluationResult(
            primary_score=primary,
            metrics={
                'room_exact_accuracy': room_exact_accuracy,
                'entity_room_accuracy': entity_room_accuracy,
                'entity_room_precision': entity_room_precision,
                'entity_room_f1': entity_room_f1,
                'all_rooms_exact': room_exact_accuracy == 1.0,
                'format_valid': format_valid,
                'predicted_room_count': len(pred_rooms),
                'predicted_entity_count': len(pred_entity_set),
                'ground_truth_entity_count': len(gt_entity_set),
                'hallucinated_entity_count': len(pred_entity_set - gt_entity_set),
                'missing_entity_count': len(gt_entity_set - pred_entity_set),
                'duplicate_entity_count': len(duplicate_entities),
                'duplicate_entities': duplicate_entities,
            },
        )
