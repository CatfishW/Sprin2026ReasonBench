from __future__ import annotations

from collections import defaultdict

from reasonbench.evaluators.base import Evaluator
from reasonbench.types import EvaluationResult, Example
from reasonbench.utils.text import normalize_text, parse_room_lines


class RoomAssignmentEvaluator(Evaluator):
    def _entity_room_map(self, assignments: dict[str, list[str]]) -> dict[str, str]:
        entity_room: dict[str, str] = {}
        for room_id, occupants in assignments.items():
            for occupant in occupants:
                entity_room[occupant] = room_id
        return entity_room

    def evaluate(self, example: Example, prediction: str) -> EvaluationResult:
        ground_truth = str(example.reference.get('completion') or '')
        gt_rooms = parse_room_lines(ground_truth)
        pred_rooms = parse_room_lines(prediction)

        gt_entities = self._entity_room_map(gt_rooms)
        pred_entities = self._entity_room_map(pred_rooms)

        room_ids = sorted(gt_rooms)
        exact_matches = 0
        for room_id in room_ids:
            if sorted(gt_rooms.get(room_id, [])) == sorted(pred_rooms.get(room_id, [])):
                exact_matches += 1
        room_exact_accuracy = exact_matches / max(len(room_ids), 1)

        entity_correct = 0
        for entity, room_id in gt_entities.items():
            if pred_entities.get(entity) == room_id:
                entity_correct += 1
        entity_room_accuracy = entity_correct / max(len(gt_entities), 1)

        primary = 0.5 * room_exact_accuracy + 0.5 * entity_room_accuracy
        return EvaluationResult(
            primary_score=primary,
            metrics={
                'room_exact_accuracy': room_exact_accuracy,
                'entity_room_accuracy': entity_room_accuracy,
                'all_rooms_exact': room_exact_accuracy == 1.0,
                'format_valid': bool(pred_rooms),
                'predicted_room_count': len(pred_rooms),
            },
        )
