from reasonbench.evaluators.room_assignment import RoomAssignmentEvaluator
from reasonbench.types import Example


def test_room_assignment_perfect_score():
    example = Example(
        example_id="1",
        dataset_name="room_assignment",
        split="train",
        turns=["dummy"],
        reference={"completion": "room 1: Alice\nroom 2: Bob"},
    )
    result = RoomAssignmentEvaluator().evaluate(example, "room 1: Alice\nroom 2: Bob")
    assert result.primary_score == 1.0
    assert result.metrics["all_rooms_exact"] is True
