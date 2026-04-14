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


def test_room_assignment_strips_reasoning_prefix_when_provided():
    example = Example(
        example_id="2",
        dataset_name="room_assignment",
        split="train",
        turns=["dummy"],
        reference={"completion": "room 1: Alice\nroom 2: Bob"},
    )
    reasoning = "Drafting possibilities: room 1 maybe Bob, room 2 maybe Alice."
    prediction = reasoning + "\n\nroom 1: Alice\nroom 2: Bob"
    result = RoomAssignmentEvaluator().evaluate(example, prediction, reasoning_content=reasoning)
    assert result.primary_score == 1.0


def test_room_assignment_penalizes_extra_room_and_entity():
    example = Example(
        example_id="3",
        dataset_name="room_assignment",
        split="train",
        turns=["dummy"],
        reference={"completion": "room 1: Alice\nroom 2: Bob"},
    )
    result = RoomAssignmentEvaluator().evaluate(example, "room 1: Alice\nroom 2: Bob\nroom 3: Charlie")
    assert result.metrics["room_exact_accuracy"] < 1.0
    assert result.metrics["entity_room_precision"] < 1.0
    assert result.metrics["hallucinated_entity_count"] == 1
    assert result.primary_score < 1.0


def test_room_assignment_marks_duplicate_entity_as_invalid_format():
    example = Example(
        example_id="4",
        dataset_name="room_assignment",
        split="train",
        turns=["dummy"],
        reference={"completion": "room 1: Alice\nroom 2: Bob"},
    )
    result = RoomAssignmentEvaluator().evaluate(example, "room 1: Alice\nroom 2: Bob, Alice")
    assert result.metrics["duplicate_entity_count"] == 1
    assert result.metrics["format_valid"] is False
