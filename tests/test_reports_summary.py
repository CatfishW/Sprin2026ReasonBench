from reasonbench.runtime.reports import build_summary
from reasonbench.types import ExperimentRecord


def test_summary_excludes_unscorable_from_primary_mean() -> None:
    records = [
        ExperimentRecord(
            example_id="ex-1",
            dataset_name="livebench",
            strategy_name="direct",
            final_text="ok",
            primary_score=1.0,
            metrics={"is_unscorable": False, "format_valid": True},
            api_calls=1,
            wall_time_s=1.0,
        ),
        ExperimentRecord(
            example_id="ex-2",
            dataset_name="livebench",
            strategy_name="direct",
            final_text="n/a",
            primary_score=0.0,
            metrics={"is_unscorable": True, "format_valid": True},
            api_calls=1,
            wall_time_s=1.0,
        ),
    ]

    rows = build_summary(records)
    assert len(rows) == 1
    assert rows[0]["examples"] == 2
    assert rows[0]["scorable_examples"] == 1
    assert rows[0]["mean_primary_score"] == 1.0
    assert rows[0]["unscorable_rate"] == 0.5
