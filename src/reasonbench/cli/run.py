from __future__ import annotations

import argparse
from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.runtime.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a ReasonBench experiment from a TOML config.')
    parser.add_argument('--config', required=True, help='Path to experiment TOML file.')
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    runner = ExperimentRunner(config)
    result = runner.run()

    print(f"Completed {result['manifest']['completed_records']} records.")
    print(f"Summary CSV: {config.output.summary_path}")
    print(f"Manifest:    {config.output.manifest_path}")
