from __future__ import annotations

import argparse
import copy
from pathlib import Path

from reasonbench.config import load_experiment_config
from reasonbench.runtime.runner import ExperimentRunner
from reasonbench.strategies.registry import available_strategies


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a sweep with strategy overrides.')
    parser.add_argument('--config', required=True, help='Base experiment TOML file.')
    parser.add_argument('--strategies', nargs='+', default=None, help='Optional strategy override list.')
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.strategies:
        config.strategies = [type(config.strategies[0])(name=name, params={}) for name in args.strategies]
    runner = ExperimentRunner(config)
    runner.run()
    print(f'Sweep complete. Available strategies: {", ".join(available_strategies())}')
