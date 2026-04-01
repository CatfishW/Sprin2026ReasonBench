from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence


def run_official_truthfulqa(repo_path: str, input_path: str, models: Sequence[str], metrics: Sequence[str]) -> subprocess.CompletedProcess:
    repo = Path(repo_path)
    cmd = [
        'python', 'truthfulqa/evaluate.py',
        '--input_path', input_path,
        '--models', *models,
        '--metrics', *metrics,
    ]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True)
