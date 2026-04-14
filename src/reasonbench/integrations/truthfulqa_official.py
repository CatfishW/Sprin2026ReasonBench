from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence


def run_official_truthfulqa(
    repo_path: str,
    input_path: str,
    models: Sequence[str],
    metrics: Sequence[str],
    output_path: str | None = None,
) -> subprocess.CompletedProcess:
    repo = Path(repo_path)
    cmd = [
        sys.executable, '-m', 'truthfulqa.evaluate',
        '--input_path', input_path,
        '--models', *models,
        '--metrics', *metrics,
    ]
    if output_path:
        cmd.extend(['--output_path', output_path])
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True)
