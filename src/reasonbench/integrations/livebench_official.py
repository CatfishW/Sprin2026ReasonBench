from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence


def run_official_livebench(
    repo_path: str,
    model_name: str,
    bench_name: str,
    release_option: str = '2024-11-25',
    parallel_requests: int = 1,
    api_base: str | None = None,
    api_key_env: str = 'OPENAI_API_KEY',
) -> subprocess.CompletedProcess:
    repo = Path(repo_path)
    cmd = [
        'python', 'run_livebench.py',
        '--model', model_name,
        '--bench-name', bench_name,
        '--livebench-release-option', release_option,
        '--parallel-requests', str(parallel_requests),
    ]
    if api_base:
        cmd.extend(['--api-base', api_base, '--api-key-name', api_key_env])
    return subprocess.run(cmd, cwd=repo / 'livebench', check=True, capture_output=True, text=True)
