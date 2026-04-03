from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class BaseExperiment:
    display_name: str
    base_session_name: str
    config_path: str


BASE_EXPERIMENTS = [
    BaseExperiment(
        display_name="Room Assignment (4B)",
        base_session_name="rb_room_assignment",
        config_path="configs/experiments/full_room_assignment_all_strategies.toml",
    ),
    BaseExperiment(
        display_name="TruthfulQA (4B)",
        base_session_name="rb_truthfulqa",
        config_path="configs/experiments/full_truthfulqa_all_strategies.toml",
    ),
    BaseExperiment(
        display_name="LiveBench (27B)",
        base_session_name="rb_livebench",
        config_path="configs/experiments/full_livebench_all_strategies.toml",
    ),
    BaseExperiment(
        display_name="AI2-ARC (4B)",
        base_session_name="rb_ai2_arc",
        config_path="configs/experiments/full_ai2_arc_all_strategies.toml",
    ),
    BaseExperiment(
        display_name="Room Assignment (27B)",
        base_session_name="rb_room_assignment_27b",
        config_path="configs/experiments/full_room_assignment_all_strategies_27b.toml",
    ),
    BaseExperiment(
        display_name="TruthfulQA (27B)",
        base_session_name="rb_truthfulqa_27b",
        config_path="configs/experiments/full_truthfulqa_all_strategies_27b.toml",
    ),
    BaseExperiment(
        display_name="AI2-ARC (27B)",
        base_session_name="rb_ai2_arc_27b",
        config_path="configs/experiments/full_ai2_arc_all_strategies_27b.toml",
    ),
]


def _default_run_tag() -> str:
    return f"spring26_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"


def _short_suffix(run_tag: str) -> str:
    digits = re.sub(r"[^0-9]", "", run_tag)
    if len(digits) >= 12:
        tail = digits[-12:]
        return f"{tail[:6]}_{tail[6:]}"
    return datetime.now().strftime("%y%m%d_%H%M%S")


def _replace_toml_setting(text: str, key: str, value: str) -> str:
    pattern = rf'^{re.escape(key)}\s*=\s*"[^"]*"'
    replacement = f'{key} = "{value}"'
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE, count=1)
    if count != 1:
        raise ValueError(f"Could not find TOML setting: {key}")
    return updated


def _output_dir_basename_from_config(repo_root: Path, config_rel: str) -> str:
    text = (repo_root / config_rel).read_text(encoding="utf-8")
    output_match = re.search(r'^output_dir\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not output_match:
        raise ValueError(f"Missing output_dir in {config_rel}")
    return Path(output_match.group(1)).name


def _unique_archive_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    suffix = 1
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _move_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _archive_previous_assets(
    repo_root: Path,
    run_tag: str,
    generated_root_rel: str,
    archive_root_rel: str,
) -> tuple[Path | None, list[str]]:
    moved: list[tuple[Path, Path]] = []

    generated_root = repo_root / generated_root_rel
    if generated_root.exists():
        for child in sorted(generated_root.iterdir()):
            if child.name == run_tag:
                continue
            moved.append((child, repo_root / archive_root_rel / generated_root_rel / child.name))

    output_basenames = {
        _output_dir_basename_from_config(repo_root, exp.config_path)
        for exp in BASE_EXPERIMENTS
    }
    outputs_root = repo_root / "outputs"
    if outputs_root.exists():
        for child in sorted(outputs_root.iterdir()):
            if not child.is_dir():
                continue

            name = child.name
            is_known_output = any(name == base or name.startswith(f"{base}__") for base in output_basenames)
            if not is_known_output:
                continue

            if name.endswith(f"__{run_tag}"):
                continue

            moved.append((child, repo_root / archive_root_rel / "outputs" / child.name))

    if not moved:
        return None, []

    archive_batch_dir = _unique_archive_dir(repo_root / archive_root_rel / f"before_{run_tag}")
    moved_rel_paths: list[str] = []

    for src, default_dst in moved:
        if not src.exists():
            continue

        if default_dst.is_relative_to(repo_root / archive_root_rel):
            relative_suffix = default_dst.relative_to(repo_root / archive_root_rel)
            dst = archive_batch_dir / relative_suffix
        else:
            dst = archive_batch_dir / src.name

        _move_path(src, dst)
        moved_rel_paths.append(str(src.relative_to(repo_root)))

    return archive_batch_dir, moved_rel_paths


def _prepare_generated_config(repo_root: Path, src_config_rel: str, run_tag: str, generated_root_rel: str) -> tuple[str, str, str, str]:
    src_path = repo_root / src_config_rel
    text = src_path.read_text(encoding="utf-8")

    output_match = re.search(r'^output_dir\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not output_match:
        raise ValueError(f"Missing output_dir in {src_config_rel}")
    output_basename = Path(output_match.group(1)).name

    experiment_match = re.search(r'^experiment_name\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not experiment_match:
        raise ValueError(f"Missing experiment_name in {src_config_rel}")
    base_experiment_name = experiment_match.group(1)

    timestamped_output_dir = f"outputs/{output_basename}__{run_tag}"
    timestamped_experiment_name = f"{base_experiment_name}__{run_tag}"

    text = _replace_toml_setting(text, "experiment_name", timestamped_experiment_name)
    text = _replace_toml_setting(text, "output_dir", timestamped_output_dir)
    text = _replace_toml_setting(text, "cache_path", f"{timestamped_output_dir}/cache.sqlite")
    text = _replace_toml_setting(text, "checkpoint_path", f"{timestamped_output_dir}/results.jsonl")
    text = _replace_toml_setting(text, "summary_path", f"{timestamped_output_dir}/summary.csv")
    text = _replace_toml_setting(text, "manifest_path", f"{timestamped_output_dir}/manifest.json")

    generated_rel = f"{generated_root_rel}/{run_tag}/{Path(src_config_rel).name}"
    generated_path = repo_root / generated_rel
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(text, encoding="utf-8")

    checkpoint_rel = f"{timestamped_output_dir}/results.jsonl"
    return generated_rel, timestamped_output_dir, checkpoint_rel, timestamped_experiment_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate timestamped experiment configs and an active run manifest.")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--run-tag", default=None, help="Explicit run tag. Default: spring26_<timestamp>")
    parser.add_argument(
        "--generated-config-root",
        default="configs/experiments/generated",
        help="Relative directory where generated configs are written",
    )
    parser.add_argument(
        "--manifest",
        default="web/session-monitor/active_run.json",
        help="Relative manifest output path",
    )
    parser.add_argument(
        "--archive-root",
        default="previous_configs_results",
        help="Relative root directory where previous generated configs and outputs are archived",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Skip archiving previous generated configs and outputs",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Archive previous generated configs and outputs, then exit without preparing a new run",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_tag = args.run_tag or _default_run_tag()

    if not args.skip_archive:
        archive_dir, moved_rel_paths = _archive_previous_assets(
            repo_root=repo_root,
            run_tag=run_tag,
            generated_root_rel=args.generated_config_root,
            archive_root_rel=args.archive_root,
        )
        if archive_dir:
            print(f"Archived previous configs/results to: {archive_dir}")
            print(f"Archived items: {len(moved_rel_paths)}")
        else:
            print("No previous configs/results to archive.")

    if args.archive_only:
        print("Archive-only mode complete.")
        return

    short_suffix = _short_suffix(run_tag)

    sessions: list[dict[str, str]] = []
    for exp in BASE_EXPERIMENTS:
        generated_config_rel, output_dir_rel, checkpoint_rel, experiment_name = _prepare_generated_config(
            repo_root=repo_root,
            src_config_rel=exp.config_path,
            run_tag=run_tag,
            generated_root_rel=args.generated_config_root,
        )
        session_name = f"{exp.base_session_name}__{short_suffix}"
        sessions.append(
            {
                "display_name": exp.display_name,
                "base_session_name": exp.base_session_name,
                "session_name": session_name,
                "config_path": generated_config_rel,
                "output_dir": output_dir_rel,
                "checkpoint_path": checkpoint_rel,
                "log_dir": f"logs/{session_name}",
                "run_tag": run_tag,
                "experiment_name": experiment_name,
            }
        )

    manifest = {
        "run_tag": run_tag,
        "run_title": f"Spring 2026 • {run_tag}",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "sessions": sessions,
    }

    manifest_path = (repo_root / args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Run tag: {run_tag}")
    print(f"Manifest: {manifest_path}")
    print(f"Sessions prepared: {len(sessions)}")


if __name__ == "__main__":
    main()