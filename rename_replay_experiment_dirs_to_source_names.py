#!/usr/bin/env python
"""Rename replay experiment directories to match their source experiment names exactly.

This is intended as a post-processing step for AgentLab replay studies. It uses
`replay_manifest.json` to map each replayed task back to the original
`source_exp_dir` basename, renames the replay directory, updates the stored
`exp_args.pkl` `exp_dir` field, and optionally regenerates aggregate reports.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


TASK_ID_PATTERN = re.compile(r"_on_(webarena\.\d+(?:_\d+)?)$")


def _task_id_from_name(exp_name: str) -> str | None:
    match = TASK_ID_PATTERN.search(exp_name)
    if match is None:
        return None
    return match.group(1)


def _task_id_from_exp_dir(exp_dir: Path) -> str:
    task_id = _task_id_from_name(exp_dir.name)
    if task_id is not None:
        return task_id

    with open(exp_dir / "exp_args.pkl", "rb") as f:
        exp_args = pickle.load(f)

    task_name = exp_args.env_args.task_name
    task_seed = getattr(exp_args.env_args, "task_seed", None)
    if task_seed is None:
        return task_name
    return f"{task_name}_{task_seed}"


def _load_manifest_sources(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for item in payload.get("sources", []):
        task_id = item["source_task_id"]
        source_name = Path(item["source_exp_dir"]).name
        if task_id in mapping and mapping[task_id] != source_name:
            raise SystemExit(
                f"Manifest contains conflicting source directory names for {task_id}: "
                f"{mapping[task_id]!r} vs {source_name!r}"
            )
        mapping[task_id] = source_name
    if not mapping:
        raise SystemExit(f"No sources found in manifest: {manifest_path}")
    return mapping


def _iter_replay_exp_dirs(study_dir: Path) -> list[Path]:
    exp_dirs: list[Path] = []
    for exp_args_path in sorted(study_dir.rglob("exp_args.pkl")):
        exp_dir = exp_args_path.parent
        if exp_dir.parent != study_dir:
            continue
        exp_dirs.append(exp_dir)
    return exp_dirs


def _rewrite_exp_args_path(exp_dir: Path) -> None:
    exp_args_path = exp_dir / "exp_args.pkl"
    with open(exp_args_path, "rb") as f:
        exp_args = pickle.load(f)
    exp_args.exp_dir = exp_dir
    with open(exp_args_path, "wb") as f:
        pickle.dump(exp_args, f)


def _rename_replay_dirs(study_dir: Path, source_name_by_task_id: dict[str, str], dry_run: bool):
    plan: list[tuple[Path, Path, str]] = []
    for exp_dir in _iter_replay_exp_dirs(study_dir):
        task_id = _task_id_from_exp_dir(exp_dir)
        target_name = source_name_by_task_id.get(task_id)
        if target_name is None:
            raise SystemExit(f"Task {task_id!r} from {exp_dir} is missing from manifest.")
        target_dir = study_dir / target_name
        plan.append((exp_dir, target_dir, task_id))

    target_names = [target_dir.name for _, target_dir, _ in plan]
    if len(target_names) != len(set(target_names)):
        raise SystemExit("Rename plan would create duplicate destination directory names.")

    for current_dir, target_dir, task_id in plan:
        if current_dir == target_dir:
            continue
        if target_dir.exists():
            raise SystemExit(
                f"Destination already exists for task {task_id}: {target_dir}. "
                "Refusing to overwrite."
            )

    applied: list[dict[str, str]] = []
    for current_dir, target_dir, task_id in plan:
        if current_dir == target_dir:
            applied.append(
                {
                    "task_id": task_id,
                    "old_name": current_dir.name,
                    "new_name": target_dir.name,
                    "status": "unchanged",
                }
            )
            continue

        if dry_run:
            applied.append(
                {
                    "task_id": task_id,
                    "old_name": current_dir.name,
                    "new_name": target_dir.name,
                    "status": "planned",
                }
            )
            continue

        current_dir.rename(target_dir)
        _rewrite_exp_args_path(target_dir)
        applied.append(
            {
                "task_id": task_id,
                "old_name": current_dir.name,
                "new_name": target_dir.name,
                "status": "renamed",
            }
        )

    return applied


def _write_reports(study_dir: Path) -> None:
    from main_webarena_recorded_action_replay import _write_result_reports

    _write_result_reports(study_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename replay experiment directories to their source experiment names."
    )
    parser.add_argument("--study-dir", required=True, help="Replay study directory to normalize.")
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional replay manifest path. Defaults to <study-dir>/replay_manifest.json.",
    )
    parser.add_argument(
        "--skip-report-rewrite",
        action="store_true",
        help="Do not rewrite result_df.csv, summary_df.csv, and error_report.md after renaming.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the rename plan and exit.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study_dir = Path(args.study_dir).expanduser().resolve()
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else study_dir / "replay_manifest.json"
    )

    source_name_by_task_id = _load_manifest_sources(manifest_path)
    applied = _rename_replay_dirs(study_dir, source_name_by_task_id, dry_run=args.dry_run)

    rename_map_path = study_dir / "replay_dir_name_map.json"
    rename_map_path.write_text(json.dumps(applied, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Planned {len(applied)} replay directory name updates in {study_dir}")
        return

    if not args.skip_report_rewrite:
        _write_reports(study_dir)

    renamed = sum(item["status"] == "renamed" for item in applied)
    unchanged = sum(item["status"] == "unchanged" for item in applied)
    print(
        f"Normalized replay directory names in {study_dir}: "
        f"{renamed} renamed, {unchanged} already matched source names."
    )


if __name__ == "__main__":
    main()
