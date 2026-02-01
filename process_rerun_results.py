#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_task_id(run_dir: Path) -> str | None:
    name = run_dir.name
    if "_on_" not in name:
        return None
    return name.split("_on_", 1)[1]


def load_summary(run_dir: Path) -> dict | None:
    for filename in ("summary_info.json", "summary.json"):
        summary_path = run_dir / filename
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None


def iter_successes(base_dir: Path) -> list[tuple[str, Path]]:
    successes: list[tuple[str, Path]] = []
    for entry in os.scandir(base_dir):
        if not entry.is_dir():
            continue
        run_dir = Path(entry.path)
        summary = load_summary(run_dir)
        if not summary:
            continue
        if summary.get("cum_reward") != 1:
            continue
        task_id = parse_task_id(run_dir)
        if task_id is None:
            continue
        successes.append((task_id, run_dir))
    return successes


def update_failed_ids(failed_path: Path, success_ids: set[str]) -> tuple[int, int]:
    with failed_path.open("r", encoding="utf-8") as f:
        failed_ids = json.load(f)
    original_count = len(failed_ids)
    new_failed = [task_id for task_id in failed_ids if task_id not in success_ids]
    if new_failed != failed_ids:
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(new_failed, f, indent=4)
            f.write("\n")
    return original_count, len(new_failed)


def copy_successes(
    successes: list[tuple[str, Path]],
    dest_root: Path,
    overwrite: bool,
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    dest_root.mkdir(parents=True, exist_ok=True)
    for _, run_dir in successes:
        dest_dir = dest_root / run_dir.name
        if dest_dir.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copytree(run_dir, dest_dir, dirs_exist_ok=overwrite)
        copied += 1
    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove successful task IDs from failed list and copy successful runs."
        )
    )
    parser.add_argument(
        "--runs-dir",
        default=(
            "/local/jerryji/agentlab/2026-01-27_22-32-32_genericagent-gpt-5-mini-2025-08-07-on-workarena-l2-agent-curriculum-eval"
        ),
        help="Directory containing per-task run subfolders.",
    )
    parser.add_argument(
        "--runs-dir-name",
        default=None,
        help="Run directory name under --runs-root (e.g., 2026-01-27_..._eval).",
    )
    parser.add_argument(
        "--runs-root",
        default="/local/jerryji/agentlab",
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--failed-json",
        default="/local/jerryji/AgentLab-Custom/failed_l2_task_ids.json",
        help="Path to failed_l2_task_ids.json.",
    )
    parser.add_argument(
        "--dest-root",
        default="/local/diwu/longmemeval-v2-data/workarena/successful_rerun_trajectories/l2",
        help="Destination root to copy successful run folders into.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination folders.",
    )
    parser.add_argument(
        "--update-script",
        default=str(Path(__file__).resolve().parent / "update_successful_rerun_journal.py"),
        help="Path to update_successful_rerun_journal.py.",
    )
    parser.add_argument(
        "--run-update-dry-run",
        action="store_true",
        help="Run update_successful_rerun_journal.py --dry-run after copying.",
    )
    args = parser.parse_args()

    if args.runs_dir_name:
        runs_dir = Path(args.runs_root) / args.runs_dir_name
    else:
        runs_dir = Path(args.runs_dir)
    failed_path = Path(args.failed_json)
    dest_root = Path(args.dest_root)

    successes = iter_successes(runs_dir)
    success_ids = {task_id for task_id, _ in successes}

    print(f"Found {len(successes)} successful runs (cum_reward == 1).")
    if success_ids:
        original_count, new_count = update_failed_ids(failed_path, success_ids)
        print(
            f"failed_l2_task_ids.json: {original_count} -> {new_count} "
            f"(removed {original_count - new_count})."
        )
    else:
        print("No successful task IDs to remove from failed_l2_task_ids.json.")

    copied, skipped = copy_successes(successes, dest_root, args.overwrite)
    print(f"Copied {copied} run folders. Skipped {skipped} existing folders.")

    if args.run_update_dry_run:
        subprocess.run(
            [sys.executable, args.update_script, "--dry-run"],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
