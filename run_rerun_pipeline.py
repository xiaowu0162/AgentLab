#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


def load_failed_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise SystemExit(f"Expected a JSON list of strings in {path}")
    return data


def load_failure_counts(journal_path: Path) -> dict[str, int]:
    if not journal_path.exists():
        return {}
    with journal_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}
    counts: dict[str, int] = {}
    for entry in data:
        task_id = entry.get("task_id")
        if not task_id:
            continue
        for rec in entry.get("trajectory_records", []):
            if rec.get("reward") == 0:
                counts[task_id] = counts.get(task_id, 0) + 1
    return counts


def filter_overfailed(task_ids: List[str], fail_counts: dict[str, int], max_failures: int) -> List[str]:
    if max_failures <= 0:
        return []
    return [task_id for task_id in task_ids if fail_counts.get(task_id, 0) <= max_failures]


def select_task_ids(task_ids: List[str], limit: int) -> List[str]:
    if limit <= 0:
        return []
    return task_ids[:limit]


def write_task_ids(task_ids: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(task_ids, f, indent=2)
        f.write("\n")


def list_dirs(root: Path) -> Set[Path]:
    if not root.exists():
        return set()
    return {p for p in root.iterdir() if p.is_dir()}


def pick_run_dir(root: Path, before: Set[Path]) -> Path:
    if not root.exists():
        raise SystemExit(f"Runs root does not exist after rerun: {root}")
    after = list_dirs(root)
    new_dirs = list(after - before)
    if len(new_dirs) == 1:
        return new_dirs[0]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    if not after:
        raise SystemExit(f"No run directories found under {root}")
    return max(after, key=lambda p: p.stat().st_mtime)


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True)


def run_command_capture(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_failed = script_dir / "failed_l2_task_ids.json"
    default_main = script_dir / "main_workarena_generic_rerun.py"
    default_process = script_dir / "process_rerun_results.py"
    default_update = script_dir / "update_successful_rerun_journal.py"

    default_runs_root = os.environ.get("AGENTLAB_EXP_ROOT")
    if default_runs_root is None:
        default_runs_root = str((script_dir / ".." / "agentlab_results").resolve())

    parser = argparse.ArgumentParser(
        description="Automate WorkArena L2 rerun pipeline."
    )
    parser.add_argument("--failed-json", type=Path, default=default_failed)
    parser.add_argument("--limit", type=int, default=62)
    parser.add_argument(
        "--task-ids-out",
        type=Path,
        default=script_dir / "rerun_task_ids.json",
        help="Where to write selected task IDs.",
    )
    parser.add_argument("--main-script", type=Path, default=default_main)
    parser.add_argument("--process-script", type=Path, default=default_process)
    parser.add_argument("--update-script", type=Path, default=default_update)
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=script_dir / "longmemevalv2_trajectory_collection_journal.json",
        help="Journal used to count prior failures per task.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=15,
        help="Skip task IDs with more than this many recorded failures.",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=40 * 60,
        help="Per-task wall-clock timeout in seconds (0 or negative to disable).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(default_runs_root),
        help="Root directory where run folders are created.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(
            "/local/diwu/longmemeval-v2-data/workarena/successful_rerun_trajectories/l2"
        ),
        help="Destination root for successful rerun copies.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination folders.",
    )
    args = parser.parse_args()

    failed_ids = load_failed_ids(args.failed_json)
    if not failed_ids:
        print("No failed task IDs found; nothing to run.")
        return 0

    fail_counts = load_failure_counts(args.journal_path)
    filtered_ids = filter_overfailed(failed_ids, fail_counts, args.max_failures)
    skipped = len(failed_ids) - len(filtered_ids)
    if skipped:
        print(f"Skipped {skipped} task IDs with > {args.max_failures} failures.")
    selected_ids = select_task_ids(filtered_ids, min(args.limit, len(filtered_ids)))
    if not selected_ids:
        print("No task IDs selected; nothing to run.")
        return 0

    write_task_ids(selected_ids, args.task_ids_out)
    print(f"Selected {len(selected_ids)} task IDs -> {args.task_ids_out}")

    runs_root = args.runs_root
    before_dirs = list_dirs(runs_root)

    run_command(
        [
            sys.executable,
            str(args.main_script),
            "--task-ids-json",
            str(args.task_ids_out),
            "--task-timeout-seconds",
            str(args.task_timeout_seconds),
        ],
        cwd=script_dir,
    )

    run_dir = pick_run_dir(runs_root, before_dirs)
    print(f"Using run directory: {run_dir}")

    process_cmd = [
        sys.executable,
        str(args.process_script),
        "--runs-dir",
        str(run_dir),
        "--failed-json",
        str(args.failed_json),
        "--dest-root",
        str(args.dest_root),
    ]
    if args.overwrite:
        process_cmd.append("--overwrite")
    run_command(process_cmd, cwd=script_dir)

    update_result = run_command_capture(
        [sys.executable, str(args.update_script), "--dry-run"],
        cwd=script_dir,
    )
    if update_result.stdout:
        print(update_result.stdout, end="")
    if update_result.stderr:
        print(update_result.stderr, end="", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
