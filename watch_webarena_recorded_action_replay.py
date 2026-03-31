"""Poll a WebArena replay batch until it finishes.

This watches an AgentLab replay study directory, prints periodic progress
updates, and exits with a clear final message when the batch is done or when
the replay process dies early.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll a WebArena recorded-action replay batch until completion."
    )
    parser.add_argument(
        "--study-dir",
        required=True,
        help="Replay study directory under web/agentlab_results_replay.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=0,
        help="Optional replay process PID. If provided, fail fast when it exits early.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=30.0,
        help="Polling interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one status snapshot and exit immediately.",
    )
    return parser.parse_args()


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_status(study_dir: Path) -> dict:
    manifest = _load_json(study_dir / "replay_manifest.json") or {}
    expected_count = manifest.get("selected_source_count")

    exp_dirs = sorted(path for path in study_dir.iterdir() if path.is_dir()) if study_dir.exists() else []
    counters = Counter()
    error_examples: list[tuple[str, str]] = []

    for exp_dir in exp_dirs:
        summary = _load_json(exp_dir / "summary_info.json")
        if summary is None:
            counters["running"] += 1
            continue

        err_msg = summary.get("err_msg")
        if err_msg:
            counters["error"] += 1
            if len(error_examples) < 3:
                error_examples.append((exp_dir.name, str(err_msg).splitlines()[0]))
        else:
            counters["done"] += 1

    completed_count = counters["done"] + counters["error"]
    summary_df_exists = (study_dir / "summary_df.csv").exists()
    result_df_exists = (study_dir / "result_df.csv").exists()

    return {
        "expected_count": expected_count,
        "exp_dir_count": len(exp_dirs),
        "done_count": counters["done"],
        "error_count": counters["error"],
        "running_count": counters["running"],
        "completed_count": completed_count,
        "summary_df_exists": summary_df_exists,
        "result_df_exists": result_df_exists,
        "error_examples": error_examples,
    }


def _format_status_line(study_dir: Path, status: dict, *, pid: int) -> str:
    expected = status["expected_count"]
    completed = status["completed_count"]
    expected_str = "?" if expected is None else str(expected)
    progress = "?"
    if isinstance(expected, int) and expected > 0:
        progress = f"{completed / expected:.0%}"

    pid_state = "n/a"
    if pid > 0:
        pid_state = "alive" if _is_pid_alive(pid) else "dead"

    return (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{study_dir.name}: completed={completed}/{expected_str} ({progress}), "
        f"done={status['done_count']}, error={status['error_count']}, "
        f"running={status['running_count']}, exp_dirs={status['exp_dir_count']}, "
        f"summary_df={'yes' if status['summary_df_exists'] else 'no'}, pid={pid_state}"
    )


def main() -> int:
    args = _parse_args()
    study_dir = Path(args.study_dir).expanduser().resolve()

    if not study_dir.exists():
        print(f"Study directory does not exist yet: {study_dir}", file=sys.stderr)
        return 2

    while True:
        status = _collect_status(study_dir)
        print(_format_status_line(study_dir, status, pid=args.pid), flush=True)

        if status["error_examples"]:
            for exp_name, err in status["error_examples"]:
                print(f"  sample error: {exp_name}: {err}", flush=True)

        expected = status["expected_count"]
        all_task_summaries_present = isinstance(expected, int) and status["completed_count"] >= expected

        if status["summary_df_exists"]:
            print(
                f"Replay batch finished cleanly: {study_dir} "
                f"(done={status['done_count']}, error={status['error_count']}).",
                flush=True,
            )
            return 0

        if args.once:
            return 0

        pid_alive = _is_pid_alive(args.pid) if args.pid > 0 else None

        if all_task_summaries_present and pid_alive is False:
            print(
                f"Replay process exited after all task summaries were written, "
                f"but final aggregate reports are missing: {study_dir}",
                flush=True,
            )
            return 1

        if args.pid > 0 and pid_alive is False and not all_task_summaries_present:
            print(
                f"Replay process {args.pid} exited before the batch completed: {study_dir}",
                flush=True,
            )
            return 1

        time.sleep(max(args.interval_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
