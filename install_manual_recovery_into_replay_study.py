#!/usr/bin/env python3
"""Install an accepted WebArena trajectory into a final replay study.

This helper archives the existing replay trajectory for a source experiment,
copies in an accepted converted trajectory, rewrites `exp_args.pkl.exp_dir`,
and regenerates the aggregate replay reports.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_summary(path: Path) -> dict:
    summary_path = path / "summary_info.json"
    if not summary_path.exists():
        raise SystemExit(f"summary_info.json not found under {path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _rewrite_exp_args_path(exp_dir: Path) -> None:
    exp_args_path = exp_dir / "exp_args.pkl"
    with open(exp_args_path, "rb") as f:
        exp_args = pickle.load(f)
    exp_args.exp_dir = exp_dir
    with open(exp_args_path, "wb") as f:
        pickle.dump(exp_args, f)


def _write_reports(study_dir: Path) -> None:
    from main_webarena_recorded_action_replay import _write_result_reports

    _write_result_reports(study_dir)


def _log_install(
    study_dir: Path,
    *,
    source_exp_dir: Path,
    accepted_exp_dir: Path,
    installed_dir: Path,
    archived_dir: Path | None,
    dry_run: bool,
) -> None:
    log_path = study_dir / "manual_recovery_install_log.jsonl"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "source_exp_dir": str(source_exp_dir),
        "accepted_exp_dir": str(accepted_exp_dir),
        "installed_dir": str(installed_dir),
        "archived_dir": None if archived_dir is None else str(archived_dir),
        "dry_run": dry_run,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive a bad replay dir and install an accepted trajectory in its place."
    )
    parser.add_argument(
        "--study-dir",
        required=True,
        help=(
            "Final replay study directory to update. "
            "If it does not exist yet, it will be created."
        ),
    )
    parser.add_argument(
        "--source-exp-dir",
        required=True,
        help="Original source trajectory dir whose basename should be preserved in the replay study.",
    )
    parser.add_argument(
        "--accepted-exp-dir",
        "--manual-exp-dir",
        dest="accepted_exp_dir",
        required=True,
        help=(
            "Accepted trajectory dir to copy into the final replay study. "
            "This can be an accepted auto replay from to_verify or a successful manual recovery."
        ),
    )
    parser.add_argument(
        "--backup-root",
        default=str(
            Path(__file__).resolve().parents[2] / "web" / "agentlab_results_replay" / "failed_runs"
        ),
        help="Root directory for archived bad replay trajectories.",
    )
    parser.add_argument(
        "--skip-report-rewrite",
        action="store_true",
        help="Do not regenerate result_df.csv, summary_df.csv, and error_report.md.",
    )
    parser.add_argument(
        "--skip-summary-check",
        action="store_true",
        help="Do not require summary_info.json to exist under the manual recovery dir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned operations only.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study_dir = Path(args.study_dir).expanduser().resolve()
    source_exp_dir = Path(args.source_exp_dir).expanduser().resolve()
    accepted_exp_dir = Path(args.accepted_exp_dir).expanduser().resolve()
    backup_root = Path(args.backup_root).expanduser().resolve()

    if not source_exp_dir.exists():
        raise SystemExit(f"Source experiment dir not found: {source_exp_dir}")
    if not accepted_exp_dir.exists():
        raise SystemExit(f"Accepted trajectory dir not found: {accepted_exp_dir}")
    if not args.skip_summary_check:
        _load_summary(accepted_exp_dir)

    target_name = source_exp_dir.name
    target_dir = study_dir / target_name

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = None
    if target_dir.exists():
        archive_dir = backup_root / study_dir.name / f"{target_name}.replaced_{timestamp}"

    print(f"Study dir:      {study_dir}")
    print(f"Source exp dir: {source_exp_dir}")
    print(f"Accepted dir:   {accepted_exp_dir}")
    print(f"Target name:    {target_name}")
    if archive_dir is not None:
        print(f"Archive replay: {target_dir} -> {archive_dir}")
    else:
        print("Archive replay: no existing target dir found; manual recovery will be added fresh")
    print(f"Install target: {accepted_exp_dir} -> {target_dir}")

    if args.dry_run:
        print("Dry run only; no files changed.")
        return

    study_dir.mkdir(parents=True, exist_ok=True)

    if archive_dir is not None:
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_dir), str(archive_dir))

    if target_dir.exists():
        raise SystemExit(f"Target dir still exists after archive step: {target_dir}")

    shutil.copytree(accepted_exp_dir, target_dir)
    _rewrite_exp_args_path(target_dir)

    if not args.skip_report_rewrite:
        _write_reports(study_dir)

    _log_install(
        study_dir,
        source_exp_dir=source_exp_dir,
        accepted_exp_dir=accepted_exp_dir,
        installed_dir=target_dir,
        archived_dir=archive_dir,
        dry_run=False,
    )
    print(f"Installed accepted manual recovery into {target_dir}")


if __name__ == "__main__":
    main()
