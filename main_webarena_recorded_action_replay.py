"""Replay recorded WebArena actions with a new viewport.

This script regenerates AgentLab-style trajectories by replaying previously
saved BrowserGym action strings from one experiment directory or one whole
study directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)
os.environ.setdefault(
    "AGENTLAB_EXP_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "web", "agentlab_results_replay"),
)

import bgym
from dotenv import load_dotenv

from agentlab.agents.replay_action_agent import ReplayActionAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments.exp_utils import add_dependencies
from agentlab.experiments.launch_exp import run_experiments
from agentlab.experiments.loop import ExpArgs
from agentlab.experiments.webarena_eval_patch import (
    install_webarena_html_evaluator_patch,
    install_webarena_string_evaluator_patch,
)


DEFAULT_BENCHMARK = "webarena"
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 1000
DEFAULT_N_JOBS = max(1, min(12, os.cpu_count() or 1))
DEFAULT_PARALLEL_BACKEND = "ray"
DEFAULT_TASK_TIMEOUT_SECONDS = 50 * 60
DEFAULT_AVG_STEP_TIMEOUT = 1200
DEFAULT_MAX_STEPS_BUFFER = 3

REQUIRED_WA_ENV_VARS = (
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_HOMEPAGE",
)

DEFAULT_WEBARENA_METADATA_JSON = (
    Path(__file__).resolve().parents[2] / "web" / "data" / "webarena.test.raw.json"
)


def _str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "t", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _task_number(task_name: str) -> int | None:
    match = re.search(r"(\d+)$", task_name)
    if match is None:
        return None
    return int(match.group(1))


def _full_task_id(env_args) -> str:
    task_seed = getattr(env_args, "task_seed", None)
    if task_seed is None:
        return env_args.task_name
    return f"{env_args.task_name}_{task_seed}"


def _print_wa_env_warning() -> None:
    missing = [key for key in REQUIRED_WA_ENV_VARS if not os.environ.get(key)]
    if missing:
        print(
            "Warning: missing WebArena env vars: "
            + ", ".join(missing)
            + ". Replay runs will fail until they are configured."
        )


def _load_webarena_metadata_by_task_id(metadata_json_path: str) -> dict[int, dict]:
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list):
        raise SystemExit(f"Metadata JSON {metadata_json_path!r} must contain a list of tasks.")

    metadata_by_task_id: dict[int, dict] = {}
    for item in metadata:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if isinstance(task_id, int):
            metadata_by_task_id[task_id] = item
    return metadata_by_task_id


def _load_source_exp_args(exp_dir: Path):
    with open(exp_dir / "exp_args.pkl", "rb") as f:
        return pickle.load(f)


def _find_source_exp_dirs(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if (path / "exp_args.pkl").exists() and (path / "step_0.pkl.gz").exists():
        return [path]

    exp_dirs: list[Path] = []
    for exp_args_path in sorted(path.rglob("exp_args.pkl")):
        exp_dir = exp_args_path.parent
        if (exp_dir / "step_0.pkl.gz").exists():
            exp_dirs.append(exp_dir)
    return exp_dirs


@dataclass
class ReplaySource:
    source_exp_dir: Path
    source_task_id: str
    source_task_name: str
    source_task_seed: int | None
    action_count: int


@dataclass
class SkippedDuplicateSource:
    source_task_name: str
    kept_source_exp_dir: Path
    skipped_source_exp_dir: Path
    reason: str


@dataclass
class SkippedInvalidSource:
    source_exp_dir: Path
    reason: str


def _collect_replay_sources(exp_dirs: list[Path], *, print_state: bool, max_state_chars: int):
    replay_sources: list[tuple[ReplaySource, ReplayActionAgentArgs, object]] = []
    skipped_invalid_sources: list[SkippedInvalidSource] = []
    for exp_dir in exp_dirs:
        source_exp_args = _load_source_exp_args(exp_dir)
        env_args = source_exp_args.env_args
        if not getattr(env_args, "task_name", "").startswith("webarena"):
            raise SystemExit(f"Source experiment is not a WebArena run: {exp_dir}")

        try:
            replay_agent = ReplayActionAgentArgs.from_exp_dir(
                exp_dir,
                print_state=print_state,
                dump_screenshot=False,
                max_state_chars=max_state_chars,
            )
        except ValueError as exc:
            skipped_invalid_sources.append(
                SkippedInvalidSource(
                    source_exp_dir=exp_dir,
                    reason=str(exc),
                )
            )
            continue
        replay_sources.append(
            (
                ReplaySource(
                    source_exp_dir=exp_dir,
                    source_task_id=_full_task_id(env_args),
                    source_task_name=env_args.task_name,
                    source_task_seed=getattr(env_args, "task_seed", None),
                    action_count=len(replay_agent.actions),
                ),
                replay_agent,
                source_exp_args,
            )
        )
    return replay_sources, skipped_invalid_sources


def _load_summary_info(exp_dir: Path) -> dict:
    summary_path = exp_dir / "summary_info.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _replay_source_sort_key(
    replay_source: tuple[ReplaySource, ReplayActionAgentArgs, object],
) -> tuple[int, int, int, int, str]:
    source, _, _ = replay_source
    task_number = _task_number(source.source_task_name)
    return (
        1 if task_number is None else 0,
        task_number if task_number is not None else 10**9,
        source.source_task_seed if source.source_task_seed is not None else -1,
        -source.action_count,
        source.source_exp_dir.name,
    )


def _replay_source_preference_key(
    replay_source: tuple[ReplaySource, ReplayActionAgentArgs, object],
) -> tuple[int, int, int, int, str]:
    source, _, _ = replay_source
    summary_info = _load_summary_info(source.source_exp_dir)
    err_msg = summary_info.get("err_msg")
    n_steps = summary_info.get("n_steps")
    return (
        1 if not source.source_exp_dir.name.startswith("_") else 0,
        1 if not err_msg else 0,
        source.action_count,
        n_steps if isinstance(n_steps, int) else -1,
        source.source_exp_dir.name,
    )


def _dedupe_replay_sources(
    replay_sources: list[tuple[ReplaySource, ReplayActionAgentArgs, object]],
) -> tuple[
    list[tuple[ReplaySource, ReplayActionAgentArgs, object]],
    list[SkippedDuplicateSource],
]:
    best_by_task_name: dict[str, tuple[ReplaySource, ReplayActionAgentArgs, object]] = {}
    skipped_duplicates: list[SkippedDuplicateSource] = []

    for replay_source in replay_sources:
        source, _, _ = replay_source
        current = best_by_task_name.get(source.source_task_name)
        if current is None:
            best_by_task_name[source.source_task_name] = replay_source
            continue

        if _replay_source_preference_key(replay_source) > _replay_source_preference_key(current):
            kept, skipped = replay_source, current
            reason = "preferred non-underscore or successful source experiment"
            best_by_task_name[source.source_task_name] = replay_source
        else:
            kept, skipped = current, replay_source
            reason = "kept preferred existing source experiment for duplicate task"

        skipped_duplicates.append(
            SkippedDuplicateSource(
                source_task_name=source.source_task_name,
                kept_source_exp_dir=kept[0].source_exp_dir,
                skipped_source_exp_dir=skipped[0].source_exp_dir,
                reason=reason,
            )
        )

    deduped_sources = sorted(best_by_task_name.values(), key=_replay_source_sort_key)
    return deduped_sources, skipped_duplicates


def _restrict_task_dependencies(
    task_dependencies: dict[str, list[str]],
    task_names: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    task_name_set = set(task_names)
    filtered_dependencies: dict[str, list[str]] = {}
    dropped_dependencies: dict[str, list[str]] = {}

    for task_name in task_names:
        dependencies = list(task_dependencies.get(task_name, []))
        filtered_dependencies[task_name] = [dep for dep in dependencies if dep in task_name_set]
        missing_dependencies = [dep for dep in dependencies if dep not in task_name_set]
        if missing_dependencies:
            dropped_dependencies[task_name] = missing_dependencies

    return filtered_dependencies, dropped_dependencies


def _write_manifest(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    source_arg: Path,
    replay_sources: list[tuple[ReplaySource, ReplayActionAgentArgs, object]],
    skipped_invalid_sources: list[SkippedInvalidSource],
    skipped_duplicates: list[SkippedDuplicateSource],
    dropped_dependencies: dict[str, list[str]],
    study_dir_name: str,
) -> None:
    payload = {
        "benchmark": args.benchmark,
        "source_path": str(source_arg),
        "study_dir_name": study_dir_name,
        "viewport": {"width": args.viewport_width, "height": args.viewport_height},
        "headless": args.headless,
        "record_video": args.record_video,
        "pre_observation_delay": args.pre_observation_delay,
        "reddit_pre_observation_delay": args.reddit_pre_observation_delay,
        "task_timeout_seconds": args.task_timeout_seconds,
        "avg_step_timeout": args.avg_step_timeout,
        "max_steps_buffer": args.max_steps_buffer,
        "parallel_backend": args.parallel_backend,
        "n_jobs": args.n_jobs,
        "ignore_dependencies": args.ignore_dependencies,
        "selected_source_count": len(replay_sources),
        "skipped_invalid_count": len(skipped_invalid_sources),
        "skipped_invalid_sources": [
            {
                "source_exp_dir": str(item.source_exp_dir),
                "reason": item.reason,
            }
            for item in skipped_invalid_sources
        ],
        "skipped_duplicate_count": len(skipped_duplicates),
        "skipped_duplicates": [
            {
                "source_task_name": item.source_task_name,
                "kept_source_exp_dir": str(item.kept_source_exp_dir),
                "skipped_source_exp_dir": str(item.skipped_source_exp_dir),
                "reason": item.reason,
            }
            for item in skipped_duplicates
        ],
        "dropped_dependency_edges": dropped_dependencies,
        "sources": [
            {
                "source_exp_dir": str(source.source_exp_dir),
                "source_task_id": source.source_task_id,
                "source_task_name": source.source_task_name,
                "source_task_seed": source.source_task_seed,
                "action_count": source.action_count,
            }
            for source, _, _ in replay_sources
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_hashable_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _normalize_unhashable_value(value: Any) -> Any:
    if _is_hashable_value(value):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        value = sorted(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return repr(value)


def _load_result_df_with_replay_fallback(study_dir: Path):
    try:
        return inspect_results.load_result_df(study_dir)
    except TypeError as exc:
        if "unhashable type" not in str(exc):
            raise

        logging.warning(
            "Retrying result dataframe load for replay study %s after normalizing "
            "unhashable metadata columns.",
            study_dir,
        )
        result_df = inspect_results.load_result_df(study_dir, set_index=False)
        if result_df is None:
            return None

        for column in result_df.columns:
            if result_df[column].map(_is_hashable_value).all():
                continue
            result_df[column] = result_df[column].map(_normalize_unhashable_value)

        inspect_results.set_index_from_variables(result_df)
        return result_df


def _write_result_reports(study_dir: Path) -> None:
    result_df = _load_result_df_with_replay_fallback(study_dir)
    if result_df is None:
        return

    summary_df = inspect_results.summarize_study(result_df)
    error_report = inspect_results.error_report(result_df, max_stack_trace=3, use_log=True)

    result_df.to_csv(study_dir / "result_df.csv")
    summary_df.to_csv(study_dir / "summary_df.csv")
    (study_dir / "error_report.md").write_text(error_report, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded WebArena actions from prior AgentLab trajectories."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-exp-dir",
        default=None,
        help="Single source experiment directory containing exp_args.pkl and step_*.pkl.gz.",
    )
    source_group.add_argument(
        "--source-study-dir",
        default=None,
        help="Source AgentLab study directory; all contained experiment dirs will be replayed.",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="Benchmark name for the replay environment (default: webarena).",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=DEFAULT_VIEWPORT_WIDTH,
        help="Replay viewport width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=DEFAULT_VIEWPORT_HEIGHT,
        help="Replay viewport height in pixels (default: 1000).",
    )
    parser.add_argument(
        "--headless",
        type=_str2bool,
        default=True,
        help="Run replay browser headless (default: true).",
    )
    parser.add_argument(
        "--record-video",
        type=_str2bool,
        default=False,
        help="Record replay videos in addition to screenshots (default: false).",
    )
    parser.add_argument(
        "--pre-observation-delay",
        type=float,
        default=None,
        help="Optional BrowserGym pre_observation_delay in seconds for all replayed tasks.",
    )
    parser.add_argument(
        "--reddit-pre-observation-delay",
        type=float,
        default=4.0,
        help="Optional reddit-specific pre_observation_delay override in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help="Per-task wall-clock timeout in seconds (0 or negative disables).",
    )
    parser.add_argument(
        "--avg-step-timeout",
        type=int,
        default=DEFAULT_AVG_STEP_TIMEOUT,
        help="Average step timeout used by AgentLab run_experiments (default: 1200).",
    )
    parser.add_argument(
        "--max-steps-buffer",
        type=int,
        default=DEFAULT_MAX_STEPS_BUFFER,
        help="Extra max_steps to add on top of the recorded action count (default: 3).",
    )
    parser.add_argument(
        "--metadata-json-path",
        default=str(DEFAULT_WEBARENA_METADATA_JSON),
        help="Path to WebArena metadata JSON used for reddit delay overrides.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("AGENTLAB_EXP_ROOT"),
        help="Root directory for replay outputs (defaults to AGENTLAB_EXP_ROOT).",
    )
    parser.add_argument(
        "--study-dir-name",
        default=None,
        help="Override the replay study directory name under --output-root.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Max parallel replay workers when using Ray (default: {DEFAULT_N_JOBS}).",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("ray", "sequential"),
        default=DEFAULT_PARALLEL_BACKEND,
        help="Replay backend (default: ray).",
    )
    parser.add_argument(
        "--ignore-dependencies",
        action="store_true",
        help="Run replay tasks without WebArena dependency ordering.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many deduplicated source experiments before replaying.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Replay at most this many deduplicated source experiments (0 means no limit).",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print current replay observation state before each action.",
    )
    parser.add_argument(
        "--max-state-chars",
        type=int,
        default=12_000,
        help="Max chars to print per state field when --print-state is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected replay sources and exit without launching replay.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    _print_wa_env_warning()
    args = _parse_args()

    source_arg = Path(args.source_exp_dir or args.source_study_dir).expanduser().resolve()
    source_exp_dirs = _find_source_exp_dirs(source_arg)
    if not source_exp_dirs:
        raise SystemExit(f"No replayable experiment directories found under {source_arg}")

    replay_sources, skipped_invalid_sources = _collect_replay_sources(
        source_exp_dirs,
        print_state=args.print_state,
        max_state_chars=args.max_state_chars,
    )
    replay_sources, skipped_duplicates = _dedupe_replay_sources(
        replay_sources
    )

    if args.offset > 0:
        replay_sources = replay_sources[args.offset :]
    if args.limit > 0:
        replay_sources = replay_sources[: args.limit]
    if not replay_sources:
        raise SystemExit("No source experiments remain after offset/limit filtering.")

    metadata_by_task_id = _load_webarena_metadata_by_task_id(args.metadata_json_path)
    benchmark = bgym.DEFAULT_BENCHMARKS[args.benchmark]()
    if install_webarena_html_evaluator_patch():
        print("Installed AgentLab WebArena HTML evaluator patch.")
    if install_webarena_string_evaluator_patch():
        print("Installed AgentLab WebArena string evaluator patch.")
    study_dir_name = args.study_dir_name or source_arg.name
    planned_study_dir = Path(args.output_root).expanduser().resolve() / study_dir_name

    print(
        f"Selected {len(replay_sources)} source experiments for replay "
        f"at viewport {args.viewport_width}x{args.viewport_height}:"
    )
    if skipped_invalid_sources:
        print(
            f"Skipped {len(skipped_invalid_sources)} non-replayable source experiments "
            "with no recorded action sequence."
        )
    if skipped_duplicates:
        print(f"Skipped {len(skipped_duplicates)} duplicate source experiments by task name.")
    for source, _, _ in replay_sources[:20]:
        print(
            f"  - {source.source_task_id} from {source.source_exp_dir.name} "
            f"({source.action_count} actions)"
        )
    if len(replay_sources) > 20:
        print(f"  ... and {len(replay_sources) - 20} more")
    print(
        f"Replay backend: {args.parallel_backend} with up to {args.n_jobs} workers. "
        f"Planned output directory: {planned_study_dir}"
    )

    if args.dry_run:
        print("Dry run only; replay not launched.")
        return

    exp_args_list: list[ExpArgs] = []
    for order, (source, replay_agent, source_exp_args) in enumerate(replay_sources):
        replay_agent.set_benchmark(benchmark, demo_mode=False)

        env_args = copy.deepcopy(source_exp_args.env_args)
        env_args.headless = args.headless
        env_args.record_video = args.record_video
        env_args.wait_for_user_message = False
        env_args.viewport = {"width": args.viewport_width, "height": args.viewport_height}
        env_args.max_steps = max(
            int(getattr(env_args, "max_steps", 0) or 0),
            source.action_count + args.max_steps_buffer,
        )

        task_num = _task_number(source.source_task_name)
        if args.pre_observation_delay is not None:
            env_args.pre_observation_delay = args.pre_observation_delay
        if (
            args.reddit_pre_observation_delay is not None
            and task_num is not None
            and "reddit" in metadata_by_task_id.get(task_num, {}).get("sites", [])
        ):
            env_args.pre_observation_delay = args.reddit_pre_observation_delay

        exp_args = ExpArgs(
            agent_args=replay_agent,
            env_args=env_args,
            logging_level=logging.INFO,
            logging_level_stdout=logging.INFO,
        )
        exp_args.order = order
        if args.task_timeout_seconds and args.task_timeout_seconds > 0:
            exp_args.episode_timeout = args.task_timeout_seconds
        exp_args_list.append(exp_args)

    dropped_dependencies: dict[str, list[str]] = {}
    if args.ignore_dependencies:
        print("Warning: running replay without WebArena dependency ordering.")
    else:
        filtered_dependencies, dropped_dependencies = _restrict_task_dependencies(
            benchmark.dependency_graph_over_tasks(),
            [exp_args.env_args.task_name for exp_args in exp_args_list],
        )
        if dropped_dependencies:
            print(
                "Warning: dropping dependency edges that point to tasks outside this replay batch:"
            )
            for task_name, missing in list(dropped_dependencies.items())[:20]:
                print(f"  - {task_name} depends on missing tasks: {', '.join(missing)}")
            if len(dropped_dependencies) > 20:
                print(f"  ... and {len(dropped_dependencies) - 20} more tasks with dropped edges")
        exp_args_list = add_dependencies(exp_args_list, filtered_dependencies)

    study_dir = planned_study_dir
    study_dir.parent.mkdir(parents=True, exist_ok=True)
    if study_dir.exists():
        raise SystemExit(
            f"Replay output directory already exists: {study_dir}. "
            "Move or remove it before replaying this batch again."
        )
    study_dir.mkdir(parents=True, exist_ok=False)

    _write_manifest(
        study_dir / "replay_manifest.json",
        args=args,
        source_arg=source_arg,
        replay_sources=replay_sources,
        skipped_invalid_sources=skipped_invalid_sources,
        skipped_duplicates=skipped_duplicates,
        dropped_dependencies=dropped_dependencies,
        study_dir_name=study_dir_name,
    )

    print(f"Replay output directory: {study_dir}")
    parallel_backend = args.parallel_backend
    n_jobs = args.n_jobs
    if len(exp_args_list) == 1 and parallel_backend == "ray":
        parallel_backend = "sequential"
        n_jobs = 1

    run_experiments(
        n_jobs=n_jobs,
        exp_args_list=exp_args_list,
        study_dir=study_dir,
        parallel_backend=parallel_backend,
        avg_step_timeout=args.avg_step_timeout,
    )
    _write_result_reports(study_dir)
    print("Replay finished.")


if __name__ == "__main__":
    main()
