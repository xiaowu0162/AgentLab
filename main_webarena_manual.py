"""Run a single WebArena task with the terminal-driven manual agent.

This mirrors the WorkArena manual entrypoint, but targets WebArena and writes
its raw interactive attempts to a separate results root so accepted recoveries
can be copied into the canonical replay study later.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Iterable

os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)
os.environ.setdefault(
    "AGENTLAB_EXP_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "web", "agentlab_results_manual"),
)

import bgym
from dotenv import load_dotenv

from agentlab.agents.manual_action_agent import ManualActionAgentArgs
from agentlab.experiments.study import make_study
from agentlab.experiments.webarena_eval_patch import (
    install_webarena_html_evaluator_patch,
    install_webarena_string_evaluator_patch,
)


DEFAULT_BENCHMARK = "webarena"
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 1000
DEFAULT_MAX_STEPS = 80
DEFAULT_TASK_TIMEOUT_SECONDS = 50 * 60
DEFAULT_AVG_STEP_TIMEOUT = 1200
DEFAULT_REDDIT_PRE_OBSERVATION_DELAY = 4.0
DEFAULT_WEBARENA_METADATA_JSON = (
    Path(__file__).resolve().parents[2] / "web" / "data" / "webarena.test.raw.json"
)

REQUIRED_WA_ENV_VARS = (
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_HOMEPAGE",
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


def _print_available_task_ids(env_args_list: Iterable) -> None:
    print("Available task IDs:")
    for env_args in env_args_list:
        print(f"  - {_full_task_id(env_args)}")


def _print_wa_env_warning() -> None:
    missing = [key for key in REQUIRED_WA_ENV_VARS if not os.environ.get(key)]
    if missing:
        print(
            "Warning: missing WebArena env vars: "
            + ", ".join(missing)
            + ". Manual runs may fail until they are configured."
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


def _task_id_from_source_exp_dir(exp_dir: str | Path) -> str:
    exp_dir = Path(exp_dir).expanduser().resolve()
    exp_args_path = exp_dir / "exp_args.pkl"
    if not exp_args_path.exists():
        raise SystemExit(f"exp_args.pkl not found under source experiment dir: {exp_dir}")

    with open(exp_args_path, "rb") as f:
        exp_args = pickle.load(f)

    env_args = exp_args.env_args
    return _full_task_id(env_args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one WebArena task interactively with ManualActionAgent."
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"Benchmark name in bgym.DEFAULT_BENCHMARKS (default: {DEFAULT_BENCHMARK}).",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Full task id including seed, e.g. webarena.201_23.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Task name without seed, used together with --seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Task seed, used together with --task-name.",
    )
    parser.add_argument(
        "--source-exp-dir",
        default=None,
        help="Optional source trajectory dir; if set, infer the task id from exp_args.pkl.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all task IDs for the selected benchmark and exit.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run browser headless (default: True).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max environment steps for the selected task (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help=(
            "Episode timeout in seconds for the selected task "
            f"(default: {DEFAULT_TASK_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--avg-step-timeout",
        type=int,
        default=DEFAULT_AVG_STEP_TIMEOUT,
        help=f"Study avg_step_timeout in seconds (default: {DEFAULT_AVG_STEP_TIMEOUT}).",
    )
    parser.add_argument(
        "--state-view",
        choices=["axtree", "pruned_html", "both"],
        default="both",
        help="Which state text to print each step.",
    )
    parser.add_argument(
        "--show-goal-each-step",
        action="store_true",
        help="Print goal text every step instead of only step 0.",
    )
    parser.add_argument(
        "--dump-screenshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save current-step screenshots for operator reference (default: True).",
    )
    parser.add_argument(
        "--screenshot-dir",
        default="/tmp/agentlab_manual_previews",
        help="Directory root for manual preview screenshots.",
    )
    parser.add_argument(
        "--max-state-chars",
        type=int,
        default=12_000,
        help="Max chars to print for each large state text field (default: 12000).",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=DEFAULT_VIEWPORT_WIDTH,
        help=f"Viewport width (default: {DEFAULT_VIEWPORT_WIDTH}).",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=DEFAULT_VIEWPORT_HEIGHT,
        help=f"Viewport height (default: {DEFAULT_VIEWPORT_HEIGHT}).",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record browser video during the manual run (default: False).",
    )
    parser.add_argument(
        "--pre-observation-delay",
        type=float,
        default=None,
        help="Override pre_observation_delay for all selected tasks.",
    )
    parser.add_argument(
        "--reddit-pre-observation-delay",
        type=float,
        default=DEFAULT_REDDIT_PRE_OBSERVATION_DELAY,
        help=(
            "Override pre_observation_delay for reddit tasks "
            f"(default: {DEFAULT_REDDIT_PRE_OBSERVATION_DELAY})."
        ),
    )
    parser.add_argument(
        "--ignore-dependencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ignore WebArena task dependencies for this single-task manual run "
            "(default: True)."
        ),
    )
    parser.add_argument(
        "--metadata-json-path",
        default=str(DEFAULT_WEBARENA_METADATA_JSON),
        help="WebArena metadata JSON used for site-aware delay logic.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    _print_wa_env_warning()
    args = _parse_args()

    benchmark = bgym.DEFAULT_BENCHMARKS[args.benchmark]()
    env_args_list = list(benchmark.env_args_list)
    metadata_by_task_id = _load_webarena_metadata_by_task_id(args.metadata_json_path)

    if args.list_tasks:
        _print_available_task_ids(env_args_list)
        return

    provided = [args.task_id is not None, args.source_exp_dir is not None, args.task_name is not None]
    if sum(provided) > 1:
        raise SystemExit(
            "Provide only one of --task-id, --source-exp-dir, or --task-name/--seed."
        )

    target_task_id = None
    if args.task_id:
        target_task_id = args.task_id.strip()
    elif args.source_exp_dir:
        target_task_id = _task_id_from_source_exp_dir(args.source_exp_dir)
    elif args.task_name is not None and args.seed is not None:
        target_task_id = f"{args.task_name}_{args.seed}"

    if target_task_id is None:
        raise SystemExit(
            "Please provide either --task-id, --source-exp-dir, or both --task-name and --seed. "
            "Use --list-tasks to inspect valid IDs."
        )

    filtered = [env_args for env_args in env_args_list if _full_task_id(env_args) == target_task_id]
    if not filtered:
        print(f"Task ID not found: {target_task_id}\n")
        _print_available_task_ids(env_args_list)
        raise SystemExit(1)

    benchmark.env_args_list = filtered[:1]

    if install_webarena_html_evaluator_patch():
        print("Installed AgentLab WebArena HTML evaluator patch.")
    if install_webarena_string_evaluator_patch():
        print("Installed AgentLab WebArena string evaluator patch.")

    for env_args in benchmark.env_args_list:
        env_args.headless = args.headless
        env_args.max_steps = args.max_steps
        env_args.record_video = args.record_video
        env_args.wait_for_user_message = False
        env_args.viewport = {"width": args.viewport_width, "height": args.viewport_height}
        task_num = _task_number(env_args.task_name)
        if args.pre_observation_delay is not None:
            env_args.pre_observation_delay = args.pre_observation_delay
        elif (
            args.reddit_pre_observation_delay is not None
            and task_num is not None
            and "reddit" in metadata_by_task_id.get(task_num, {}).get("sites", [])
        ):
            env_args.pre_observation_delay = args.reddit_pre_observation_delay

    manual_agent = ManualActionAgentArgs(
        state_view=args.state_view,
        show_goal_each_step=args.show_goal_each_step,
        dump_screenshot=args.dump_screenshot,
        screenshot_dir=args.screenshot_dir,
        max_state_chars=args.max_state_chars,
    )

    print(f"Running interactive manual WebArena task: {target_task_id}")
    print(f"Benchmark: {args.benchmark}")
    print(
        "Execution mode forced to n_jobs=1, parallel_backend=sequential, "
        f"ignore_dependencies={args.ignore_dependencies}."
    )
    if args.source_exp_dir:
        print(f"Reference source trajectory: {Path(args.source_exp_dir).expanduser().resolve()}")

    study = make_study(
        benchmark=benchmark,
        agent_args=[manual_agent],
        comment="manual interactive WebArena trajectory recovery",
        ignore_dependencies=args.ignore_dependencies,
    )
    study.avg_step_timeout = args.avg_step_timeout
    for exp_args in study.exp_args_list:
        exp_args.episode_timeout = args.task_timeout_seconds

    study.run(
        n_jobs=1,
        parallel_backend="sequential",
        strict_reproducibility=False,
        n_relaunch=1,
    )


if __name__ == "__main__":
    main()
