"""Run a single WorkArena task with a terminal-driven manual agent.

This script preserves standard AgentLab output artifacts while letting a human
or coding operator provide each next action interactively.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

# Keep runtime behavior aligned with the other WorkArena convenience scripts.
os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)
os.environ.setdefault(
    "AGENTLAB_EXP_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "agentlab_results"),
)

import bgym

from agentlab.agents.manual_action_agent import ManualActionAgentArgs
from agentlab.experiments.study import make_study


def _full_task_id(env_args) -> str:
    task_seed = getattr(env_args, "task_seed", None)
    if task_seed is None:
        return env_args.task_name
    return f"{env_args.task_name}_{task_seed}"


def _print_available_task_ids(env_args_list: Iterable) -> None:
    print("Available task IDs:")
    for env_args in env_args_list:
        print(f"  - {_full_task_id(env_args)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one WorkArena task interactively with ManualActionAgent."
    )
    parser.add_argument(
        "--benchmark",
        default="workarena_l3_agent_curriculum_eval",
        help="Benchmark name in bgym.DEFAULT_BENCHMARKS (default: workarena_l3_agent_curriculum_eval).",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Full task id including seed, e.g. workarena.servicenow.some-task_123.",
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
        default=50,
        help="Max environment steps for the selected task (default: 50).",
    )
    parser.add_argument(
        "--avg-step-timeout",
        type=int,
        default=1200,
        help="Study avg_step_timeout in seconds (default: 1200).",
    )
    parser.add_argument(
        "--state-view",
        choices=["axtree", "pruned_html", "both"],
        default="axtree",
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
    return parser.parse_args()


def main():
    args = _parse_args()

    benchmark = bgym.DEFAULT_BENCHMARKS[args.benchmark]()
    env_args_list = list(benchmark.env_args_list)

    if args.list_tasks:
        _print_available_task_ids(env_args_list)
        return

    target_task_id = None
    if args.task_id:
        target_task_id = args.task_id
    elif args.task_name is not None and args.seed is not None:
        target_task_id = f"{args.task_name}_{args.seed}"

    if target_task_id is None:
        raise SystemExit(
            "Please provide either --task-id, or both --task-name and --seed. "
            "Use --list-tasks to inspect valid IDs."
        )

    filtered = [env_args for env_args in env_args_list if _full_task_id(env_args) == target_task_id]
    if not filtered:
        print(f"Task ID not found: {target_task_id}\n")
        _print_available_task_ids(env_args_list)
        raise SystemExit(1)

    # Single-task mode by design.
    benchmark.env_args_list = filtered[:1]

    for env_args in benchmark.env_args_list:
        env_args.headless = args.headless
        env_args.max_steps = args.max_steps

    manual_agent = ManualActionAgentArgs(
        state_view=args.state_view,
        show_goal_each_step=args.show_goal_each_step,
        dump_screenshot=args.dump_screenshot,
        screenshot_dir=args.screenshot_dir,
        max_state_chars=args.max_state_chars,
    )

    print(f"Running interactive manual task: {target_task_id}")
    print(f"Benchmark: {args.benchmark}")
    print("Execution mode forced to n_jobs=1, parallel_backend=sequential.")

    study = make_study(
        benchmark=benchmark,
        agent_args=[manual_agent],
        comment="manual interactive trajectory",
    )
    study.avg_step_timeout = args.avg_step_timeout
    study.run(
        n_jobs=1,
        parallel_backend="sequential",
        strict_reproducibility=False,
        n_relaunch=1,
    )


if __name__ == "__main__":
    main()
