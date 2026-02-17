"""
Run GenericAgent on an environment-driven subset of WebArena tasks.

This script is designed for remote/shared WebArena setups where task selection
is often controlled through environment variables.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
from copy import deepcopy
from pathlib import Path

# Keep runtime behavior aligned with local workspace conventions.
os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)
os.environ.setdefault(
    "AGENTLAB_EXP_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "agentlab_results"),
)

import bgym

from agentlab.agents.generic_agent.agent_configs import AGENT_GPT5_MINI
from agentlab.experiments.loop import log_reasoning_effort_reminder
from agentlab.experiments.study import make_study
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

DEFAULT_BENCHMARK = "webarena"
DEFAULT_MODEL_NAME = "openai/gpt-5-mini-2025-08-07"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_N_JOBS = 4
DEFAULT_PARALLEL_BACKEND = "ray"
DEFAULT_MAX_STEPS = 30
DEFAULT_TASK_TIMEOUT_SECONDS = 50 * 60

MODEL_NAME_ALIASES = {
    "gpt-5-mini": "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-5-mini": "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-5-2": "openai/gpt-5.2",
    "gpt-5-2": "openai/gpt-5.2",
    "gpt-5.2": "openai/gpt-5.2",
}

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

START_URL_FILTER_ALIASES = {
    # Convenience alias used in local workflows.
    "__SHOP__": ("__SHOPPING__", "__SHOPPING_ADMIN__"),
}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return _str2bool(value)


def _str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "t", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _full_task_id(env_args) -> str:
    task_seed = getattr(env_args, "task_seed", None)
    if task_seed is None:
        return env_args.task_name
    return f"{env_args.task_name}_{task_seed}"


def _task_number(task_name: str) -> int | None:
    # WebArena task names are usually of form "webarena.<id>".
    match = re.search(r"(\d+)$", task_name)
    if match is None:
        return None
    return int(match.group(1))


def _parse_id_range(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    raw = raw.strip()
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
    if match is None:
        raise SystemExit(
            f"Invalid --task-id-range {raw!r}. Expected format: START-END (e.g., 1-50)."
        )
    start = int(match.group(1))
    end = int(match.group(2))
    if end < start:
        raise SystemExit(f"Invalid --task-id-range {raw!r}: end must be >= start.")
    return start, end


def _load_task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_ids_json and args.task_ids:
        raise SystemExit("Use only one of --task-ids-json or --task-ids.")

    if args.task_ids_json:
        with open(args.task_ids_json, "r", encoding="utf-8") as f:
            ids = json.load(f)
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise SystemExit("--task-ids-json must contain a JSON list of strings.")
        return [item.strip() for item in ids if item.strip()]

    if args.task_ids:
        return [item.strip() for item in args.task_ids.split(",") if item.strip()]

    return []


def _parse_start_url_filters(raw: str | None) -> list[str]:
    if raw is None:
        return []
    stripped = raw.strip()
    if not stripped:
        return []

    values: list[str]
    if stripped.startswith("["):
        parsed = None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise SystemExit(
                "--start-url-filters must be a list of strings, e.g. "
                '\'["__GITLAB__", "__SHOP__"]\'.'
            )
        values = parsed
    else:
        values = stripped.split(",")

    filters = [item.strip() for item in values if item and item.strip()]
    return filters


def _expand_start_url_filters(filters: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in filters:
        expanded.append(token)
        for alias_token in START_URL_FILTER_ALIASES.get(token, ()):
            expanded.append(alias_token)
    # Preserve order while deduplicating.
    return list(dict.fromkeys(expanded))


def _matching_webarena_indices_from_metadata(
    metadata_json_path: str, filters: list[str]
) -> set[int]:
    try:
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except OSError as exc:
        raise SystemExit(f"Failed reading metadata JSON {metadata_json_path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid metadata JSON {metadata_json_path!r}: {exc}") from exc

    if not isinstance(metadata, list):
        raise SystemExit(f"Metadata JSON {metadata_json_path!r} must contain a list of tasks.")

    expanded_filters = _expand_start_url_filters(filters)
    matched_indices: set[int] = set()
    for idx, item in enumerate(metadata):
        if not isinstance(item, dict):
            continue
        start_url = item.get("start_url", "")
        if isinstance(start_url, str) and any(token in start_url for token in expanded_filters):
            matched_indices.add(idx)

    print(
        "Metadata start_url filter "
        f"{filters} -> expanded {expanded_filters}: matched {len(matched_indices)} indices"
    )
    return matched_indices


def _print_wa_env_warning() -> None:
    missing = [key for key in REQUIRED_WA_ENV_VARS if not os.environ.get(key)]
    if missing:
        print(
            "Warning: missing WebArena env vars: "
            + ", ".join(missing)
            + ". If your instance is not configured elsewhere, runs may fail."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-5-mini GenericAgent on a filtered WebArena subset."
    )
    parser.add_argument(
        "--benchmark",
        default=os.environ.get("WEBARENA_BENCHMARK", DEFAULT_BENCHMARK),
        help="Benchmark key in bgym.DEFAULT_BENCHMARKS (default: webarena).",
    )
    parser.add_argument(
        "--task-ids-json",
        default=os.environ.get("WEBARENA_TASK_IDS_JSON"),
        help="Path to JSON list of full task IDs.",
    )
    parser.add_argument(
        "--task-ids",
        default=os.environ.get("WEBARENA_TASK_IDS"),
        help="Comma-separated full task IDs.",
    )
    parser.add_argument(
        "--task-name-glob",
        default=os.environ.get("WEBARENA_TASK_NAME_GLOB"),
        help='Glob for task_name, e.g. "webarena.1*"',
    )
    parser.add_argument(
        "--task-id-range",
        default=os.environ.get("WEBARENA_TASK_ID_RANGE"),
        help='Numeric WebArena id range "START-END", e.g. "1-100".',
    )
    parser.add_argument(
        "--start-url-filters",
        default=os.environ.get("WEBARENA_START_URL_FILTERS"),
        help=(
            "List or comma-separated start_url substrings from webarena.test.raw.json, "
            "e.g. '[\"__GITLAB__\", \"__SHOP__\"]' or '__GITLAB__,__SHOP__'."
        ),
    )
    parser.add_argument(
        "--metadata-json-path",
        default=os.environ.get("WEBARENA_METADATA_JSON", str(DEFAULT_WEBARENA_METADATA_JSON)),
        help="Path to WebArena metadata JSON (default: web/data/webarena.test.raw.json).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=_env_int("WEBARENA_OFFSET", 0),
        help="Skip this many tasks after filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int("WEBARENA_LIMIT", 0),
        help="Max number of tasks to keep after offset (0 means no cap).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=_env_int("WEBARENA_NUM_SHARDS"),
        help="Shard count for deterministic partitioning (optional).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=_env_int("WEBARENA_SHARD_INDEX"),
        help="0-based shard index when --num-shards is set.",
    )
    parser.add_argument(
        "--allow-full-benchmark",
        type=_str2bool,
        default=_env_bool("WEBARENA_ALLOW_FULL_BENCHMARK", False),
        help="Allow running full benchmark when no subset selector is given.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("WEBARENA_MODEL_NAME", DEFAULT_MODEL_NAME),
        help="Model key from CHAT_MODEL_ARGS_DICT.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.environ.get("WEBARENA_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        help="Reasoning effort for GPT-5-family models.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_env_int("WEBARENA_N_JOBS", DEFAULT_N_JOBS),
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--parallel-backend",
        default=os.environ.get("WEBARENA_PARALLEL_BACKEND", DEFAULT_PARALLEL_BACKEND),
        help='Parallel backend ("ray", "sequential", "joblib").',
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_env_int("WEBARENA_MAX_STEPS", DEFAULT_MAX_STEPS),
        help="Max steps per task.",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=_env_int("WEBARENA_TASK_TIMEOUT_SECONDS", DEFAULT_TASK_TIMEOUT_SECONDS),
        help="Per-task wall-clock timeout (<=0 disables).",
    )
    parser.add_argument(
        "--avg-step-timeout",
        type=int,
        default=_env_int("WEBARENA_AVG_STEP_TIMEOUT", 1200),
        help="Study-level avg_step_timeout used by Ray cancellation logic.",
    )
    parser.add_argument(
        "--headless",
        type=_str2bool,
        default=_env_bool("WEBARENA_HEADLESS", True),
        help="Run browser headless.",
    )
    parser.add_argument(
        "--ignore-dependencies",
        type=_str2bool,
        default=_env_bool("WEBARENA_IGNORE_DEPENDENCIES", False),
        help="Ignore WebArena task dependency graph for higher parallelism.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected tasks; do not launch study.",
    )
    return parser.parse_args()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    _print_wa_env_warning()

    args = _parse_args()
    task_ids = _load_task_ids(args)
    task_id_range = _parse_id_range(args.task_id_range)
    start_url_filters = _parse_start_url_filters(args.start_url_filters)

    has_selector = any(
        (
            bool(task_ids),
            bool(args.task_name_glob),
            task_id_range is not None,
            bool(start_url_filters),
            (args.num_shards is not None and args.shard_index is not None),
            args.offset > 0,
            args.limit > 0,
        )
    )
    if not has_selector and not args.allow_full_benchmark:
        raise SystemExit(
            "No subset selector provided. Set one of --task-ids-json/--task-ids/"
            "--task-name-glob/--task-id-range/--start-url-filters/"
            "--num-shards+--shard-index/--limit, "
            "or pass --allow-full-benchmark true."
        )

    benchmark = bgym.DEFAULT_BENCHMARKS[args.benchmark]()
    env_args_list = list(benchmark.env_args_list)
    print(f"Loaded benchmark {benchmark.name!r} with {len(env_args_list)} tasks")

    if task_ids:
        task_id_set = set(task_ids)
        env_args_list = [ea for ea in env_args_list if _full_task_id(ea) in task_id_set]
        print(f"After task ID filter: {len(env_args_list)} tasks")

    if args.task_name_glob:
        env_args_list = [
            ea
            for ea in env_args_list
            if fnmatch.fnmatch(getattr(ea, "task_name", ""), args.task_name_glob)
        ]
        print(f"After task_name glob filter: {len(env_args_list)} tasks")

    if task_id_range is not None:
        start, end = task_id_range
        env_args_list = [
            ea
            for ea in env_args_list
            if (
                (task_num := _task_number(getattr(ea, "task_name", ""))) is not None
                and start <= task_num <= end
            )
        ]
        print(f"After task ID range filter: {len(env_args_list)} tasks")

    if start_url_filters:
        matched_indices = _matching_webarena_indices_from_metadata(
            args.metadata_json_path, start_url_filters
        )
        env_args_list = [
            ea
            for ea in env_args_list
            if (
                (task_num := _task_number(getattr(ea, "task_name", ""))) is not None
                and task_num in matched_indices
            )
        ]
        print(f"After start_url metadata filter: {len(env_args_list)} tasks")

    # Stable ordering keeps sharding deterministic.
    env_args_list = sorted(env_args_list, key=_full_task_id)

    if args.num_shards is not None or args.shard_index is not None:
        if args.num_shards is None or args.shard_index is None:
            raise SystemExit("Use --num-shards and --shard-index together.")
        if args.num_shards <= 0:
            raise SystemExit("--num-shards must be > 0.")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards.")
        env_args_list = [
            ea for idx, ea in enumerate(env_args_list) if idx % args.num_shards == args.shard_index
        ]
        print(
            "After shard filter: "
            f"{len(env_args_list)} tasks (shard {args.shard_index}/{args.num_shards})"
        )

    if args.offset > 0:
        env_args_list = env_args_list[args.offset :]
        print(f"After offset={args.offset}: {len(env_args_list)} tasks")

    if args.limit > 0:
        env_args_list = env_args_list[: args.limit]
        print(f"After limit={args.limit}: {len(env_args_list)} tasks")

    if not env_args_list:
        raise SystemExit("No tasks selected after filtering.")

    model_name = MODEL_NAME_ALIASES.get(args.model_name, args.model_name)
    if model_name != args.model_name:
        print(f"Normalized model alias {args.model_name!r} -> {model_name!r}")
    if model_name not in CHAT_MODEL_ARGS_DICT:
        raise SystemExit(
            f"Unknown model {model_name!r}. Add it to CHAT_MODEL_ARGS_DICT "
            "or pass a supported key."
        )

    generic_agent = deepcopy(AGENT_GPT5_MINI)
    generic_agent.chat_model_args = deepcopy(CHAT_MODEL_ARGS_DICT[model_name])
    if model_name.startswith("openai/gpt-5"):
        generic_agent.chat_model_args.reasoning_effort = args.reasoning_effort
        print(f"Using reasoning_effort={args.reasoning_effort} for {model_name}")
    generic_agent.agent_name = f"GenericAgent-{generic_agent.chat_model_args.model_name}".replace(
        "/", "_"
    )
    log_reasoning_effort_reminder(generic_agent)

    benchmark.env_args_list = env_args_list
    for env_args in benchmark.env_args_list:
        env_args.headless = args.headless
        env_args.max_steps = args.max_steps

    print(f"Final selected tasks: {len(benchmark.env_args_list)}")
    preview = [_full_task_id(ea) for ea in benchmark.env_args_list[:20]]
    print("Task preview (up to 20):")
    for task_id in preview:
        print(f"  - {task_id}")

    if args.dry_run:
        print("Dry run only; study not launched.")
        return

    study = make_study(
        benchmark=benchmark,
        agent_args=[generic_agent],
        ignore_dependencies=args.ignore_dependencies,
        comment=f"webarena subset eval ({model_name}, reasoning={args.reasoning_effort})",
    )
    if args.task_timeout_seconds and args.task_timeout_seconds > 0:
        for exp_args in study.exp_args_list:
            exp_args.episode_timeout = args.task_timeout_seconds
        print(f"Per-task timeout set to {args.task_timeout_seconds}s")
    study.avg_step_timeout = args.avg_step_timeout
    study.run(
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
        strict_reproducibility=False,
        n_relaunch=3,
    )


if __name__ == "__main__":
    main()
