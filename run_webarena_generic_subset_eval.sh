#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_webarena_generic_subset_eval.sh <host> [extra args for python script]
#
# Example:
#   ./run_webarena_generic_subset_eval.sh 10.0.0.12 --task-id-range 1-100 --limit 40 --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

HOST="${1:-${WA_HOST:-}}"
if [[ -z "${HOST}" ]]; then
  echo "Missing host."
  echo "Usage: $0 <host> [extra args for python script]"
  echo "Or set WA_HOST and run without the first positional argument."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  shift
fi

# Save AgentLab outputs under repo-local web/agentlab_results by default.
export AGENTLAB_EXP_ROOT="${AGENTLAB_EXP_ROOT:-${SCRIPT_DIR}/../../web/agentlab_results}"
mkdir -p "${AGENTLAB_EXP_ROOT}"

# Ports aligned with:
#   web/explorations/20251003_webarena_env_prep/00_vars.sh
SHOPPING_PORT="${SHOPPING_PORT:-9082}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-9083}"
REDDIT_PORT="${REDDIT_PORT:-9080}"
GITLAB_PORT="${GITLAB_PORT:-9001}"
WIKIPEDIA_PORT="${WIKIPEDIA_PORT:-9081}"
MAP_PORT="${MAP_PORT:-9443}"
HOMEPAGE_PORT="${HOMEPAGE_PORT:-9090}"
RESET_PORT="${RESET_PORT:-9565}"

# WebArena environment URLs
export WA_SHOPPING="http://${HOST}:${SHOPPING_PORT}"
export WA_SHOPPING_ADMIN="http://${HOST}:${SHOPPING_ADMIN_PORT}/admin"
export WA_REDDIT="http://${HOST}:${REDDIT_PORT}/forums/all"
export WA_GITLAB="http://${HOST}:${GITLAB_PORT}/explore"
export WA_WIKIPEDIA="http://${HOST}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://${HOST}:${MAP_PORT}"
export WA_HOMEPAGE="http://${HOST}:${HOMEPAGE_PORT}"
# Disable WebArena full-reset endpoint for eval runs unless you re-export it manually.
# unset WA_FULL_RESET

# Runner defaults (override via env if desired)
export WEBARENA_BENCHMARK="${WEBARENA_BENCHMARK:-webarena}"
export WEBARENA_MODEL_NAME="${WEBARENA_MODEL_NAME:-openai/gpt-5-mini-2025-08-07}"
export WEBARENA_REASONING_EFFORT="${WEBARENA_REASONING_EFFORT:-high}"
export WEBARENA_N_JOBS="${WEBARENA_N_JOBS:-5}"
export WEBARENA_MAX_STEPS="${WEBARENA_MAX_STEPS:-50}"
export WEBARENA_TASK_TIMEOUT_SECONDS="${WEBARENA_TASK_TIMEOUT_SECONDS:-3000}"
export WEBARENA_PARALLEL_BACKEND="${WEBARENA_PARALLEL_BACKEND:-ray}"
export WEBARENA_HEADLESS="${WEBARENA_HEADLESS:-true}"
export WEBARENA_TASK_ID_RANGE="1-200"  # inclusive

# Optional metadata-based subset selector. Example:
export WEBARENA_START_URL_FILTERS='["__SHOP__"]'  # '["__GITLAB__", "__SHOP__"]'
EXTRA_ARGS=("$@")
HAS_START_URL_FILTER_FLAG=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--start-url-filters" || "${arg}" == --start-url-filters=* ]]; then
    HAS_START_URL_FILTER_FLAG=1
    break
  fi
done

if [[ -n "${WEBARENA_START_URL_FILTERS:-}" && "${HAS_START_URL_FILTER_FLAG}" -eq 0 ]]; then
  EXTRA_ARGS+=(--start-url-filters "${WEBARENA_START_URL_FILTERS}")
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" main_webarena_generic_subset_eval.py "${EXTRA_ARGS[@]}"
