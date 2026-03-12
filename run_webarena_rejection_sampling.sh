#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_webarena_rejection_sampling.sh <host> [extra args for python script]
#
# Example:
#   ./run_webarena_rejection_sampling.sh 10.0.0.12 --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Keep older Chromium revisions from being garbage-collected when multiple
# Playwright versions coexist on the same machine.
export PLAYWRIGHT_SKIP_BROWSER_GC="${PLAYWRIGHT_SKIP_BROWSER_GC:-1}"
# Isolate this harness from global ~/.cache/ms-playwright churn.
PLAYWRIGHT_BROWSERS_PATH_DEFAULT="${HOME}/.cache/ms-playwright-agentlab-pw144"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-${PLAYWRIGHT_BROWSERS_PATH_DEFAULT}}"
mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}"

if ! PLAYWRIGHT_CHECK_OUTPUT="$("${PYTHON_BIN}" - <<'PY' 2>&1
import os
import sys
from importlib import metadata

try:
    import playwright
    from playwright.sync_api import sync_playwright
except Exception as exc:
    print(f"PLAYWRIGHT_IMPORT_ERROR={exc}")
    raise SystemExit(2)

with sync_playwright() as pw:
    chromium_executable = pw.chromium.executable_path

print(f"PYTHON_EXECUTABLE={sys.executable}")
print(f"PLAYWRIGHT_VERSION={metadata.version('playwright')}")
print(f"PLAYWRIGHT_BROWSERS_PATH={os.environ.get('PLAYWRIGHT_BROWSERS_PATH', '')}")
print(f"CHROMIUM_EXECUTABLE={chromium_executable}")
print(f"CHROMIUM_EXISTS={int(os.path.exists(chromium_executable))}")
PY
)"; then
  echo "Failed Playwright preflight check using PYTHON_BIN=${PYTHON_BIN}:"
  echo "${PLAYWRIGHT_CHECK_OUTPUT}"
  echo
  echo "Install Playwright with the same interpreter used by this script:"
  echo "  ${PYTHON_BIN} -m pip install \"playwright==1.44.0\""
  echo "  PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH} ${PYTHON_BIN} -m playwright install chromium"
  exit 1
fi

echo "${PLAYWRIGHT_CHECK_OUTPUT}"

if [[ "${PLAYWRIGHT_CHECK_OUTPUT}" == *"CHROMIUM_EXISTS=0"* ]]; then
  echo "Missing Chromium binary for this Playwright environment."
  echo "Install it (into the isolated cache) with:"
  echo "  PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH} ${PYTHON_BIN} -m playwright install chromium"
  exit 1
fi

# HOST="${1:-${WA_HOST:-}}"
HOST=localhost
#if [[ -z "${HOST}" ]]; then
#  echo "Missing host."
#  echo "Usage: $0 <host> [extra args for python script]"
#  echo "Or set WA_HOST and run without the first positional argument."
#  exit 1
#fi

#if [[ $# -gt 0 ]]; then
#  shift
#fi

# TASK_IDS_JSON="${WEBARENA_TASK_IDS_JSON:-${SCRIPT_DIR}/../../web/agentlab_results/20260221_webarena_rejection_sampling.json}"
TASK_IDS_JSON=$1
if [[ ! -f "${TASK_IDS_JSON}" ]]; then
  echo "Task ID JSON not found: ${TASK_IDS_JSON}"
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

# WebArena environment URLs
export WA_SHOPPING="http://${HOST}:${SHOPPING_PORT}"
export WA_SHOPPING_ADMIN="http://${HOST}:${SHOPPING_ADMIN_PORT}/admin"
export WA_REDDIT="http://${HOST}:${REDDIT_PORT}/forums/all"
export WA_GITLAB="http://${HOST}:${GITLAB_PORT}/explore"
export WA_WIKIPEDIA="http://${HOST}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://${HOST}:${MAP_PORT}"
export WA_HOMEPAGE="http://${HOST}:${HOMEPAGE_PORT}"

# Runner defaults (override via env if desired)
export WEBARENA_BENCHMARK="${WEBARENA_BENCHMARK:-webarena}"
# export WEBARENA_MODEL_NAME="${WEBARENA_MODEL_NAME:-openai/gpt-5-mini-2025-08-07}"
export WEBARENA_MODEL_NAME="${WEBARENA_MODEL_NAME:-openai/gpt-5.2}"
# export WEBARENA_MODEL_NAME="${WEBARENA_MODEL_NAME:-openai/gpt-4.1-mini}"
export WEBARENA_REASONING_EFFORT="${WEBARENA_REASONING_EFFORT:-high}"
export WEBARENA_N_JOBS="${WEBARENA_N_JOBS:-5}"
export WEBARENA_MAX_STEPS="${WEBARENA_MAX_STEPS:-50}"
export WEBARENA_TASK_TIMEOUT_SECONDS="${WEBARENA_TASK_TIMEOUT_SECONDS:-3000}"
export WEBARENA_PARALLEL_BACKEND="${WEBARENA_PARALLEL_BACKEND:-ray}"
export WEBARENA_HEADLESS="${WEBARENA_HEADLESS:-true}"
export WEBARENA_IGNORE_DEPENDENCIES="${WEBARENA_IGNORE_DEPENDENCIES:-true}"
# Postmill mutations often persist slightly after the visible click state changes.
# Use a longer settle before validation for reddit tasks to reduce false negatives.
export WEBARENA_REDDIT_PRE_OBSERVATION_DELAY="${WEBARENA_REDDIT_PRE_OBSERVATION_DELAY:-4.0}"

EXTRA_ARGS=("$@")
HAS_TASK_IDS_JSON_FLAG=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--task-ids-json" || "${arg}" == --task-ids-json=* ]]; then
    HAS_TASK_IDS_JSON_FLAG=1
    break
  fi
done

if [[ "${HAS_TASK_IDS_JSON_FLAG}" -eq 0 ]]; then
  EXTRA_ARGS+=(--task-ids-json "${TASK_IDS_JSON}")
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" main_webarena_generic_subset_eval.py "${EXTRA_ARGS[@]}"
