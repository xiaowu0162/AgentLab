#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_webarena_recorded_action_replay.sh <host> <source-study-dir> [extra args for python script]
#
# Example:
#   ./run_webarena_recorded_action_replay.sh 10.0.0.12 \
#     ../../web/agentlab_results/2026-02-16_18-13-18_genericagent-gpt-5-mini-2025-08-07-on-webarena \
#     --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

HOST="${1:-${WA_HOST:-}}"
if [[ -z "${HOST}" ]]; then
  echo "Missing host."
  echo "Usage: $0 <host> <source-study-dir> [extra args for python script]"
  echo "Or set WA_HOST and pass only the source study dir."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  shift
fi

SOURCE_STUDY_DIR="${1:-${SOURCE_STUDY_DIR:-}}"
if [[ -z "${SOURCE_STUDY_DIR}" ]]; then
  echo "Missing source study dir."
  echo "Usage: $0 <host> <source-study-dir> [extra args for python script]"
  echo "Or set SOURCE_STUDY_DIR and omit the second positional argument."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  shift
fi

export AGENTLAB_EXP_ROOT="${AGENTLAB_EXP_ROOT:-${SCRIPT_DIR}/../../web/agentlab_results_replay}"
mkdir -p "${AGENTLAB_EXP_ROOT}"

SHOPPING_PORT="${SHOPPING_PORT:-9082}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-9083}"
REDDIT_PORT="${REDDIT_PORT:-9080}"
GITLAB_PORT="${GITLAB_PORT:-9001}"
WIKIPEDIA_PORT="${WIKIPEDIA_PORT:-9081}"
MAP_PORT="${MAP_PORT:-9443}"
HOMEPAGE_PORT="${HOMEPAGE_PORT:-9090}"

export WA_SHOPPING="http://${HOST}:${SHOPPING_PORT}"
export WA_SHOPPING_ADMIN="http://${HOST}:${SHOPPING_ADMIN_PORT}/admin"
export WA_REDDIT="http://${HOST}:${REDDIT_PORT}/forums/all"
export WA_GITLAB="http://${HOST}:${GITLAB_PORT}/explore"
export WA_WIKIPEDIA="http://${HOST}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://${HOST}:${MAP_PORT}"
export WA_HOMEPAGE="http://${HOST}:${HOMEPAGE_PORT}"

export WEBARENA_REPLAY_VIEWPORT_WIDTH="${WEBARENA_REPLAY_VIEWPORT_WIDTH:-1280}"
export WEBARENA_REPLAY_VIEWPORT_HEIGHT="${WEBARENA_REPLAY_VIEWPORT_HEIGHT:-1000}"
export WEBARENA_REPLAY_N_JOBS="${WEBARENA_REPLAY_N_JOBS:-12}"
export WEBARENA_REPLAY_PARALLEL_BACKEND="${WEBARENA_REPLAY_PARALLEL_BACKEND:-ray}"
export WEBARENA_REPLAY_HEADLESS="${WEBARENA_REPLAY_HEADLESS:-true}"
export WEBARENA_REPLAY_RECORD_VIDEO="${WEBARENA_REPLAY_RECORD_VIDEO:-false}"
export WEBARENA_REPLAY_TASK_TIMEOUT_SECONDS="${WEBARENA_REPLAY_TASK_TIMEOUT_SECONDS:-3000}"
export WEBARENA_REPLAY_REDDIT_PRE_OBSERVATION_DELAY="${WEBARENA_REPLAY_REDDIT_PRE_OBSERVATION_DELAY:-4.0}"
export WEBARENA_REPLAY_AVG_STEP_TIMEOUT="${WEBARENA_REPLAY_AVG_STEP_TIMEOUT:-1200}"

EXTRA_ARGS=("$@")

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" main_webarena_recorded_action_replay.py \
  --source-study-dir "${SOURCE_STUDY_DIR}" \
  --viewport-width "${WEBARENA_REPLAY_VIEWPORT_WIDTH}" \
  --viewport-height "${WEBARENA_REPLAY_VIEWPORT_HEIGHT}" \
  --n-jobs "${WEBARENA_REPLAY_N_JOBS}" \
  --parallel-backend "${WEBARENA_REPLAY_PARALLEL_BACKEND}" \
  --headless "${WEBARENA_REPLAY_HEADLESS}" \
  --record-video "${WEBARENA_REPLAY_RECORD_VIDEO}" \
  --task-timeout-seconds "${WEBARENA_REPLAY_TASK_TIMEOUT_SECONDS}" \
  --reddit-pre-observation-delay "${WEBARENA_REPLAY_REDDIT_PRE_OBSERVATION_DELAY}" \
  --avg-step-timeout "${WEBARENA_REPLAY_AVG_STEP_TIMEOUT}" \
  "${EXTRA_ARGS[@]}"
