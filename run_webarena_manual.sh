#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_webarena_manual.sh <host> [args for main_webarena_manual.py]
#
# Example:
#   ./run_webarena_manual.sh localhost \
#     --source-exp-dir ../../web/agentlab_results/<study>/<trajectory>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

HOST="${1:-${WA_HOST:-}}"
if [[ -z "${HOST}" ]]; then
  echo "Missing host."
  echo "Usage: $0 <host> [args for main_webarena_manual.py]"
  echo "Or set WA_HOST and omit the first positional argument."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  shift
fi

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

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" main_webarena_manual.py "$@"
