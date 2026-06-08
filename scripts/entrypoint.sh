#!/usr/bin/env bash
# entrypoint.sh — the script harbor runs inside the container.
#
# Usage:
#   entrypoint.sh "your task description"
#   entrypoint.sh --max-iter 500 "your task description"
#
# It just wraps `mianmi-headless run` with sensible defaults for
# long-running benchmark tasks.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: entrypoint.sh [--max-iter N] [--model MODEL] <instruction>" >&2
  exit 2
fi

# Defaults tuned for hours-long benchmark runs.
MAX_ITER="${MIANMI_HEADLESS_MAX_ITER:-500}"
MODEL="${MIANMI_HEADLESS_MODEL:-gpt-5.5}"

exec mianmi-headless run \
  --max-iter "$MAX_ITER" \
  --model "$MODEL" \
  "$@"
