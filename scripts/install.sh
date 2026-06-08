#!/usr/bin/env bash
# install.sh — install mianmi-headless in a harbor container.
#
# harbor calls this from BaseInstalledAgent.install(). It does:
#   1. Update pip + install the package (from PyPI or a git URL).
#   2. Verify the entry point works.
#
# The caller can override the install source with MIANMI_HEADLESS_REPO
# (default: install from the current directory / mounted source).

set -euo pipefail

REPO="${MIANMI_HEADLESS_REPO:-.}"
EXTRA_INDEX_URL="${MIANMI_HEADLESS_INDEX_URL:-}"
PINNED_VERSION="${MIANMI_HEADLESS_VERSION:-}"

echo "[mianmi-headless] installing from: $REPO"

# Upgrade pip first so PEP 517 builds work cleanly.
python3 -m pip install --upgrade pip wheel setuptools

# Install the package.
if [[ -n "$PINNED_VERSION" ]]; then
  echo "[mianmi-headless] installing pinned version: $PINNED_VERSION"
  python3 -m pip install \
    ${EXTRA_INDEX_URL:+--extra-index-url "$EXTRA_INDEX_URL"} \
    "mianmi-headless==$PINNED_VERSION"
else
  if [[ -d "$REPO" && -f "$REPO/pyproject.toml" ]]; then
    echo "[mianmi-headless] installing from local source: $REPO"
    python3 -m pip install -e "$REPO"
  else
    echo "[mianmi-headless] installing from PyPI"
    python3 -m pip install \
      ${EXTRA_INDEX_URL:+--extra-index-url "$EXTRA_INDEX_URL"} \
      "mianmi-headless"
  fi
fi

# Verify the entry point is callable.
mianmi-headless version
echo "[mianmi-headless] install OK"
