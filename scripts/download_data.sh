#!/usr/bin/env bash
#
# Download the externally hosted data bundles:
#
#   1. kitchen_assets.tar.gz        -- vendored RoboCasa/RoboSuite meshes and
#      textures for the kitchen scene (~146 MB), extracted to ./meshes/
#   2. startingTrajectories.tar.gz  -- GRAB-derived demonstrations, extracted
#      to ./startingTrajectories/ (only if STARTING_TRAJECTORIES_URL is set)
#
# The demonstrations are not hosted publicly because the GRAB and MANO licenses
# don't allow it. If you have a licensed copy, point STARTING_TRAJECTORIES_URL
# at it. The kitchen assets come from this repo's GitHub release unless
# KITCHEN_ASSETS_URL is set.
#
# Usage:
#   bash scripts/download_data.sh
#   STARTING_TRAJECTORIES_URL=https://... bash scripts/download_data.sh
#
# If you don't have curl/wget, use the Python fallback:
#   python scripts/download_data.py
set -euo pipefail

cd "$(dirname "$0")/.."

RELEASE_BASE="${RELEASE_BASE:-https://github.com/hyojp13/trajectoryWarping/releases/download/v1.0.0}"
STARTING_TRAJECTORIES_URL="${STARTING_TRAJECTORIES_URL:-}"
KITCHEN_ASSETS_URL="${KITCHEN_ASSETS_URL:-${RELEASE_BASE}/kitchen_assets.tar.gz}"

fetch() {
  # fetch <url> <output_path>
  local url="$1" out="$2"
  echo "Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
  else
    echo "ERROR: neither curl nor wget is available. Use: python scripts/download_data.py" >&2
    exit 1
  fi
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

if [ -d startingTrajectories ] && [ -n "$(ls -A startingTrajectories 2>/dev/null)" ]; then
  echo "startingTrajectories/ already present, skipping."
elif [ -z "${STARTING_TRAJECTORIES_URL}" ]; then
  echo "STARTING_TRAJECTORIES_URL is not set, skipping demonstrations (see the Data section of the README)."
else
  fetch "${STARTING_TRAJECTORIES_URL}" "${tmpdir}/startingTrajectories.tar.gz"
  echo "Extracting startingTrajectories.tar.gz ..."
  tar -xzf "${tmpdir}/startingTrajectories.tar.gz" -C .
fi

if [ -d meshes/robocasa ] && [ -d meshes/robosuite ]; then
  echo "Kitchen assets already present, skipping."
else
  fetch "${KITCHEN_ASSETS_URL}" "${tmpdir}/kitchen_assets.tar.gz"
  echo "Extracting kitchen_assets.tar.gz ..."
  mkdir -p meshes
  tar -xzf "${tmpdir}/kitchen_assets.tar.gz" -C meshes
fi

echo "Done."
