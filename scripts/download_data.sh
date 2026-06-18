#!/usr/bin/env bash
#
# Download the externally hosted data bundle required to run trajwarp:
#
#   1. startingTrajectories.tar.gz  -- the demonstration trajectories (~533 MB),
#      extracted to ./startingTrajectories/
#   2. kitchen_assets.tar.gz        -- vendored RoboCasa/RoboSuite meshes and
#      textures for the kitchen scene (~146 MB), extracted to ./meshes/
#
# Both are gitignored. The defaults point at this repo's GitHub Releases; upload
# the bundles there (or set the env vars) once they are hosted.
#
# Usage:
#   bash scripts/download_data.sh
#   STARTING_TRAJECTORIES_URL=https://... KITCHEN_ASSETS_URL=https://... \
#       bash scripts/download_data.sh
#
# If you don't have curl/wget, use the Python fallback:
#   python scripts/download_data.py
set -euo pipefail

cd "$(dirname "$0")/.."

# Defaults assume the bundles are attached to a GitHub Release of this repo.
RELEASE_BASE="${RELEASE_BASE:-https://github.com/hyojp13/trajectoryRetargeting/releases/download/v1.0.0}"
STARTING_TRAJECTORIES_URL="${STARTING_TRAJECTORIES_URL:-${RELEASE_BASE}/startingTrajectories.tar.gz}"
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

echo "Done. Data is in ./startingTrajectories/ and ./meshes/{robocasa,robosuite}/."
