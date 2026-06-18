#!/usr/bin/env python
"""Cross-platform fallback for scripts/download_data.sh (no curl/wget needed).

Downloads and extracts the externally hosted data bundle:

  * startingTrajectories.tar.gz -> ./startingTrajectories/   (~533 MB)
  * kitchen_assets.tar.gz       -> ./meshes/{robocasa,robosuite}/  (~146 MB)

Both are gitignored. The defaults point at this repo's GitHub Releases; upload
the bundles there (or set the STARTING_TRAJECTORIES_URL / KITCHEN_ASSETS_URL
environment variables) once they are hosted.

Usage:
    python scripts/download_data.py
"""
import os
import sys
import tarfile
import tempfile
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Defaults assume the bundles are attached to a GitHub Release of this repo.
RELEASE_BASE = os.environ.get(
    "RELEASE_BASE",
    "https://github.com/hyojp13/trajectoryRetargeting/releases/download/v1.0.0")
STARTING_TRAJECTORIES_URL = os.environ.get(
    "STARTING_TRAJECTORIES_URL",
    f"{RELEASE_BASE}/startingTrajectories.tar.gz")
KITCHEN_ASSETS_URL = os.environ.get(
    "KITCHEN_ASSETS_URL",
    f"{RELEASE_BASE}/kitchen_assets.tar.gz")


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    sys.stdout.write(f"\r  {pct:5.1f}%  ({downloaded // (1024 * 1024)} MB)")
    sys.stdout.flush()


def fetch_and_extract(url, dest_dir, marker):
    """Download ``url`` and extract it into ``dest_dir`` unless ``marker`` exists."""
    if os.path.exists(marker):
        print(f"{os.path.relpath(marker, REPO_ROOT)} already present, skipping.")
        return
    os.makedirs(dest_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, tmp_path, _progress)
        print("\nExtracting ...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    fetch_and_extract(
        STARTING_TRAJECTORIES_URL,
        REPO_ROOT,
        os.path.join(REPO_ROOT, "startingTrajectories"))
    fetch_and_extract(
        KITCHEN_ASSETS_URL,
        os.path.join(REPO_ROOT, "meshes"),
        os.path.join(REPO_ROOT, "meshes", "robocasa"))
    print("Done. Data is in ./startingTrajectories/ and ./meshes/{robocasa,robosuite}/.")


if __name__ == "__main__":
    main()
