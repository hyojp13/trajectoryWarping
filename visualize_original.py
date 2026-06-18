#!/usr/bin/env python
"""Visualize the ORIGINAL demonstration for a config (before any warping).

Opens an interactive MuJoCo viewer playing the source hand/object trajectory
with the contact correspondences overlaid. Useful for inspecting a demonstration
before retargeting it.

Usage:
    python visualize_original.py retargeting_configs/mug_pass_wall.json
"""
import argparse

from trajwarp.viz.viewer import play_demonstration


def main():
    parser = argparse.ArgumentParser(
        description="Replay the original demonstration referenced by a config file.")
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()

    play_demonstration(args.config)


if __name__ == "__main__":
    main()
