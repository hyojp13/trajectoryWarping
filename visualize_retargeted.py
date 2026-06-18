#!/usr/bin/env python
"""Visualize a saved RETARGETED trajectory for a config.

Opens an interactive MuJoCo viewer playing back the hand/object trajectory that
``retarget.py`` saved under ``final_trajectories_<threshold>/``. No computation
is performed; the trajectory is loaded from disk.

Usage:
    python visualize_retargeted.py retargeting_configs/mug_pass_wall.json
    python visualize_retargeted.py retargeting_configs/mug_pass_wall.json \\
        --trajectory-dir final_trajectories_0.0035
"""
import argparse

from trajwarp.viz.viewer import play_saved


def main():
    parser = argparse.ArgumentParser(
        description="Replay a saved retargeted trajectory referenced by a config file.")
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--trajectory-dir', type=str, default='final_trajectories_0.0035',
                        help='Directory containing the saved <config>_hand.npy / _object.npy')
    args = parser.parse_args()

    play_saved(args.config, trajectory_dir=args.trajectory_dir)


if __name__ == "__main__":
    main()
