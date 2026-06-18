#!/usr/bin/env python
"""Naive linear-spline trajectory retargeting baseline.

Identical object warping to ``retarget.py``, but the hand is recovered by
keyframing the exact rigid hand transform at each input contact frame and
interpolating between keyframes with a cubic B-spline. This exposes SE(3)
interpolation non-equivariance. Provided as a comparison baseline in the paper.

Usage:
    python retarget_naive_spline.py retargeting_configs/mug_pass_wall.json
    python retarget_naive_spline.py retargeting_configs/mug_pass_wall.json --no-view

See ``retargeting_configs/schema.json`` for the configuration format.
"""
import argparse

from trajwarp.pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Run the naive linear-spline trajectory-retargeting baseline from a config file.")
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--initial', action='store_true',
                        help='Load initial data from initial_trajectories/ instead of startingTrajectories/')
    parser.add_argument('--no-view', action='store_true',
                        help='Skip the interactive viewer after retargeting')
    parser.add_argument('--loss-threshold', type=float, default=None,
                        help='Override the contact loss threshold (and output directory)')
    args = parser.parse_args()

    run(args.config, strategy="naive_spline", initial=args.initial,
        view=not args.no_view, loss_threshold=args.loss_threshold)


if __name__ == "__main__":
    main()
