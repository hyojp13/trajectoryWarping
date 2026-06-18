#!/usr/bin/env python
"""Contact-based trajectory retargeting (our method).

Repurposes a contact-rich manipulation demonstration to new spatial waypoints,
environmental barriers, terminal configurations, and temporal constraints, then
recovers the hand trajectory by optimizing for the original hand-object contacts.

Usage:
    python retarget.py retargeting_configs/mug_pass_wall.json
    python retarget.py retargeting_configs/mug_pass_wall.json --no-view

See ``retargeting_configs/schema.json`` for the configuration format.
"""
import argparse

from trajwarp.pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Run contact-based trajectory retargeting from a config file.")
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--initial', action='store_true',
                        help='Load initial data from initial_trajectories/ instead of startingTrajectories/')
    parser.add_argument('--no-view', action='store_true',
                        help='Skip the interactive viewer after retargeting')
    parser.add_argument('--loss-threshold', type=float, default=None,
                        help='Override the contact loss threshold (and output directory)')
    args = parser.parse_args()

    run(args.config, strategy="contact_based", initial=args.initial,
        view=not args.no_view, loss_threshold=args.loss_threshold)


if __name__ == "__main__":
    main()
