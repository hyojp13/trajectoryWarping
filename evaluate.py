#!/usr/bin/env python
"""Evaluate saved retargeting results and build the paper's comparison tables.

Reads the artifacts ``retarget.py`` writes under ``final_trajectories_<threshold>/``
(``*_hand.npy``, ``*_object.npy``, ``*_metrics.npy``, ``*_timewarp.npy``,
``*_data.txt``) and reports, per trajectory: object-to-wrist distance, contact
distance (avg/max), contact-window length, object/hand path-length ratios vs the
initial demonstration, temporal-warp RMS discrepancy, and retargeting time.

Usage:
    # Evaluate specific trajectories in the default results directory
    python evaluate.py mug_pass_wall teapot_pour_cups

    # Evaluate every config, write Markdown + CSV tables
    python evaluate.py --all --dir final_trajectories_0.0035 \\
        --markdown results/table.md --csv results/table.csv

    # Evaluate the naive baselines (saved with the _naive suffix)
    python evaluate.py --all --suffix _naive
"""
import argparse
import glob
import os

from trajwarp.viz.metrics import (
    collect_metrics,
    results_to_csv,
    results_to_markdown,
)


def _all_config_names(configs_dir='retargeting_configs'):
    names = []
    for path in sorted(glob.glob(os.path.join(configs_dir, '*.json'))):
        base = os.path.splitext(os.path.basename(path))[0]
        if base == 'schema':
            continue
        names.append(base)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trajectories', nargs='*',
                        help='Trajectory (config) names to evaluate')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate every config in retargeting_configs/')
    parser.add_argument('--dir', '-d', default='final_trajectories_0.0035',
                        help='Directory with saved retargeted trajectories')
    parser.add_argument('--initial-dir', default='initial_trajectories',
                        help='Directory with initial trajectories (for path-length ratios)')
    parser.add_argument('--suffix', default='',
                        help="Filename suffix of the saved run (e.g. '_naive' for baselines)")
    parser.add_argument('--markdown', '-m', default=None,
                        help='Write the table to this Markdown file')
    parser.add_argument('--csv', default=None,
                        help='Write the table to this CSV file')
    args = parser.parse_args()

    names = _all_config_names() if args.all else args.trajectories
    if not names:
        parser.error('Provide trajectory names or pass --all.')

    results = collect_metrics(names, data_dir=args.dir,
                              initial_dir=args.initial_dir, suffix=args.suffix)

    markdown = results_to_markdown(results)
    print(markdown)

    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, 'w') as f:
            f.write(markdown)
        print(f"Wrote Markdown table to {args.markdown}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, 'w') as f:
            f.write(results_to_csv(results))
        print(f"Wrote CSV table to {args.csv}")


if __name__ == "__main__":
    main()
