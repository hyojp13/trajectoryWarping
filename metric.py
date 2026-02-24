"""
Metric analysis tool for trajectory retargeting.
Supports contact distance metrics and object-to-wrist distance metrics.
"""

import argparse
import numpy as np
import os


def load_metrics(metrics_path):
    """Load contact metrics from a .npy file."""
    metrics = np.load(metrics_path, allow_pickle=True).item()
    return metrics


def load_trajectory(traj_path):
    """Load trajectory data from a .npy file."""
    return np.load(traj_path)


def compute_distance_metrics(hand_traj, object_traj, start_idx=None, end_idx=None):
    """
    Compute distance metrics between object position and wrist position.

    Args:
        hand_traj: Hand trajectory array (hand_dof, num_frames) or (num_frames, hand_dof)
        object_traj: Object trajectory array (num_frames, 7) where [:3] is position
        start_idx: Start frame index for contact period (optional)
        end_idx: End frame index for contact period (optional)

    Returns:
        dict with mean, min, max distances and per-frame distances
    """
    # Handle transposed hand trajectory - hand_traj is saved as (hand_dof, num_frames)
    # but we need (num_frames, hand_dof)
    if hand_traj.shape[0] < hand_traj.shape[1]:
        hand_traj = hand_traj.T

    # Use contact range if provided, otherwise use full trajectory
    if start_idx is not None and end_idx is not None:
        contact_range = slice(start_idx, end_idx + 1)
    else:
        contact_range = slice(None)

    # Extract object positions (first 3 dimensions)
    object_positions = object_traj[contact_range, :3]

    # Extract wrist positions (first 3 dimensions of hand trajectory)
    wrist_positions = hand_traj[contact_range, :3]

    # Compute Euclidean distances for each frame
    distances = np.linalg.norm(object_positions - wrist_positions, axis=1)

    return {
        'mean': np.mean(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'distances': distances,
        'num_frames': len(distances),
        'start_idx': start_idx,
        'end_idx': end_idx
    }


def print_metrics(metrics, name=None):
    """Print detailed metrics for a single trajectory."""
    if name:
        print(f"\n{'='*60}")
        print(f"Trajectory: {name}")
        print(f"{'='*60}")

    start_idx = metrics['start_idx']
    end_idx = metrics['end_idx']
    num_frames = end_idx - start_idx + 1

    print(f"\nContact Range: frames {start_idx} to {end_idx} ({num_frames} frames)")

    avg_distances = metrics['average_distances']
    num_contacts = metrics['num_contacts']

    # Overall statistics
    overall_avg = np.mean(avg_distances)
    overall_max = np.max(avg_distances)
    overall_min = np.min(avg_distances[avg_distances > 0]) if np.any(avg_distances > 0) else 0.0
    overall_std = np.std(avg_distances)

    print(f"\nOverall Statistics:")
    print(f"  Average distance: {overall_avg:.6f}")
    print(f"  Max distance:     {overall_max:.6f}")
    print(f"  Min distance:     {overall_min:.6f}")
    print(f"  Std deviation:    {overall_std:.6f}")

    # Per-frame details
    print(f"\nPer-Frame Details:")
    print(f"  {'Frame':<8} {'Contacts':<10} {'Avg Dist':<12} {'Max Dist':<12} {'Min Dist':<12}")
    print(f"  {'-'*54}")

    for i, frame_idx in enumerate(metrics['frame_indices']):
        per_pair = metrics['per_pair_distances'][i]
        n_contacts = num_contacts[i]
        avg_dist = avg_distances[i]

        if len(per_pair) > 0:
            max_dist = np.max(per_pair)
            min_dist = np.min(per_pair)
        else:
            max_dist = 0.0
            min_dist = 0.0

        print(f"  {frame_idx:<8} {n_contacts:<10} {avg_dist:<12.6f} {max_dist:<12.6f} {min_dist:<12.6f}")


def print_distance_metrics(metrics, name=None):
    """Print distance metrics between object and wrist."""
    if name:
        print(f"\n{'='*60}")
        print(f"Trajectory: {name}")
        print(f"{'='*60}")

    if metrics.get('start_idx') is not None and metrics.get('end_idx') is not None:
        print(f"\nContact Range: frames {metrics['start_idx']} to {metrics['end_idx']} ({metrics['num_frames']} frames)")

    print(f"\nObject-to-Wrist Distance Statistics:")
    print(f"  Mean distance: {metrics['mean']:.6f}")
    print(f"  Min distance:  {metrics['min']:.6f}")
    print(f"  Max distance:  {metrics['max']:.6f}")
    print(f"  Total frames:  {metrics['num_frames']}")


def print_summary(all_metrics):
    """Print summary comparison across multiple trajectories."""
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'Trajectory':<40} {'Avg Dist':<12} {'Min Dist':<12} {'Max Dist':<12} {'Frames':<8}")
    print(f"  {'-'*84}")

    for name, metrics in all_metrics.items():
        avg_distances = metrics['average_distances']
        overall_avg = np.mean(avg_distances)
        overall_min = np.min(avg_distances[avg_distances > 0]) if np.any(avg_distances > 0) else 0.0
        overall_max = np.max(avg_distances)
        num_frames = metrics['end_idx'] - metrics['start_idx'] + 1

        # Truncate name if too long
        display_name = name if len(name) <= 38 else name[:35] + "..."
        print(f"  {display_name:<40} {overall_avg:<12.6f} {overall_min:<12.6f} {overall_max:<12.6f} {num_frames:<8}")


def print_distance_summary(all_metrics):
    """Print summary comparison of distance metrics across multiple trajectories."""
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON - OBJECT-TO-WRIST DISTANCES")
    print(f"{'='*60}")

    print(f"\n  {'Trajectory':<40} {'Mean':<12} {'Min':<12} {'Max':<12} {'Frames':<8}")
    print(f"  {'-'*84}")

    for name, metrics in all_metrics.items():
        # Truncate name if too long
        display_name = name if len(name) <= 38 else name[:35] + "..."
        print(f"  {display_name:<40} {metrics['mean']:<12.6f} {metrics['min']:<12.6f} {metrics['max']:<12.6f} {metrics['num_frames']:<8}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze metrics from trajectory retargeting'
    )
    parser.add_argument(
        'trajectories',
        nargs='+',
        help='Trajectory names or paths to files. '
             'Can be trajectory names (e.g., "fryingpan_into_dishwasher") '
             'or full paths to trajectory files.'
    )
    parser.add_argument(
        '--metric',
        choices=['contacts', 'distance'],
        default='contacts',
        help='Metric type to compute: "contacts" for contact distances, '
             '"distance" for object-to-wrist distances (default: contacts)'
    )
    parser.add_argument(
        '--dir',
        default='final_trajectories',
        help='Directory containing trajectory files (default: final_trajectories)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary comparison, skip per-trajectory details'
    )

    args = parser.parse_args()

    all_metrics = {}

    if args.metric == 'contacts':
        # Original contacts metric behavior
        for traj in args.trajectories:
            # Determine the metrics file path
            if traj.endswith('.npy'):
                metrics_path = traj
                name = os.path.basename(traj).replace('_metrics.npy', '')
            else:
                metrics_path = os.path.join(args.dir, f"{traj}_metrics.npy")
                name = traj

            if not os.path.exists(metrics_path):
                print(f"Warning: Metrics file not found: {metrics_path}")
                continue

            try:
                metrics = load_metrics(metrics_path)
                all_metrics[name] = metrics

                if not args.summary_only:
                    print_metrics(metrics, name)
            except Exception as e:
                print(f"Error loading {metrics_path}: {e}")

        if len(all_metrics) > 1 or (len(all_metrics) == 1 and args.summary_only):
            print_summary(all_metrics)
        elif len(all_metrics) == 0:
            print("No valid metrics files found.")

    elif args.metric == 'distance':
        # Distance metric between object and wrist
        for traj in args.trajectories:
            # Determine trajectory file paths
            if traj.endswith('_hand.npy') or traj.endswith('_object.npy'):
                # Full path provided
                base_name = os.path.basename(traj).replace('_hand.npy', '').replace('_object.npy', '')
                base_path = os.path.join(os.path.dirname(traj), base_name)
                name = base_name
            else:
                # Trajectory name provided
                base_path = os.path.join(args.dir, traj)
                name = traj

            hand_path = f"{base_path}_hand.npy"
            object_path = f"{base_path}_object.npy"
            metrics_path = f"{base_path}_metrics.npy"

            if not os.path.exists(hand_path):
                print(f"Warning: Hand trajectory file not found: {hand_path}")
                continue
            if not os.path.exists(object_path):
                print(f"Warning: Object trajectory file not found: {object_path}")
                continue

            # Load contact metrics to get start and end indices
            start_idx = None
            end_idx = None
            if os.path.exists(metrics_path):
                try:
                    contact_metrics = load_metrics(metrics_path)
                    start_idx = contact_metrics['start_idx']
                    end_idx = contact_metrics['end_idx']
                except Exception as e:
                    print(f"Warning: Could not load contact metrics from {metrics_path}: {e}")
                    print("Computing distance for entire trajectory instead.")
            else:
                print(f"Warning: Metrics file not found: {metrics_path}")
                print("Computing distance for entire trajectory instead.")

            try:
                hand_traj = load_trajectory(hand_path)
                object_traj = load_trajectory(object_path)
                metrics = compute_distance_metrics(hand_traj, object_traj, start_idx, end_idx)
                all_metrics[name] = metrics

                if not args.summary_only:
                    print_distance_metrics(metrics, name)
            except Exception as e:
                print(f"Error processing {name}: {e}")

        if len(all_metrics) > 1 or (len(all_metrics) == 1 and args.summary_only):
            print_distance_summary(all_metrics)
        elif len(all_metrics) == 0:
            print("No valid trajectory files found.")


if __name__ == "__main__":
    main()
