"""
Metric analysis tools for trajectory retargeting.

Two layers live here:

* Low-level metrics (``compute_distance_metrics``, ``compute_path_length``,
  ``compute_timewarp_discrepancy``) operating on raw arrays.
* Aggregation helpers (``evaluate_trajectory``, ``collect_metrics``,
  ``results_to_markdown``, ``results_to_csv``) that read the artifacts
  ``retarget.py`` saves under ``final_trajectories_<threshold>/`` and assemble
  the paper's comparison tables. ``evaluate.py`` is the thin CLI on top.
"""

import argparse
import csv
import glob
import io
import os
import re

import numpy as np


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


# ---------------------------------------------------------------------------
# Path-length and temporal-distortion metrics (paper Tables II-V).
# ---------------------------------------------------------------------------

def compute_path_length(positions):
    """Total path length: sum of consecutive frame-to-frame distances."""
    positions = np.asarray(positions)
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def compute_path_length_ratio(original_positions, retargeted_positions):
    """Ratio retargeted/original path length (>1 = longer, <1 = shorter)."""
    original_length = compute_path_length(original_positions)
    retargeted_length = compute_path_length(retargeted_positions)
    if original_length == 0:
        return float('inf') if retargeted_length > 0 else 1.0
    return retargeted_length / original_length


def compute_timewarp_discrepancy(timewarp_array):
    """RMS deviation of the timewarp from a uniform (linear) resampling.

    Higher values mean more temporal distortion was applied to satisfy the
    temporal waypoints (paper Sec. III-C)."""
    timewarp_array = np.asarray(timewarp_array)
    if len(timewarp_array) < 2:
        return 0.0
    linear_params = np.linspace(timewarp_array.min(), timewarp_array.max(), len(timewarp_array))
    return float(np.sqrt(np.mean((timewarp_array - linear_params) ** 2)))


# ---------------------------------------------------------------------------
# Aggregation over saved retargeting artifacts.
# ---------------------------------------------------------------------------

def _contact_distance_summary(metrics_dict):
    """Reduce a saved contact-metrics dict to avg/min/max average distances."""
    avg_distances = np.asarray(metrics_dict['average_distances'])
    positive = avg_distances[avg_distances > 0]
    return {
        'avg': float(np.mean(avg_distances)),
        'min': float(np.min(positive)) if positive.size else 0.0,
        'max': float(np.max(avg_distances)),
    }


def read_retargeting_time(data_txt_path):
    """Parse 'Retargeting time: X.XX seconds' from a run-summary text file."""
    if not os.path.exists(data_txt_path):
        return None
    with open(data_txt_path) as f:
        match = re.search(r'Retargeting time:\s+([\d.]+)\s+seconds', f.read())
    return float(match.group(1)) if match else None


def find_matching_initial_trajectory(name, initial_dir='initial_trajectories'):
    """Find the longest initial-trajectory name that is a substring of ``name``."""
    hand_files = glob.glob(os.path.join(initial_dir, '*_hand.npy'))
    candidates = [os.path.basename(f).replace('_hand.npy', '') for f in hand_files]
    best, best_len = None, 0
    for cand in candidates:
        if cand in name and len(cand) > best_len:
            best, best_len = cand, len(cand)
    return best


def _as_frames_first(arr):
    """Return ``arr`` shaped (num_frames, dof)."""
    return arr.T if arr.shape[0] < arr.shape[1] else arr


def evaluate_trajectory(name, data_dir, initial_dir='initial_trajectories', suffix=''):
    """Assemble all evaluation metrics for one saved trajectory from disk.

    Returns a dict of metrics, with ``None``/empty entries where the required
    artifacts are missing (so the table still renders)."""
    base = os.path.join(data_dir, f"{name}{suffix}")
    hand_path = f"{base}_hand.npy"
    object_path = f"{base}_object.npy"
    metrics_path = f"{base}_metrics.npy"
    timewarp_path = f"{base}_timewarp.npy"
    data_txt_path = f"{base}_data.txt"

    result = {
        'output_distance': {}, 'contacts': {}, 'retargeting_time': None,
        'num_contact_frames': None, 'path_length_ratio_obj': None,
        'path_length_ratio_hand': None, 'timewarp_discrepancy': None,
    }

    if not os.path.exists(hand_path) or not os.path.exists(object_path):
        return result

    hand_traj = load_trajectory(hand_path)
    object_traj = load_trajectory(object_path)

    start_idx = end_idx = None
    if os.path.exists(metrics_path):
        contact_metrics = load_metrics(metrics_path)
        start_idx = contact_metrics.get('start_idx')
        end_idx = contact_metrics.get('end_idx')
        result['contacts'] = _contact_distance_summary(contact_metrics)
        if start_idx is not None and end_idx is not None:
            result['num_contact_frames'] = end_idx - start_idx + 1

    result['output_distance'] = compute_distance_metrics(hand_traj, object_traj, start_idx, end_idx)

    result['retargeting_time'] = read_retargeting_time(data_txt_path)

    if os.path.exists(timewarp_path):
        timewarp_data = load_metrics(timewarp_path)
        contact_timewarp = timewarp_data.get('contact_timewarp')
        if contact_timewarp is not None:
            result['timewarp_discrepancy'] = compute_timewarp_discrepancy(contact_timewarp)

    initial_name = find_matching_initial_trajectory(name, initial_dir)
    if initial_name:
        init_hand_path = os.path.join(initial_dir, f"{initial_name}_hand.npy")
        init_object_path = os.path.join(initial_dir, f"{initial_name}_object.npy")
        if os.path.exists(init_hand_path) and os.path.exists(init_object_path):
            init_hand = _as_frames_first(np.load(init_hand_path))
            init_object = np.load(init_object_path)
            retarg_hand = _as_frames_first(hand_traj)
            result['path_length_ratio_obj'] = compute_path_length_ratio(
                init_object[:, :3], object_traj[:, :3])
            result['path_length_ratio_hand'] = compute_path_length_ratio(
                init_hand[:, :3], retarg_hand[:, :3])

    return result


def collect_metrics(names, data_dir, initial_dir='initial_trajectories', suffix=''):
    """Evaluate every trajectory in ``names``; returns an ordered dict."""
    results = {}
    for name in names:
        results[name] = evaluate_trajectory(name, data_dir, initial_dir, suffix)
    return results


_TABLE_COLUMNS = [
    ('trajectory', 'Trajectory'),
    ('out_dist_mean', 'Out Dist (mean)'),
    ('contact_avg', 'Contact (avg)'),
    ('contact_max', 'Contact (max)'),
    ('num_contact_frames', '#Contact'),
    ('plr_obj', 'PLR Obj'),
    ('plr_hand', 'PLR Hand'),
    ('timewarp', 'Timewarp RMS'),
    ('time_s', 'Time (s)'),
]


def _row_values(name, m):
    def fmt(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) and x is not None else "N/A"
    out = m.get('output_distance', {})
    cont = m.get('contacts', {})
    return {
        'trajectory': name,
        'out_dist_mean': fmt(out.get('mean')),
        'contact_avg': fmt(cont.get('avg'), 6),
        'contact_max': fmt(cont.get('max'), 6),
        'num_contact_frames': str(m.get('num_contact_frames')) if m.get('num_contact_frames') is not None else "N/A",
        'plr_obj': fmt(m.get('path_length_ratio_obj')),
        'plr_hand': fmt(m.get('path_length_ratio_hand')),
        'timewarp': fmt(m.get('timewarp_discrepancy'), 6),
        'time_s': fmt(m.get('retargeting_time'), 2),
    }


def results_to_markdown(results):
    """Render ``results`` as a GitHub-flavored Markdown table."""
    keys = [k for k, _ in _TABLE_COLUMNS]
    headers = [h for _, h in _TABLE_COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for name, m in results.items():
        row = _row_values(name, m)
        lines.append("| " + " | ".join(row[k] for k in keys) + " |")
    return "\n".join(lines) + "\n"


def results_to_csv(results):
    """Render ``results`` as CSV text."""
    keys = [k for k, _ in _TABLE_COLUMNS]
    headers = [h for _, h in _TABLE_COLUMNS]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    for name, m in results.items():
        row = _row_values(name, m)
        writer.writerow([row[k] for k in keys])
    return buffer.getvalue()


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
