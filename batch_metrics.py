"""
Batch metric collection script for trajectory retargeting.
Collects initial trajectory distance, output trajectory distance, contact metrics,
and retargeting time for multiple trajectories.
"""

import argparse
import subprocess
import re
import os
import json
import glob
import numpy as np
import mujoco
import trimesh

# Import from codebase
from load_contacts import load_contacts_lcexp, get_contact_frame_range
from contacts import process_contacts
from handContacts import get_mesh_for_body, get_local_pos, local_to_global
from optimize_contacts import compute_global_object_contacts
from main import build_env_xml
from parse_splines import parseSplines
from trajectory import trajectoryConstraintsPolyline
from smoothspline import create_smoothing_bspline
from scipy.spatial.transform import Rotation as R


def compute_path_length(positions):
    """
    Compute total path length (sum of frame-to-frame distances).

    Args:
        positions: (num_frames, 3) array of XYZ positions

    Returns:
        float: total distance traveled along the trajectory
    """
    if len(positions) < 2:
        return 0.0
    displacements = np.diff(positions, axis=0)  # (num_frames-1, 3)
    distances = np.linalg.norm(displacements, axis=1)  # (num_frames-1,)
    return float(np.sum(distances))


def compute_path_length_ratio(original_positions, retargeted_positions):
    """
    Compute ratio of path lengths: retargeted / original.

    Args:
        original_positions: (num_frames, 3) array of original XYZ positions
        retargeted_positions: (num_frames, 3) array of retargeted XYZ positions

    Returns:
        float: ratio of path lengths (>1 means path got longer, <1 means shorter)
    """
    original_length = compute_path_length(original_positions)
    retargeted_length = compute_path_length(retargeted_positions)

    if original_length == 0:
        return float('inf') if retargeted_length > 0 else 1.0

    return retargeted_length / original_length


def compute_timewarp_discrepancy(timewarp_array):
    """
    Compute RMS discrepancy between actual timewarp and linear sampling.

    Args:
        timewarp_array: (num_frames,) array of timewarp parameter values

    Returns:
        float: RMS discrepancy (higher = more temporal distortion)
    """
    if len(timewarp_array) < 2:
        return 0.0

    linear_params = np.linspace(
        timewarp_array.min(),
        timewarp_array.max(),
        len(timewarp_array)
    )
    discrepancy = np.sqrt(np.mean((timewarp_array - linear_params) ** 2))
    return float(discrepancy)


def compute_timewarp_for_trajectory(traj_name, frames):
    """
    Compute the timewarp array for a trajectory by replicating main.py logic.

    Args:
        traj_name: trajectory name (to load config)
        frames: number of frames in the trajectory

    Returns:
        timewarp_array or None if computation fails
    """
    config_path = f"retargeting_configs/{traj_name}.json"
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    agent = config['agent']
    task = config['task']

    # Load object splines
    spline_path = f"startingTrajectories/{agent}/{task}/object.smexp"
    if not os.path.exists(spline_path):
        return None

    objectSplines, _, _ = parseSplines(spline_path)
    sim_time = np.linspace(0, 1, frames)
    object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])
    object_qpos_spline_data = object_qpos_spline_data[:, :, 1]

    # Convert to quaternions
    object_qpos = np.zeros((7, frames))
    object_qpos[:3, :] = object_qpos_spline_data[:3, :]
    for j in range(frames):
        rotation = R.from_euler('xyz', object_qpos_spline_data[3:6, j], degrees=False)
        quat = rotation.as_quat()
        object_qpos[4:7, j] = quat[:3]
        object_qpos[3, j] = quat[3]

    # Process waypoints
    new_start_pos_shift = np.array(config['new_start_pos_shift'])
    waypts_config = config['waypts']
    start_pos = object_qpos[:3, 0].copy()
    newStartPos = start_pos + new_start_pos_shift

    waypts = []
    for w in waypts_config:
        waypt_pos = newStartPos + np.array(w[0])
        if len(w) == 4:
            waypts.append((waypt_pos, w[1], w[2], w[3]))
        else:
            waypts.append((waypt_pos, w[1], w[2]))

    end_final_pos_shift = np.array(config['end_final_pos_shift'])
    end_obj_pos_shift = np.array(config['end_obj_pos_shift'])
    end_pos = object_qpos[:3, frames - 1].copy()
    endFinalPos = end_pos + end_final_pos_shift
    endObjPos = endFinalPos + end_obj_pos_shift

    # Get waypoint indices from trajectory constraints
    _, _, _, waypts_idx = trajectoryConstraintsPolyline(
        object_qpos[:3, :], startPos=newStartPos, endPos=endObjPos,
        floor_height=start_pos[2], waypts=waypts
    )

    final_waypt_timesteps = np.array([w[2] for w in waypts])

    # Read the curve positions to get the new trajectory
    curve_path = "scene/curve_positions.obj"
    if not os.path.exists(curve_path):
        return None

    from barrier import read_obj
    trajectory = read_obj(curve_path)
    new_frames = trajectory.shape[0]

    new_object_qpos = np.zeros((7, new_frames))
    new_object_qpos[:3, :] = trajectory.T

    # Create smoothing bspline to get timewarp
    _, contact_timewarp = create_smoothing_bspline(
        new_object_qpos[:3, :].T,
        parameterization="waypts",
        waypts_info=(waypts_idx, final_waypt_timesteps)
    )

    return contact_timewarp


def compute_initial_contact_metrics(traj_name, initial_traj_name, start_idx, end_idx):
    """
    Compute contact metrics for the initial trajectory.

    Args:
        traj_name: Name of the final trajectory (to load config)
        initial_traj_name: Name of the initial trajectory
        start_idx: Start frame index for contact period
        end_idx: End frame index for contact period

    Returns:
        dict with 'avg', 'min', 'max' contact distances, or None on error
    """
    # Load config file
    config_path = f"retargeting_configs/{traj_name}.json"
    if not os.path.exists(config_path):
        print(f"    Warning: Config file not found: {config_path}")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    agent = config['agent']
    # Map 'trajectories' to 'MANO_right' for MuJoCo model loading
    agent_for_mujoco = 'MANO_right' if agent == 'trajectories' else agent
    task = config['task']
    object_mesh_file = config['object_mesh_file']

    # Load initial trajectories
    hand_path = os.path.join('initial_trajectories', f"{initial_traj_name}_hand.npy")
    object_path = os.path.join('initial_trajectories', f"{initial_traj_name}_object.npy")

    if not os.path.exists(hand_path) or not os.path.exists(object_path):
        print(f"    Warning: Initial trajectory files not found")
        return None

    hand_traj = np.load(hand_path)
    object_traj = np.load(object_path)

    # Handle transposed hand trajectory
    if hand_traj.shape[0] < hand_traj.shape[1]:
        hand_traj = hand_traj.T

    # Load contacts from lcexp file
    contacts_path = f"startingTrajectories/{agent}/{task}/contacts.lcexp"
    if not os.path.exists(contacts_path):
        print(f"    Warning: Contacts file not found: {contacts_path}")
        return None

    contacts_lcexp = load_contacts_lcexp(contacts_path)

    # Process contacts
    hand_components_len = 16
    hand_component_offset = 2
    hand_contacts, object_contacts = process_contacts(contacts_lcexp, hand_components_len)

    # Load object mesh
    if not os.path.exists(object_mesh_file):
        print(f"    Warning: Object mesh not found: {object_mesh_file}")
        return None

    object_mesh = trimesh.load(object_mesh_file, process=False)

    # Create object.xml with the correct mesh (same as main.py)
    # Use relative path from tasks/ to meshes/
    mesh_relative_path = f"../meshes/{os.path.basename(object_mesh_file)}"
    object_xml_content = f"""<mujoco>
    <asset>
      <mesh name="object_mesh" file="{mesh_relative_path}" scale="1 1 1"/>
    </asset>

    <worldbody>
      <body name="object" mocap="true">
        <geom type="mesh" mesh="object_mesh" group="6" rgba="0.8 0.6 0.4 1" mass="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  """
    with open('tasks/object.xml', 'w') as f:
        f.write(object_xml_content)

    # Copy the object mesh to the meshes directory if needed
    import shutil
    mesh_dest = f"meshes/{os.path.basename(object_mesh_file)}"
    if not os.path.exists(mesh_dest):
        os.makedirs('meshes', exist_ok=True)
        shutil.copy(object_mesh_file, mesh_dest)

    # Build and load MuJoCo model
    build_env_xml(agent_for_mujoco, "object")
    m = mujoco.MjModel.from_xml_path("env.xml")
    d = mujoco.MjData(m)

    # Get hand component meshes
    hand_components = [None] * hand_components_len
    for j in range(hand_components_len):
        hand_components[j] = get_mesh_for_body(m, j + hand_component_offset)

    # Compute contact distances for each frame in contact range
    all_distances = []

    for frame_idx in range(start_idx, end_idx + 1):
        # Set the hand pose in MuJoCo
        d.qpos = hand_traj[frame_idx, :]
        mujoco.mj_forward(m, d)

        # Compute global object contacts
        object_qpos = object_traj[frame_idx, :]
        try:
            global_object_contacts_np = compute_global_object_contacts(
                object_qpos, object_mesh, object_contacts, frame_idx
            )
        except Exception:
            continue

        if len(global_object_contacts_np) == 0:
            continue

        # Get local hand contacts and compute global positions
        for hand_component_id in range(hand_components_len):
            contacts_this_frame = hand_contacts[hand_component_id][frame_idx]
            if contacts_this_frame is None:
                continue

            for contact in contacts_this_frame:
                face_id, bary_coords, object_contact_idx = contact
                # Get local position on hand mesh
                local_pos = get_local_pos(face_id, bary_coords,
                                          hand_components[hand_component_id][0],
                                          hand_components[hand_component_id][1])
                # Convert to global position
                global_hand_pos = local_to_global(local_pos, hand_component_id + hand_component_offset, d)

                # Get corresponding object contact position
                if object_contact_idx < len(global_object_contacts_np):
                    obj_pos = global_object_contacts_np[object_contact_idx]
                    distance = np.linalg.norm(global_hand_pos - obj_pos)
                    all_distances.append(distance)

    if len(all_distances) == 0:
        return None

    all_distances = np.array(all_distances)
    return {
        'avg': float(np.mean(all_distances)),
        'min': float(np.min(all_distances)),
        'max': float(np.max(all_distances)),
    }


def compute_distance_metrics(hand_traj, object_traj, start_idx=None, end_idx=None):
    """
    Compute distance metrics between object position and wrist position.
    Same logic as metric.py.
    """
    # Handle transposed hand trajectory
    if hand_traj.shape[0] < hand_traj.shape[1]:
        hand_traj = hand_traj.T

    # Use contact range if provided
    if start_idx is not None and end_idx is not None:
        contact_range = slice(start_idx, end_idx + 1)
    else:
        contact_range = slice(None)

    object_positions = object_traj[contact_range, :3]
    wrist_positions = hand_traj[contact_range, :3]
    distances = np.linalg.norm(object_positions - wrist_positions, axis=1)

    return {
        'mean': float(np.mean(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
    }


def find_matching_initial_trajectory(trajectory_name, initial_dir='initial_trajectories'):
    """
    Find an initial trajectory that matches the given trajectory name using substring matching.

    For example, if trajectory_name is 'mug_pass_wall' and initial_dir contains 'mug_pass_hand.npy',
    it will return 'mug_pass' since 'mug_pass' is a substring of 'mug_pass_wall'.
    """
    # Get all unique trajectory names in the initial directory
    hand_files = glob.glob(os.path.join(initial_dir, '*_hand.npy'))
    initial_names = [os.path.basename(f).replace('_hand.npy', '') for f in hand_files]

    # Find the longest matching initial trajectory name that is a substring of trajectory_name
    best_match = None
    best_length = 0

    for init_name in initial_names:
        if init_name in trajectory_name and len(init_name) > best_length:
            best_match = init_name
            best_length = len(init_name)

    return best_match


def parse_distance_output(output):
    """Parse the distance metric output from metric.py."""
    metrics = {}

    # Look for the summary line or individual metrics
    mean_match = re.search(r'Mean distance:\s+([\d.]+)', output)
    min_match = re.search(r'Min distance:\s+([\d.]+)', output)
    max_match = re.search(r'Max distance:\s+([\d.]+)', output)

    if mean_match:
        metrics['mean'] = float(mean_match.group(1))
    if min_match:
        metrics['min'] = float(min_match.group(1))
    if max_match:
        metrics['max'] = float(max_match.group(1))

    return metrics


def parse_contact_output(output):
    """Parse the contact metric output from metric.py --summary-only."""
    metrics = {}

    # Look for summary table format
    # Format: name    Avg Dist    Min Dist    Max Dist    Frames
    lines = output.strip().split('\n')
    for line in lines:
        # Skip header and separator lines
        if 'Avg Dist' in line or '---' in line or '===' in line:
            continue

        # Try to parse data line
        parts = line.split()
        if len(parts) >= 4:
            try:
                # Last 4 values should be avg, min, max, frames
                metrics['avg'] = float(parts[-4])
                metrics['min'] = float(parts[-3])
                metrics['max'] = float(parts[-2])
                break
            except (ValueError, IndexError):
                continue

    return metrics


def get_retargeting_time(trajectory_name, data_dir='final_trajectories_0.01'):
    """Read retargeting time from the _data.txt file."""
    data_path = os.path.join(data_dir, f"{trajectory_name}_data.txt")

    if not os.path.exists(data_path):
        print(f"Warning: Data file not found: {data_path}")
        return None

    with open(data_path, 'r') as f:
        content = f.read()

    # Look for "Retargeting time: X.XX seconds"
    match = re.search(r'Retargeting time:\s+([\d.]+)\s+seconds', content)
    if match:
        return float(match.group(1))

    return None


def run_metric_command(args_list):
    """Run metric.py with given arguments and return output."""
    cmd = ['python', 'metric.py'] + args_list
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)) or '.')
        return result.stdout + result.stderr
    except Exception as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return ""


def collect_metrics(trajectory_names, data_dir='final_trajectories_0.01'):
    """
    Collect metrics for a list of trajectories.

    Args:
        trajectory_names: list of trajectory names to process
        data_dir: directory containing retargeted trajectory files

    Returns a dictionary with structure:
    {
        'trajectory_name': {
            'initial_distance': {'mean': ..., 'min': ..., 'max': ...},
            'output_distance': {'mean': ..., 'min': ..., 'max': ...},
            'contacts': {'avg': ..., 'min': ..., 'max': ...},
            'retargeting_time': ...
        },
        ...
    }
    """
    print(f"Using data directory: {data_dir}")
    results = {}

    for traj_name in trajectory_names:
        print(f"\nCollecting metrics for: {traj_name}")
        results[traj_name] = {}

        # 1. Initial trajectory distance
        print(f"  Running initial trajectory distance metric...")
        # Find matching initial trajectory using substring matching
        initial_traj_name = find_matching_initial_trajectory(traj_name)
        if initial_traj_name:
            print(f"    Using initial trajectory: {initial_traj_name}")
            # Load start/end indices from final trajectory's metrics file
            metrics_path = os.path.join(data_dir, f"{traj_name}_metrics.npy")
            start_idx, end_idx = None, None
            if os.path.exists(metrics_path):
                try:
                    contact_metrics = np.load(metrics_path, allow_pickle=True).item()
                    start_idx = contact_metrics['start_idx']
                    end_idx = contact_metrics['end_idx']
                except Exception as e:
                    print(f"    Warning: Could not load contact metrics: {e}")

            # Load initial trajectory files
            hand_path = os.path.join('initial_trajectories', f"{initial_traj_name}_hand.npy")
            object_path = os.path.join('initial_trajectories', f"{initial_traj_name}_object.npy")

            if os.path.exists(hand_path) and os.path.exists(object_path):
                hand_traj = np.load(hand_path)
                object_traj = np.load(object_path)
                initial_metrics = compute_distance_metrics(hand_traj, object_traj, start_idx, end_idx)
                results[traj_name]['initial_distance'] = initial_metrics
                print(f"    Mean: {initial_metrics.get('mean', 'N/A')}, Min: {initial_metrics.get('min', 'N/A')}, Max: {initial_metrics.get('max', 'N/A')}")

                # Compute initial contact metrics
                print(f"  Running initial contact metric...")
                if start_idx is not None and end_idx is not None:
                    initial_contact_metrics = compute_initial_contact_metrics(
                        traj_name, initial_traj_name, start_idx, end_idx
                    )
                    results[traj_name]['initial_contacts'] = initial_contact_metrics if initial_contact_metrics else {}
                    if initial_contact_metrics:
                        print(f"    Avg: {initial_contact_metrics.get('avg', 'N/A'):.6f}, Min: {initial_contact_metrics.get('min', 'N/A'):.6f}, Max: {initial_contact_metrics.get('max', 'N/A'):.6f}")
                    else:
                        print(f"    Warning: Could not compute initial contact metrics")
                else:
                    print(f"    Warning: No start/end indices available for initial contact metrics")
                    results[traj_name]['initial_contacts'] = {}
            else:
                print(f"    Warning: Could not find initial trajectory files")
                results[traj_name]['initial_distance'] = {}
                results[traj_name]['initial_contacts'] = {}
        else:
            print(f"    Warning: No matching initial trajectory found for '{traj_name}'")
            results[traj_name]['initial_distance'] = {}
            results[traj_name]['initial_contacts'] = {}

        # 2. Output trajectory distance
        print(f"  Running output trajectory distance metric...")
        output = run_metric_command([traj_name, '--metric', 'distance', '--dir', data_dir])
        output_metrics = parse_distance_output(output)
        results[traj_name]['output_distance'] = output_metrics
        if output_metrics:
            print(f"    Mean: {output_metrics.get('mean', 'N/A')}, Min: {output_metrics.get('min', 'N/A')}, Max: {output_metrics.get('max', 'N/A')}")
        else:
            print(f"    Warning: Could not parse output distance metrics")

        # 3. Contact metrics
        print(f"  Running contact metric...")
        output = run_metric_command([traj_name, '--summary-only', '--metric', 'contacts', '--dir', data_dir])
        contact_metrics = parse_contact_output(output)
        results[traj_name]['contacts'] = contact_metrics
        if contact_metrics:
            print(f"    Avg: {contact_metrics.get('avg', 'N/A')}, Min: {contact_metrics.get('min', 'N/A')}, Max: {contact_metrics.get('max', 'N/A')}")
        else:
            print(f"    Warning: Could not parse contact metrics")

        # 4. Retargeting time
        print(f"  Reading retargeting time...")
        retargeting_time = get_retargeting_time(traj_name, data_dir=data_dir)
        results[traj_name]['retargeting_time'] = retargeting_time
        if retargeting_time is not None:
            print(f"    Time: {retargeting_time:.2f} seconds")
        else:
            print(f"    Warning: Could not read retargeting time")

        # 5. Path length ratios (object and hand)
        print(f"  Computing path length ratios...")
        results[traj_name]['path_length_ratio_obj'] = None
        results[traj_name]['path_length_ratio_hand'] = None

        # Load retargeted trajectories
        retargeted_hand_path = os.path.join(data_dir, f"{traj_name}_hand.npy")
        retargeted_object_path = os.path.join(data_dir, f"{traj_name}_object.npy")

        if initial_traj_name and os.path.exists(hand_path) and os.path.exists(object_path):
            if os.path.exists(retargeted_hand_path) and os.path.exists(retargeted_object_path):
                # Load trajectories
                initial_hand = np.load(hand_path)
                initial_object = np.load(object_path)
                retargeted_hand = np.load(retargeted_hand_path)
                retargeted_object = np.load(retargeted_object_path)

                # Handle transposed trajectories
                if initial_hand.shape[0] < initial_hand.shape[1]:
                    initial_hand = initial_hand.T
                if retargeted_hand.shape[0] < retargeted_hand.shape[1]:
                    retargeted_hand = retargeted_hand.T

                # Object path length ratio (positions are first 3 columns)
                plr_obj = compute_path_length_ratio(initial_object[:, :3], retargeted_object[:, :3])
                results[traj_name]['path_length_ratio_obj'] = plr_obj
                print(f"    Object PLR: {plr_obj:.4f}")

                # Hand (wrist) path length ratio (positions are first 3 columns)
                plr_hand = compute_path_length_ratio(initial_hand[:, :3], retargeted_hand[:, :3])
                results[traj_name]['path_length_ratio_hand'] = plr_hand
                print(f"    Hand PLR: {plr_hand:.4f}")
            else:
                print(f"    Warning: Retargeted trajectory files not found")
        else:
            print(f"    Warning: Initial trajectory files not found for path length computation")

        # 6. Timewarp discrepancy
        print(f"  Computing timewarp discrepancy...")
        results[traj_name]['timewarp_discrepancy'] = None

        if os.path.exists(retargeted_object_path):
            retargeted_object = np.load(retargeted_object_path)
            frames = retargeted_object.shape[0]
            timewarp = compute_timewarp_for_trajectory(traj_name, frames)
            if timewarp is not None:
                print(f"    Timewarp range: [{timewarp.min():.6f}, {timewarp.max():.6f}]")
                tw_discrepancy = compute_timewarp_discrepancy(timewarp)
                results[traj_name]['timewarp_discrepancy'] = tw_discrepancy
                print(f"    Timewarp discrepancy: {tw_discrepancy:.6f}")
            else:
                print(f"    Warning: Could not compute timewarp")
        else:
            print(f"    Warning: Retargeted object trajectory not found")

    return results


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 180)
    print("SUMMARY")
    print("=" * 180)

    header = f"{'Trajectory':<30} | {'Init Dist (m/m/M)':<20} | {'Out Dist (m/m/M)':<20} | {'Init Cont (a/m/M)':<20} | {'Out Cont (a/m/M)':<20} | {'PLR Obj':<8} | {'PLR Hand':<8} | {'Timewarp':<8} | {'Time(s)':<8}"
    print(header)
    print("-" * 180)

    for traj_name, metrics in results.items():
        init = metrics.get('initial_distance', {})
        out = metrics.get('output_distance', {})
        init_cont = metrics.get('initial_contacts', {})
        cont = metrics.get('contacts', {})
        time_val = metrics.get('retargeting_time')
        plr_obj = metrics.get('path_length_ratio_obj')
        plr_hand = metrics.get('path_length_ratio_hand')
        tw_disc = metrics.get('timewarp_discrepancy')

        init_str = f"{init.get('mean', 0):.4f}/{init.get('min', 0):.4f}/{init.get('max', 0):.4f}" if init else "N/A"
        out_str = f"{out.get('mean', 0):.4f}/{out.get('min', 0):.4f}/{out.get('max', 0):.4f}" if out else "N/A"
        init_cont_str = f"{init_cont.get('avg', 0):.4f}/{init_cont.get('min', 0):.4f}/{init_cont.get('max', 0):.4f}" if init_cont else "N/A"
        cont_str = f"{cont.get('avg', 0):.4f}/{cont.get('min', 0):.4f}/{cont.get('max', 0):.4f}" if cont else "N/A"
        time_str = f"{time_val:.2f}" if time_val is not None else "N/A"
        plr_obj_str = f"{plr_obj:.4f}" if plr_obj is not None else "N/A"
        plr_hand_str = f"{plr_hand:.4f}" if plr_hand is not None else "N/A"
        tw_disc_str = f"{tw_disc:.6f}" if tw_disc is not None else "N/A"

        # Truncate trajectory name if too long
        display_name = traj_name if len(traj_name) <= 28 else traj_name[:25] + "..."
        print(f"{display_name:<30} | {init_str:<20} | {out_str:<20} | {init_cont_str:<20} | {cont_str:<20} | {plr_obj_str:<8} | {plr_hand_str:<8} | {tw_disc_str:<8} | {time_str:<8}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect metrics for multiple trajectories'
    )
    parser.add_argument(
        'trajectories',
        nargs='+',
        help='List of trajectory names to collect metrics for'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file to save results (optional)'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='final_trajectories_0.01',
        help='Directory containing retargeted trajectory files (default: final_trajectories_0.01)'
    )

    args = parser.parse_args()

    # Collect metrics
    results = collect_metrics(args.trajectories, data_dir=args.dir)

    # Print summary
    print_summary(results)

    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
