"""
Loads initial object and hand trajectories from spline files and saves them
to initial_trajectories folder in the same format as final_trajectories.
"""
import os
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from parse_splines import parseSplines
from load_contacts import load_contacts_lcexp, get_contact_frame_range


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def convert_to_quaternions_MANO(qpos_spline_data):
    # 51 dofs, [3:] are for rotations
    qpos = np.zeros((67, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]

    for j in range(qpos_spline_data.shape[1]):
        for i in range(3, 51, 3):
            rotation = R.from_euler('xyz', qpos_spline_data[i:i+3, j], degrees=False)
            qpos[int((i/3 - 1) * 4 + 4): int(i/3*4 + 3), j] = rotation.as_quat()[:3]
            qpos[int((i/3 - 1) * 4 + 3), j] = rotation.as_quat()[3]

    return qpos


def convert_to_quaternions_Allegro(qpos_spline_data):
    # 22 dofs, [3:6] need to be converted to quaternions
    qpos = np.zeros((23, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]
    qpos[7:, :] = qpos_spline_data[6:, :]

    for j in range(qpos_spline_data.shape[1]):
        rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
        qpos[4:7, j] = rotation.as_quat()[:3]
        qpos[3, j] = rotation.as_quat()[3]

    return qpos


def convert_to_quaternions_object(qpos_spline_data):
    # 6 dofs, [3:6] need to be converted to quaternions
    qpos = np.zeros((7, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]

    for j in range(qpos_spline_data.shape[1]):
        rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
        qpos[4:7, j] = rotation.as_quat()[:3]
        qpos[3, j] = rotation.as_quat()[3]

    return qpos


def rotate_keyframe_angles(keyframes, rotation):
    """
    Add rotation to existing orientations in keyframes
    keyframes: (6, n_frames) array
    rotation: [rx, ry, rz] to add to existing rotations
    """
    keyframes_modified = keyframes.copy()

    additional_rot = R.from_euler('xyz', np.radians(rotation))

    for i in range(keyframes.shape[1]):
        existing_euler = keyframes[3:6, i]
        existing_rot = R.from_euler('xyz', existing_euler)
        combined_rot = additional_rot * existing_rot
        keyframes_modified[3:6, i] = combined_rot.as_euler('xyz')

    return keyframes_modified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save initial trajectories from spline files')
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    AGENT = config['agent']
    TASK = config['task']
    rotation = config['rotation']

    # Retrieve splines
    splines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/hand.smexp')
    objectSplines, objectSeconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')

    # Load contacts to determine frame count
    contacts_lcexp = load_contacts_lcexp('startingTrajectories/' + AGENT + '/' + TASK + '/contacts.lcexp')
    frames = len(contacts_lcexp)

    sim_time = np.linspace(0, 1, frames)

    # Read splines to numpy
    qpos_spline_data = np.array([spline(sim_time) for spline in splines])
    object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])

    qpos_spline_data = qpos_spline_data[:, :, 1]  # (51, frames)
    object_qpos_spline_data = object_qpos_spline_data[:, :, 1]  # (6, frames)

    # Apply rotation
    qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
    object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)

    # Convert to quaternions based on agent type
    if AGENT == 'MANO_right' or AGENT == 'trajectories':
        qpos = convert_to_quaternions_MANO(qpos_spline_data)
    elif AGENT == 'Allegro_right':
        qpos = convert_to_quaternions_Allegro(qpos_spline_data)
    else:
        raise ValueError(f"Unknown agent type: {AGENT}")

    object_qpos = convert_to_quaternions_object(object_qpos_spline_data)

    # Transpose object_qpos to match final trajectory format (frames, 7)
    object_trajectory = object_qpos.T

    # Save initial trajectories
    trajectory_dir = "initial_trajectories"
    os.makedirs(trajectory_dir, exist_ok=True)

    # Extract trajectory name from config file
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    hand_traj_path = os.path.join(trajectory_dir, f"{config_name}_hand.npy")
    object_traj_path = os.path.join(trajectory_dir, f"{config_name}_object.npy")

    np.save(hand_traj_path, qpos)
    np.save(object_traj_path, object_trajectory)

    print(f"\nInitial trajectories saved:")
    print(f"  Hand: {hand_traj_path} (shape: {qpos.shape})")
    print(f"  Object: {object_traj_path} (shape: {object_trajectory.shape})")
    print(f"  Frames: {frames}")
