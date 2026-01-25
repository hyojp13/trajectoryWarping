"""
Playback Adroit hand trajectories from DexMV demonstration file.

This script visualizes demonstrations from DexMV (https://github.com/yzqin/dexmv-sim)
using the Adroit hand model. It loads trajectory data and plays it back in MuJoCo.

Expected data format:
- Torch file (.pt) contains dict: {'robot_states': tensor, 'object_poses': tensor}
- robot_states: (n_frames, 30) = [pos(3), quat(4), 23 joint angles]
  - pos: XYZ position of hand base
  - quat: Orientation as quaternion [w, x, y, z]
  - joints: 23 hand joint angles (missing one compared to full Adroit model)
- object_poses: (n_frames, 7) = [pos(3), quat(4)]

Model structure (DexMV adroit_relocate.xml):
- 30 DOFs for hand: 6 forearm (3 translation + 3 rotation) + 24 hand joints
- 7 DOFs for object: freejoint (3 position + 4 quaternion)
- Total: 37 qpos DOFs

Usage:
  mjpython playback_adroit.py <path_to_demo.pt> --speed 0.5 --object-mesh path/to/mesh.obj
"""

import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import torch


def load_adroit_trajectory(data_path):
    """
    Load Adroit trajectory from DexMV demonstration file.

    Returns:
        hand_qpos: (n_dofs, n_frames) - Adroit hand joint positions (mocap controlled)
        object_qpos: (7, n_frames) - Object position and quaternion
        n_frames: Number of frames
    """
    data = torch.load(data_path, map_location='cpu', weights_only=False)

    # Extract data from dict
    if isinstance(data, dict):
        robot_states = data['robot_states']
        object_states = data['object_poses']
    else:
        raise ValueError("Expected dict with 'robot_states' and 'object_poses' keys")

    if torch.is_tensor(robot_states):
        robot_states = robot_states.cpu().numpy()
    if torch.is_tensor(object_states):
        object_states = object_states.cpu().numpy()

    n_frames = robot_states.shape[0]

    print(f"Loaded trajectory with {n_frames} frames")
    print(f"  Robot states shape: {robot_states.shape}")
    print(f"  Object states shape: {object_states.shape}")

    # Parse robot states: [pos(3), 27 joint angles]
    # Total: 30 values per frame
    robot_pos = robot_states[:, :3]  # (n_frames, 3)
    robot_joints_all = robot_states[:, 3:30]  # (n_frames, 27)

    print(f"  Robot position range: {robot_pos.min(axis=0)} to {robot_pos.max(axis=0)}")
    print(f"  Robot joints shape: {robot_joints_all.shape}")
    print(f"  Robot joints range: [{robot_joints_all.min():.3f}, {robot_joints_all.max():.3f}]")

    # Parse object states: [pos(3), quat(4)]
    object_pos = object_states[:, :3]  # (n_frames, 3)
    object_quat = object_states[:, 3:7]  # (n_frames, 4)

    print(f"  Object position range: {object_pos.min(axis=0)} to {object_pos.max(axis=0)}")

    # Build hand_qpos: [ARTx, ARTy, ARTz, ARRx, ARRy, ARRz, 24 hand joints] = 30 DOFs
    # - Indices 0-2: ARTx, ARTy, ARTz (from robot_pos)
    # - Indices 3-5: ARRx, ARRy, ARRz
    # - Indices 6-29: 24 hand joints
    hand_qpos = np.zeros((30, n_frames))
    hand_qpos[0:3, :] = robot_pos.T  # ARTx, ARTy, ARTz (translation)
    hand_qpos[3:6, :] = robot_joints_all[:, 0:3].T  # ARRx, ARRy, ARRz (rotation)
    hand_qpos[6:30, :] = robot_joints_all[:, 3:27].T  # 24 hand joints (WRJ1, WRJ0, FFJ3-FFJ0, ...)

    # Build object_qpos: [pos(3), quat(4)] = 7 DOFs (freejoint)
    object_qpos = np.zeros((7, n_frames))
    object_qpos[:3, :] = object_pos.T

    # Normalize and convert quaternions to [w, x, y, z] format for MuJoCo
    for frame_idx in range(n_frames):
        quat = object_quat[frame_idx]
        quat_norm = quat / np.linalg.norm(quat)
        object_qpos[3:7, frame_idx] = quat_norm

    return hand_qpos, object_qpos, n_frames


def create_adroit_scene_xml(object_mesh_path=None):
    """
    Create a MuJoCo XML for Adroit hand with object from DexMV model.
    Creates a temporary modified XML file and returns the path.
    """
    import os
    import xml.etree.ElementTree as ET

    tree = ET.parse('Adroit/adroit_relocate.xml')
    root = tree.getroot()

    # Add visual assets
    asset = root.find('asset')
    if asset is not None:
        # Add ground texture
        ET.SubElement(asset, 'texture', name='texplane', type='2d', builtin='checker',
                     rgb1='0.2 0.3 0.4', rgb2='0.1 0.15 0.2', width='512', height='512')
        ET.SubElement(asset, 'material', name='MatGnd', reflectance='0.5',
                     texture='texplane', texrepeat='2 2', texuniform='true')

    worldbody = root.find('worldbody')

    # Add lighting and ground plane
    if worldbody is not None:
        ET.SubElement(worldbody, 'light', directional='false', diffuse='0.8 0.8 0.8',
                     specular='0.3 0.3 0.3', pos='0 0 4.0', dir='0 0 -1')
        ET.SubElement(worldbody, 'geom', name='ground', pos='0 -0.7 0', size='2 2 0.1',
                     material='MatGnd', type='plane', contype='1', conaffinity='1')

    # Add object as a free-floating body
    # Find the forearm body and add object as a sibling
    if object_mesh_path:
        asset = root.find('asset')
        mesh_name = os.path.basename(object_mesh_path).replace('.obj', '')
        # Get absolute path for mesh
        mesh_file = os.path.abspath(object_mesh_path)
        ET.SubElement(asset, 'mesh', name=mesh_name, file=mesh_file)

        # Add new object with free joint
        object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', pos='0 0 0.3')
        ET.SubElement(object_body, 'freejoint')
        ET.SubElement(object_body, 'geom', type='mesh', mesh=mesh_name,
                     rgba='0.8 0.6 0.4 1', mass='0.1')
    else:
        # Add mustard bottle sized cylinder with free joint
        object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', pos='0 0 0.3')
        ET.SubElement(object_body, 'freejoint')
        ET.SubElement(object_body, 'geom', type='cylinder', size='0.035 0.08',
                     rgba='0.9 0.8 0.2 1', mass='0.1')

    # Write to temporary file in Adroit directory so relative paths work
    temp_xml_path = 'Adroit/temp_playback_scene.xml'
    tree.write(temp_xml_path, encoding='unicode')

    return temp_xml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Playback Adroit hand trajectory from DexMV demonstration file'
    )
    parser.add_argument('data_path', type=str,
                       help='Path to .pt file with trajectory data')
    parser.add_argument('--object-mesh', type=str, default=None,
                       help='Path to object mesh file (.obj) (optional)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier (default: 1.0)')
    args = parser.parse_args()

    print(f"Loading trajectory from: {args.data_path}")
    hand_qpos, object_qpos, n_frames = load_adroit_trajectory(args.data_path)

    # Create MuJoCo model
    print(f"Creating Adroit scene...")
    if args.object_mesh:
        print(f"Using object mesh: {args.object_mesh}")
    xml_path = create_adroit_scene_xml(args.object_mesh)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    dt = 0.03  # ~30 Hz
    m.opt.timestep = dt / args.speed

    print(f"\nPlayback info:")
    print(f"  Frames: {n_frames}")
    print(f"  Hand DOFs: {hand_qpos.shape[0]}")
    print(f"  Model nq (position DOFs): {m.nq}")
    print(f"  Model nmocap: {m.nmocap}")
    print(f"  Timestep: {m.opt.timestep:.4f}s")
    print(f"  Speed: {args.speed}x")
    print(f"  Total duration: {n_frames * m.opt.timestep:.2f}s")

    # print("\nJoint info:")
    # print(f"  Hand joints: 0-29 (6 forearm DOFs + 24 hand joints)")
    # print(f"  Object joint: {30 if m.nq > 30 else 'not found'} (freejoint, 7 DOFs)")

    # print("\nStarting playback... (close viewer to exit)")

    # Find qpos addresses
    hand_qpos_start = 0
    hand_qpos_end = 30

    # Find object freejoint qpos address (it's the freejoint after all hand joints)
    # The freejoint is at index 30 (type 0 = freejoint)
    object_qpos_start = 30
    if m.nq >= 37:
        print(f"Object qpos range: {object_qpos_start} to {object_qpos_start + 7}")
    else:
        object_qpos_start = -1
        print("Warning: Object qpos not found in model")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()

            if i >= n_frames:
                i = 0

            d.qpos[hand_qpos_start:hand_qpos_end] = hand_qpos[:, i]

            if object_qpos_start >= 0:
                d.qpos[object_qpos_start:object_qpos_start+7] = object_qpos[:, i]

            mujoco.mj_forward(m, d)

            geometry_count = 0
            max_geoms = viewer.user_scn.maxgeom

            # Draw object trajectory
            trajectory_skip = max(1, n_frames // 100)
            for j in range(0, min(i + 1, n_frames), trajectory_skip):
                if geometry_count >= max_geoms:
                    break

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],
                    pos=object_qpos[:3, j],
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 1, 0.5])
                )
                geometry_count += 1

            viewer.user_scn.ngeom = geometry_count
            viewer.sync()
            i += 1

            elapsed = time.time() - step_start
            time_until_next_step = m.opt.timestep - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)