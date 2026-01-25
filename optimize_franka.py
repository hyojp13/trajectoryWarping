"""
Optimize Franka trajectory with contact constraints.
Creates dummy contact data (two contact points on opposite sides of the box)
and optimizes the arm configuration for each frame.
"""

import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
import trimesh
from scipy.spatial.transform import Rotation as R

from differentiable_fk import extract_kinematic_tree, precompute_kinematic_tree_tensors, precompute_local_hand_contacts
from optimize_contacts import optimize_frame


def load_franka_trajectory(pkl_path):
    """
    Load Franka trajectory from roboverse torch file.

    Returns:
        hand_qpos: (n_dofs, n_frames) - Franka hand joint positions
        object_qpos: (7, n_frames) - Object position and quaternion (w,x,y,z)
        n_frames: Number of frames
    """
    data = torch.load(pkl_path, map_location='cpu', weights_only=False)

    # Handle both tuple and dict formats
    if isinstance(data, dict):
        robot_states = data['robot_states']
        object_states = data['object_poses']
    else:
        robot_states, object_states = data

    # Convert from torch tensors if needed
    if torch.is_tensor(robot_states):
        robot_states = robot_states.cpu().numpy()
    if torch.is_tensor(object_states):
        object_states = object_states.cpu().numpy()

    n_frames = robot_states.shape[0]

    print(f"Loaded trajectory with {n_frames} frames")
    print(f"  Robot states shape: {robot_states.shape}")
    print(f"  Object states shape: {object_states.shape}")

    # Parse robot states: [pos(3), quat(4), 9 joints]
    robot_pos = robot_states[:, :3]  # (n_frames, 3)

    robot_shift = [3, -4, 0.95]
    robot_pos += robot_shift

    robot_quat = robot_states[:, 3:7]  # (n_frames, 4)
    robot_joints = robot_states[:, 7:16]  # (n_frames, 9)

    # Parse object states: [pos(3), rot(4)]
    object_pos = object_states[:, :3]  # (n_frames, 3)
    object_pos += robot_shift
    object_rot = object_states[:, 3:7]  # (n_frames, 4)

    # Build hand_qpos: [pos(3), quat(4), joints(9)] = 16 DOFs
    hand_qpos = np.zeros((16, n_frames))
    hand_qpos[:3, :] = robot_pos.T

    # Normalize quaternions
    for frame_idx in range(n_frames):
        quat = robot_quat[frame_idx]
        quat_norm = quat / np.linalg.norm(quat)
        hand_qpos[3:7, frame_idx] = quat_norm

    hand_qpos[7:16, :] = robot_joints.T

    # Build object_qpos: [pos(3), quat(4)] = 7 DOFs
    object_qpos = np.zeros((7, n_frames))
    object_qpos[:3, :] = object_pos.T

    # Normalize quaternions
    for frame_idx in range(n_frames):
        quat = object_rot[frame_idx]
        quat_norm = quat / np.linalg.norm(quat)
        object_qpos[3:7, frame_idx] = quat_norm

    return hand_qpos, object_qpos, n_frames


def create_dummy_contacts(object_mesh, n_frames, hand_components, left_finger_id, right_finger_id, hand_component_offset):
    """
    Create dummy contact data for Franka gripper using actual mesh vertices.

    For a box mesh, we'll create two contact points:
    - One on the left face (x = -0.025)
    - One on the right face (x = 0.025)

    This simulates a parallel jaw gripper grasping the box from opposite sides.

    Args:
        object_mesh: trimesh object
        n_frames: number of frames
        hand_components: list of (faces, verts) for each body
        left_finger_id: body ID of left finger
        right_finger_id: body ID of right finger
        hand_component_offset: starting body ID

    Returns:
        hand_contacts: dict mapping component_id -> list of contacts per frame
        object_contacts: list of vertex indices per frame
    """
    # Find vertices on left and right faces (x = ±0.025)
    vertices = object_mesh.vertices

    # Find vertex closest to center of left face (x=-0.025, y=0, z=0)
    left_face_center = np.array([-0.025, 0.0, 0.0])
    left_distances = np.linalg.norm(vertices - left_face_center, axis=1)
    left_vertex_idx = np.argmin(left_distances)

    # Find vertex closest to center of right face (x=0.025, y=0, z=0)
    right_face_center = np.array([0.025, 0.0, 0.0])
    right_distances = np.linalg.norm(vertices - right_face_center, axis=1)
    right_vertex_idx = np.argmin(right_distances)

    print(f"\nObject contact vertices:")
    print(f"  Left: index {left_vertex_idx}, position {vertices[left_vertex_idx]}")
    print(f"  Right: index {right_vertex_idx}, position {vertices[right_vertex_idx]}")

    # Object contacts: same two vertices for all frames
    object_contacts = [np.array([left_vertex_idx, right_vertex_idx])] * n_frames

    # Hand contacts: Use actual vertices from finger meshes
    # Find component indices for fingers
    left_finger_component_idx = left_finger_id - hand_component_offset
    right_finger_component_idx = right_finger_id - hand_component_offset

    print(f"\nFinger component indices:")
    print(f"  Left finger: component {left_finger_component_idx} (body {left_finger_id})")
    print(f"  Right finger: component {right_finger_component_idx} (body {right_finger_id})")

    # Get finger meshes
    left_finger_faces, left_finger_verts = hand_components[left_finger_component_idx]
    right_finger_faces, right_finger_verts = hand_components[right_finger_component_idx]

    print(f"\nFinger mesh info:")
    print(f"  Left finger: {len(left_finger_verts)} vertices, {len(left_finger_faces)} faces")
    print(f"  Right finger: {len(right_finger_verts)} vertices, {len(right_finger_faces)} faces")

    # Find contact vertex on left finger (inner surface at fingertip)
    # For parallel jaw gripper, we want contacts at the fingertip on inner surface
    # Find the vertex with maximum Z (tip of finger) and Y close to 0 (inner surface)
    left_tip_z = left_finger_verts[:, 2].max()
    left_tip_candidates = left_finger_verts[left_finger_verts[:, 2] > left_tip_z - 0.005]  # Within 5mm of tip
    # Among tip candidates, find the one closest to Y=0 (inner surface)
    left_contact_vertex_idx_local = np.argmin(np.abs(left_tip_candidates[:, 1]))
    # Find this vertex in the original array
    left_contact_vertex_idx = np.where((left_finger_verts == left_tip_candidates[left_contact_vertex_idx_local]).all(axis=1))[0][0]
    left_contact_vertex = left_finger_verts[left_contact_vertex_idx]

    # Find contact vertex on right finger (inner surface at fingertip)
    right_tip_z = right_finger_verts[:, 2].max()
    right_tip_candidates = right_finger_verts[right_finger_verts[:, 2] > right_tip_z - 0.005]  # Within 5mm of tip
    # Among tip candidates, find the one closest to Y=0 (inner surface)
    right_contact_vertex_idx_local = np.argmin(np.abs(right_tip_candidates[:, 1]))
    right_contact_vertex_idx = np.where((right_finger_verts == right_tip_candidates[right_contact_vertex_idx_local]).all(axis=1))[0][0]
    right_contact_vertex = right_finger_verts[right_contact_vertex_idx]

    print(f"\nHand contact vertices:")
    print(f"  Left finger: vertex {left_contact_vertex_idx}, position {left_contact_vertex}")
    print(f"  Right finger: vertex {right_contact_vertex_idx}, position {right_contact_vertex}")

    # Create hand contacts using actual vertex positions
    # hand_contacts[component_id][frame_idx] = [(face_id, bary_coords, object_contact_idx), ...]
    # We need to find which face contains the contact vertex

    def find_face_for_vertex(vertex_idx, faces):
        """Find first face that contains this vertex"""
        for face_idx, face in enumerate(faces):
            if vertex_idx in face:
                # Create barycentric coordinates with full weight on this vertex
                bary = np.zeros(3)
                vertex_pos_in_face = np.where(face == vertex_idx)[0][0]
                bary[vertex_pos_in_face] = 1.0
                return face_idx, bary
        # If not found, use face 0 with center barycentric coords
        return 0, np.array([0.33, 0.33, 0.34])

    left_face_idx, left_bary = find_face_for_vertex(left_contact_vertex_idx, left_finger_faces)
    right_face_idx, right_bary = find_face_for_vertex(right_contact_vertex_idx, right_finger_faces)

    print(f"\nContact faces:")
    print(f"  Left finger: face {left_face_idx}, bary {left_bary}")
    print(f"  Right finger: face {right_face_idx}, bary {right_bary}")

    hand_contacts = {}

    # For parallel jaw gripper:
    # Left finger (Y > 0) should contact RIGHT side of box (x = +0.025)
    # Right finger (Y < 0) should contact LEFT side of box (x = -0.025)
    # This is because fingers approach from opposite sides

    # Left finger touches RIGHT object vertex
    hand_contacts[left_finger_component_idx] = []
    for frame_idx in range(n_frames):
        hand_contacts[left_finger_component_idx].append(
            [(left_face_idx, left_bary, 1)]  # 1 = right object vertex (x=+0.025)
        )

    # Right finger touches LEFT object vertex
    hand_contacts[right_finger_component_idx] = []
    for frame_idx in range(n_frames):
        hand_contacts[right_finger_component_idx].append(
            [(right_face_idx, right_bary, 0)]  # 0 = left object vertex (x=-0.025)
        )

    return hand_contacts, object_contacts


def create_franka_xml_for_optimization():
    """
    Create a simplified MuJoCo XML for Franka optimization.
    Uses the standalone Franka model with upsampled box.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse('franka_model/panda.xml')
    root = tree.getroot()

    # Update meshdir for Franka meshes
    compiler = root.find('compiler')
    if compiler is not None:
        compiler.set('meshdir', 'franka_model/assets')

    # Remove keyframes
    keyframe = root.find('keyframe')
    if keyframe is not None:
        root.remove(keyframe)

    # Find worldbody and link0
    worldbody = root.find('worldbody')
    link0 = worldbody.find(".//body[@name='link0']")

    # Insert freejoint
    freejoint = ET.Element('freejoint', name='panda_root')
    link0.insert(0, freejoint)

    # Add upsampled box mesh (use absolute path to override meshdir)
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')

    # Use absolute path from franka_model directory
    import os
    box_path = os.path.abspath('meshes/box_upsampled.obj')
    ET.SubElement(asset, 'mesh', name='box_upsampled', file=box_path)

    # Add object to worldbody
    object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', mocap='true')
    ET.SubElement(object_body, 'geom', type='mesh', mesh='box_upsampled',
                  rgba='0.8 0.6 0.4 1', mass='0.1')

    xml_string = ET.tostring(root, encoding='unicode')
    return xml_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize Franka trajectory with contact constraints')
    parser.add_argument('pkl_path', type=str, help='Path to pickle file with trajectory data')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--n-iter', type=int, default=100, help='Number of iterations (default: 100)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for optimization: cpu, cuda, or mps (default: cpu)')
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from: {args.pkl_path}")
    hand_qpos, object_qpos, n_frames = load_franka_trajectory(args.pkl_path)
    qpos_copy = hand_qpos.copy()  # Save original for comparison

    # Load upsampled box mesh
    object_mesh = trimesh.load('meshes/box_upsampled.obj', process=False)
    print(f"Object mesh: {len(object_mesh.vertices)} vertices, {len(object_mesh.faces)} faces")

    # Create MuJoCo model
    xml_string = create_franka_xml_for_optimization()
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    print(f"\nModel info:")
    print(f"  DOFs: {m.nq}")
    print(f"  Actuators: {m.nu}")
    print(f"  Bodies: {m.nbody}")

    # Find hand component information
    # For Franka, we'll use the hand and finger bodies
    # The kinematic structure is: link0 -> link1 -> ... -> link7 -> hand -> fingers

    # Find body IDs (use names from panda.xml)
    hand_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    left_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
    right_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')

    print(f"\nBody IDs:")
    print(f"  Hand: {hand_body_id}")
    print(f"  Left finger: {left_finger_id}")
    print(f"  Right finger: {right_finger_id}")

    # For optimization, we need ALL bodies from link0 to fingers for proper FK
    # This is the complete kinematic chain
    link0_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'link0')
    hand_component_offset = link0_id  # Start from base
    hand_components_len = right_finger_id - link0_id + 1  # All bodies in chain

    print(f"\nHand components: {hand_components_len} bodies (from {link0_id} to {right_finger_id})")

    # Extract actual meshes for ALL bodies in the kinematic chain
    from handContacts import get_mesh_for_body
    hand_components = []
    for body_id in range(hand_component_offset, hand_component_offset + hand_components_len):
        try:
            faces, verts = get_mesh_for_body(m, body_id)
            hand_components.append((faces, verts))
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"  Body {body_id} ({body_name}): {len(verts)} vertices, {len(faces)} faces")
        except:
            # Some bodies might not have meshes, create dummy
            dummy_verts = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
            dummy_faces = np.array([[0, 1, 2]])
            hand_components.append((dummy_faces, dummy_verts))
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"  Body {body_id} ({body_name}): no mesh, using dummy")

    # Create dummy contact data using actual finger mesh vertices
    print("\nCreating contact data...")
    hand_contacts, object_contacts = create_dummy_contacts(
        object_mesh, n_frames, hand_components,
        left_finger_id, right_finger_id, hand_component_offset
    )

    # Extract kinematic tree
    print(f"\nExtracting kinematic tree...")
    kinematic_tree = extract_kinematic_tree(m, hand_component_offset, hand_components_len)

    # Precompute kinematic tree tensors
    print(f"Precomputing kinematic tree tensors on device: {args.device}")
    kinematic_tree_torch = precompute_kinematic_tree_tensors(
        kinematic_tree, hand_component_offset, hand_components_len, args.device
    )

    ##### Object Retargeting #####
    print("\n=== Object Retargeting ===")

    # Define barriers: a box wall and a destination box
    # Box wall: navigate OVER this barrier
    wall_barrier = ('rect', {
        'pos': np.array([3.3, -3.81, 1]),  # Center position
        'dims': np.array([0.4, 0.05, 0.5])    # Dimensions: width, depth, height
    })

    # # Destination box: place object ON TOP of this
    # dest_box_barrier = ('rect', {
    #     'pos': np.array([3.2, -4.0, 0.98]),  # Slightly offset from robot
    #     'dims': np.array([0.15, 0.15, 0.06])  # Small box to place on
    # })

    # barriers = [wall_barrier, dest_box_barrier]
    barriers = [wall_barrier]
    # barriers = []

    # Define trajectory waypoints
    # Format: (relative_pos, timestep_start, timestep_end, optional_rotation)
    # Waypoint 1: Go up and over the wall barrier
    # Waypoint 2: Come down to place on destination box

    start_pos = object_qpos[:3, 0].copy()
    end_pos = object_qpos[:3, n_frames-1].copy()

    # New start and end positions
    new_start_pos_shift = np.array([0.0, 0.0, 0.0])  # Keep original start
    end_final_pos_shift = np.array([0.0, 0.2, -0.25])  # Place on top of dest_box
    end_obj_pos_shift = np.array([0.0, 0.0, 0.0])  # No intermediate offset

    newStartPos = start_pos + new_start_pos_shift
    endFinalPos = end_pos + end_final_pos_shift
    endObjPos = endFinalPos + end_obj_pos_shift

    # Waypoints to navigate over the wall barrier
    # Note: Waypoint times must be within the contact frame range [startIdx/n_frames, endIdx/n_frames]
    # With startIdx=85 and n_frames=158, the minimum time is 85/158 = 0.538
    waypts = [
        #(np.array([0.0, 0.0, 0.15]), 0.6, 0.6),
        (np.array([0.0, 0.3, 0.4]), 0.8, 0.6)
        # (np.array([0.15, 0.15, 0.0]), 0.7, 0.7),
        # (np.array([0.15, 0.0, 0.0]), 0.8, 0.8)
    ]

    # Process waypoints: convert relative positions to absolute positions
    waypts_processed = []
    for waypt in waypts:
        waypt_pos = newStartPos + waypt[0]  # Add waypoint offset to newStartPos
        if len(waypt) == 4:
            # Include rotation if provided
            waypts_processed.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
        else:
            waypts_processed.append((waypt_pos, waypt[1], waypt[2]))
    waypts = waypts_processed

    print(f"Start position: {newStartPos}")
    print(f"End position: {endObjPos}")
    print(f"Waypoints: {len(waypts)}")
    for i, wp in enumerate(waypts):
        print(f"  Waypoint {i}: pos={wp[0]}, t_start={wp[1]}, t_end={wp[2]}")

    # Apply trajectory constraints with waypoints
    from trajectory import apply_waypoint_rotations, transformSplines
    from barrier import barrierConstraints, process_barriers, read_obj, moveToEndPt

    # Process barriers (apply scale and position transformations)
    barriers = process_barriers(barriers)

    # Object radius for collision avoidance (box half-diagonal)
    object_radius = 0.035  # ~3.5cm safety margin around box

    # For Franka, the object is always in contact with the gripper
    # So we treat all frames as contact frames
    startIdx = 90
    endIdx = n_frames - 1

    # Get waypoint frame indices
    waypts_idx = np.zeros(len(waypts), dtype=int)
    times = np.linspace(0, 1, n_frames)
    for j in range(len(waypts)):
        waypts_idx[j] = np.abs(times - waypts[j][1]).argmin()

    print(f"\nContact frames: start={startIdx}, end={endIdx} (all frames)")
    print(f"Waypoint frame indices: {waypts_idx}")

    # Transform trajectory segments between waypoints
    pos_waypt_constrained = object_qpos[:3, :].copy()

    # Process segments
    if len(waypts) == 0:
        # No waypoints: transform entire trajectory from start to end
        print(f"Transforming trajectory: frames {startIdx}-{endIdx + 1} (no waypoints)")
        pos_waypt_constrained[:, startIdx:endIdx + 1] = transformSplines(
            pos_waypt_constrained[:, startIdx:endIdx + 1],
            newStartPos,
            endObjPos
        )
    else:
        # With waypoints: transform segments between waypoints
        for i in range(len(waypts) + 1):
            if i == 0:
                # First segment: start to first waypoint
                segment_start = startIdx
                segment_end = waypts_idx[i]
                target_start = newStartPos
                target_end = waypts[i][0]
            elif i == len(waypts):
                # Last segment: last waypoint to end
                segment_start = waypts_idx[i-1]
                segment_end = endIdx + 1
                target_start = waypts[i-1][0]
                target_end = endObjPos
            else:
                # Middle segments: waypoint to waypoint
                segment_start = waypts_idx[i-1]
                segment_end = waypts_idx[i]
                target_start = waypts[i-1][0]
                target_end = waypts[i][0]

            print(f"Transforming segment {i}: frames {segment_start}-{segment_end}")
            pos_waypt_constrained[:, segment_start:segment_end] = transformSplines(
                pos_waypt_constrained[:, segment_start:segment_end],
                target_start,
                target_end
            )

    # Apply rotations at waypoints if specified
    object_rotation_qpos = apply_waypoint_rotations(
        object_qpos[3:7, :], waypts, waypts_idx, startIdx, n_frames
    )

    # Apply barrier constraints to keep trajectory away from barriers
    barrierConstraints(pos_waypt_constrained.T, object_radius, barriers)

    # Read back the corrected trajectory
    trajectory = read_obj("scene/curve_positions.obj")  # (n_frames, 3)

    # Ensure trajectory ends at end position
    extra_pt_count = 0
    # if not np.isclose(trajectory[-1, :], endFinalPos, atol=1e-3).all():
    #     print(f"Trajectory doesn't end at target, adding interpolation points")
    #     print(f"  Current end: {trajectory[-1, :]}")
    #     print(f"  Target end: {endFinalPos}")
    #     moveToEndPt("scene/curve_positions.obj", endFinalPos, 5)
    #     trajectory = read_obj("scene/curve_positions.obj")
    #     extra_pt_count = 5

    # Create new object trajectory with corrected positions and rotations
    new_object_qpos = np.zeros((7, trajectory.shape[0]))
    new_object_qpos[:3, :] = trajectory.T

    # Handle rotation with potential extra points
    if extra_pt_count == 0:
        new_object_qpos[3:, :] = object_rotation_qpos
    else:
        new_object_qpos[3:, :n_frames] = object_rotation_qpos
        # Repeat last rotation for extra points
        new_object_qpos[3:, n_frames:] = object_rotation_qpos[:, -1].reshape(4, 1)

    # Update object_qpos and n_frames for optimization
    old_n_frames = hand_qpos.shape[1]
    object_qpos = new_object_qpos
    n_frames = object_qpos.shape[1]

    print(f"\nRetargeted object trajectory: {n_frames} frames (was {old_n_frames})")
    print(f"Start: {object_qpos[:3, 0]}")
    print(f"End: {object_qpos[:3, -1]}")

    # Update hand_qpos to match new frame count
    if n_frames != old_n_frames:
        # Extend hand_qpos by repeating the last frame
        hand_qpos_new = np.zeros((hand_qpos.shape[0], n_frames))
        hand_qpos_new[:, :old_n_frames] = hand_qpos
        for i in range(old_n_frames, n_frames):
            hand_qpos_new[:, i] = hand_qpos[:, -1]
        hand_qpos = hand_qpos_new

        # Extend object_contacts by repeating the last frame
        for i in range(old_n_frames, n_frames):
            object_contacts.append(object_contacts[-1])

        # Extend hand_contacts by repeating the last frame for each component
        for component_id in hand_contacts:
            last_contacts = hand_contacts[component_id][-1]
            for i in range(old_n_frames, n_frames):
                hand_contacts[component_id].append(last_contacts)

        print(f"Extended hand trajectory and contacts to {n_frames} frames")

    # Optimize each frame
    print(f"\n=== Optimizing {n_frames} frames ===")

    for frame in range(n_frames):
        print(f"\nFrame {frame}/{n_frames}")

        # Precompute local hand contacts for this frame
        local_contacts_cache = precompute_local_hand_contacts(
            hand_contacts, hand_components, hand_components_len,
            frame, args.device
        )

        # Use previous frame's optimized result as initialization (warm start)
        if frame == 0:
            initial_qpos = hand_qpos[:, frame]
        else:
            # Use previous frame's optimized qpos as initialization
            initial_qpos = hand_qpos[:, frame - 1].copy()

        # Object qpos for this frame
        object_qpos_frame = object_qpos[:, frame]

        # Debug: Compute initial contact error
        print(f"\nInitial state analysis:")
        d.qpos[:16] = initial_qpos
        mujoco.mj_forward(m, d)

        from handContacts import get_local_pos
        for hand_component_id in hand_contacts:
            contacts_this_frame = hand_contacts[hand_component_id][frame]
            if contacts_this_frame is None:
                continue
            component_faces, component_verts = hand_components[hand_component_id]
            body_id = hand_component_id + hand_component_offset

            for contact in contacts_this_frame:
                face_id, bary_coords, object_contact_idx = contact
                local_pos = get_local_pos(face_id, bary_coords, component_faces, component_verts)
                body_pos = d.xpos[body_id]
                body_rot = d.xmat[body_id].reshape(3, 3)
                global_hand_pos = body_rot @ local_pos + body_pos

                # Get corresponding object contact
                from optimize_contacts import compute_global_object_contacts
                global_object_contacts = compute_global_object_contacts(
                    object_qpos_frame, object_mesh, object_contacts, frame
                )
                object_pos = global_object_contacts[object_contact_idx]

                distance = np.linalg.norm(global_hand_pos - object_pos)
                print(f"  Component {hand_component_id}, Contact {object_contact_idx}:")
                print(f"    Hand:   {global_hand_pos}")
                print(f"    Object: {object_pos}")
                print(f"    Distance: {distance:.6f}")

        # Optimize
        hand_qpos[:, frame] = optimize_frame(
            initial_qpos,
            object_qpos_frame,
            m, d, hand_contacts, object_contacts,
            hand_components, hand_component_offset, object_mesh, frame,
            kinematic_tree, lr=args.lr, n_iter=args.n_iter,
            optimize_wrist=False,  # Keep robot base fixed, only optimize joints
            optimize_joints=True,
            agent_type='Franka',
            print_logs=True,
            device=args.device,
            kinematic_tree_torch=kinematic_tree_torch,
            local_contacts_cache=local_contacts_cache
        )

        # Debug: Compute final contact error
        print(f"\nFinal state analysis:")
        d.qpos[:16] = hand_qpos[:, frame]
        mujoco.mj_forward(m, d)

        for hand_component_id in hand_contacts:
            contacts_this_frame = hand_contacts[hand_component_id][frame]
            if contacts_this_frame is None:
                continue
            component_faces, component_verts = hand_components[hand_component_id]
            body_id = hand_component_id + hand_component_offset

            for contact in contacts_this_frame:
                face_id, bary_coords, object_contact_idx = contact
                local_pos = get_local_pos(face_id, bary_coords, component_faces, component_verts)
                body_pos = d.xpos[body_id]
                body_rot = d.xmat[body_id].reshape(3, 3)
                global_hand_pos = body_rot @ local_pos + body_pos

                from optimize_contacts import compute_global_object_contacts
                global_object_contacts = compute_global_object_contacts(
                    object_qpos_frame, object_mesh, object_contacts, frame
                )
                object_pos = global_object_contacts[object_contact_idx]

                distance = np.linalg.norm(global_hand_pos - object_pos)
                print(f"  Component {hand_component_id}, Contact {object_contact_idx}:")
                print(f"    Hand:   {global_hand_pos}")
                print(f"    Object: {object_pos}")
                print(f"    Distance: {distance:.6f}")

    print("\nOptimization complete!")

    # Smooth arm trajectory using smooth_hand_trajectory
    print("\n=== Smoothing Arm Trajectory ===")
    from smoothspline import smooth_hand_trajectory

    # Use same smoothing as main.py: window_length=21, polyorder=3
    hand_qpos = smooth_hand_trajectory(hand_qpos, n_frames, 'Franka',
                                       window_length=5, polyorder=3, visualize=False)
    print("Trajectory smoothing complete!")

    # Save optimized trajectory
    import os
    output_dir = "final_trajectories"
    os.makedirs(output_dir, exist_ok=True)

    output_name = os.path.splitext(os.path.basename(args.pkl_path))[0]
    hand_traj_path = os.path.join(output_dir, f"{output_name}_optimized_hand.npy")
    object_traj_path = os.path.join(output_dir, f"{output_name}_object.npy")

    np.save(hand_traj_path, hand_qpos)
    np.save(object_traj_path, object_qpos)

    print(f"\nTrajectories saved:")
    print(f"  Hand: {hand_traj_path}")
    print(f"  Object: {object_traj_path}")

    # Visualize
    print("\nLaunching viewer...")

    # Update object.xml to use upsampled box
    object_xml_content = """<mujoco>
    <asset>
      <mesh name="object_mesh" file="box_upsampled.obj" scale="1 1 1"/>
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

    # Recreate model with kitchen scene for visualization
    import xml.etree.ElementTree as ET
    tree = ET.parse('franka_in_kitchen.xml')
    root = tree.getroot()
    xml_vis = ET.tostring(root, encoding='unicode')

    m_vis = mujoco.MjModel.from_xml_string(xml_vis)
    d_vis = mujoco.MjData(m_vis)

    # Get body IDs for visualization (names differ in franka_in_kitchen.xml)
    try:
        left_finger_id_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, 'panda_leftfinger')
        right_finger_id_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, 'panda_rightfinger')
    except:
        # Try alternative names
        left_finger_id_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
        right_finger_id_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')

    # Find the base link for visualization model
    try:
        link0_id_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, 'panda_link0')
    except:
        link0_id_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, 'link0')

    hand_component_offset_vis = link0_id_vis

    print(f"Visualization body IDs:")
    print(f"  Link0: {link0_id_vis}")
    print(f"  Left finger: {left_finger_id_vis}")
    print(f"  Right finger: {right_finger_id_vis}")
    print(f"  Offset: {hand_component_offset_vis}")

    dt = 0.03
    m_vis.opt.timestep = dt

    with mujoco.viewer.launch_passive(m_vis, d_vis) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()

            if i >= n_frames:
                i = 0

            # Set hand pose
            d_vis.qpos[:16] = hand_qpos[:, i]

            # Set object pose
            if m_vis.nmocap > 0:
                d_vis.mocap_pos[0] = object_qpos[:3, i]
                d_vis.mocap_quat[0] = object_qpos[3:7, i]

            mujoco.mj_forward(m_vis, d_vis)

            # Visualize contact points
            geometry_count = 0

            # Draw object trajectory
            max_geoms = viewer.user_scn.maxgeom
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

            # Draw contact points on object (red)
            quat_scipy = np.array([object_qpos[4, i], object_qpos[5, i],
                                  object_qpos[6, i], object_qpos[3, i]])
            rotation = R.from_quat(quat_scipy)
            rotation_matrix = rotation.as_matrix()

            for contact_idx in object_contacts[i]:
                if geometry_count >= max_geoms:
                    break
                local_vertex = object_mesh.vertices[contact_idx]
                world_vertex = rotation_matrix @ local_vertex + object_qpos[:3, i]

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.008, 0, 0],
                    pos=world_vertex,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1, 0, 0, 1])
                )
                geometry_count += 1

            # Draw contact points on hand/fingers (green)
            # Use actual contact vertices from hand_contacts
            from handContacts import get_local_pos, local_to_global

            # Process hand contacts for this frame
            for hand_component_id in hand_contacts:
                contacts_this_frame = hand_contacts[hand_component_id][i]
                if contacts_this_frame is None:
                    continue

                # Get the mesh for this component
                component_faces, component_verts = hand_components[hand_component_id]

                # Convert component ID to body ID for visualization
                body_id_vis = hand_component_id + hand_component_offset_vis

                for contact in contacts_this_frame:
                    if geometry_count >= max_geoms:
                        break

                    face_id, bary_coords, object_contact_idx = contact

                    # Get local position from face and barycentric coords
                    local_pos = get_local_pos(face_id, bary_coords, component_faces, component_verts)

                    # Transform to global coordinates
                    global_pos = local_to_global(local_pos, body_id_vis, d_vis)

                    # Draw contact point (green)
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.008, 0, 0],
                        pos=global_pos,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 1, 0, 1])
                    )
                    geometry_count += 1

            # Draw barriers (gray boxes)
            for barrier in barriers:
                if geometry_count >= max_geoms:
                    break
                if isinstance(barrier, str):
                    # Skip mesh barriers for now
                    continue
                barr_type, config = barrier
                if barr_type == 'rect':
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=np.array(config['dims']) / 2,  # MuJoCo uses half-sizes
                        pos=config['pos'],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.5, 0.5, 0.5, 0.7])  # Semi-transparent gray
                    )
                    geometry_count += 1
                elif barr_type == 'sphere':
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[config['rad'], 0, 0],
                        pos=config['pos'],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.5, 0.5, 0.5, 0.7])
                    )
                    geometry_count += 1

            # Draw waypoints (black spheres)
            if waypts is not None:
                for j in range(len(waypts)):
                    if geometry_count >= max_geoms:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=waypts[j][0],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 1])
                    )
                    geometry_count += 1

            viewer.user_scn.ngeom = geometry_count
            viewer.sync()
            i += 1

            time_until_next_step = m_vis.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
