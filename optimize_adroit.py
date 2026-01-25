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


def load_adroit_trajectory(data_path):
    """
    Load Adroit trajectory from DexMV torch file.

    Returns:
        hand_qpos: (30, n_frames) - Adroit hand joint positions
        object_qpos: (7, n_frames) - Object position and quaternion (w,x,y,z)
        n_frames: Number of frames
    """
    data = torch.load(data_path, map_location='cpu', weights_only=False)

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
    robot_pos = robot_states[:, :3]  # (n_frames, 3)
    robot_joints_all = robot_states[:, 3:30]  # (n_frames, 27)

    object_pos = object_states[:, :3]
    object_quat = object_states[:, 3:7]

    # Build hand_qpos: [ARTx, ARTy, ARTz, ARRx, ARRy, ARRz, 24 hand joints] = 30 DOFs
    hand_qpos = np.zeros((30, n_frames))
    hand_qpos[0:3, :] = robot_pos.T
    hand_qpos[3:6, :] = robot_joints_all[:, 0:3].T
    hand_qpos[6:30, :] = robot_joints_all[:, 3:27].T

    # Build object_qpos: [pos(3), quat(4)] = 7 DOFs
    object_qpos = np.zeros((7, n_frames))
    object_qpos[:3, :] = object_pos.T

    # Normalize quaternions
    for frame_idx in range(n_frames):
        quat = object_quat[frame_idx]
        quat_norm = quat / np.linalg.norm(quat)
        object_qpos[3:7, frame_idx] = quat_norm

    return hand_qpos, object_qpos, n_frames


def create_real_contacts_from_frame200(object_mesh, n_frames, hand_components,
                                      thumb_tip_id, index_tip_id, middle_tip_id,
                                      hand_component_offset):
    """
    Create contact data using real contacts extracted from frame 200 of DexMV demo.

    This uses actual contact points from the demonstration where the hand makes
    good contact with the object using thumb, index, and middle fingers.

    Args:
        object_mesh: trimesh object
        n_frames: number of frames
        hand_components: list of (faces, verts) for each body
        thumb_tip_id: body ID of thumb distal
        index_tip_id: body ID of index distal (ffdistal)
        middle_tip_id: body ID of middle distal (mfdistal)
        hand_component_offset: starting body ID

    Returns:
        hand_contacts: dict mapping component_id -> list of contacts per frame
        object_contacts: list of face+bary tuples per frame
    """
    index_local_pos = np.array([0.006651739484779224, -0.006402379537698843, 0.006885730529664066])
    middle_local_pos = np.array([0.006651739484779224, -0.006402379537698843, 0.0058857305296640666])
    thumb_local_pos = np.array([-0.007826289852307097, 0.007158050976496451, 0.0227564963921879])

    index_obj_face = 106
    index_obj_bary = np.array([0.18382274836363896, 0.46547735681921143, 0.3506998948171496])

    middle_obj_face = 105
    middle_obj_bary = np.array([0.5212853508428175, -3.0305818271387574e-16, 0.4787146491571828])

    thumb_obj_face = 53
    thumb_obj_bary = np.array([0.2903301081130494, 0.2585083665795384, 0.4511615253074122])

    # Get component indices
    thumb_component_idx = thumb_tip_id - hand_component_offset
    index_component_idx = index_tip_id - hand_component_offset
    middle_component_idx = middle_tip_id - hand_component_offset

    # print(f"\nFingertip component indices:")
    # print(f"  Thumb: component {thumb_component_idx} (body {thumb_tip_id})")
    # print(f"  Index: component {index_component_idx} (body {index_tip_id})")
    # print(f"  Middle: component {middle_component_idx} (body {middle_tip_id})")

    # Get fingertip meshes
    thumb_faces, thumb_verts = hand_components[thumb_component_idx]
    index_faces, index_verts = hand_components[index_component_idx]
    middle_faces, middle_verts = hand_components[middle_component_idx]

    # print(f"\nFingertip mesh info:")
    # print(f"  Thumb: {len(thumb_verts)} vertices, {len(thumb_faces)} faces")
    # print(f"  Index: {len(index_verts)} vertices, {len(index_faces)} faces")
    # print(f"  Middle: {len(middle_verts)} vertices, {len(middle_faces)} faces")

    def find_face_for_local_pos(local_pos, faces, verts):
        """Find closest face to a local position and compute barycentric coords"""
        # Find closest vertex
        distances = np.linalg.norm(verts - local_pos, axis=1)
        closest_vertex_idx = np.argmin(distances)

        # Find first face containing this vertex
        for face_idx, face in enumerate(faces):
            if closest_vertex_idx in face:
                # Create barycentric coordinates with full weight on this vertex
                bary = np.zeros(3)
                vertex_pos_in_face = np.where(face == closest_vertex_idx)[0][0]
                bary[vertex_pos_in_face] = 1.0
                return face_idx, bary

        # Fallback: use first face with center coords
        return 0, np.array([0.33, 0.33, 0.34])

    thumb_face_idx, thumb_bary = find_face_for_local_pos(thumb_local_pos, thumb_faces, thumb_verts)
    index_face_idx, index_bary = find_face_for_local_pos(index_local_pos, index_faces, index_verts)
    middle_face_idx, middle_bary = find_face_for_local_pos(middle_local_pos, middle_faces, middle_verts)

    # print(f"\nHand contact faces:")
    # print(f"  Thumb: face {thumb_face_idx}, bary {thumb_bary}, local_pos {thumb_local_pos}")
    # print(f"  Index: face {index_face_idx}, bary {index_bary}, local_pos {index_local_pos}")
    # print(f"  Middle: face {middle_face_idx}, bary {middle_bary}, local_pos {middle_local_pos}")

    # print(f"\nObject contact faces:")
    # print(f"  Thumb: face {thumb_obj_face}, bary {thumb_obj_bary}")
    # print(f"  Index: face {index_obj_face}, bary {index_obj_bary}")
    # print(f"  Middle: face {middle_obj_face}, bary {middle_obj_bary}")

    object_contacts = [
        (thumb_obj_face, thumb_obj_bary),    # Contact 0: thumb
        (index_obj_face, index_obj_bary),    # Contact 1: index
        (middle_obj_face, middle_obj_bary),  # Contact 2: middle
    ]
    object_contacts = [object_contacts] * n_frames  # Repeat for all frames

    hand_contacts = {}

    # Thumb contact
    hand_contacts[thumb_component_idx] = []
    for frame_idx in range(n_frames):
        hand_contacts[thumb_component_idx].append(
            [(thumb_face_idx, thumb_bary, 0)]
        )

    # Index finger contact
    hand_contacts[index_component_idx] = []
    for frame_idx in range(n_frames):
        hand_contacts[index_component_idx].append(
            [(index_face_idx, index_bary, 1)]
        )

    # Middle finger contact
    hand_contacts[middle_component_idx] = []
    for frame_idx in range(n_frames):
        hand_contacts[middle_component_idx].append(
            [(middle_face_idx, middle_bary, 2)]
        )

    return hand_contacts, object_contacts


def create_adroit_xml_for_optimization():
    """
    Create a simplified MuJoCo XML for Adroit optimization.
    Uses the DexMV Adroit relocate model with cylinder object.
    Returns the path to the temporary XML file.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse('Adroit/adroit_relocate.xml')
    root = tree.getroot()

    # Add ground and lighting
    asset = root.find('asset')
    if asset is not None:
        ET.SubElement(asset, 'texture', name='texplane', type='2d', builtin='checker',
                     rgb1='0.2 0.3 0.4', rgb2='0.1 0.15 0.2', width='512', height='512')
        ET.SubElement(asset, 'material', name='MatGnd', reflectance='0.5',
                     texture='texplane', texrepeat='2 2', texuniform='true')

    worldbody = root.find('worldbody')
    if worldbody is not None:
        ET.SubElement(worldbody, 'light', directional='false', diffuse='0.8 0.8 0.8',
                     specular='0.3 0.3 0.3', pos='0 0 4.0', dir='0 0 -1')
        ET.SubElement(worldbody, 'geom', name='ground', pos='0 -0.7 0', size='2 2 0.1',
                     material='MatGnd', type='plane', contype='1', conaffinity='1')

        # Add object with freejoint (cylinder matching DexMV mustard bottle)
        object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', pos='0 0 0.3')
        ET.SubElement(object_body, 'freejoint')
        ET.SubElement(object_body, 'geom', type='cylinder', size='0.035 0.08',
                     rgba='0.9 0.8 0.2 1', mass='0.1')

    # Write to temporary file in Adroit directory so relative paths work
    temp_xml_path = 'Adroit/temp_optimization_scene.xml'
    tree.write(temp_xml_path, encoding='unicode')

    return temp_xml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize Adroit trajectory with contact constraints')
    parser.add_argument('data_path', type=str, help='Path to .pt file with trajectory data')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--n-iter', type=int, default=100, help='Number of iterations (default: 100)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for optimization: cpu, cuda, or mps (default: cpu)')
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from: {args.data_path}")
    hand_qpos, object_qpos, n_frames = load_adroit_trajectory(args.data_path)
    qpos_copy = hand_qpos.copy()
    object_qpos_original = object_qpos.copy() 

    # Create cylinder object mesh
    object_mesh = trimesh.creation.cylinder(radius=0.035, height=0.16, sections=32)
    print(f"Object mesh: {len(object_mesh.vertices)} vertices, {len(object_mesh.faces)} faces")

    # Create MuJoCo model
    xml_path = create_adroit_xml_for_optimization()
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # print(f"\nModel info:")
    # print(f"  nq: {m.nq}")
    # print(f"  nv: {m.nv}")
    # print(f"  nbody: {m.nbody}")

    # Find fingertip body IDs
    thumb_distal_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'thdistal')
    index_distal_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'ffdistal')
    middle_distal_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'mfdistal')

    # print(f"\nFingertip body IDs:")
    # print(f"  Thumb distal: {thumb_distal_id}")
    # print(f"  Index distal: {index_distal_id}")
    # print(f"  Middle distal: {middle_distal_id}")

    # Determine hand component offset
    hand_component_offset = 1  # Start from forearm (body 0 is world)
    hand_components_len = m.nbody - 2  # Exclude world and object

    # print(f"\nKinematic info:")
    # print(f"  Hand component offset: {hand_component_offset}")
    # print(f"  Hand components length: {hand_components_len}")

    # Extract meshes for all hand bodies
    from handContacts import get_mesh_for_body
    hand_components = []
    for body_id in range(hand_component_offset, hand_component_offset + hand_components_len):
        try:
            faces, verts = get_mesh_for_body(m, body_id)
            hand_components.append((faces, verts))
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"  Body {body_id} ({body_name}): {len(verts)} vertices, {len(faces)} faces")
        except Exception as e:
            # Some bodies might not have meshes, create dummy
            dummy_verts = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
            dummy_faces = np.array([[0, 1, 2]])
            hand_components.append((dummy_faces, dummy_verts))
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"  Body {body_id} ({body_name}): No mesh, using dummy")

    # print("\nCreating contact data from frame 200...")
    hand_contacts, object_contacts = create_real_contacts_from_frame200(
        object_mesh, n_frames, hand_components,
        thumb_distal_id, index_distal_id, middle_distal_id,
        hand_component_offset
    )

    kinematic_tree = extract_kinematic_tree(m, hand_component_offset, hand_components_len)

    # Precompute kinematic tree tensors
    kinematic_tree_torch = precompute_kinematic_tree_tensors(
        kinematic_tree, hand_component_offset, hand_components_len, args.device
    )

    ##### Object Retargeting #####
    wall_barrier = ('rect', {
        'pos': np.array([0.1, 0.1, 0.5]),
        'dims': np.array([0.3, 0.15, 0.02])
    })

    barriers = [wall_barrier]
    # barriers = []

    # (pos, t_start, t_end, optional_rotation)
    start_pos = object_qpos[:3, 0].copy()
    end_pos = object_qpos[:3, n_frames-1].copy()

    # New start and end positions
    new_start_pos_shift = np.array([0.0, 0.0, 0.0])
    end_final_pos_shift = np.array([0.0, 0.0, 0.6])
    end_obj_pos_shift = np.array([0.0, 0.0, 0.0])

    newStartPos = start_pos + new_start_pos_shift
    endFinalPos = end_pos + end_final_pos_shift
    endObjPos = endFinalPos + end_obj_pos_shift


    waypts = [
        # (np.array([0.0, 0.0, 0.3]), 0.5, 0.5),
    ]

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

    barriers = process_barriers(barriers)

    object_radius = 0.1

    visual_destination_marker = ('rect', {
        'pos': np.array([endObjPos[0] - 0.09, 0.1, endObjPos[2] -1.4*object_radius]),
        'dims': np.array([0.3, 0.15, 0.02])
    })
    visual_barriers = [visual_destination_marker]


    startIdx = 0
    endIdx = n_frames - 1

    # Get waypoint frame indices
    waypts_idx = np.zeros(len(waypts), dtype=int)
    times = np.linspace(0, 1, n_frames)
    for j in range(len(waypts)):
        waypts_idx[j] = np.abs(times - waypts[j][1]).argmin()

    print(f"\nContact frames: start={startIdx}, end={endIdx}")
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
                segment_start = startIdx
                segment_end = waypts_idx[i]
                target_start = newStartPos
                target_end = waypts[i][0]
            elif i == len(waypts):
                segment_start = waypts_idx[i-1]
                segment_end = endIdx + 1
                target_start = waypts[i-1][0]
                target_end = endObjPos
            else:
                segment_start = waypts_idx[i-1]
                segment_end = waypts_idx[i]
                target_start = waypts[i-1][0]
                target_end = waypts[i][0]

            # print(f"Transforming segment {i}: frames {segment_start}-{segment_end}")
            pos_waypt_constrained[:, segment_start:segment_end] = transformSplines(
                pos_waypt_constrained[:, segment_start:segment_end],
                target_start,
                target_end
            )

    object_rotation_qpos = apply_waypoint_rotations(
        object_qpos[3:7, :], waypts, waypts_idx, startIdx, n_frames
    )

    barrierConstraints(pos_waypt_constrained.T, object_radius, barriers)

    trajectory = read_obj("scene/curve_positions.obj")  # (n_frames, 3)

    extra_pt_count = 0

    new_object_qpos = np.zeros((7, trajectory.shape[0]))
    new_object_qpos[:3, :] = trajectory.T

    # Handle rotation with potential extra points
    if extra_pt_count == 0:
        new_object_qpos[3:, :] = object_rotation_qpos
    else:
        new_object_qpos[3:, :n_frames] = object_rotation_qpos
        new_object_qpos[3:, n_frames:] = object_rotation_qpos[:, -1].reshape(4, 1)

    old_n_frames = hand_qpos.shape[1]
    object_qpos = new_object_qpos
    n_frames = object_qpos.shape[1]


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


    ###### Hand Optimization
    original_hand_to_object = object_qpos_original[:3, 0] - qpos_copy[0:3, 0]

    for frame in range(n_frames):
        hand_qpos[0:3, frame] = object_qpos[:3, frame] - original_hand_to_object

    # Reset wrist rotation to a neutral pose (approximately horizontal)
    for frame in range(n_frames):
        hand_qpos[3:6, frame] = np.array([0.0, 0.0, 0.0])  # Neutral rotation

    for frame in range(n_frames):
        print(f"\nFrame {frame}/{n_frames}")

        # Precompute local hand contacts for this frame
        local_contacts_cache = precompute_local_hand_contacts(
            hand_contacts, hand_components, hand_components_len,
            frame, args.device
        )

        # Use previous frame's optimized result as initialization
        if frame == 0:
            initial_qpos = hand_qpos[:, frame]
        else:
            # Use previous frame's optimized qpos as initialization
            initial_qpos = hand_qpos[:, frame - 1].copy()

        object_qpos_frame = object_qpos[:, frame]

        # print(f"\nInitial state analysis:")
        # d.qpos[:30] = initial_qpos
        # d.qpos[30:37] = object_qpos_frame
        # mujoco.mj_forward(m, d)

        # from handContacts import get_local_pos
        # for hand_component_id in hand_contacts:
        #     contacts_this_frame = hand_contacts[hand_component_id][frame]
        #     if contacts_this_frame is None:
        #         continue
        #     component_faces, component_verts = hand_components[hand_component_id]
        #     body_id = hand_component_id + hand_component_offset

        #     for contact in contacts_this_frame:
        #         face_id, bary_coords, object_contact_idx = contact
        #         local_pos = get_local_pos(face_id, bary_coords, component_faces, component_verts)
        #         body_pos = d.xpos[body_id]
        #         body_rot = d.xmat[body_id].reshape(3, 3)
        #         global_hand_pos = body_rot @ local_pos + body_pos

        #         # Get corresponding object contact
        #         from optimize_contacts import compute_global_object_contacts
        #         global_object_contacts = compute_global_object_contacts(
        #             object_qpos_frame, object_mesh, object_contacts, frame
        #         )
        #         object_pos = global_object_contacts[object_contact_idx]

        #         distance = np.linalg.norm(global_hand_pos - object_pos)
        #         print(f"  Component {hand_component_id}, Contact {object_contact_idx}:")
        #         print(f"    Hand:   {global_hand_pos}")
        #         print(f"    Object: {object_pos}")
        #         print(f"    Distance: {distance:.6f}")


        hand_qpos[:, frame] = optimize_frame(
            initial_qpos,
            object_qpos_frame,
            m, d, hand_contacts, object_contacts,
            hand_components, hand_component_offset, object_mesh, frame,
            kinematic_tree, lr=args.lr, n_iter=args.n_iter,
            optimize_wrist=True,
            optimize_joints=True,
            agent_type='Adroit',
            print_logs=True,
            device=args.device,
            kinematic_tree_torch=kinematic_tree_torch,
            local_contacts_cache=local_contacts_cache
        )

        # print(f"\nFinal state analysis:")
        # d.qpos[:30] = hand_qpos[:, frame]
        # d.qpos[30:37] = object_qpos_frame
        # mujoco.mj_forward(m, d)

        # for hand_component_id in hand_contacts:
        #     contacts_this_frame = hand_contacts[hand_component_id][frame]
        #     if contacts_this_frame is None:
        #         continue
        #     component_faces, component_verts = hand_components[hand_component_id]
        #     body_id = hand_component_id + hand_component_offset

        #     for contact in contacts_this_frame:
        #         face_id, bary_coords, object_contact_idx = contact
        #         local_pos = get_local_pos(face_id, bary_coords, component_faces, component_verts)
        #         body_pos = d.xpos[body_id]
        #         body_rot = d.xmat[body_id].reshape(3, 3)
        #         global_hand_pos = body_rot @ local_pos + body_pos

        #         from optimize_contacts import compute_global_object_contacts
        #         global_object_contacts = compute_global_object_contacts(
        #             object_qpos_frame, object_mesh, object_contacts, frame
        #         )
        #         object_pos = global_object_contacts[object_contact_idx]

        #         distance = np.linalg.norm(global_hand_pos - object_pos)
        #         print(f"  Component {hand_component_id}, Contact {object_contact_idx}:")
        #         print(f"    Hand:   {global_hand_pos}")
        #         print(f"    Object: {object_pos}")
        #         print(f"    Distance: {distance:.6f}")


    # Smooth the optimized trajectory
    from smoothspline import smooth_adroit_trajectory
    hand_qpos = smooth_adroit_trajectory(
        hand_qpos,
        n_frames,
        window_length=21,
        polyorder=3,
        visualize=False
    )

    # Save optimized trajectory
    import os
    output_dir = "final_trajectories"
    os.makedirs(output_dir, exist_ok=True)

    output_name = os.path.splitext(os.path.basename(args.data_path))[0]
    hand_traj_path = os.path.join(output_dir, f"{output_name}_optimized_hand.npy")
    object_traj_path = os.path.join(output_dir, f"{output_name}_object.npy")

    np.save(hand_traj_path, hand_qpos)
    np.save(object_traj_path, object_qpos)


    dt = 0.03
    m.opt.timestep = dt

    from handContacts import get_local_pos, local_to_global

    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()

            if i >= n_frames:
                i = 0

            d.qpos[:30] = hand_qpos[:, i]
            d.qpos[30:37] = object_qpos[:, i]

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

            # Draw contact points on object
            quat_scipy = np.array([object_qpos[4, i], object_qpos[5, i],
                                  object_qpos[6, i], object_qpos[3, i]])
            rotation = R.from_quat(quat_scipy)
            rotation_matrix = rotation.as_matrix()

            # Handle both formats: vertex indices or (face_id, bary_coords)
            for contact_data in object_contacts[i]:
                if geometry_count >= max_geoms:
                    break

                if isinstance(contact_data, tuple) and len(contact_data) == 2:
                    # New format: (face_id, bary_coords)
                    face_id, bary_coords = contact_data
                    face = object_mesh.faces[face_id]
                    v0 = object_mesh.vertices[face[0]]
                    v1 = object_mesh.vertices[face[1]]
                    v2 = object_mesh.vertices[face[2]]
                    local_vertex = bary_coords[0] * v0 + bary_coords[1] * v1 + bary_coords[2] * v2
                else:
                    # Legacy format: vertex index
                    local_vertex = object_mesh.vertices[contact_data]

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

            # Draw hand contacts
            for hand_component_id in hand_contacts:
                contacts_this_frame = hand_contacts[hand_component_id][i]
                if contacts_this_frame is None:
                    continue

                # Get the mesh for this component
                component_faces, component_verts = hand_components[hand_component_id]

                # Convert component ID to body ID
                body_id = hand_component_id + hand_component_offset

                for contact in contacts_this_frame:
                    if geometry_count >= max_geoms:
                        break

                    face_id, bary_coords, object_contact_idx = contact

                    local_pos = get_local_pos(face_id, bary_coords, component_faces, component_verts)
                    global_pos = local_to_global(local_pos, body_id, d)

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.008, 0, 0],
                        pos=global_pos,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 1, 0, 1])
                    )
                    geometry_count += 1

            # Draw barriers
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
                        size=np.array(config['dims']) / 2,
                        pos=config['pos'],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.5, 0.5, 0.5, 0.7])
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

            # Draw visual-only barriers
            for barrier in visual_barriers:
                if geometry_count >= max_geoms:
                    break
                if isinstance(barrier, str):
                    continue
                barr_type, config = barrier
                if barr_type == 'rect':
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=np.array(config['dims']) / 2,
                        pos=config['pos'],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.2, 0.6, 1.0, 0.3])
                    )
                    geometry_count += 1

            # Draw waypoints
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

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)