"""
Optimization module for minimizing hand-object contact correspondence error.
Uses differentiable forward kinematics implemented in PyTorch for optimization.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from differentiable_fk import extract_kinematic_tree


def compute_global_object_contacts(object_qpos, object_mesh, object_contacts, frame_idx):
    """
    Compute global positions of object contacts.

    Args:
        object_qpos: (7,) numpy array [pos(3), quat(4)]
        object_mesh: trimesh object
        object_contacts: list of contact data per frame
                        Each frame contains either:
                        - vertex indices (legacy format)
                        - list of (face_id, bary_coords) tuples (new format)
        frame_idx: frame index to process

    Returns:
        numpy array of global contact positions (N, 3)
    """
    obj_pos = object_qpos[:3]
    obj_quat = object_qpos[3:]  # [w, x, y, z]

    # Convert MuJoCo quat [w,x,y,z] to SciPy quat [x,y,z,w]
    quat_scipy = np.array([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])
    rotation = R.from_quat(quat_scipy)
    rotation_matrix = rotation.as_matrix()  # (3, 3)

    contacts_this_frame = object_contacts[frame_idx]

    # Check format: list of tuples (face, bary) or vertex indices
    if isinstance(contacts_this_frame, list) and len(contacts_this_frame) > 0:
        if isinstance(contacts_this_frame[0], tuple) and len(contacts_this_frame[0]) == 2:
            # New format: list of (face_id, bary_coords) tuples
            local_positions = []
            for face_id, bary_coords in contacts_this_frame:
                # Get face vertices
                face = object_mesh.faces[face_id]
                v0 = object_mesh.vertices[face[0]]
                v1 = object_mesh.vertices[face[1]]
                v2 = object_mesh.vertices[face[2]]

                # Compute position using barycentric coordinates
                local_pos = bary_coords[0] * v0 + bary_coords[1] * v1 + bary_coords[2] * v2
                local_positions.append(local_pos)

            local_vertices = np.array(local_positions)  # (N, 3)
        else:
            # Legacy format: vertex indices
            contact_vertex_indices = np.array(contacts_this_frame)
            local_vertices = object_mesh.vertices[contact_vertex_indices]  # (N, 3)
    else:
        # Legacy format: numpy array of vertex indices
        contact_vertex_indices = np.array(contacts_this_frame)
        local_vertices = object_mesh.vertices[contact_vertex_indices]  # (N, 3)

    global_vertices = (rotation_matrix @ local_vertices.T).T + obj_pos  # (N, 3)

    return global_vertices


def compute_contact_loss_torch(global_hand_contacts_dict, global_object_contacts,
                               hand_contacts, frame_idx):
    """
    Compute loss between hand contacts and object contacts.
    Assumes 1-to-1 correspondence: each object contact has exactly one hand contact.

    Args:
        global_hand_contacts_dict: dict mapping hand_component_id -> torch tensor (N, 3)
        global_object_contacts: torch tensor (M, 3)
        hand_contacts: dict mapping hand_component_id -> list of contacts per frame
        frame_idx: frame index

    Returns:
        scalar torch tensor loss
    """
    # Build a mapping from object_contact_idx to hand contact position
    # hand_contacts[component_id][frame] = [(face_id, bary_coords, object_contact_idx), ...]
    # Each object_contact_idx should appear exactly once across all hand components

    contact_correspondences = {}  # object_contact_idx -> hand contact position (3,)

    for hand_component_id in global_hand_contacts_dict:
        contacts_this_frame = hand_contacts[hand_component_id][frame_idx]
        if contacts_this_frame is None:
            continue

        global_positions = global_hand_contacts_dict[hand_component_id]  # (N, 3)

        for i, contact in enumerate(contacts_this_frame):
            face_id, bary_coords, object_contact_idx = contact
            # Since we have 1-to-1 correspondence, each object_contact_idx should only appear once
            contact_correspondences[object_contact_idx] = global_positions[i]

    # DEBUG: Print contact info on first call
    import __main__
    if not hasattr(__main__, '_contact_corr_debug_printed'):
        print(f"\n[DEBUG compute_contact_loss_torch - correspondences]")
        print(f"  Hand component IDs with contacts: {list(global_hand_contacts_dict.keys())}")
        for hand_component_id in global_hand_contacts_dict:
            global_positions = global_hand_contacts_dict[hand_component_id]
            print(f"  Component {hand_component_id}: {len(global_positions)} contact(s)")
            print(f"    Positions: {global_positions}")
        print(f"  Contact correspondences: {list(contact_correspondences.keys())}")
        __main__._contact_corr_debug_printed = True

    # Compute loss: for each object contact, compute distance to its corresponding hand contact
    losses = []
    for obj_idx in range(len(global_object_contacts)):
        obj_pos = global_object_contacts[obj_idx]  # (3,)

        if obj_idx not in contact_correspondences:
            raise Exception("No corresponding hand contact")
            # losses.append(torch.tensor(1.0, device=obj_pos.device, requires_grad=True))
            # continue

        hand_pos = contact_correspondences[obj_idx]  # (3,)

        # Compute distance between hand contact and object contact
        distance = torch.norm(hand_pos - obj_pos)
        losses.append(distance)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True)

    total_loss = torch.stack(losses).mean()

    # DEBUG: Print loss details on first call
    import __main__
    if not hasattr(__main__, '_contact_loss_debug_printed'):
        print(f"\n[DEBUG compute_contact_loss_torch]")
        print(f"  Number of object contacts: {len(global_object_contacts)}")
        print(f"  Number of contact correspondences: {len(contact_correspondences)}")
        print(f"  Individual distances: {[f'{d.item():.6f}' for d in losses]}")
        print(f"  Mean loss: {total_loss.item():.6f}")
        __main__._contact_loss_debug_printed = True

    return total_loss


def optimize_frame(qpos_init, object_qpos, m, d, hand_contacts, object_contacts,
                  hand_components, hand_component_offset, object_mesh, frame_idx,
                  kinematic_tree, lr=0.01, n_iter=100, optimize_wrist=True, optimize_joints=True,
                  agent_type='MANO_right', print_logs=False, device=None,
                  kinematic_tree_torch=None, local_contacts_cache=None,
                  barriers=None, barrier_weight=1.0, barrier_margin=0.01, barrier_n=2.0):
    """
    Optimize hand qpos for a single frame to minimize contact correspondence error.
    Uses differentiable forward kinematics (PyTorch autograd) with GPU acceleration.
    Optimized version with cached data structures to reduce CPU-GPU transfers.

    Args:
        qpos_init: (nq,) initial qpos (numpy array)
        object_qpos: (7,) object pose (numpy array)
        m: MuJoCo model
        d: MuJoCo data
        hand_contacts: dict of hand contacts per frame
        object_contacts: list of object contact vertex indices per frame
        hand_components: list of (faces, verts) tuples
        hand_component_offset: starting body_id for hand components
        object_mesh: trimesh object mesh
        frame_idx: frame index to optimize
        kinematic_tree: dict from extract_kinematic_tree
        lr: learning rate
        n_iter: number of optimization iterations
        optimize_wrist: whether to optimize wrist position (qpos[:3]) and rotation (qpos[3:7])
        optimize_joints: whether to optimize joint rotations
        agent_type: 'MANO_right' or 'Allegro_right' - affects quaternion handling
        print_logs: whether to print optimization logs
        device: torch device to use (None for auto-detection: MPS on M-series, CUDA on NVIDIA, CPU otherwise)
        kinematic_tree_torch: optional pre-computed kinematic tree with torch tensors
        local_contacts_cache: optional pre-computed local contact positions
        barriers: optional list of barrier specifications for collision avoidance
        barrier_weight: weight for barrier collision loss term (default: 1.0)
        barrier_margin: distance threshold for barrier penalty (default: 0.01)
        barrier_n: steepness parameter for barrier penalty function (default: 2.0)

    Returns:
        optimized qpos (numpy array)
    """
    from differentiable_fk import (
        precompute_kinematic_tree_tensors,
        precompute_local_hand_contacts,
        compute_global_hand_contacts_torch
    )

    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    if print_logs:
        print(f"  Using device: {device}")

    hand_components_len = len(hand_components)
    root_body_id = hand_component_offset  # Wrist is root

    # Determine which parts of qpos to optimize
    if agent_type == 'MANO_right' or agent_type == 'trajectories':
        if optimize_wrist and optimize_joints:
            optimize_mask = np.ones(len(qpos_init), dtype=bool)
        elif optimize_wrist:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[:7] = True  # Translation + wrist rotation
        elif optimize_joints:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[7:] = True  # Only joints
        else:
            return qpos_init
    elif agent_type == 'Franka':
        # Franka: qpos = [pos(3), quat(4), 7 arm joints, 2 finger joints]
        if optimize_wrist and optimize_joints:
            optimize_mask = np.ones(len(qpos_init), dtype=bool)
        elif optimize_wrist:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[:7] = True  # Translation + base rotation
        elif optimize_joints:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[7:] = True  # Arm and finger joints
        else:
            return qpos_init
    elif agent_type == 'Adroit':
        # Adroit: qpos = [ARTx, ARTy, ARTz, ARRx, ARRy, ARRz, 24 hand joints]
        if optimize_wrist and optimize_joints:
            optimize_mask = np.ones(len(qpos_init), dtype=bool)
        elif optimize_wrist:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[:6] = True  # Forearm translation + rotation
        elif optimize_joints:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[6:] = True  # Hand joints only
        else:
            return qpos_init
    else:  # Allegro_right
        if optimize_wrist and optimize_joints:
            optimize_mask = np.ones(len(qpos_init), dtype=bool)
        elif optimize_wrist:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[:7] = True
        elif optimize_joints:
            optimize_mask = np.zeros(len(qpos_init), dtype=bool)
            optimize_mask[7:] = True
        else:
            return qpos_init

    # Pre-compute kinematic tree tensors if not provided
    if kinematic_tree_torch is None:
        kinematic_tree_torch = precompute_kinematic_tree_tensors(
            kinematic_tree, hand_component_offset, hand_components_len, device
        )

    # Pre-compute local hand contacts if not provided
    if local_contacts_cache is None:
        local_contacts_cache = precompute_local_hand_contacts(
            hand_contacts, hand_components, hand_components_len, frame_idx, device
        )

    # Create PyTorch parameter for full qpos on the specified device
    qpos_torch = torch.tensor(qpos_init.copy(), requires_grad=True, dtype=torch.float32, device=device)

    # Compute object contacts (constant) - move to device once
    global_object_contacts_np = compute_global_object_contacts(
        object_qpos, object_mesh, object_contacts, frame_idx
    )
    global_object_contacts_torch = torch.tensor(
        global_object_contacts_np, dtype=torch.float32, requires_grad=False, device=device
    )

    optimizer = torch.optim.Adam([qpos_torch], lr=lr)

    # Precompute joint limits for Franka (avoid recreating tensor in every iteration)
    franka_joint_limits = None
    if agent_type == 'Franka':
        franka_joint_limits = torch.tensor([
            # [-2.8973, 2.8973],   # joint1
            # [-1.7628, 1.7628],   # joint2
            # [-2.8973, 2.8973],   # joint3
            # [-3.0718, -0.0698],  # joint4
            # [-2.8973, 2.8973],   # joint5
            # [-0.0175, 3.7525],   # joint6
            # [-2.8973, 2.8973],   # joint7
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0.0, 0.04],         # finger_joint1
            [0.0, 0.04]          # finger_joint2
        ], device=device, dtype=torch.float32)

    # Build combined barriers list including object mesh as a dynamic barrier
    combined_barriers = []
    if barriers is not None:
        combined_barriers.extend(barriers)

    # # Add object mesh as a dynamic barrier
    # # Debug: print object mesh info
    # if print_logs and frame_idx == 0:
    #     print(f"  Object mesh bounds (local): {object_mesh.bounds}")
    #     print(f"  Object qpos: {object_qpos}")
    #     print(f"  Object mesh vertices: {len(object_mesh.vertices)}, faces: {len(object_mesh.faces)}")

    # object_barrier = ('mesh', {
    #     'vertices': object_mesh.vertices,
    #     'faces': object_mesh.faces,
    #     'qpos': object_qpos
    # })
    # combined_barriers.append(object_barrier)


    for iteration in range(n_iter):
        optimizer.zero_grad()

        # Compute global hand contacts using differentiable FK with cached data
        global_hand_contacts_torch = compute_global_hand_contacts_torch(
            qpos_torch, kinematic_tree_torch, hand_contacts,
            hand_components, hand_component_offset, hand_components_len,
            root_body_id, frame_idx, local_contacts_cache=local_contacts_cache
        )

        # Compute contact correspondence loss
        contact_loss = compute_contact_loss_torch(
            global_hand_contacts_torch,
            global_object_contacts_torch,
            hand_contacts,
            frame_idx
        )

        # Compute barrier collision loss if barriers are provided
        barrier_loss = torch.tensor(0.0, device=device)

        if len(combined_barriers) > 0:
            from barrier_sdf import compute_hand_component_vertices_torch, compute_hand_barrier_loss

            # Compute hand component vertices for barrier checking
            hand_vertices = compute_hand_component_vertices_torch(
                qpos_torch, kinematic_tree_torch, hand_components,
                hand_component_offset, hand_components_len,
                root_body_id, device
            )

            # Compute barrier collision loss (includes both static barriers and object)
            barrier_loss = compute_hand_barrier_loss(
                hand_vertices, combined_barriers,
                n=barrier_n, margin=barrier_margin, device=device
            )

        loss = contact_loss + barrier_weight * barrier_loss
        loss.backward()

        if print_logs and iteration % 10 == 0:
            print(f"  Iteration {iteration}, Loss: {loss.item():.6f}, Contact: {contact_loss.item():.6f}")

        # Zero gradients for non-optimized parameters
        with torch.no_grad():
            for i in range(len(qpos_init)):
                if not optimize_mask[i]:
                    qpos_torch.grad[i] = 0.0

        optimizer.step()

        # Normalize quaternions after update
        with torch.no_grad():
            if agent_type == 'MANO_right' or agent_type == 'trajectories':
                # Normalize wrist quaternion
                if optimize_mask[3] and optimize_mask[4] and optimize_mask[5] and optimize_mask[6]:
                    qpos_torch.data[3:7] = qpos_torch.data[3:7] / torch.norm(qpos_torch.data[3:7])
                    if qpos_torch.data[3] < 0:
                        qpos_torch.data[3:7] = -qpos_torch.data[3:7]
                # Normalize joint quaternions
                for i in range(7, len(qpos_init), 4):
                    if i + 4 <= len(qpos_init) and np.all(optimize_mask[i:i+4]):
                        qpos_torch.data[i:i+4] = qpos_torch.data[i:i+4] / torch.norm(qpos_torch.data[i:i+4])
                        if qpos_torch.data[i] < 0:
                            qpos_torch.data[i:i+4] = -qpos_torch.data[i:i+4]
            elif agent_type == 'Franka':
                # Normalize base quaternion only (Franka has regular joint angles, not quaternions)
                if optimize_mask[3] and optimize_mask[4] and optimize_mask[5] and optimize_mask[6]:
                    qpos_torch.data[3:7] = qpos_torch.data[3:7] / torch.norm(qpos_torch.data[3:7])
                    if qpos_torch.data[3] < 0:
                        qpos_torch.data[3:7] = -qpos_torch.data[3:7]

                # Clamp joint angles to their limits (from MuJoCo model)
                # Apply limits to joints (qpos[7:16])
                for i in range(7, 9):  # 7 arm joints + 2 finger joints
                    qpos_idx = 7 + i
                    if qpos_idx < len(qpos_init) and optimize_mask[qpos_idx]:
                        qpos_torch.data[qpos_idx] = torch.clamp(
                            qpos_torch.data[qpos_idx],
                            franka_joint_limits[i, 0],
                            franka_joint_limits[i, 1]
                        )
            elif agent_type == 'Adroit':
                # Adroit uses Euler angles for forearm rotation (ARRx, ARRy, ARRz at indices 3:6)
                if optimize_wrist:
                    # Clamp forearm rotation angles to [-pi, pi]
                    for i in range(3, 6):
                        if optimize_mask[i]:
                            qpos_torch.data[i] = torch.clamp(
                                qpos_torch.data[i],
                                -np.pi,
                                np.pi
                            )
                # Note: Hand joint limits would go here if needed, but typically
                # MuJoCo handles this through the model definition
            else:  # Allegro_right
                # Normalize wrist quaternion only
                if optimize_mask[3] and optimize_mask[4] and optimize_mask[5] and optimize_mask[6]:
                    qpos_torch.data[3:7] = qpos_torch.data[3:7] / torch.norm(qpos_torch.data[3:7])
                    if qpos_torch.data[3] < 0:
                        qpos_torch.data[3:7] = -qpos_torch.data[3:7]

        if print_logs and (iteration+1) % 30 == 0:
            if len(combined_barriers) > 0:
                print(f"  Iteration {iteration}, Total Loss: {loss.item():.6f}, Contact: {contact_loss.item():.6f}, Collision: {barrier_loss.item():.6f}")
            else:
                print(f"  Iteration {iteration}, Loss: {loss.item():.6f}")

        # Early termination if loss is effectively zero
        if loss.item() < 1e-3:
            if print_logs:
                print(f"  Early termination at iteration {iteration} (loss: {loss.item():.6e})")
            break

    return qpos_torch.detach().cpu().numpy()