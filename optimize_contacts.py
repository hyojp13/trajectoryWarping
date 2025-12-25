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
        object_contacts: list of vertex indices per frame
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
    
    contact_vertex_indices = object_contacts[frame_idx]
    if not isinstance(contact_vertex_indices, np.ndarray):
        contact_vertex_indices = np.array(contact_vertex_indices)

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
    return total_loss


def optimize_frame(qpos_init, object_qpos, m, d, hand_contacts, object_contacts,
                  hand_components, hand_component_offset, object_mesh, frame_idx,
                  kinematic_tree, lr=0.01, n_iter=100, optimize_wrist=True, optimize_joints=True,
                  agent_type='MANO_right', print_logs=False, device=None,
                  kinematic_tree_torch=None, local_contacts_cache=None):
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

    for iteration in range(n_iter):
        optimizer.zero_grad()

        # Compute global hand contacts using differentiable FK with cached data
        global_hand_contacts_torch = compute_global_hand_contacts_torch(
            qpos_torch, kinematic_tree_torch, hand_contacts,
            hand_components, hand_component_offset, hand_components_len,
            root_body_id, frame_idx, local_contacts_cache=local_contacts_cache
        )

        # Compute loss
        loss = compute_contact_loss_torch(
            global_hand_contacts_torch,
            global_object_contacts_torch,
            hand_contacts,
            frame_idx
        )

        # Backpropagate
        loss.backward()

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
            else:  # Allegro_right
                # Normalize wrist quaternion only
                if optimize_mask[3] and optimize_mask[4] and optimize_mask[5] and optimize_mask[6]:
                    qpos_torch.data[3:7] = qpos_torch.data[3:7] / torch.norm(qpos_torch.data[3:7])
                    if qpos_torch.data[3] < 0:
                        qpos_torch.data[3:7] = -qpos_torch.data[3:7]

        if print_logs and (iteration+1) % 30 == 0:
            print(f"  Iteration {iteration}, Loss: {loss.item():.6f}")

    return qpos_torch.detach().cpu().numpy()




def optimize_trajectory(qpos_trajectory, object_qpos_trajectory, m, d, hand_contacts,
                       object_contacts, hand_components, hand_component_offset,
                       object_mesh, agent_type='MANO_right', optimize_wrist=True,
                       optimize_joints=True, lr=0.01, n_iter=100, start_frame=0, end_frame=None,
                       device=None):
    """
    Optimize hand trajectory across specified frame range with GPU acceleration.
    Optimized version with cached data structures.

    Args:
        qpos_trajectory: (nq, n_frames) numpy array
        object_qpos_trajectory: (7, n_frames) numpy array
        m: MuJoCo model
        d: MuJoCo data
        hand_contacts: dict of hand contacts
        object_contacts: list of object contacts per frame
        hand_components: list of mesh components
        hand_component_offset: starting body_id
        object_mesh: trimesh object
        agent_type: 'MANO_right' or 'Allegro_right'
        optimize_wrist: optimize wrist position and rotation
        optimize_joints: optimize joint rotations
        lr: learning rate
        n_iter: iterations per frame
        start_frame: first frame to optimize (inclusive), default 0
        end_frame: last frame to optimize (inclusive), default None (optimize all)
        device: torch device to use (None for auto-detection: MPS on M-series, CUDA on NVIDIA, CPU otherwise)

    Returns:
        optimized qpos_trajectory (nq, n_frames)
    """
    from differentiable_fk import precompute_kinematic_tree_tensors, precompute_local_hand_contacts

    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f"Using device: {device}")
    n_frames = qpos_trajectory.shape[1]

    # Set end_frame if not specified
    if end_frame is None:
        end_frame = n_frames - 1

    # Validate frame range
    start_frame = max(0, start_frame)
    end_frame = min(n_frames - 1, end_frame)

    if start_frame > end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be <= end_frame ({end_frame})")

    qpos_optimized = qpos_trajectory.copy()
    num_frames_to_optimize = end_frame - start_frame + 1

    kinematic_tree = extract_kinematic_tree(m, hand_component_offset, len(hand_components))

    # Pre-compute kinematic tree tensors once (reused across all frames)
    kinematic_tree_torch = precompute_kinematic_tree_tensors(
        kinematic_tree, hand_component_offset, len(hand_components), device
    )

    print(f"Optimizing frames {start_frame} to {end_frame} (inclusive, {num_frames_to_optimize} frames total)")

    for frame_idx in range(start_frame, end_frame + 1):
        frame_num = frame_idx - start_frame + 1
        print(f"Optimizing frame {frame_idx} ({frame_num}/{num_frames_to_optimize})")

        # Pre-compute local hand contacts for this frame
        local_contacts_cache = precompute_local_hand_contacts(
            hand_contacts, hand_components, len(hand_components), frame_idx, device
        )

        qpos_opt = optimize_frame(
            qpos_trajectory[:, frame_idx],
            object_qpos_trajectory[:, frame_idx],
            m, d, hand_contacts, object_contacts,
            hand_components, hand_component_offset, object_mesh, frame_idx, kinematic_tree,
            lr=lr, n_iter=n_iter, optimize_wrist=optimize_wrist,
            optimize_joints=optimize_joints, agent_type=agent_type, device=device,
            kinematic_tree_torch=kinematic_tree_torch,
            local_contacts_cache=local_contacts_cache
        )

        qpos_optimized[:, frame_idx] = qpos_opt

    return qpos_optimized
