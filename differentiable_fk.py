"""
Differentiable forward kinematics implementation in PyTorch.
Extracts kinematic tree structure from MuJoCo and computes transformations manually.
"""

import numpy as np
import torch
import mujoco


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix (3x3).
    Differentiable in PyTorch.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)])
    ])
    return R


def extract_kinematic_tree(model, hand_component_offset, hand_components_len):
    """
    Extract kinematic tree structure from MuJoCo model.
    
    Returns:
        body_info: dict mapping body_id -> {
            'parent_id': parent body id,
            'joints': list of {
                'qpos_start': starting qpos index,
                'qpos_count': number of qpos elements,
                'type': 'ball', 'hinge', 'slide',
                'axis': axis vector (for hinge/slide)
            },
            'body_pos': local position relative to parent,
            'body_quat': local orientation relative to parent (w, x, y, z)
        }
    """
    body_info = {}
    
    # Get body IDs for hand components (including root)
    # Root body is typically hand_component_offset
    root_body_id = hand_component_offset
    hand_body_ids = list(range(hand_component_offset, hand_component_offset + hand_components_len))
    
    for body_id in hand_body_ids:
        parent_id = model.body_parentid[body_id]
        
        # Get body position and orientation relative to parent
        body_pos = model.body_pos[body_id].copy()
        body_quat = model.body_quat[body_id].copy()  # (w, x, y, z)
        
        # Find joints attached to this body
        joints = []
        
        for jnt_id in range(model.njnt):
            if model.jnt_bodyid[jnt_id] == body_id:
                qposadr = model.jnt_qposadr[jnt_id]
                jnt_type = model.jnt_type[jnt_id]
                axis = model.jnt_axis[jnt_id].copy() if jnt_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE] else None
                
                if jnt_type == mujoco.mjtJoint.mjJNT_FREE:  # 6 DOF (3 translation + 4 quaternion rotation)
                    joints.append({
                        'qpos_start': qposadr,
                        'qpos_count': 7,
                        'type': 'free',
                        'axis': None
                    })
                elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:  # 3D rotation (quaternion: 4 DOFs)
                    joints.append({
                        'qpos_start': qposadr,
                        'qpos_count': 4,
                        'type': 'ball',
                        'axis': None
                    })
                elif jnt_type == mujoco.mjtJoint.mjJNT_HINGE:  # 1 DOF rotation
                    joints.append({
                        'qpos_start': qposadr,
                        'qpos_count': 1,
                        'type': 'hinge',
                        'axis': axis
                    })
                elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:  # 1 DOF translation
                    joints.append({
                        'qpos_start': qposadr,
                        'qpos_count': 1,
                        'type': 'slide',
                        'axis': axis
                    })
        
        body_info[body_id] = {
            'parent_id': parent_id,
            'joints': joints,
            'body_pos': body_pos,
            'body_quat': body_quat
        }
    
    return body_info


def precompute_kinematic_tree_tensors(kinematic_tree, hand_component_offset, hand_components_len, device):
    """
    Pre-convert kinematic tree numpy arrays to torch tensors on the target device.
    This avoids repeated conversions during optimization.

    Returns:
        kinematic_tree_torch: modified kinematic tree with torch tensors
    """
    kinematic_tree_torch = {}

    for body_id in range(hand_component_offset, hand_component_offset + hand_components_len):
        if body_id not in kinematic_tree:
            continue

        info = kinematic_tree[body_id].copy()

        # Convert body pos and quat to tensors
        info['body_pos'] = torch.tensor(info['body_pos'], dtype=torch.float32, device=device)
        info['body_quat'] = torch.tensor(info['body_quat'], dtype=torch.float32, device=device)

        # Convert joint axes to tensors
        joints_torch = []
        for joint_info in info['joints']:
            joint_torch = joint_info.copy()
            if joint_info['axis'] is not None:
                joint_torch['axis'] = torch.tensor(joint_info['axis'], dtype=torch.float32, device=device)
            joints_torch.append(joint_torch)
        info['joints'] = joints_torch

        kinematic_tree_torch[body_id] = info

    return kinematic_tree_torch


def compute_forward_kinematics_torch(qpos_torch, kinematic_tree, hand_component_offset, hand_components_len, root_body_id):
    """
    Compute forward kinematics in PyTorch for all hand bodies.
    Optimized version with pre-allocated tensors and reduced dynamic allocations.

    Args:
        qpos_torch: (nq,) torch tensor of joint positions
        kinematic_tree: dict from extract_kinematic_tree (with pre-converted torch tensors)
        hand_component_offset: starting body_id for hand components
        hand_components_len: number of hand components
        root_body_id: body ID of root (wrist)

    Returns:
        transforms: dict mapping body_id -> {
            'pos': (3,) world position,
            'rot': (3, 3) world rotation matrix
        }
    """
    transforms = {}
    hand_body_ids = list(range(hand_component_offset, hand_component_offset + hand_components_len))
    device = qpos_torch.device
    dtype = qpos_torch.dtype

    # Pre-allocate commonly used tensors
    zeros_3 = torch.zeros(3, dtype=dtype, device=device)
    eye_3 = torch.eye(3, dtype=dtype, device=device)

    # Process bodies in order (parent before children)
    # For now, we'll process them in order and assume parent is always processed first
    # This works for tree structures

    def get_transforms_recursive(body_id):
        if body_id in transforms:
            return transforms[body_id]

        if body_id == root_body_id:
            # Root body: transformation comes from root joints
            info = kinematic_tree[body_id]

            # Get body's local rotation from XML (relative to parent/world)
            body_quat_local = info['body_quat']
            body_rot_local = quaternion_to_rotation_matrix(body_quat_local)
            body_pos_local = info['body_pos']

            # Process joints in body-local frame
            joint_pos = zeros_3.clone()
            joint_rot = eye_3.clone()

            for joint_info in info['joints']:
                qpos_start = joint_info['qpos_start']
                qpos_count = joint_info['qpos_count']
                jnt_type = joint_info['type']
                axis = joint_info['axis']  # Axis in body-local frame

                if qpos_start + qpos_count > len(qpos_torch):
                    continue

                if jnt_type == 'free':
                    # Free joint: qpos[start:start+7] = [x, y, z, qw, qx, qy, qz]
                    joint_pos = joint_pos + qpos_torch[qpos_start:qpos_start+3]
                    root_quat = qpos_torch[qpos_start+3:qpos_start+7]
                    root_quat = root_quat / torch.norm(root_quat)  # Normalize
                    joint_rot = quaternion_to_rotation_matrix(root_quat)
                elif jnt_type == 'slide':
                    # Slide joint: translation along axis (in body-local frame)
                    if axis is not None:
                        joint_pos = joint_pos + qpos_torch[qpos_start] * axis
                elif jnt_type == 'ball':
                    # Ball joint: rotation as quaternion
                    root_quat = qpos_torch[qpos_start:qpos_start+4]
                    root_quat = root_quat / torch.norm(root_quat)  # Normalize
                    joint_rot = quaternion_to_rotation_matrix(root_quat)
                elif jnt_type == 'hinge':
                    # Hinge joint: rotation around axis (in body-local frame)
                    if axis is not None:
                        angle = qpos_torch[qpos_start]
                        axis_normalized = axis / torch.norm(axis)
                        K = torch.stack([
                            torch.stack([zeros_3[0], -axis_normalized[2], axis_normalized[1]]),
                            torch.stack([axis_normalized[2], zeros_3[0], -axis_normalized[0]]),
                            torch.stack([-axis_normalized[1], axis_normalized[0], zeros_3[0]])
                        ])
                        c, s = torch.cos(angle), torch.sin(angle)
                        rot_hinge = eye_3 + s * K + (1 - c) * (K @ K)
                        joint_rot = joint_rot @ rot_hinge

            # Transform from body-local to world:
            # pos_world = body_rot_local @ joint_pos + body_pos_local
            # rot_world = body_rot_local @ joint_rot
            pos = body_rot_local @ joint_pos + body_pos_local
            rot = body_rot_local @ joint_rot

            transforms[body_id] = {'pos': pos, 'rot': rot}
            return transforms[body_id]

        if body_id < hand_component_offset:
            # Body outside hand - world transform
            transforms[body_id] = {'pos': zeros_3.clone(), 'rot': eye_3.clone()}
            return transforms[body_id]

        # Get parent transform
        info = kinematic_tree[body_id]
        parent_id = info['parent_id']

        # Recursively get parent transform
        if parent_id not in transforms and parent_id >= hand_component_offset:
            get_transforms_recursive(parent_id)

        if parent_id >= hand_component_offset and parent_id in transforms:
            parent_transform = transforms[parent_id]
            parent_pos = parent_transform['pos']
            parent_rot = parent_transform['rot']
        else:
            # Parent is outside hand (world)
            parent_pos = zeros_3
            parent_rot = eye_3

        # Body's local transform from XML (relative to parent) - already tensors
        body_pos_local = info['body_pos']
        body_quat_local = info['body_quat']
        body_rot_local = quaternion_to_rotation_matrix(body_quat_local)

        # Joint transform (compose all joints for this body)
        joint_pos = zeros_3.clone()
        joint_rot = eye_3.clone()

        for joint_info in info['joints']:
            qpos_start = joint_info['qpos_start']
            qpos_count = joint_info['qpos_count']
            jnt_type = joint_info['type']
            axis = joint_info['axis']

            if qpos_start + qpos_count > len(qpos_torch):
                continue

            if jnt_type == 'free':
                # Free joint: qpos[qpos_start:qpos_start+7] = [x, y, z, qw, qx, qy, qz]
                # Translation (3 DOFs)
                joint_pos = joint_pos + qpos_torch[qpos_start:qpos_start+3]
                # Rotation (4 DOFs - quaternion)
                joint_quat = qpos_torch[qpos_start+3:qpos_start+7]
                joint_quat = joint_quat / torch.norm(joint_quat)  # Normalize
                joint_rot_jnt = quaternion_to_rotation_matrix(joint_quat)
                joint_rot = joint_rot @ joint_rot_jnt  # Compose rotations
            elif jnt_type == 'ball':
                # Quaternion joint: qpos[qpos_start:qpos_start+4] = [w, x, y, z]
                joint_quat = qpos_torch[qpos_start:qpos_start+4]
                joint_quat = joint_quat / torch.norm(joint_quat)  # Normalize
                joint_rot_jnt = quaternion_to_rotation_matrix(joint_quat)
                joint_rot = joint_rot @ joint_rot_jnt  # Compose rotations
            elif jnt_type == 'slide':
                # Translation along axis
                # NOTE: axis is in parent frame. Since we apply T_joint BEFORE T_body,
                # the axis should NOT be transformed by body_rot_local
                if axis is not None:
                    translation = qpos_torch[qpos_start] * axis
                    joint_pos = joint_pos + translation
            elif jnt_type == 'hinge':
                # Rotation around axis (Rodrigues' rotation formula)
                # NOTE: axis is in parent frame. Since we apply T_joint BEFORE T_body,
                # the axis should NOT be transformed by body_rot_local
                if axis is not None:
                    angle = qpos_torch[qpos_start]
                    axis_normalized = axis / torch.norm(axis)  # Normalize
                    # Rodrigues' formula - pre-compute skew-symmetric matrix
                    K = torch.stack([
                        torch.stack([zeros_3[0], -axis_normalized[2], axis_normalized[1]]),
                        torch.stack([axis_normalized[2], zeros_3[0], -axis_normalized[0]]),
                        torch.stack([-axis_normalized[1], axis_normalized[0], zeros_3[0]])
                    ])
                    c, s = torch.cos(angle), torch.sin(angle)
                    rot_hinge = eye_3 + s * K + (1 - c) * (K @ K)
                    joint_rot = joint_rot @ rot_hinge

        # Compute world transform: T_world = T_parent * T_body_local * T_joint
        # The joint is defined in the body's local frame (after body transform)
        # For positions: p_world = parent_rot @ (body_pos_local + body_rot_local @ joint_pos) + parent_pos
        # For rotations: R_world = parent_rot @ body_rot_local @ joint_rot

        # DEBUG: Print for link0
        if body_id == hand_component_offset and False:  # Disabled by default
            print(f'DEBUG link0 (body {body_id}):')
            print(f'  body_pos_local: {body_pos_local}')
            print(f'  joint_pos: {joint_pos}')
            print(f'  parent_pos: {parent_pos}')

        world_pos = parent_rot @ (body_pos_local + body_rot_local @ joint_pos) + parent_pos
        world_rot = parent_rot @ body_rot_local @ joint_rot

        transforms[body_id] = {'pos': world_pos, 'rot': world_rot}
        return transforms[body_id]

    # Process all bodies
    for body_id in hand_body_ids:
        get_transforms_recursive(body_id)

    return transforms


def precompute_local_hand_contacts(hand_contacts, hand_components, hand_components_len,
                                    frame_idx, device):
    """
    Pre-compute local hand contact positions and cache them as torch tensors.
    This avoids repeated computation and CPU-GPU transfers during optimization.

    Args:
        hand_contacts: dict of hand contacts
        hand_components: list of (faces, verts) tuples
        hand_components_len: number of components
        frame_idx: frame index
        device: torch device

    Returns:
        dict mapping hand_component_id -> torch tensor (N, 3) of local positions
    """
    from handContacts import get_local_pos

    local_hand_contacts_cache = {}

    for hand_component_id in range(hand_components_len):
        # Skip if this component doesn't have contacts
        if hand_component_id not in hand_contacts:
            continue

        contacts_this_frame = hand_contacts[hand_component_id][frame_idx]
        if contacts_this_frame is None:
            continue

        # Compute local positions
        local_positions = []
        for contact in contacts_this_frame:
            face_id, bary_coords, object_contact_idx = contact
            local_pos = get_local_pos(
                face_id, bary_coords,
                hand_components[hand_component_id][0],
                hand_components[hand_component_id][1]
            )
            local_positions.append(local_pos)

        if len(local_positions) > 0:
            local_positions_torch = torch.tensor(
                np.array(local_positions),
                dtype=torch.float32,
                device=device
            )  # (N, 3)
            local_hand_contacts_cache[hand_component_id] = local_positions_torch

    return local_hand_contacts_cache


def compute_global_hand_contacts_torch(qpos_torch, kinematic_tree, hand_contacts,
                                       hand_components, hand_component_offset,
                                       hand_components_len, root_body_id, frame_idx,
                                       local_contacts_cache=None):
    """
    Compute global hand contact positions using differentiable forward kinematics.
    Optimized version with cached local positions.

    Args:
        qpos_torch: (nq,) torch tensor with requires_grad=True
        kinematic_tree: dict from extract_kinematic_tree (with pre-converted torch tensors)
        hand_contacts: dict of hand contacts
        hand_components: list of (faces, verts) tuples
        hand_component_offset: starting body_id
        hand_components_len: number of components
        root_body_id: root body ID
        frame_idx: frame index
        local_contacts_cache: optional pre-computed local contact positions

    Returns:
        dict mapping hand_component_id -> torch tensor (N, 3) of global positions
    """
    # Compute forward kinematics
    transforms = compute_forward_kinematics_torch(
        qpos_torch, kinematic_tree, hand_component_offset,
        hand_components_len, root_body_id
    )

    global_hand_contacts = {}

    # Use cached local positions if available
    if local_contacts_cache is not None:
        for hand_component_id in local_contacts_cache:
            body_id = hand_component_id + hand_component_offset

            if body_id not in transforms:
                continue

            body_transform = transforms[body_id]
            body_pos = body_transform['pos']  # (3,)
            body_rot = body_transform['rot']  # (3, 3)

            local_positions_torch = local_contacts_cache[hand_component_id]

            # Transform to global: global = R @ local.T + pos
            global_positions = (body_rot @ local_positions_torch.T).T + body_pos.unsqueeze(0)  # (N, 3)
            global_hand_contacts[hand_component_id] = global_positions
    else:
        # Fallback to computing local positions on the fly
        from handContacts import get_local_pos

        for hand_component_id in range(hand_components_len):
            contacts_this_frame = hand_contacts[hand_component_id][frame_idx]
            if contacts_this_frame is None:
                continue

            body_id = hand_component_id + hand_component_offset

            if body_id not in transforms:
                continue

            body_transform = transforms[body_id]
            body_pos = body_transform['pos']  # (3,)
            body_rot = body_transform['rot']  # (3, 3)

            # Compute local positions
            local_positions = []
            for contact in contacts_this_frame:
                face_id, bary_coords, object_contact_idx = contact
                local_pos = get_local_pos(
                    face_id, bary_coords,
                    hand_components[hand_component_id][0],
                    hand_components[hand_component_id][1]
                )
                local_positions.append(local_pos)

            if len(local_positions) > 0:
                local_positions_torch = torch.tensor(
                    np.array(local_positions),
                    dtype=qpos_torch.dtype,
                    device=qpos_torch.device
                )  # (N, 3)

                # Transform to global: global = R @ local.T + pos
                global_positions = (body_rot @ local_positions_torch.T).T + body_pos.unsqueeze(0)  # (N, 3)
                global_hand_contacts[hand_component_id] = global_positions

    return global_hand_contacts
