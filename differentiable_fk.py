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
                
                if jnt_type == mujoco.mjtJoint.mjJNT_BALL:  # 3D rotation (quaternion: 4 DOFs)
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


def compute_forward_kinematics_torch(qpos_torch, kinematic_tree, hand_component_offset, hand_components_len, root_body_id):
    """
    Compute forward kinematics in PyTorch for all hand bodies.
    
    Args:
        qpos_torch: (nq,) torch tensor of joint positions
        kinematic_tree: dict from extract_kinematic_tree
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
    
    # Process bodies in order (parent before children)
    # For now, we'll process them in order and assume parent is always processed first
    # This works for tree structures
    
    def get_transforms_recursive(body_id):
        if body_id in transforms:
            return transforms[body_id]
        
        if body_id == root_body_id:
            # Root body: transformation comes from root joints (qpos[:3] for translation, qpos[3:7] for rotation)
            info = kinematic_tree[body_id]
            pos = torch.zeros(3, dtype=qpos_torch.dtype, device=qpos_torch.device)
            rot = torch.eye(3, dtype=qpos_torch.dtype, device=qpos_torch.device)
            
            # Process root joints
            for joint_info in info['joints']:
                qpos_start = joint_info['qpos_start']
                qpos_count = joint_info['qpos_count']
                jnt_type = joint_info['type']
                axis = joint_info['axis']
                
                if qpos_start + qpos_count > len(qpos_torch):
                    continue
                
                if jnt_type == 'slide':
                    # Root translation joints (typically indices 0, 1, 2)
                    if axis is not None:
                        axis_torch = torch.tensor(axis, dtype=qpos_torch.dtype, device=qpos_torch.device)
                        pos = pos + qpos_torch[qpos_start] * axis_torch
                elif jnt_type == 'ball':
                    # Root rotation joint (typically indices 3:7)
                    root_quat = qpos_torch[qpos_start:qpos_start+4]
                    root_quat = root_quat / torch.norm(root_quat)  # Normalize
                    rot = quaternion_to_rotation_matrix(root_quat)
            
            transforms[body_id] = {'pos': pos, 'rot': rot}
            return transforms[body_id]
        
        if body_id < hand_component_offset:
            # Body outside hand - world transform
            transforms[body_id] = {
                'pos': torch.zeros(3, dtype=qpos_torch.dtype, device=qpos_torch.device),
                'rot': torch.eye(3, dtype=qpos_torch.dtype, device=qpos_torch.device)
            }
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
            parent_pos = torch.zeros(3, dtype=qpos_torch.dtype, device=qpos_torch.device)
            parent_rot = torch.eye(3, dtype=qpos_torch.dtype, device=qpos_torch.device)
        
        # Body's local transform from XML (relative to parent)
        body_pos_local = torch.tensor(info['body_pos'], dtype=qpos_torch.dtype, device=qpos_torch.device)
        body_quat_local = torch.tensor(info['body_quat'], dtype=qpos_torch.dtype, device=qpos_torch.device)
        body_rot_local = quaternion_to_rotation_matrix(body_quat_local)
        
        # Joint transform (compose all joints for this body)
        joint_pos = torch.zeros(3, dtype=qpos_torch.dtype, device=qpos_torch.device)
        joint_rot = torch.eye(3, dtype=qpos_torch.dtype, device=qpos_torch.device)
        
        for joint_info in info['joints']:
            qpos_start = joint_info['qpos_start']
            qpos_count = joint_info['qpos_count']
            jnt_type = joint_info['type']
            axis = joint_info['axis']
            
            if qpos_start + qpos_count > len(qpos_torch):
                continue
            
            if jnt_type == 'ball':
                # Quaternion joint: qpos[qpos_start:qpos_start+4] = [w, x, y, z]
                joint_quat = qpos_torch[qpos_start:qpos_start+4]
                joint_quat = joint_quat / torch.norm(joint_quat)  # Normalize
                joint_rot_jnt = quaternion_to_rotation_matrix(joint_quat)
                joint_rot = joint_rot @ joint_rot_jnt  # Compose rotations
            elif jnt_type == 'slide':
                # Translation along axis
                if axis is not None:
                    axis_torch = torch.tensor(axis, dtype=qpos_torch.dtype, device=qpos_torch.device)
                    translation = qpos_torch[qpos_start] * axis_torch
                    joint_pos = joint_pos + translation
            elif jnt_type == 'hinge':
                # Rotation around axis (Rodrigues' rotation formula)
                if axis is not None:
                    angle = qpos_torch[qpos_start]
                    axis_torch = torch.tensor(axis, dtype=qpos_torch.dtype, device=qpos_torch.device)
                    axis_torch = axis_torch / torch.norm(axis_torch)  # Normalize
                    # Rodrigues' formula
                    K = torch.tensor([
                        [0, -axis_torch[2], axis_torch[1]],
                        [axis_torch[2], 0, -axis_torch[0]],
                        [-axis_torch[1], axis_torch[0], 0]
                    ], dtype=qpos_torch.dtype, device=qpos_torch.device)
                    c, s = torch.cos(angle), torch.sin(angle)
                    rot_hinge = torch.eye(3, dtype=qpos_torch.dtype, device=qpos_torch.device) + \
                               s * K + (1 - c) * (K @ K)
                    joint_rot = joint_rot @ rot_hinge
        
        # Compute world transform: T_world = T_parent * T_body_local * T_joint
        # For positions: p_world = parent_rot @ (body_pos_local + joint_pos) + parent_pos
        # For rotations: R_world = parent_rot @ body_rot_local @ joint_rot
        
        world_pos = parent_rot @ (body_pos_local + joint_pos) + parent_pos
        world_rot = parent_rot @ body_rot_local @ joint_rot
        
        transforms[body_id] = {'pos': world_pos, 'rot': world_rot}
        return transforms[body_id]
    
    # Process all bodies
    for body_id in hand_body_ids:
        get_transforms_recursive(body_id)
    
    return transforms


def compute_global_hand_contacts_torch(qpos_torch, kinematic_tree, hand_contacts,
                                       hand_components, hand_component_offset, 
                                       hand_components_len, root_body_id, frame_idx):
    """
    Compute global hand contact positions using differentiable forward kinematics.
    
    Args:
        qpos_torch: (nq,) torch tensor with requires_grad=True
        kinematic_tree: dict from extract_kinematic_tree
        hand_contacts: dict of hand contacts
        hand_components: list of (faces, verts) tuples
        hand_component_offset: starting body_id
        hand_components_len: number of components
        root_body_id: root body ID
        frame_idx: frame index
        
    Returns:
        dict mapping hand_component_id -> torch tensor (N, 3) of global positions
    """
    # Compute forward kinematics
    transforms = compute_forward_kinematics_torch(
        qpos_torch, kinematic_tree, hand_component_offset, 
        hand_components_len, root_body_id
    )
    
    from handContacts import get_local_pos
    
    global_hand_contacts = {}
    
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
