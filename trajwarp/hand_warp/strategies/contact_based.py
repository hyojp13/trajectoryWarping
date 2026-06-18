"""Contact-based hand warping (ours, paper Sec. III-D).

Recovers the hand trajectory by optimizing the hand configuration at every output
timestep so that hand-object contact correspondences (taken from the original
demonstration via the timewarp) are reproduced. Pre- and post-contact segments
are rigidly shifted and smoothed.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from trajwarp.object_warp.smoothing import smooth_hand_trajectory
from trajwarp.hand_warp.fk import (
    extract_kinematic_tree,
    precompute_kinematic_tree_tensors,
    precompute_local_hand_contacts,
)
from trajwarp.hand_warp.hand_mesh import get_closest_original_frame
from trajwarp.hand_warp.optimizer import optimize_frame
from trajwarp.hand_warp.strategies._common import finalize_noncontact_hand


def warp_hand(state):
    """Optimize the hand trajectory in place on ``state.qpos``."""
    config = state.config
    m, d = state.model, state.data
    qpos, qpos_copy = state.qpos, state.qpos_copy
    retargeted_spline_pos = state.retargeted_spline_pos
    contact_timewarp = state.contact_timewarp
    hand_contacts, object_contacts = state.hand_contacts, state.object_contacts
    hand_components = state.hand_components
    hand_component_body_ids = state.hand_component_body_ids
    hand_components_len = state.hand_components_len
    obj = state.object_mesh
    startIdx, endIdx = state.start_idx, state.end_idx
    frames = state.frames
    start_frame_count = state.start_frame_count
    contact_frame_count = state.contact_frame_count
    end_frame_count = state.end_frame_count
    start_pos = state.start_pos
    barriers = state.barriers
    AGENT = state.agent
    device = config.optimization_device

    kinematic_tree, root_body_id = extract_kinematic_tree(m, hand_component_body_ids)
    kinematic_tree_torch = precompute_kinematic_tree_tensors(
        kinematic_tree, hand_component_body_ids, device)

    for frame in range(contact_frame_count):
        closet_original_frame = get_closest_original_frame(
            contact_timewarp, (frame + 1) / contact_frame_count)

        if frame % 1 == 0:
            print(frame, "/", contact_frame_count, ":", closet_original_frame,
                  start_frame_count, frames + state.extra_pt_count)
        if closet_original_frame > endIdx:
            closet_original_frame = endIdx

        local_contacts_cache = precompute_local_hand_contacts(
            hand_contacts, hand_components, hand_components_len,
            closet_original_frame, device)

        if frame == 0:
            wrist_init_pos = retargeted_spline_pos[start_frame_count, :3].T
            wrist_init_quat = qpos_copy[3:7, closet_original_frame]
        else:
            prev_obj_pos = retargeted_spline_pos[start_frame_count + frame - 1, :3]
            curr_obj_pos = retargeted_spline_pos[start_frame_count + frame, :3]
            obj_displacement = curr_obj_pos - prev_obj_pos
            wrist_init_pos = qpos[:3, start_frame_count + frame - 1] + obj_displacement

            prev_obj_quat_mj = retargeted_spline_pos[start_frame_count + frame - 1, 3:7]
            curr_obj_quat_mj = retargeted_spline_pos[start_frame_count + frame, 3:7]

            prev_obj_rot = R.from_quat([prev_obj_quat_mj[1], prev_obj_quat_mj[2], prev_obj_quat_mj[3], prev_obj_quat_mj[0]])
            curr_obj_rot = R.from_quat([curr_obj_quat_mj[1], curr_obj_quat_mj[2], curr_obj_quat_mj[3], curr_obj_quat_mj[0]])
            delta_rotation = curr_obj_rot * prev_obj_rot.inv()

            prev_wrist_quat_mj = qpos[3:7, start_frame_count + frame - 1]
            prev_wrist_rot = R.from_quat([prev_wrist_quat_mj[1], prev_wrist_quat_mj[2], prev_wrist_quat_mj[3], prev_wrist_quat_mj[0]])
            new_wrist_rot = delta_rotation * prev_wrist_rot
            new_wrist_quat_scipy = new_wrist_rot.as_quat()
            wrist_init_quat = np.array([new_wrist_quat_scipy[3], new_wrist_quat_scipy[0], new_wrist_quat_scipy[1], new_wrist_quat_scipy[2]])

        initial_qpos = np.concatenate((wrist_init_pos, wrist_init_quat, qpos_copy[7:, closet_original_frame]))

        qpos[:, start_frame_count + frame] = optimize_frame(
            initial_qpos,
            retargeted_spline_pos[start_frame_count + frame, :],
            m, d, hand_contacts, object_contacts,
            hand_components, hand_component_body_ids, obj, closet_original_frame,
            kinematic_tree, lr=config.learning_rate,
            n_iter=config.n_iter + (config.first_frame_iter - config.n_iter) * (frame == 0),
            optimize_wrist=True, optimize_joints=True, agent_type=AGENT,
            print_logs=True, device=device,
            kinematic_tree_torch=kinematic_tree_torch,
            local_contacts_cache=local_contacts_cache,
            loss_threshold=config.loss_threshold,
            root_body_id=root_body_id,
        )

    # Shift/re-orient/re-spline the non-contact segments (shared with the baselines).
    finalize_noncontact_hand(state)

    # Light smoothing of the full hand trajectory.
    state.qpos = smooth_hand_trajectory(qpos, frames, AGENT, window_length=21, polyorder=3, visualize=False)
    return state
