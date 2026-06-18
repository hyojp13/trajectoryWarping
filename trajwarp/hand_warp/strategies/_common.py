"""Shared post-processing for the non-contact hand segments.

After a strategy has set the hand configuration over the contact window, the
pre- and post-contact segments are rigidly shifted to the new contact
boundaries, the post-contact wrist rotation offset is carried through, and both
segments are routed around barriers and re-splined. This logic is identical
across the contact-based and naive baselines.
"""
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R

from trajwarp.object_warp.barriers import barrierConstraints, read_obj, save_obj
from trajwarp.object_warp.smoothing import create_smoothing_bspline


def finalize_noncontact_hand(state):
    """Shift, re-orient, and re-spline the pre/post-contact hand segments."""
    qpos, qpos_copy = state.qpos, state.qpos_copy
    startIdx, endIdx = state.start_idx, state.end_idx
    frames = state.frames
    start_frame_count = state.start_frame_count
    end_frame_count = state.end_frame_count
    start_pos = state.start_pos
    barriers = state.barriers
    hand_boundary_radius = state.config.hand_boundary_radius

    # Rigidly shift pre- and post-contact translation to the new boundaries.
    hand_shift_start = qpos[:3, startIdx] - qpos_copy[:3, startIdx]
    hand_shift_end = qpos[:3, endIdx] - qpos_copy[:3, endIdx]
    qpos[:3, :startIdx] += hand_shift_start[:, np.newaxis]
    qpos[:3, endIdx + 1:] += hand_shift_end[:, np.newaxis]

    # Carry the contact-boundary wrist rotation change into the post-contact frames.
    wrist_rot_end_orig_mj = qpos_copy[3:7, endIdx]
    wrist_rot_end_orig = R.from_quat([wrist_rot_end_orig_mj[1], wrist_rot_end_orig_mj[2], wrist_rot_end_orig_mj[3], wrist_rot_end_orig_mj[0]])
    wrist_rot_end_new_mj = qpos[3:7, endIdx]
    wrist_rot_end_new = R.from_quat([wrist_rot_end_new_mj[1], wrist_rot_end_new_mj[2], wrist_rot_end_new_mj[3], wrist_rot_end_new_mj[0]])
    delta_rot_end = wrist_rot_end_new * wrist_rot_end_orig.inv()

    for i in range(endIdx + 1, frames):
        orig_quat_mj = qpos_copy[3:7, i]
        orig_rot = R.from_quat([orig_quat_mj[1], orig_quat_mj[2], orig_quat_mj[3], orig_quat_mj[0]])
        new_rot = delta_rot_end * orig_rot
        new_quat_scipy = new_rot.as_quat()
        qpos[3:7, i] = np.array([new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]])

    # Route the pre- and post-contact wrist path around barriers and re-spline.
    qpos_start = qpos[:3, :start_frame_count]
    qpos_end = qpos[:3, -end_frame_count:]

    barrierConstraints(qpos_start.T, hand_boundary_radius, barriers=barriers,
                       traj_path="scene/hand_start_positions.obj")
    obj_start = read_obj("scene/hand_start_positions.obj")
    obj_start[:, 2] = np.maximum(obj_start[:, 2], start_pos[2])
    save_obj(obj_start, "scene/hand_start_positions.obj")

    retargeted_start_trajectory = read_obj("scene/hand_start_positions.obj")
    retargeted_start_spline, _ = create_smoothing_bspline(retargeted_start_trajectory, parameterization="uniform")
    retargeted_sim_time = np.linspace(0, 1, start_frame_count)
    qpos[:3, :start_frame_count] = np.array(scipy.interpolate.splev(retargeted_sim_time, retargeted_start_spline))

    barrierConstraints(qpos_end.T, hand_boundary_radius, barriers=barriers,
                       traj_path="scene/hand_end_positions.obj")
    obj_end = read_obj("scene/hand_end_positions.obj")
    obj_end[:, 2] = np.maximum(obj_end[:, 2], start_pos[2])
    save_obj(obj_end, "scene/hand_end_positions.obj")

    retargeted_end_trajectory = read_obj("scene/hand_end_positions.obj")
    retargeted_end_spline, _ = create_smoothing_bspline(retargeted_end_trajectory, parameterization="uniform")
    retargeted_sim_time = np.linspace(0, 1, end_frame_count)
    qpos[:3, -end_frame_count:] = np.array(scipy.interpolate.splev(retargeted_sim_time, retargeted_end_spline))

    return qpos
