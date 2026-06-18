"""Naive linear-spline hand-warp baseline.

Computes the exact rigid hand transform at each input contact frame (as in the
SE(3) baseline), stores them as keyframes at their warped output times, then
fits a cubic B-spline through the keyframes and resamples uniformly. This
exposes SE(3) interpolation non-equivariance: keyframes are exact but
interpolated frames between them do not preserve contact geometry. Used as a
comparison baseline in the paper.
"""
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R

from trajwarp.object_warp.smoothing import break_adjacent_duplicates
from trajwarp.hand_warp.strategies._common import finalize_noncontact_hand


def _fit_and_sample_bspline(data, times, sample_times, smoothing_factor=0.001):
    """Fit a cubic B-spline to ``data`` (n_points, n_dims) parameterized by
    ``times`` and sample at ``sample_times``. ``splprep`` has a 10-dimension
    limit, so dimensions are batched in groups of <=10."""
    n_dims = data.shape[1]
    result = np.zeros((len(sample_times), n_dims))
    batch_size = 10
    for start in range(0, n_dims, batch_size):
        end = min(start + batch_size, n_dims)
        batch = data[:, start:end]
        batch = break_adjacent_duplicates(batch)
        batch_list = [*batch.T]
        spline, _ = scipy.interpolate.splprep(batch_list, u=times, s=smoothing_factor, k=3)
        sampled = np.array(scipy.interpolate.splev(sample_times, spline)).T
        result[:, start:end] = sampled
    return result


def warp_hand(state):
    """Keyframe-and-interpolate the rigid hand transform over the contact window."""
    qpos, qpos_copy = state.qpos, state.qpos_copy
    object_qpos_copy = state.object_qpos_copy
    contact_timewarp = state.contact_timewarp
    pos_spline, quat_spline = state.pos_spline, state.quat_spline
    startIdx, endIdx = state.start_idx, state.end_idx
    start_frame_count = state.start_frame_count
    contact_frame_count = state.contact_frame_count

    contact_timewarp_subset = contact_timewarp[startIdx:endIdx + 1]
    n_input_frames = len(contact_timewarp_subset)

    keyframe_times = np.zeros(n_input_frames)
    keyframe_hand_pos = np.zeros((n_input_frames, 3))
    keyframe_hand_quat_scipy = np.zeros((n_input_frames, 4))
    keyframe_fingers = np.zeros((n_input_frames, qpos_copy.shape[0] - 7))

    for i in range(n_input_frames):
        orig_frame = startIdx + i
        output_time = contact_timewarp_subset[i]
        keyframe_times[i] = output_time

        original_obj_pos = object_qpos_copy[:3, orig_frame]
        original_obj_quat_mj = object_qpos_copy[3:7, orig_frame]
        original_obj_rot = R.from_quat([original_obj_quat_mj[1], original_obj_quat_mj[2], original_obj_quat_mj[3], original_obj_quat_mj[0]])

        original_hand_pos = qpos_copy[:3, orig_frame]
        original_hand_quat_mj = qpos_copy[3:7, orig_frame]
        original_hand_rot = R.from_quat([original_hand_quat_mj[1], original_hand_quat_mj[2], original_hand_quat_mj[3], original_hand_quat_mj[0]])

        retargeted_obj_pos_at_t = np.array(scipy.interpolate.splev(output_time, pos_spline)).flatten()
        retargeted_obj_quat_scipy_at_t = np.array(scipy.interpolate.splev(output_time, quat_spline)).flatten()
        retargeted_obj_quat_scipy_at_t = retargeted_obj_quat_scipy_at_t / np.linalg.norm(retargeted_obj_quat_scipy_at_t)
        retargeted_obj_rot = R.from_quat(retargeted_obj_quat_scipy_at_t)

        hand_pos_local = original_obj_rot.inv().apply(original_hand_pos - original_obj_pos)
        hand_rot_local = original_obj_rot.inv() * original_hand_rot

        new_hand_pos = retargeted_obj_pos_at_t + retargeted_obj_rot.apply(hand_pos_local)
        new_hand_rot = retargeted_obj_rot * hand_rot_local

        keyframe_hand_pos[i] = new_hand_pos
        keyframe_hand_quat_scipy[i] = new_hand_rot.as_quat()
        keyframe_fingers[i] = qpos_copy[7:, orig_frame]

    # Enforce strictly increasing keyframe times for splprep.
    kf_times = keyframe_times.copy()
    for i in range(1, len(kf_times)):
        if kf_times[i] <= kf_times[i - 1]:
            kf_times[i] = kf_times[i - 1] + 1e-10

    output_times = np.linspace(kf_times[0], kf_times[-1], contact_frame_count)

    interp_hand_pos = _fit_and_sample_bspline(keyframe_hand_pos, kf_times, output_times)
    interp_fingers = _fit_and_sample_bspline(keyframe_fingers, kf_times, output_times)

    # Quaternion hemisphere consistency before interpolation.
    aligned_quats = keyframe_hand_quat_scipy.copy()
    for i in range(1, n_input_frames):
        if np.dot(aligned_quats[i], aligned_quats[i - 1]) < 0:
            aligned_quats[i] = -aligned_quats[i]

    eps = 1e-8
    for i in range(1, len(aligned_quats)):
        if np.allclose(aligned_quats[i], aligned_quats[i - 1], atol=eps):
            aligned_quats[i] += eps * np.array([1.0, 0.0, 0.0, 0.0])
            aligned_quats[i] /= np.linalg.norm(aligned_quats[i])

    interp_hand_quat_scipy = _fit_and_sample_bspline(aligned_quats, kf_times, output_times)
    interp_hand_quat_scipy = interp_hand_quat_scipy / np.linalg.norm(interp_hand_quat_scipy, axis=1, keepdims=True)
    interp_hand_quat_mujoco = np.column_stack([interp_hand_quat_scipy[:, 3], interp_hand_quat_scipy[:, 0:3]])

    qpos[:3, start_frame_count:start_frame_count + contact_frame_count] = interp_hand_pos.T
    qpos[3:7, start_frame_count:start_frame_count + contact_frame_count] = interp_hand_quat_mujoco.T
    qpos[7:, start_frame_count:start_frame_count + contact_frame_count] = interp_fingers.T

    finalize_noncontact_hand(state)
    return state
