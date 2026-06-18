"""
Plot comparison of contact-based vs naive hand retargeting.

Shows:
1. Object position (retargeted) - same for both methods
2. Hand position from contact-based optimization (final_trajectories_0.01)
3. Hand position from naive method (final_trajectories)
4. Marks the 6 waypoint positions where the timewarp has its anchor points
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from trajwarp.io.spline_io import parseSplines
from trajwarp.io.contact_io import load_contacts_lcexp, get_contact_frame_range
from trajwarp.object_warp.smoothing import create_smoothing_bspline
from trajwarp.object_warp.spatial import trajectoryConstraintsPolyline
from trajwarp.object_warp.barriers import read_obj
from scipy.spatial.transform import Rotation as R


def convert_to_quaternions_object(qpos_spline_data):
    """Convert object euler angles to quaternions."""
    qpos = np.zeros((7, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]

    for j in range(qpos_spline_data.shape[1]):
        rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
        qpos[4:7, j] = rotation.as_quat()[:3]
        qpos[3, j] = rotation.as_quat()[3]

    return qpos


def convert_to_quaternions_MANO(qpos_spline_data):
    """Convert MANO hand euler angles (51 DOF) to quaternions (67 DOF)."""
    qpos = np.zeros((67, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]

    for j in range(qpos_spline_data.shape[1]):
        for i in range(3, 51, 3):
            rotation = R.from_euler('xyz', qpos_spline_data[i:i+3, j], degrees=False)
            qpos[int((i/3 - 1) * 4 + 4): int(i/3*4 + 3), j] = rotation.as_quat()[:3]
            qpos[int((i/3 - 1) * 4 + 3), j] = rotation.as_quat()[3]

    return qpos


def main():
    # Load config
    config_path = 'retargeting_configs/teapot_pour_cup_long.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    AGENT = config['agent']
    TASK = config['task']

    # ============================================================
    # Load trajectories from both methods
    # ============================================================

    # Contact-based method (final_trajectories_0.01)
    hand_contact = np.load('final_trajectories_0.0035/teapot_pour_cup_long_hand.npy')  # (67, frames)
    object_contact = np.load('final_trajectories_0.0035/teapot_pour_cup_long_object.npy')  # (frames, 7)

    # Naive method (final_trajectories)
    hand_naive = np.load('final_trajectories/teapot_pour_cup_long_hand_naive.npy')  # (67, frames)
    object_naive = np.load('final_trajectories/teapot_pour_cup_long_object_naive.npy')  # (frames, 7)

    frames = hand_contact.shape[1]
    print(f"Total frames: {frames}")

    # ============================================================
    # Load contact range and recreate timewarp to find waypoint positions
    # ============================================================

    contacts_lcexp = load_contacts_lcexp(f'startingTrajectories/{AGENT}/{TASK}/contacts.lcexp', 'MANO')
    startIdx, endIdx = get_contact_frame_range(contacts_lcexp)

    print(f"Contact range: {startIdx} to {endIdx}")

    # Recreate the timewarp to find waypoint positions
    objectSplines, _, _ = parseSplines(f'startingTrajectories/{AGENT}/{TASK}/object.smexp')
    handSplines, _, _ = parseSplines(f'startingTrajectories/{AGENT}/{TASK}/hand.smexp')
    sim_time = np.linspace(0, 1, frames)
    object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])
    object_qpos_spline_data = object_qpos_spline_data[:, :, 1]
    object_qpos_copy = convert_to_quaternions_object(object_qpos_spline_data)

    hand_qpos_spline_data = np.array([spline(sim_time) for spline in handSplines])
    hand_qpos_spline_data = hand_qpos_spline_data[:, :, 1]
    qpos_copy = convert_to_quaternions_MANO(hand_qpos_spline_data)

    # Process waypoints
    new_start_pos_shift = np.array(config['new_start_pos_shift'])
    waypts_config = config['waypts']
    start_pos = object_qpos_copy[:3, 0].copy()
    newStartPos = start_pos + new_start_pos_shift

    waypts = []
    for w in waypts_config:
        waypt_pos = newStartPos + np.array(w[0])
        if len(w) == 4:
            waypts.append((waypt_pos, w[1], w[2], w[3]))
        else:
            waypts.append((waypt_pos, w[1], w[2]))

    end_final_pos_shift = np.array(config['end_final_pos_shift'])
    end_obj_pos_shift = np.array(config['end_obj_pos_shift'])
    end_pos = object_qpos_copy[:3, frames-1].copy()
    endFinalPos = end_pos + end_final_pos_shift
    endObjPos = endFinalPos + end_obj_pos_shift

    # Get timewarp
    pos_waypt_constrained, _, _, waypts_idx = trajectoryConstraintsPolyline(
        object_qpos_copy[:3, :], startPos=newStartPos, endPos=endObjPos,
        floor_height=start_pos[2], waypts=waypts
    )

    trajectory = read_obj("scene/curve_positions.obj")
    new_object_qpos = np.zeros((7, frames))
    new_object_qpos[:3, :] = trajectory.T

    final_waypt_timesteps = np.array([w[2] for w in waypts])
    _, contact_timewarp = create_smoothing_bspline(
        new_object_qpos[:3, :].T,
        parameterization="waypts",
        waypts_info=(waypts_idx, final_waypt_timesteps)
    )

    # Contact frame info
    start_frame_count = startIdx
    contact_frame_count = endIdx - startIdx + 1
    end_frame_count = frames - start_frame_count - contact_frame_count

    print(f"Frames before contact: {start_frame_count}")
    print(f"Frames of contact: {contact_frame_count}")
    print(f"Frames after contact: {end_frame_count}")

    # Get timewarp subset for contact region
    contact_timewarp_subset = contact_timewarp[startIdx:endIdx+1]

    # Linear sampling params used by naive method
    linear_sample_params = np.linspace(contact_timewarp_subset.min(), contact_timewarp_subset.max(), contact_frame_count)

    # Find which OUTPUT frames correspond to WAYPOINT positions
    # waypts_idx contains the original frame indices where waypoints occur
    print(f"Waypoint indices (original frames): {waypts_idx}")
    print(f"Waypoint final timesteps: {final_waypt_timesteps}")

    # Convert waypoint original frame indices to output frame indices
    # The waypoints are at specific parameter values in the timewarp
    waypoint_output_indices = []
    for wp_idx in waypts_idx:
        if wp_idx >= startIdx and wp_idx <= endIdx:
            # Get the timewarp parameter value at this waypoint
            wp_param = contact_timewarp[wp_idx]
            # Find the closest output frame (linear sampling)
            closest_output = np.argmin(np.abs(linear_sample_params - wp_param))
            waypoint_output_indices.append(closest_output + startIdx)

    waypoint_output_indices = np.array(waypoint_output_indices)
    print(f"Waypoint output frame indices: {waypoint_output_indices}")

    # Compute output frame indices for all timewarp sample points (ground truth positions)
    # These are where the naive method's rigid body transforms are exact
    timewarp_output_indices = []
    for orig_idx in range(startIdx, endIdx + 1):
        tw_param = contact_timewarp[orig_idx]
        closest_output = np.argmin(np.abs(linear_sample_params - tw_param))
        timewarp_output_indices.append(closest_output + startIdx)
    timewarp_output_indices = np.array(timewarp_output_indices)

    # ============================================================
    # Create separate plots - X, Y, Z positions + 4 quaternion components
    # ============================================================

    import os
    output_dir = 'visuals/timewarp_plots'
    os.makedirs(output_dir, exist_ok=True)

    frame_indices = np.arange(frames)

    # Position labels and indices
    pos_labels = ['X', 'Y', 'Z']
    quat_labels = ['W', 'Qx', 'Qy', 'Qz']

    # --- Plot positions (X, Y, Z) ---
    for i, label in enumerate(pos_labels):
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        ax.plot(frame_indices, object_contact[:, i], 'k-', linewidth=2,
                label=f'Object {label}', alpha=0.7)
        ax.plot(frame_indices, hand_contact[i, :], 'b-', linewidth=2,
                label=f'Hand {label} (contact)', alpha=0.8)
        ax.plot(frame_indices, hand_naive[i, :], 'r--', linewidth=2,
                label=f'Hand {label} (naive)', alpha=0.8)
        ax.scatter(waypoint_output_indices, hand_naive[i, waypoint_output_indices],
                   s=60, c='green', zorder=5, marker='o', label='Waypoints')
        ax.axvspan(startIdx, endIdx, alpha=0.1, color='yellow')
        for tw_out_idx in timewarp_output_indices[::2]:
            ax.axvline(x=tw_out_idx, color='purple', linestyle='-', alpha=0.15, linewidth=0.5)
        ax.set_xlabel('Frame index', fontsize=12)
        ax.set_ylabel(f'{label} position (m)', fontsize=12)
        ax.set_title(f'{label} Position: Contact vs Naive', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, frames + 1, 50))

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'position_{label.lower()}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        if label == 'Z':
            plt.show()
        plt.close(fig)
        print(f"Saved: {filepath}")

    # --- Plot quaternions (W, X, Y, Z) ---
    # Hand quaternions are at indices 3:7 (w, x, y, z in MuJoCo format)
    for i, label in enumerate(quat_labels):
        fig, (ax, ax_orig) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # --- Top: retargeted output ---
        ax.plot(frame_indices, object_contact[:, 3+i], 'k-', linewidth=2,
                label=f'Object {label}', alpha=0.7)
        ax.plot(frame_indices, hand_contact[3+i, :], 'b-', linewidth=2,
                label=f'Hand {label} (contact)', alpha=0.8)
        ax.plot(frame_indices, hand_naive[3+i, :], 'r--', linewidth=2,
                label=f'Hand {label} (naive)', alpha=0.8)
        ax.scatter(waypoint_output_indices, hand_naive[3+i, waypoint_output_indices],
                   s=60, c='green', zorder=5, marker='o', label='Waypoints')
        ax.axvspan(startIdx, endIdx, alpha=0.1, color='yellow')
        for tw_out_idx in timewarp_output_indices[::2]:
            ax.axvline(x=tw_out_idx, color='purple', linestyle='-', alpha=0.15, linewidth=0.5)
        # orig_x = timewarp_output_indices[::2]
        # orig_frames = np.arange(startIdx, endIdx + 1)[::2]
        # ax.scatter(orig_x, object_qpos_copy[3 + i, orig_frames],
        #            s=20, c='orange', zorder=4, marker='x', linewidths=1.2,
        #            label=f'Original object {label}')
        # ax.scatter(orig_x, qpos_copy[3 + i, orig_frames],
        #            s=20, c='cyan', zorder=4, marker='+', linewidths=1.2,
        #            label=f'Original hand {label}')
        ax.set_ylabel(f'{label}', fontsize=13)
        ax.set_title(f'Wrist Rotation {label}: Contact vs Naive', fontsize=15)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # --- Bottom: original demo trajectory ---
        ax_orig.plot(frame_indices, object_qpos_copy[3 + i, :], 'k-', linewidth=2,
                     label=f'Object {label} (original)', alpha=0.8)
        ax_orig.plot(frame_indices, qpos_copy[3 + i, :], 'b-', linewidth=2,
                     label=f'Hand {label} (original)', alpha=0.8)
        ax_orig.axvspan(startIdx, endIdx, alpha=0.1, color='yellow', label='Contact region')
        ax_orig.set_xlabel('Frame index', fontsize=13)
        ax_orig.set_ylabel(f'{label}', fontsize=13)
        ax_orig.set_title(f'Wrist Rotation {label}: Original Demo', fontsize=15)
        ax_orig.legend(loc='best', fontsize=11)
        ax_orig.grid(True, alpha=0.3)
        ax_orig.set_xticks(np.arange(0, frames + 1, 50))

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'quaternion_{label.lower()}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")

    # --- Plot finger joint quaternions (15 joints, each in a 2x4 subplot figure) ---
    # Joint j (0-indexed, 0..14): DOFs [j*4+7 .. j*4+10] = [w, qx, qy, qz]
    # Row 0: retargeted output (contact vs naive); Row 1: original demo

    for fj in range(15):
        base_dof = fj * 4 + 7  # first DOF (w) for this finger joint
        fig, axes = plt.subplots(2, 4, figsize=(24, 7), sharey=False)
        fig.suptitle(f'Finger Joint {fj+1} Quaternion', fontsize=14)

        for comp_idx, comp_label in enumerate(quat_labels):
            dof = base_dof + comp_idx

            # --- Top row: retargeted output ---
            ax = axes[0, comp_idx]
            ax.plot(frame_indices, hand_contact[dof, :], 'b-', linewidth=1.5,
                    label='Contact', alpha=0.8)
            ax.plot(frame_indices, hand_naive[dof, :], 'r--', linewidth=1.5,
                    label='Naive', alpha=0.8)
            ax.scatter(waypoint_output_indices, hand_naive[dof, waypoint_output_indices],
                       s=40, c='green', zorder=5, marker='o', label='Waypoints')
            ax.axvspan(startIdx, endIdx, alpha=0.1, color='yellow')
            for tw_out_idx in timewarp_output_indices[::2]:
                ax.axvline(x=tw_out_idx, color='purple', linestyle='-', alpha=0.15, linewidth=0.5)
            # orig_x = timewarp_output_indices[::2]
            # orig_frames = np.arange(startIdx, endIdx + 1)[::2]
            # ax.scatter(orig_x, qpos_copy[dof, orig_frames],
            #            s=15, c='cyan', zorder=4, marker='+', linewidths=1.0,
            #            label='Original hand')
            ax.set_ylabel(comp_label, fontsize=11)
            ax.set_title(f'{comp_label} (output)', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.arange(0, frames + 1, 100))
            ax.tick_params(labelsize=10)

            # --- Bottom row: original demo ---
            ax_orig = axes[1, comp_idx]
            ax_orig.plot(frame_indices, qpos_copy[dof, :], 'b-', linewidth=1.5, alpha=0.8)
            ax_orig.axvspan(startIdx, endIdx, alpha=0.1, color='yellow')
            ax_orig.set_xlabel('Frame', fontsize=11)
            ax_orig.set_ylabel(comp_label, fontsize=11)
            ax_orig.set_title(f'{comp_label} (original)', fontsize=12)
            ax_orig.grid(True, alpha=0.3)
            ax_orig.set_xticks(np.arange(0, frames + 1, 100))
            ax_orig.tick_params(labelsize=10)

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'finger_joint_{fj+1:02d}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")

    # --- Plot timewarp function (codomain visualization) ---
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot the timewarp: original frame index -> output parameter value
    contact_frame_indices = np.arange(startIdx, endIdx + 1)
    ax.plot(contact_frame_indices, contact_timewarp_subset, 'b-', linewidth=2,
            label='Timewarp (non-linear)')

    # Plot linear reference (what uniform sampling would be)
    linear_reference = np.linspace(contact_timewarp_subset.min(), contact_timewarp_subset.max(), len(contact_timewarp_subset))
    ax.plot(contact_frame_indices, linear_reference, 'r--', linewidth=2,
            label='Linear reference', alpha=0.7)

    # Mark waypoints
    for wp_idx in waypts_idx:
        if wp_idx >= startIdx and wp_idx <= endIdx:
            ax.axvline(x=wp_idx, color='green', linestyle=':', alpha=0.7)
            ax.scatter([wp_idx], [contact_timewarp[wp_idx]], s=80, c='green', zorder=5, marker='o')

    ax.set_xlabel('Original frame index', fontsize=13)
    ax.set_ylabel('Timewarp parameter (codomain)', fontsize=13)
    ax.set_title('Timewarp Function: Original Frame → Output Parameter', fontsize=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(startIdx, endIdx + 1, 50))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'timewarp_function.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
