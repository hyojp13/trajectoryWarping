import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from trajectory import motionStartEnd

def break_adjacent_duplicates(points, eps=1e-8):
    x = points[:, :3]

    # Identify adjacent duplicates
    same = np.all(x[1:] == x[:-1], axis=1)

    # Count consecutive duplicate runs
    shift_count = np.zeros(len(x), dtype=int)
    shift_count[1:] = same.astype(int)
    shift_count = np.cumsum(shift_count)

    # Apply tiny shift along a fixed direction
    direction =  np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    x += shift_count[:, None] * eps * direction

    points[:, :3] = x

    return points


def remove_duplicate_points(points, waypts_idx=None):
    # points : (n, 7)
    p = points[:, :3]
    diff = np.diff(p, axis=0)

    squared_distances = np.sum(diff**2, axis=1)

    keep_mask = np.zeros(len(p), dtype=bool)
    keep_mask[0] = 1  # keep first point

    duplicate_mask = squared_distances != 0
    keep_mask[1:] = duplicate_mask

    if waypts_idx is not None:
        # map old idx -> new idx of nearest surviving duplicate
        old_to_new = np.full(len(p), -1, dtype=int)
        new_idx = -1
        for old_idx, keep in enumerate(keep_mask):
            if keep:
                new_idx += 1
            old_to_new[old_idx] = new_idx

        new_waypts_idx = np.array([old_to_new[idx] for idx in waypts_idx], dtype=int)

        return points[keep_mask], new_waypts_idx
    
    return points[keep_mask]

def chord_length_parameterize(points):
    # points : (n, 7)
    points = points[:, :3]
    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    
    cumulative_lengths = np.concatenate(([0], np.cumsum(distances)))
    
    if cumulative_lengths[-1] > 0:
        t = cumulative_lengths / cumulative_lengths[-1]
    else:   # edge case (object not moving)
        t = np.linspace(0, 1, len(points))
        
    return t


# waypts_info must be a tuple (waypts_idx, final_waypt_timesteps) where:
# waypts_idx: frame where the waypoint currently lies
# final_waypt_timesteps: at what percent of the final trajectory the waypoint should end
def create_smoothing_bspline(points, smoothing=None, degree=3, parameterization="uniform", waypts_info = None, smoothing_factor=0.001):
    if smoothing is None:
        smoothing = points.shape[0]

    # print("RUNNING:")
    # motionStartEnd(points)
    

    if parameterization == "uniform":
        points = remove_duplicate_points(points)    # (n, 7)
        t = np.linspace(0, 1, len(points))
    elif parameterization == "chord":
        points = remove_duplicate_points(points)    # (n, 7)
        t = chord_length_parameterize(points)
    elif parameterization == "waypts":
        if waypts_info == None:
            points = remove_duplicate_points(points)    # (n, 7)
            t = chord_length_parameterize(points)
        else:
            assert(len(waypts_info) == 2)
            waypts_idx, final_waypt_timesteps = waypts_info

            if waypts_idx is None:
                waypts_idx = np.array([], dtype=int)

            # points, waypts_idx = remove_duplicate_points(points, waypts_idx)    # (n, 7)
            points = break_adjacent_duplicates(points)    # (n, 7)

            # add ending point
            waypts_idx = np.append(waypts_idx, len(points))
            final_waypt_timesteps = np.append(final_waypt_timesteps, 1.0)

            t = np.zeros(len(points))

            prev_timestep = 0
            for i in range(waypts_idx.shape[0]):
                # each segment: (prev_idx:idx)
                start_idx = 0
                if i != 0:
                    start_idx = waypts_idx[i-1]
                    
                end_idx = waypts_idx[i]

                eps = 0.001
                t_section = chord_length_parameterize(points[start_idx:end_idx])
                t_section = t_section * (final_waypt_timesteps[i] - prev_timestep - eps) + prev_timestep

                # print(start_idx, end_idx, t_section[:10], t_section[-10:], final_waypt_timesteps[i], prev_timestep)

                t[start_idx:end_idx] = t_section
                prev_timestep = final_waypt_timesteps[i]


    else:
        raise Exception("unavailable parameterization format")
    

    points_list = [*points.T]

    # print(np.split(points, points.shape[0])[0].shape)
    # print(np.split(points, points.shape[0]))
    spline, u = scipy.interpolate.splprep(points_list, u=t, s=smoothing_factor, k=degree)

    # t, c, k = spline

    # splines = [scipy.interpolate.BSpline(t, c[i], k) for i in range(3)]

    return spline, t


def smooth_quaternions(quats, window_size):
    """
    Smooth quaternions using moving window averaging with hemisphere alignment.

    How it works:
    1. For each frame, we take a window of quaternions around it (e.g., 10 frames before and after)
    2. We ensure all quaternions in the window are in the same hemisphere by flipping any that
       point in the opposite direction (since q and -q represent the same rotation)
    3. We compute the simple arithmetic mean of the quaternions in the window
    4. We normalize the result to ensure it's a unit quaternion

    This works well because:
    - For small rotational changes (which we expect in a smooth trajectory), the quaternion
      space is approximately linear, so simple averaging gives good results
    - The hemisphere alignment prevents the average from being pulled toward zero
    - Normalization ensures we get a valid rotation

    Args:
        quats: (n, 4) array of quaternions in (x, y, z, w) format
        window_size: size of the moving window (larger = more smoothing)

    Returns:
        smoothed: (n, 4) array of smoothed quaternions
    """
    smoothed = np.zeros_like(quats)
    half_window = window_size // 2

    for i in range(len(quats)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(quats), i + half_window + 1)

        # Get window of quaternions
        window_quats = quats[start_idx:end_idx].copy()

        # Ensure all quaternions in window have same hemisphere (avoid long path)
        # This is critical: q and -q represent the same rotation, but averaging them
        # would give a near-zero vector. We flip quaternions so they all point the same way.
        reference = window_quats[0]
        for j in range(1, len(window_quats)):
            if np.dot(window_quats[j], reference) < 0:
                window_quats[j] = -window_quats[j]

        # Average quaternions (simple mean works well for small windows)
        avg_quat = np.mean(window_quats, axis=0)

        # Normalize to ensure unit quaternion (valid rotation)
        smoothed[i] = avg_quat / np.linalg.norm(avg_quat)

    return smoothed


def create_quaternion_bspline(quats, waypts_info=None, smoothing_factor=0.001, degree=3):
    """
    Create a B-spline for quaternion data using proper quaternion interpolation.

    Instead of treating quaternions as 4D vectors (which can cause strange rotations),
    this function:
    1. Ensures quaternions are in the same hemisphere
    2. Uses scipy's splprep on the aligned quaternions
    3. Re-normalizes sampled quaternions to ensure they remain unit quaternions

    Args:
        quats: (n, 4) array of quaternions in scipy format (x, y, z, w)
        waypts_info: Optional tuple (waypts_idx, final_waypt_timesteps) for waypoint constraints
        smoothing_factor: Smoothing parameter for splprep
        degree: Degree of the B-spline

    Returns:
        spline: B-spline representation (output will be in same format as input)
        t: Parameter values used for the spline
    """
    n = len(quats)

    # Ensure all quaternions are in the same hemisphere to avoid discontinuities
    aligned_quats = quats.copy()
    for i in range(1, n):
        if np.dot(aligned_quats[i], aligned_quats[i-1]) < 0:
            aligned_quats[i] = -aligned_quats[i]

    # Create parameterization - use the same logic as create_smoothing_bspline
    # but applied to quaternion data
    if waypts_info is None:
        # Uniform parameterization
        t = np.linspace(0, 1, n)
    else:
        waypts_idx, final_waypt_timesteps = waypts_info

        if waypts_idx is None:
            waypts_idx = np.array([], dtype=int)

        # Break adjacent duplicates in quaternion space to avoid issues with splprep
        # Check for identical quaternions
        eps = 1e-8
        for i in range(1, len(aligned_quats)):
            if np.allclose(aligned_quats[i], aligned_quats[i-1], atol=eps):
                # Add tiny perturbation
                aligned_quats[i] += eps * np.array([1.0, 0.0, 0.0, 0.0])
                aligned_quats[i] /= np.linalg.norm(aligned_quats[i])

        # Add ending point
        waypts_idx = np.append(waypts_idx, len(aligned_quats))
        final_waypt_timesteps = np.append(final_waypt_timesteps, 1.0)

        t = np.zeros(len(aligned_quats))

        prev_timestep = 0
        for i in range(waypts_idx.shape[0]):
            start_idx = 0 if i == 0 else waypts_idx[i-1]
            end_idx = waypts_idx[i]

            # Compute chord length for this segment
            segment = aligned_quats[start_idx:end_idx]
            if len(segment) > 1:
                diffs = segment[1:] - segment[:-1]
                distances = np.sqrt(np.sum(diffs**2, axis=1))
                cumulative = np.concatenate(([0], np.cumsum(distances)))

                if cumulative[-1] > 0:
                    t_section = cumulative / cumulative[-1]
                else:
                    t_section = np.linspace(0, 1, len(segment))
            else:
                t_section = np.array([0])

            eps = 0.001
            t_section = t_section * (final_waypt_timesteps[i] - prev_timestep - eps) + prev_timestep
            t[start_idx:end_idx] = t_section
            prev_timestep = final_waypt_timesteps[i]

    # Ensure t is strictly increasing (required by splprep)
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + 1e-10

    # Create B-spline using aligned quaternions
    quats_list = [aligned_quats[:, i] for i in range(4)]
    spline, u = scipy.interpolate.splprep(quats_list, u=t, s=smoothing_factor, k=degree)

    return spline, t


def smooth_hand_trajectory(qpos, frames, AGENT, window_length=5, polyorder=3, visualize=False):
    """
    Smooth hand trajectory including translation, wrist rotation, and finger joints.

    Args:
        qpos: Hand pose data array
        frames: Number of frames
        AGENT: Agent type ('MANO_right', 'trajectories', 'Allegro_right', etc.)
        window_length: Window length for smoothing (must be odd and >= polyorder+2)
        polyorder: Polynomial order for Savitzky-Golay filter (typically 2-5)
        visualize: If True, generate visualization plots (default: False)

    Returns:
        qpos: Smoothed hand pose data
    """
    # Store original values before smoothing for visualization
    if visualize:
        original_translation = qpos[:3, :].copy()

    # Smooth translation using Savitzky-Golay filter
    if frames > window_length:
        for i in range(3):
            qpos[i, :] = savgol_filter(qpos[i, :], window_length, polyorder)

    # Store smoothed values
    if visualize:
        smoothed_translation = qpos[:3, :].copy()

    # Smooth wrist rotation using quaternion averaging
    wrist_quats = qpos[3:7, :].T  # (frames, 4)

    # Store original wrist euler angles for visualization
    if visualize:
        original_wrist_rotations = R.from_quat(wrist_quats)
        original_wrist_euler = original_wrist_rotations.as_euler('xyz')

    # Smooth quaternions directly using moving window average
    if frames > window_length:
        smoothed_wrist_quats = smooth_quaternions(wrist_quats, window_length)
    else:
        smoothed_wrist_quats = wrist_quats.copy()

    # Convert smoothed quaternions to euler for visualization
    if visualize:
        smoothed_wrist_rotations = R.from_quat(smoothed_wrist_quats)
        smoothed_wrist_euler = smoothed_wrist_rotations.as_euler('xyz')

    # Update qpos with smoothed quaternions
    qpos[3:7, :] = smoothed_wrist_quats.T

    times = np.linspace(0, 1, frames)

    # Smooth finger joints (agent-specific)
    if AGENT == 'Franka':
        # Franka: qpos = [pos(3), quat(4), 7 arm joints, 2 finger joints]
        # Arm joints (7-13) and finger joints (14-15) are regular angles, not quaternions
        # Use moving window average for smooth trajectories
        if frames > window_length:
            half_window = window_length // 2

            # Smooth arm joints (7 revolute joints) using moving average
            for joint_idx in range(7):
                idx = 7 + joint_idx
                smoothed_joint = np.zeros(frames)
                for i in range(frames):
                    start_idx = max(0, i - half_window)
                    end_idx = min(frames, i + half_window + 1)
                    smoothed_joint[i] = np.mean(qpos[idx, start_idx:end_idx])
                qpos[idx, :] = smoothed_joint

            # Smooth finger joints (2 prismatic joints) using moving average
            for joint_idx in range(2):
                idx = 14 + joint_idx
                smoothed_joint = np.zeros(frames)
                for i in range(frames):
                    start_idx = max(0, i - half_window)
                    end_idx = min(frames, i + half_window + 1)
                    smoothed_joint[i] = np.mean(qpos[idx, start_idx:end_idx])
                qpos[idx, :] = smoothed_joint

    elif AGENT == 'Allegro_right':
        # Allegro: qpos = [pos(3), quat(4), 16 finger hinge joints] = 23 DOFs
        # Finger joints (7-22) are all revolute/hinge joints (regular angles)
        if frames > window_length:
            for i in range(7, 23):
                qpos[i, :] = savgol_filter(qpos[i, :], window_length, polyorder)

    elif AGENT == 'MANO_right' or AGENT == 'trajectories':
        # MANO: qpos[7:] contains 16 finger joints as quaternions (60 values = 15 joints × 4)
        num_finger_joints = 15  # (51 - 3) / 3 = 16 euler joints -> 15 quaternion joints in qpos[7:67]

        # Store original finger joint data for visualization
        if visualize:
            original_finger_joints_euler = []
            smoothed_finger_joints_euler = []

        for joint_idx in range(num_finger_joints):
            quat_start_idx = 7 + joint_idx * 4
            quat_end_idx = quat_start_idx + 4

            # Get original quaternions
            original_joint_quats = qpos[quat_start_idx:quat_end_idx, :].T  # (frames, 4)

            # Store original euler angles for visualization
            if visualize:
                original_rotations = R.from_quat(original_joint_quats)
                original_euler = original_rotations.as_euler('xyz')
                original_finger_joints_euler.append(original_euler)

            # Smooth quaternions directly using moving window average
            if frames > window_length:
                smoothed_joint_quats = smooth_quaternions(original_joint_quats, window_length)
            else:
                smoothed_joint_quats = original_joint_quats.copy()

            # Update qpos with smoothed quaternions
            qpos[quat_start_idx:quat_end_idx, :] = smoothed_joint_quats.T

            # Store smoothed euler angles for visualization
            if visualize:
                smoothed_rotations = R.from_quat(smoothed_joint_quats)
                smoothed_euler = smoothed_rotations.as_euler('xyz')
                smoothed_finger_joints_euler.append(smoothed_euler)

    # Generate visualizations if requested
    if visualize:
        import matplotlib
        matplotlib.use('Agg')  # avoids conflicts with MuJoCo viewer
        import matplotlib.pyplot as plt

        # Visualize smoothing effects for hand pose
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Hand Pose Smoothing (Translation: Savgol window={window_length}, Rotation: Quat avg window={window_length})', fontsize=14)

        # Plot translation components (x, y, z)
        translation_labels = ['X Translation', 'Y Translation', 'Z Translation']
        for i in range(3):
            axes[0, i].plot(times, original_translation[i, :], 'b-', alpha=0.5, linewidth=2, label='Original')
            axes[0, i].plot(times, smoothed_translation[i, :], 'r-', linewidth=2, label='Smoothed')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Distance (m)')
            axes[0, i].set_title(translation_labels[i])
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

        # Plot wrist rotation components (roll, pitch, yaw)
        rotation_labels = ['Wrist Roll', 'Wrist Pitch', 'Wrist Yaw']
        for i in range(3):
            axes[1, i].plot(times, original_wrist_euler[:, i], 'b-', alpha=0.5, linewidth=2, label='Original')
            axes[1, i].plot(times, smoothed_wrist_euler[:, i], 'r-', linewidth=2, label='Smoothed')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Angle (radians)')
            axes[1, i].set_title(rotation_labels[i])
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('smoothing_visualization_hand.png', dpi=150, bbox_inches='tight')
        print(f"Hand pose smoothing visualization saved to smoothing_visualization_hand.png")
        plt.close()

        # Visualize finger joint smoothing (if available)
        if AGENT == 'MANO_right' or AGENT == 'trajectories':
            # Create plots for first 6 finger joints (2 rows x 3 cols)
            # Each joint will show its 3 euler angle components
            num_joints_to_plot = min(6, num_finger_joints)

            fig, axes = plt.subplots(num_joints_to_plot, 3, figsize=(15, 3*num_joints_to_plot))
            fig.suptitle(f'Finger Joint Smoothing (Quaternion averaging: window={window_length})', fontsize=16)

            euler_labels = ['Roll', 'Pitch', 'Yaw']

            for joint_idx in range(num_joints_to_plot):
                for euler_idx in range(3):
                    ax = axes[joint_idx, euler_idx] if num_joints_to_plot > 1 else axes[euler_idx]
                    ax.plot(times, original_finger_joints_euler[joint_idx][:, euler_idx],
                            'b-', alpha=0.5, linewidth=2, label='Original')
                    ax.plot(times, smoothed_finger_joints_euler[joint_idx][:, euler_idx],
                            'r-', linewidth=2, label='Smoothed')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Angle (radians)')
                    ax.set_title(f'Joint {joint_idx} - {euler_labels[euler_idx]}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('smoothing_visualization_joints.png', dpi=150, bbox_inches='tight')
            print(f"Finger joint smoothing visualization saved to smoothing_visualization_joints.png")
            plt.close()

    return qpos


def smooth_adroit_trajectory(hand_qpos, frames, window_length=21, polyorder=3, visualize=False):
    """
    Smooth Adroit hand trajectory with proper Euler angle handling.

    Adroit hand structure (30 DOFs):
    - Indices 0-2: ARTx, ARTy, ARTz (translation)
    - Indices 3-5: ARRx, ARRy, ARRz (Euler angles for forearm rotation)
    - Indices 6-29: 24 finger joints (all revolute/hinge joints)

    Args:
        hand_qpos: (30, n_frames) hand pose data
        frames: Number of frames
        window_length: Window length for Savitzky-Golay filter (must be odd and >= polyorder+2)
        polyorder: Polynomial order for filter (typically 2-5)
        visualize: If True, generate visualization plots

    Returns:
        hand_qpos: (30, n_frames) smoothed hand pose data
    """
    if visualize:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        original_qpos = hand_qpos.copy()

    # Ensure window_length is valid
    window_length = min(window_length, frames)
    if window_length % 2 == 0:
        window_length -= 1  # Must be odd
    window_length = max(window_length, polyorder + 2)

    print(f"\nSmoothing Adroit trajectory:")
    print(f"  Frames: {frames}")
    print(f"  Window length: {window_length}")
    print(f"  Polynomial order: {polyorder}")

    # 1. Smooth translation (indices 0-2) using Savitzky-Golay filter
    if frames > window_length:
        for i in range(3):
            hand_qpos[i, :] = savgol_filter(hand_qpos[i, :], window_length, polyorder)
        print(f"  Smoothed translation (ARTx, ARTy, ARTz)")

    # 2. Smooth Euler angles (indices 3-5) - need special handling for angle wrapping
    # Convert to rotation matrices, smooth, then convert back
    if frames > window_length:
        euler_angles = hand_qpos[3:6, :].T  # (frames, 3)

        # Convert Euler angles to rotation matrices
        rotations = R.from_euler('xyz', euler_angles, degrees=False)

        # Smooth using quaternion representation (to avoid gimbal lock issues)
        quats = rotations.as_quat()  # (frames, 4) in scipy format [x, y, z, w]

        # Align quaternions to same hemisphere
        aligned_quats = quats.copy()
        for i in range(1, frames):
            if np.dot(aligned_quats[i], aligned_quats[i-1]) < 0:
                aligned_quats[i] = -aligned_quats[i]

        # Smooth quaternions using moving window average
        smoothed_quats = smooth_quaternions(aligned_quats, window_length)

        # Convert back to Euler angles
        smoothed_rotations = R.from_quat(smoothed_quats)
        smoothed_euler = smoothed_rotations.as_euler('xyz', degrees=False)

        hand_qpos[3:6, :] = smoothed_euler.T
        print(f"  Smoothed forearm rotation (ARRx, ARRy, ARRz) via quaternions")

    # 3. Smooth finger joints (indices 6-29) using Savitzky-Golay filter
    # These are all revolute joints, but we need to handle potential angle wrapping
    if frames > window_length:
        for i in range(6, 30):
            # Check if joint values wrap around (e.g., -π to π)
            joint_values = hand_qpos[i, :]
            value_range = joint_values.max() - joint_values.min()

            # If range suggests wrapping (e.g., > 5 radians ≈ 286 degrees)
            if value_range > 5.0:
                # Unwrap angles before smoothing
                unwrapped = np.unwrap(joint_values)
                smoothed = savgol_filter(unwrapped, window_length, polyorder)
                # Wrap back to [-π, π]
                hand_qpos[i, :] = np.arctan2(np.sin(smoothed), np.cos(smoothed))
            else:
                # No wrapping needed, smooth directly
                hand_qpos[i, :] = savgol_filter(joint_values, window_length, polyorder)

        print(f"  Smoothed 24 finger joints (WRJ1, WRJ0, FFJ3-FFJ0, ...)")

    # Visualization
    if visualize:
        times = np.linspace(0, 1, frames)

        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Adroit Hand Trajectory Smoothing (window={window_length}, polyorder={polyorder})',
                     fontsize=14)

        # Row 1: Translation (ARTx, ARTy, ARTz)
        translation_labels = ['ARTx (X)', 'ARTy (Y)', 'ARTz (Z)']
        for i in range(3):
            axes[0, i].plot(times, original_qpos[i, :], 'b-', alpha=0.5,
                           linewidth=2, label='Original')
            axes[0, i].plot(times, hand_qpos[i, :], 'r-',
                           linewidth=2, label='Smoothed')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Position (m)')
            axes[0, i].set_title(f'Translation: {translation_labels[i]}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

        # Row 2: Rotation (ARRx, ARRy, ARRz)
        rotation_labels = ['ARRx (Roll)', 'ARRy (Pitch)', 'ARRz (Yaw)']
        for i in range(3):
            axes[1, i].plot(times, original_qpos[3+i, :], 'b-', alpha=0.5,
                           linewidth=2, label='Original')
            axes[1, i].plot(times, hand_qpos[3+i, :], 'r-',
                           linewidth=2, label='Smoothed')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Angle (rad)')
            axes[1, i].set_title(f'Forearm Rotation: {rotation_labels[i]}')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        # Row 3: Sample finger joints (first 3 finger joints)
        finger_joint_names = ['WRJ1 (Wrist)', 'WRJ0 (Wrist)', 'FFJ3 (Index)']
        for i in range(3):
            joint_idx = 6 + i
            axes[2, i].plot(times, original_qpos[joint_idx, :], 'b-', alpha=0.5,
                           linewidth=2, label='Original')
            axes[2, i].plot(times, hand_qpos[joint_idx, :], 'r-',
                           linewidth=2, label='Smoothed')
            axes[2, i].set_xlabel('Time')
            axes[2, i].set_ylabel('Angle (rad)')
            axes[2, i].set_title(f'Finger Joint: {finger_joint_names[i]}')
            axes[2, i].legend()
            axes[2, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('smoothing_adroit_visualization.png', dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to smoothing_adroit_visualization.png")
        plt.close()

    return hand_qpos