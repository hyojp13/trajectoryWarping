import numpy as np
import scipy
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