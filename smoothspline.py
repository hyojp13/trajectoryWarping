import numpy as np
import scipy
from trajectory import motionStartEnd

import numpy as np

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
def create_smoothing_bspline(points, smoothing=None, degree=3, parameterization="uniform", waypts_info = None):
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
    spline, u = scipy.interpolate.splprep(points_list, u=t, s=0.001, k=degree)

    # t, c, k = spline

    # splines = [scipy.interpolate.BSpline(t, c[i], k) for i in range(3)]
    
    return spline, t