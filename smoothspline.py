import numpy as np
import scipy
from trajectory import motionStartEnd

def remove_duplicate_points(points):
    diff = np.diff(points, axis=0)

    squared_distances = np.sum(diff**2, axis=1)
    
    keep_mask = np.zeros(len(points), dtype=bool)
    keep_mask[0] = 1    # keep first point
    
    duplicate_mask = squared_distances != 0
    keep_mask[1:] = duplicate_mask

    return points[keep_mask]


def chord_length_parameterize(points):
    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    
    cumulative_lengths = np.concatenate(([0], np.cumsum(distances)))
    
    if cumulative_lengths[-1] > 0:
        t = cumulative_lengths / cumulative_lengths[-1]
    else:   # edge case (object not moving)
        t = np.linspace(0, 1, len(points))
        
    return t


# waypts must be provided with parameterization=='waypts'
# waypts must be a list of (pos, t_start, t_final, count) where:
# pos: position, t_start: time step in the original trajectory, t_final: time step in the final trajectory,
# count: number of points since the last waypt
def create_smoothing_bspline(points, smoothing=None, degree=3, parameterization="uniform", waypts = None):
    if smoothing is None:
        smoothing = points.shape[0]

    # print("RUNNING:")
    # motionStartEnd(points)
    
    points = remove_duplicate_points(points)
    x, y, z = points.T

    if parameterization == "uniform":
        t = np.linspace(0, 1, len(points))
    elif parameterization == "chord":
        t = chord_length_parameterize(points)
    elif parameterization == "waypts" and waypts != None:
        t = np.zeros(len(points))

        point_count = 0
        prev_timestep = 0
        for waypt in waypts:
            t_section = chord_length_parameterize(points[point_count:point_count + waypt[3]])
            t_section = t_section * (waypt[2] - prev_timestep) + waypt[2]

            t[point_count:point_count + waypt[3]] = t_section
            point_count += waypt[3]
            prev_timestep = waypt[2]

    else:
        raise Exception("unavailable parameterization format")
    

    spline, u = scipy.interpolate.splprep([x, y, z], u=t, s=0.001, k=degree)
    # u = parameterization

    t, c, k = spline

    splines = [scipy.interpolate.BSpline(t, c[i], k) for i in range(3)]
    
    return spline