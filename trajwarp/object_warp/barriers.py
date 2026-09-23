import copy
import os

import numpy as np
import trimesh
from scipy.spatial import KDTree
import open3d as o3d

from trajwarp.object_warp.spatial import create_obj
from trajwarp.io.scene_xml import generate_barrier, scale_obj

# def optimizeCurve(api, scene_file, out_pos_file, out_tangent_file):
#     return api.optimizeCurve(scene_file, out_pos_file, out_tangent_file)
# api = msr.APIWrapper()



def getEntrance(obj_file):
    mesh = trimesh.load_mesh(obj_file)
    if not mesh.is_watertight:
        print("mesh is not watertight, entrance point may be inaccurate...")
        
    hull = mesh.convex_hull
    # hull.export('scene/barrier6.obj')
    
    hull_points, hull_face_indices = hull.sample(10000, return_index=True)
    
    _, dists, _ = mesh.nearest.on_surface(hull_points)

    entrance = hull_points[np.argmax(dists)]

    return np.array([-entrance[0], entrance[2], entrance[1]])


def read_obj(file):
    vertices = []
    with open(file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == 'v':
                x, y, z = -float(parts[1]), float(parts[3]), float(parts[2])
                vertices.append([x, y, z])
    
    return np.array(vertices, dtype=np.float32)


def load_obj_vertices(file):
    """Read raw ``v`` vertices from an OBJ, keeping the file's coordinate frame.

    Trajectory OBJs written by ``save_obj`` contain only ``v`` lines (no faces),
    and ``trimesh.load_mesh`` silently drops every vertex that no face references
    -- so it returns an empty point set for these files. This parser keeps all
    vertices. Unlike ``read_obj`` it does NOT remap axes, so the points stay in
    the same frame as the barrier meshes (which trimesh loads verbatim).
    """
    vertices = []
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(vertices, dtype=np.float64)

def save_obj(trajectory, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as file:
        for vertex in trajectory:
            x, y, z = vertex
            obj_x = -x
            obj_y = z
            obj_z = y
            
            file.write(f"v {obj_x:.6f} {obj_y:.6f} {obj_z:.6f}\n")

def scale(barr_type, config, percentage):
    new_config = copy.deepcopy(config)
    if barr_type == 'sphere':
       new_config['rad'] *= percentage
    if barr_type == 'rect':
        new_config['dims'][0] *= percentage
        new_config['dims'][1] *= percentage
        new_config['dims'][2] *= percentage

    print(new_config)
    
    return new_config

def get_wayPointIdx(waypts, pos):
    if waypts is None:
        return None
    
    idxs = []

    for waypt in waypts:
        squared_distances = np.sum((pos - waypt[0]) ** 2, axis=1)
    
        idxs.append(np.argmin(squared_distances))
    
    return idxs


def trajectoryBarrierIntersectionCount(trajectoryFile, barrierFile):
    surface_mesh = trimesh.load_mesh(barrierFile)

    curve_points = load_obj_vertices(trajectoryFile)

    return int(np.sum(surface_mesh.contains(curve_points)))


# get the number of points (of the trajectory) within some radius of the surface
def trajectoryBarrierNearbyCount(trajectoryFile, barrierFile, radius):
    surface_mesh = trimesh.load_mesh(barrierFile)

    curve_points = load_obj_vertices(trajectoryFile)

    surface_tree = KDTree(surface_mesh.vertices)
    distances, _ = surface_tree.query(curve_points)

    return int(np.sum(distances <= radius))


def percentage_steps(start_percentage, end_percentage, length, linear=False):
    if linear is False:
        ratio = (end_percentage / start_percentage) ** (1 / length)
        return start_percentage * np.power(ratio, np.arange(length + 1))
    else:
        return np.linspace(start_percentage, end_percentage, length)


def correctTrajectory(trajectoryFile, barrierFile):
    surface = trimesh.load(barrierFile)
    surface = surface.convex_hull
    
    trajectory_points = load_obj_vertices(trajectoryFile)

    edges = np.array([[i, i+1] for i in range(len(trajectory_points)-1)])
    
    # check which points are inside the surface
    inside_points = surface.contains(trajectory_points)

    print("Correcting " + barrierFile + ": ", np.sum(inside_points))

    total_corrected = 0
    
    if np.any(inside_points):
        # for each inside point, find closest point on surface
        for i, is_inside in enumerate(inside_points):
            if is_inside:
                # closest_point, _, _ = trimesh.proximity.closest_point(surface, [trajectory_points[i]])
                # trajectory_points[i] = closest_point[0]

                point = trajectory_points[i]
                direction = point - surface.centroid
                
                distance = np.linalg.norm(direction)
                if distance < 1e-10:  # if point is very close to center
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = direction / distance
                
                new_point = point.copy()
                
                # move the point outward in small steps
                step_size = 0.005 * surface.scale
                # max_steps = 1000
                steps = 0
                
                while surface.contains([new_point])[0]:
                    new_point = new_point + direction * step_size
                    steps += 1
                
                if not surface.contains([new_point])[0]:
                    trajectory_points[i] = new_point
                    total_corrected += 1

    print("Corrected: ", total_corrected)

    
    with open(trajectoryFile, 'w') as f:
        for point in trajectory_points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
        for edge in edges:
            f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")


def enterEndObj(trajectoryFile, endObj, n=10):
    trajectory = read_obj(trajectoryFile)

    startpt = trajectory[-1, :]

    surface = trimesh.load(endObj)
    endpt = np.array(surface.centroid)
    endpt[0] *= -1
    temp = endpt[1]
    endpt[1] = endpt[2]
    endpt[2] = temp

    t_values = np.linspace(1.0/n, 1.0, n)
    path = np.array([startpt + t * (endpt - startpt) for t in t_values])

    save_obj(np.vstack((trajectory, path)), trajectoryFile)


# similar to enterEndObj(). Creates linear points from the last point
# in the trajectory to endpt
def moveToEndPt(trajectoryFile, endpt, n=10):
    trajectory = read_obj(trajectoryFile)

    startpt = trajectory[-1, :]

    # endpt[0] *= -1
    # temp = endpt[1]
    # endpt[1] = endpt[2]
    # endpt[2] = temp

    t_values = np.linspace(1.0/n, 1.0, n)
    path = np.array([startpt + t * (endpt - startpt) for t in t_values])

    save_obj(np.vstack((trajectory, path)), trajectoryFile)



def barrierConstraints(pts, object_radius, barriers, endObj=None, traj_path = "scene/curve_positions.obj"):
    save_obj(pts, traj_path)

    for i, barrier in enumerate(barriers):
        if not isinstance(barrier, str):
            barr_type, config = barrier
            name = 'barrier' + str(i) + '.obj'
            path = "scene/" + name
            generate_barrier(object_radius, barr_type, config, path=path)
        else:
            path = barrier
            minkowski_sum_convex_ball(path, object_radius)


        intersectionCount = trajectoryBarrierIntersectionCount(traj_path, path)

        # no preprocessing needed, object does not intersect with trajectory
        if intersectionCount == 0:
            print("No intersections with: " + path)
            continue
        
        # identify points in the intersection and move them to the surface
        else:
            correctTrajectory(traj_path, path)

    if endObj is not None:   
        scale_obj(endObj, object_radius)
        correctTrajectory(traj_path, endObj)

        enterEndObj(traj_path, endObj)



def barrierWayptsCheck(barriers, waypts, object_radius):
    if waypts is not None:
        for waypt in waypts:
            waypt = np.array(waypt[0])

            # fix orientation
            waypt[0] *= -1
            temp = waypt[2]
            waypt[2] = waypt[1]
            waypt[1] = temp

            for i, barrier in enumerate(barriers):
                if not isinstance(barrier, str):
                    barr_type, config = barrier

                    name = 'barrier' + str(i) + '.obj'
                    path = "scene/" + name
                    generate_barrier(0, barr_type, config, path=path)

                    surface_mesh = trimesh.load_mesh(path)

                    closest_point, distance, _ = surface_mesh.nearest.on_surface([waypt])

                    # print(i, closest_point, waypt, distance)

                    if distance < object_radius*2 + 0.005:
                        return False
    return True


def cleanTrajectory(floor_height, startTime, endTime, objectSplines, resolution, traj_path):
    obj = read_obj(traj_path)
    obj[:, 2] = np.maximum(obj[:, 2], floor_height)

    sim_time = np.linspace(0, 1, resolution)
    spline_data = np.array([spline(sim_time) for spline in objectSplines])[:, :, 1] # (3, resolution)
    print(spline_data.shape, startTime, endTime, obj.shape)

    obj[:startTime, 0] = spline_data[0, :startTime]
    obj[:startTime, 1] = spline_data[1, :startTime]
    obj[:startTime, 2] = spline_data[2, :startTime]

    obj[endTime:, 0] = spline_data[0, endTime:]
    obj[endTime:, 1] = spline_data[1, endTime:]
    obj[endTime:, 2] = spline_data[2, endTime:]

    save_obj(obj, traj_path)


# takes in a list of barriers and if a barrier is an obj file with scale and pos,
# create a new obj with those configurations
# returns a list of obj files (its coord axes need to be fixed) and primitive barriers
def process_barriers(barriers):
    barriers_new = []
    for i, barrier in enumerate(barriers):
        b1, b2 = barrier
        if b1 != 'rect' and b2 != 'sphere':
            input_path = b1
            scale = b2.get('scale', (1.0, 1.0, 1.0))
            pos = b2.get('pos', (0.0, 0.0, 0.0))

            # Load mesh
            mesh = trimesh.load(input_path)

            # Apply scaling
            # mesh.apply_scale([-scale[0], scale[2], scale[1]])
            mesh.apply_scale(scale)

            # Apply translation
            # mesh.apply_translation([-pos[0], pos[2], pos[1]])
            mesh.apply_translation(pos)

            # Save transformed mesh
            output_path = f"meshes/barrier_transformed_{i}.obj"
            mesh.export(output_path)
            
            barriers_new.append(output_path)
        else:
            barriers_new.append(barrier)
    return barriers_new


# takes in a list of barriers and fixes the coord axes of non-primitive barriers
# from xyz to -xzy
def correct_barrier_axes(barriers):
    barriers_new = []
    for barrier in barriers:
        if isinstance(barrier, str):
            print("Fixing axes for " + barrier)
            # Load mesh
            mesh = trimesh.load(barrier)

            # Fix coord axes for consistency
            T = np.array([
                [-1,  0, 0, 0],  # -x
                [ 0,  0, 1, 0],  # z -> y
                [ 0,  1, 0, 0],  # y -> z
                [ 0,  0, 0, 1]
            ])

            mesh.apply_transform(T)

            # Save transformed mesh
            mesh.export(barrier)

        barriers_new.append(barrier)
    return barriers_new


# compute the convex hull then performs minkowski sum with a ball
def minkowski_sum_convex_ball(mesh_path, radius):
    mesh = trimesh.load(mesh_path)
    mesh = mesh.convex_hull
    mesh.export(mesh_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    new_vertices = vertices + radius * normals
    
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    o3d.io.write_triangle_mesh(mesh_path, mesh)