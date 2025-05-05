# import Manipulation_Scene_Retargeting as msr
from trajectory import create_obj
from generate_barrier import *
import copy
import trimesh
from scipy.spatial import KDTree

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

def save_obj(trajectory, path):
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
        squared_distances = np.sum((pos - waypt) ** 2, axis=1)
    
        idxs.append(np.argmin(squared_distances))
    
    return idxs


def trajectoryBarrierIntersectionCount(trajectoryFile, barrierFile):
    surface_mesh = trimesh.load_mesh(barrierFile)

    curve_mesh = trimesh.load_mesh(trajectoryFile)
    curve_points = curve_mesh.vertices

    return int(np.sum(surface_mesh.contains(curve_points)))


# get the number of points (of the trajectory) within some radius of the surface
def trajectoryBarrierNearbyCount(trajectoryFile, barrierFile, radius):
    surface_mesh = trimesh.load_mesh(barrierFile)

    curve_mesh = trimesh.load_mesh(trajectoryFile)
    curve_points = curve_mesh.vertices

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
    
    traj_mesh = trimesh.load(trajectoryFile)
    trajectory_points = traj_mesh.vertices

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
                step_size = 0.01 * surface.scale
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



def barrierConstraints(splines, object_radius, barriers, resolution, endObj=None, traj_path = "scene/curve_positions.obj"):
    create_obj(splines, traj_path, resolution = resolution)
    create_obj(splines, traj_path[:-4] + "temp.obj", resolution = resolution)

    # if traj_path == "scene/hand_end_positions.obj":
        # return
    
    scene_file = 'curve curve_positions.obj\n'
    scene_file += 'repel_curve\n'
    scene_file += 'fix_endpoint_vertices\n'
    #scene_file += 'fix_special_tangents\n'
    #scene_file += 'repel_plane 0 ' + str(floor_height-0.05) + ' 0 0 1 0\n'
    scene_file += 'fix_length\n'


    for i, (barr_type, config) in enumerate(barriers):
        name = 'barrier' + str(i) + '.obj'
        path = "scene/" + name
        scene_file += 'repel_surface ' + name + '\n'

    # scene_file += 'fix_length'
    f = open("scene/temp.txt", "w")
    f.write(scene_file)
    f.close()


    for i, (barr_type, config) in enumerate(barriers):
        name = 'barrier' + str(i) + '.obj'
        path = "scene/" + name
        generate_barrier(object_radius, barr_type, config, path=path)

        # create_obj(splines, traj_path[:-4] + "_dense.obj", resolution = 10000)
        intersectionCount = trajectoryBarrierIntersectionCount(traj_path, path)
        # print(intersectionCount)

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
            waypt = np.array(waypt)

            # fix orientation
            waypt[0] *= -1
            temp = waypt[2]
            waypt[2] = waypt[1]
            waypt[1] = temp
            
            for i, (barr_type, config) in enumerate(barriers):
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