import numpy as np
import trimesh

# def offset_mesh(obj_filepath, radius):

#     mesh = o3d.io.read_triangle_mesh(obj_filepath)
#     mesh.compute_vertex_normals()

#     # Compute vertex normals
#     mesh.compute_vertex_normals()
#     vertices = np.asarray(mesh.vertices)
#     normals = np.asarray(mesh.vertex_normals)

#     # Move each vertex outward by distance r
#     new_vertices = vertices + radius * normals
#     new_mesh = o3d.geometry.TriangleMesh()
#     new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
#     new_mesh.triangles = mesh.triangles
#     new_mesh.compute_vertex_normals()

#     o3d.io.write_triangle_mesh(obj_filepath, new_mesh)

def scale_obj(input_file, object_radius):
    mesh = trimesh.load(input_file)
    center = mesh.centroid

    vertices_centered = mesh.vertices - center
    distances = np.linalg.norm(vertices_centered, axis=1)
    
    # find the maximum distance from center
    max_distance = np.max(distances)
    
    scale = 1 + (object_radius / max_distance)
    
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    mesh.apply_translation(center)
    
    mesh.export(input_file)

def shift_obj(input_file, offset):
    x, y, z = offset
    new_offset = (-x, z, y)
    
    mesh = trimesh.load(input_file)
    mesh.apply_translation(new_offset)
    mesh.export(input_file)

def generate_barrier(object_radius, shape, config={}, path=None):
    if path is None:
        path = "/Users/hjp/desktop/repulsive-curves/scenes/retargeting/barrier.obj"
    
    if shape == 'sphere':
        assert(len(config) == 2) # pos + radius
        c = config['pos']

        sphere = trimesh.creation.icosphere(radius=object_radius + config['rad'], subdivisions=4)

        sphere.apply_translation([-c[0], c[2], c[1]])
    
        sphere.export(path)

        # with open('sphere.obj', "r") as obj:

        #     new_obj = ''
        #     for line in obj:
        #         if line[:2] != 'v ':
        #             new_obj += line
        #         else:
        #             pos = np.array([float(i) for i in line[1:].split()])
        #             pos /= np.linalg.norm(pos) / (config['rad']+object_radius)
        #             new_obj += 'v ' + str(-c[0] + pos[0]) + ' ' + str(c[2] - pos[2]) + ' ' + str(c[1] - pos[1]) + '\n'
            
        #     with open(path, "w") as text_file:
        #             text_file.write(new_obj)       

    if shape == 'rect':
        assert(len(config) == 2) # center + (l, w, h)

        #h = config['height'] #1
        #w = config['width'] #2
        #l = config['length'] #-0
        dims = [dim + object_radius*2 for dim in config['dims']]  # (l, w, h)
        c = config['pos']

        translation = np.eye(4)
        translation[:3, 3] = [-c[0], c[2], c[1]]
        
        box = trimesh.creation.box(extents=[dims[0], dims[2], dims[1]], transform=translation)
        box.export(path)

        # out_str = 'o barrier\n'
        # out_str += 'v ' + str((-c[0]+dims[0])) + ' ' + str((c[2]-dims[2])) + ' ' + str((c[1]-dims[1])) + '\n'
        # out_str += 'v ' + str((-c[0]-dims[0])) + ' ' + str((c[2]-dims[2])) + ' ' + str((c[1]-dims[1])) + '\n'
        # out_str += 'v ' + str((-c[0]-dims[0])) + ' ' + str((c[2]-dims[2])) + ' ' + str((c[1]+dims[1])) + '\n'
        # out_str += 'v ' + str((-c[0]+dims[0])) + ' ' + str((c[2]-dims[2])) + ' ' + str((c[1]+dims[1])) + '\n'
        # out_str += 'v ' + str((-c[0]+dims[0])) + ' ' + str((c[2]+dims[2])) + ' ' + str((c[1]-dims[1])) + '\n'
        # out_str += 'v ' + str((-c[0]+dims[0])) + ' ' + str((c[2]+dims[2])) + ' ' + str((c[1]+dims[1])) + '\n'
        # out_str += 'v ' + str((-c[0]-dims[0])) + ' ' + str((c[2]+dims[2])) + ' ' + str((c[1]+dims[1])) + '\n'
        # out_str += 'v ' + str((-c[0]-dims[0])) + ' ' + str((c[2]+dims[2])) + ' ' + str((c[1]-dims[1])) + '\n'

        # out_str += '''f 1 2 3
        # f 1 3 4
        # f 5 6 7
        # f 5 7 8
        # f 5 1 4
        # f 5 4 6
        # f 3 2 8
        # f 3 8 7
        # f 6 4 3
        # f 6 3 7
        # f 2 1 5
        # f 2 5 8'''
        
        # with open(path, "w") as text_file:
        #     text_file.write(out_str)

    # offset_mesh(path, object_radius)