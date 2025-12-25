from scipy.spatial.transform import Rotation as R
import numpy as np

def get_mesh_for_body(model, body_id):
    # find all geoms that belong to this body
    geoms = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            mesh_id = model.geom_dataid[geom_id]  # mesh used by this geom
            if mesh_id == -1:
                continue
            geoms.append(geom_id)

    assert len(geoms) == 1  # should only be one geometry per body for MANO

    geom_id = geoms[0]
    mesh_id = model.geom_dataid[geom_id]
    mesh_name = model.mesh(mesh_id).name
    # print(f"Geom {mesh_id}: {mesh_name}")

    # import trimesh
    # mesh = trimesh.load('agents/MANO_right/geom_assets/' + mesh_name + '.obj')
    # verts = mesh.vertices
    # faces = mesh.faces

    v_start = model.mesh_vertadr[mesh_id]
    v_count = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[v_start : v_start + v_count]

    f_start = model.mesh_faceadr[mesh_id]
    f_count = model.mesh_facenum[mesh_id]
    faces = model.mesh_face[f_start : f_start + f_count]

    # geom transform
    geom_pos = model.geom_pos[geom_id]  # (3,)
    geom_quat = model.geom_quat[geom_id]    # xyzw
    r_geom = R.from_quat([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]).as_matrix()

    verts = (r_geom @ verts.T).T + geom_pos  # apply geom transform

    # print(verts)

    return (faces, verts)


# get 3D position from face ID and barycentric coordinates for hand contacts
def get_local_pos(face_id, bary_coords, faces, verts):
    vertex_indices = faces[face_id]
    v0 = verts[vertex_indices[0]]
    v1 = verts[vertex_indices[1]]
    v2 = verts[vertex_indices[2]]

    local_pos = (bary_coords[0] * v0 + bary_coords[1] * v1 + bary_coords[2] * v2)

    return local_pos


# convert local vertex pos to global
def local_to_global(local_vertex, body_id, data):
    body_pos = data.xpos[body_id]   # (3,)
    body_rot = data.xmat[body_id].reshape(3,3)

    vertex_in_world = body_rot @ local_vertex + body_pos
    return vertex_in_world


def get_closest_original_frame(timewarp, frame, left_biased=True):
    if left_biased:
        # index of last element <= frame
        return np.searchsorted(timewarp, frame, side='right') - 1
    else:
        # Return index of closest element (leftmost if tied)
        return np.argmin(np.abs(timewarp - frame))