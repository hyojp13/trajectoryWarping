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

    if len(geoms) == 0:
        raise ValueError(f"Body {body_id} has no mesh geoms")

    # Handle single mesh case (MANO) or multiple meshes (Franka fingers)
    if len(geoms) == 1:
        geom_id = geoms[0]
        mesh_id = model.geom_dataid[geom_id]

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

        return (faces, verts)
    else:
        # Multiple meshes: combine them all
        all_verts = []
        all_faces = []
        vertex_offset = 0

        for geom_id in geoms:
            mesh_id = model.geom_dataid[geom_id]

            v_start = model.mesh_vertadr[mesh_id]
            v_count = model.mesh_vertnum[mesh_id]
            verts = model.mesh_vert[v_start : v_start + v_count].copy()

            f_start = model.mesh_faceadr[mesh_id]
            f_count = model.mesh_facenum[mesh_id]
            faces = model.mesh_face[f_start : f_start + f_count].copy()

            # geom transform
            geom_pos = model.geom_pos[geom_id]  # (3,)
            geom_quat = model.geom_quat[geom_id]    # xyzw
            r_geom = R.from_quat([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]).as_matrix()

            verts = (r_geom @ verts.T).T + geom_pos  # apply geom transform

            # Offset face indices by current vertex count
            faces = faces + vertex_offset
            vertex_offset += len(verts)

            all_verts.append(verts)
            all_faces.append(faces)

        # Concatenate all vertices and faces
        combined_verts = np.vstack(all_verts)
        combined_faces = np.vstack(all_faces)

        return (combined_faces, combined_verts)


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