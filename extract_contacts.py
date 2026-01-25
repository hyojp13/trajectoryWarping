"""
Extract real contact points from DexMV demonstration at frame 200.

This script:
1. Loads the demonstration and sets up the scene at frame 200
2. Finds the outermost vertex on thumb, index, middle fingertip meshes
3. Finds closest point on object surface for each fingertip vertex
4. Outputs contact data in format suitable for optimize_adroit.py
"""
import numpy as np
import mujoco
import torch
import trimesh
from scipy.spatial.transform import Rotation as R


def get_mesh_for_body(model, body_id):
    """Extract mesh data for a body, handling geom transforms."""
    geoms = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            mesh_id = model.geom_dataid[geom_id]
            if mesh_id == -1:
                continue
            geoms.append(geom_id)

    if len(geoms) == 0:
        raise ValueError(f"Body {body_id} has no mesh geoms")

    if len(geoms) == 1:
        geom_id = geoms[0]
        mesh_id = model.geom_dataid[geom_id]

        v_start = model.mesh_vertadr[mesh_id]
        v_count = model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[v_start : v_start + v_count].copy()

        f_start = model.mesh_faceadr[mesh_id]
        f_count = model.mesh_facenum[mesh_id]
        faces = model.mesh_face[f_start : f_start + f_count].copy()

        # Apply geom transform
        geom_pos = model.geom_pos[geom_id]
        geom_quat = model.geom_quat[geom_id]  # xyzw
        r_geom = R.from_quat([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]).as_matrix()

        verts = (r_geom @ verts.T).T + geom_pos

        return faces, verts
    else:
        # Multiple meshes: combine them
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

            # Apply geom transform
            geom_pos = model.geom_pos[geom_id]
            geom_quat = model.geom_quat[geom_id]
            r_geom = R.from_quat([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]).as_matrix()

            verts = (r_geom @ verts.T).T + geom_pos

            faces = faces + vertex_offset
            vertex_offset += len(verts)

            all_verts.append(verts)
            all_faces.append(faces)

        combined_verts = np.vstack(all_verts)
        combined_faces = np.vstack(all_faces)

        return combined_faces, combined_verts


def local_to_global(local_vertex, body_id, data):
    """Convert local vertex position to global coordinates."""
    body_pos = data.xpos[body_id]
    body_rot = data.xmat[body_id].reshape(3, 3)
    return body_rot @ local_vertex + body_pos


def find_outermost_vertex(verts, direction):
    """
    Find the vertex that extends furthest in the given direction.

    Args:
        verts: (N, 3) array of vertex positions
        direction: (3,) unit direction vector

    Returns:
        Index of outermost vertex
    """
    # Project vertices onto direction
    projections = verts @ direction
    return np.argmax(projections)


def main():
    # Load trajectory
    data_path = 'startingTrajectories/adroit/dexmv_relocate_mustard_demo_0.pt'
    print(f"Loading trajectory from: {data_path}")

    data = torch.load(data_path, map_location='cpu', weights_only=False)
    robot_states = data['robot_states'].cpu().numpy()
    object_states = data['object_poses'].cpu().numpy()

    n_frames = robot_states.shape[0]
    print(f"Loaded {n_frames} frames")

    # Parse robot and object states
    robot_pos = robot_states[:, :3]
    robot_joints_all = robot_states[:, 3:30]

    object_pos = object_states[:, :3]
    object_quat = object_states[:, 3:7]

    # Build hand qpos
    hand_qpos = np.zeros((30, n_frames))
    hand_qpos[0:3, :] = robot_pos.T
    hand_qpos[3:6, :] = robot_joints_all[:, 0:3].T
    hand_qpos[6:30, :] = robot_joints_all[:, 3:27].T

    # Build object qpos
    object_qpos = np.zeros((7, n_frames))
    object_qpos[:3, :] = object_pos.T
    for frame_idx in range(n_frames):
        quat = object_quat[frame_idx]
        quat_norm = quat / np.linalg.norm(quat)
        object_qpos[3:7, frame_idx] = quat_norm

    # Create scene
    print("\nCreating scene...")
    from playback_adroit import create_adroit_scene_xml
    xml_path = create_adroit_scene_xml()

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Set to frame 200
    frame = 200
    print(f"\nAnalyzing frame {frame}...")

    d.qpos[0:30] = hand_qpos[:, frame]
    d.qpos[30:37] = object_qpos[:, frame]
    mujoco.mj_forward(m, d)

    # Find body IDs
    ffdistal_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'ffdistal')  # index
    mfdistal_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'mfdistal')  # middle
    thdistal_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'thdistal')  # thumb
    object_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'manipulated_object')

    print(f"Body IDs:")
    print(f"  Index finger (ffdistal): {ffdistal_id}")
    print(f"  Middle finger (mfdistal): {mfdistal_id}")
    print(f"  Thumb (thdistal): {thdistal_id}")
    print(f"  Object: {object_id}")

    # Extract meshes
    print("\nExtracting fingertip meshes...")
    ff_faces, ff_verts = get_mesh_for_body(m, ffdistal_id)
    mf_faces, mf_verts = get_mesh_for_body(m, mfdistal_id)
    th_faces, th_verts = get_mesh_for_body(m, thdistal_id)

    print(f"  Index finger: {len(ff_verts)} vertices")
    print(f"  Middle finger: {len(mf_verts)} vertices")
    print(f"  Thumb: {len(th_verts)} vertices")

    # Find outermost vertices (tip direction is roughly +Z in local frame for distal)
    # But we want the point furthest from the hand center
    object_center = d.xpos[object_id]

    # Convert local vertices to global
    ff_verts_global = np.array([local_to_global(v, ffdistal_id, d) for v in ff_verts])
    mf_verts_global = np.array([local_to_global(v, mfdistal_id, d) for v in mf_verts])
    th_verts_global = np.array([local_to_global(v, thdistal_id, d) for v in th_verts])

    # Find vertex closest to object center (likely in contact)
    ff_dists = np.linalg.norm(ff_verts_global - object_center, axis=1)
    mf_dists = np.linalg.norm(mf_verts_global - object_center, axis=1)
    th_dists = np.linalg.norm(th_verts_global - object_center, axis=1)

    ff_idx = np.argmin(ff_dists)
    mf_idx = np.argmin(mf_dists)
    th_idx = np.argmin(th_dists)

    # Get local contact positions on fingers
    ff_local_pos = ff_verts[ff_idx]
    mf_local_pos = mf_verts[mf_idx]
    th_local_pos = th_verts[th_idx]

    # Get global contact positions
    ff_global_pos = ff_verts_global[ff_idx]
    mf_global_pos = mf_verts_global[mf_idx]
    th_global_pos = th_verts_global[th_idx]

    print(f"\nFingertip contact points (global):")
    print(f"  Index: {ff_global_pos} (dist to object: {ff_dists[ff_idx]:.4f})")
    print(f"  Middle: {mf_global_pos} (dist to object: {mf_dists[mf_idx]:.4f})")
    print(f"  Thumb: {th_global_pos} (dist to object: {th_dists[th_idx]:.4f})")

    # Get object mesh (cylinder geometry)
    print("\nFinding closest points on object surface...")

    # For a cylinder, we need to generate the mesh or use closest point on surface
    # The object is a cylinder: type='cylinder', size='0.035 0.08' (radius, half-height)
    # Let's create a trimesh cylinder
    cylinder_radius = 0.035
    cylinder_height = 0.16  # 2 * half-height

    # Create cylinder mesh
    obj_mesh = trimesh.creation.cylinder(radius=cylinder_radius, height=cylinder_height, sections=32)

    # Transform to object pose
    obj_pos = d.xpos[object_id]
    obj_mat = d.xmat[object_id].reshape(3, 3)

    obj_mesh.vertices = (obj_mat @ obj_mesh.vertices.T).T + obj_pos

    # Find closest points on object for each fingertip
    ff_closest, ff_dist, ff_face_id = trimesh.proximity.closest_point(obj_mesh, [ff_global_pos])
    mf_closest, mf_dist, mf_face_id = trimesh.proximity.closest_point(obj_mesh, [mf_global_pos])
    th_closest, th_dist, th_face_id = trimesh.proximity.closest_point(obj_mesh, [th_global_pos])

    ff_closest = ff_closest[0]
    mf_closest = mf_closest[0]
    th_closest = th_closest[0]
    ff_face_id = ff_face_id[0]
    mf_face_id = mf_face_id[0]
    th_face_id = th_face_id[0]

    print(f"\nClosest points on object:")
    print(f"  Index -> {ff_closest} (face {ff_face_id}, dist: {np.linalg.norm(ff_global_pos - ff_closest):.4f})")
    print(f"  Middle -> {mf_closest} (face {mf_face_id}, dist: {np.linalg.norm(mf_global_pos - mf_closest):.4f})")
    print(f"  Thumb -> {th_closest} (face {th_face_id}, dist: {np.linalg.norm(th_global_pos - th_closest):.4f})")

    # Convert object contact points to local frame and barycentric coords
    def world_to_local_bary(world_pos, face_id, mesh, obj_pos, obj_mat):
        """Convert world position to local position and barycentric coordinates."""
        # Get face vertices
        face = mesh.faces[face_id]
        v0_world = mesh.vertices[face[0]]
        v1_world = mesh.vertices[face[1]]
        v2_world = mesh.vertices[face[2]]

        # Convert to local frame
        obj_mat_inv = obj_mat.T
        v0_local = obj_mat_inv @ (v0_world - obj_pos)
        v1_local = obj_mat_inv @ (v1_world - obj_pos)
        v2_local = obj_mat_inv @ (v2_world - obj_pos)
        world_local = obj_mat_inv @ (world_pos - obj_pos)

        # Compute barycentric coordinates
        # world_local = w0*v0 + w1*v1 + w2*v2, w0+w1+w2=1
        # Solve: [v0-v2, v1-v2] * [w0, w1]^T = world_local - v2
        A = np.column_stack([v0_local - v2_local, v1_local - v2_local])
        b = world_local - v2_local

        try:
            weights = np.linalg.lstsq(A, b, rcond=None)[0]
            w0, w1 = weights
            w2 = 1 - w0 - w1
            bary = np.array([w0, w1, w2])
        except:
            # Fallback: equal weights
            bary = np.array([1/3, 1/3, 1/3])

        return world_local, bary, face

    ff_local, ff_bary, ff_face_local = world_to_local_bary(ff_closest, ff_face_id, obj_mesh, obj_pos, obj_mat)
    mf_local, mf_bary, mf_face_local = world_to_local_bary(mf_closest, mf_face_id, obj_mesh, obj_pos, obj_mat)
    th_local, th_bary, th_face_local = world_to_local_bary(th_closest, th_face_id, obj_mesh, obj_pos, obj_mat)

    print(f"\nObject contact points (local frame):")
    print(f"  Index: face={ff_face_id}, bary={ff_bary}, local={ff_local}")
    print(f"  Middle: face={mf_face_id}, bary={mf_bary}, local={mf_local}")
    print(f"  Thumb: face={th_face_id}, bary={th_bary}, local={th_local}")

    # Output contact data
    print("\n" + "="*80)
    print("CONTACT DATA FOR OPTIMIZATION")
    print("="*80)

    print("\n# Hand contact points (local coordinates in fingertip frame)")
    print("hand_contacts = [")
    print(f"    # Index finger (ffdistal)")
    print(f"    {{")
    print(f"        'body_name': 'ffdistal',")
    print(f"        'local_pos': np.array({list(ff_local_pos)}),")
    print(f"    }},")
    print(f"    # Middle finger (mfdistal)")
    print(f"    {{")
    print(f"        'body_name': 'mfdistal',")
    print(f"        'local_pos': np.array({list(mf_local_pos)}),")
    print(f"    }},")
    print(f"    # Thumb (thdistal)")
    print(f"    {{")
    print(f"        'body_name': 'thdistal',")
    print(f"        'local_pos': np.array({list(th_local_pos)}),")
    print(f"    }},")
    print("]")

    print("\n# Object contact points (barycentric coordinates on mesh faces)")
    print("# Note: These are for a cylinder mesh generated with trimesh")
    print("# You'll need to regenerate these using the actual object mesh in optimization")
    print("object_contacts = [")
    print(f"    # Corresponding to index finger")
    print(f"    {{")
    print(f"        'face_id': {ff_face_id},")
    print(f"        'bary_coords': np.array({list(ff_bary)}),")
    print(f"    }},")
    print(f"    # Corresponding to middle finger")
    print(f"    {{")
    print(f"        'face_id': {mf_face_id},")
    print(f"        'bary_coords': np.array({list(mf_bary)}),")
    print(f"    }},")
    print(f"    # Corresponding to thumb")
    print(f"    {{")
    print(f"        'face_id': {th_face_id},")
    print(f"        'bary_coords': np.array({list(th_bary)}),")
    print(f"    }},")
    print("]")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
