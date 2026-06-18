"""Interactive MuJoCo viewers for inspecting trajectories.

Three entry points:

* ``view_result(state)``     -- live preview at the end of a retargeting run.
* ``play_demonstration(cfg)``-- replay the ORIGINAL demonstration (before warp).
* ``play_saved(cfg)``        -- replay a previously saved retargeted trajectory.

All three play the hand and object back in a passive viewer with overlay
geometry: the object path, barriers, bounding spheres, waypoints, and the
hand/object contact points.
"""
import os
import time
import colorsys

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

from trajwarp.hand_warp.hand_mesh import get_local_pos, local_to_global


def time_to_rgb(t):
    """Map ``t`` in [0, 1] to an RGB tuple along a green->red hue ramp."""
    hue = (1.0 - t) * 120.0 / 360.0
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)


_EYE = np.eye(3).flatten()


def _draw_barriers(scn, offset, barriers, alpha=1.0):
    """Draw primitive (sphere/rect) barriers; mesh barriers are already in the
    compiled model and are skipped. Returns the number of geoms added."""
    if not barriers:
        return 0
    n = 0
    for barrier in barriers:
        if isinstance(barrier, str):
            continue
        shape, params = barrier[0], barrier[1]
        if shape == 'sphere':
            mujoco.mjv_initGeom(
                scn.geoms[offset + n], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[params['rad'], 0, 0], pos=params['pos'], mat=_EYE,
                rgba=[0.5, 0.5, 0.5, alpha])
            n += 1
        elif shape == 'rect':
            mujoco.mjv_initGeom(
                scn.geoms[offset + n], type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array(params['dims']) / 2, pos=params['pos'], mat=_EYE,
                rgba=[0.5, 0.5, 0.5, alpha])
            n += 1
    return n


def _draw_waypoints(scn, offset, waypts):
    """Draw waypoint markers. Returns the number of geoms added."""
    if not waypts:
        return 0
    for j, waypt in enumerate(waypts):
        mujoco.mjv_initGeom(
            scn.geoms[offset + j], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0], pos=waypt[0], mat=_EYE, rgba=[0, 0, 0, 1])
    return len(waypts)


def _draw_object_path(scn, offset, pts, contact_start, contact_end):
    """Draw the object path: contact frames colored along a time gradient,
    pre/post frames faint gray. Returns the number of geoms added."""
    contact_len = max(contact_end - contact_start, 1)
    for j, pt in enumerate(pts):
        if contact_start <= j <= contact_end:
            r, g, b = time_to_rgb((j - contact_start) / contact_len)
            rgba = np.array([r, g, b, 1])
        else:
            rgba = np.array([0.5, 0.5, 0.5, 0.3])
        mujoco.mjv_initGeom(
            scn.geoms[offset + j], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0], pos=np.array(pt), mat=_EYE, rgba=rgba)
    return len(pts)


def _draw_hand_contacts(scn, offset, hand_contacts, hand_components,
                        hand_components_len, hand_component_body_ids, d, frame):
    """Draw the hand-side contact points for ``frame``. Returns geoms added."""
    n = 0
    for cid in range(hand_components_len):
        contacts_this_frame = hand_contacts[cid][frame]
        if contacts_this_frame is None:
            continue
        for face_id, bary_coords, _ in contacts_this_frame:
            local_pos = get_local_pos(face_id, bary_coords,
                                      hand_components[cid][0], hand_components[cid][1])
            global_vertex = local_to_global(local_pos, hand_component_body_ids[cid], d)
            mujoco.mjv_initGeom(
                scn.geoms[offset + n], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.002, 0, 0], pos=np.array(global_vertex), mat=_EYE,
                rgba=[0, 0, 1, 1])
            n += 1
    return n


def _ensure_quaternion_continuity(qpos, quat_indices=(3, 7)):
    """Flip wrist-quaternion signs so consecutive frames stay in one hemisphere
    (q and -q are the same rotation but naive playback spins between them)."""
    qpos_fixed = qpos.copy()
    start, end = quat_indices
    for i in range(1, qpos.shape[1]):
        if np.dot(qpos_fixed[start:end, i - 1], qpos_fixed[start:end, i]) < 0:
            qpos_fixed[start:end, i] = -qpos_fixed[start:end, i]
    return qpos_fixed


def view_result(state):
    """Open a passive viewer animating ``state``'s retargeted trajectory."""
    config = state.config
    m, d = state.model, state.data
    qpos = state.qpos
    retargeted_spline_pos = state.retargeted_spline_pos
    frames = state.frames
    hand_components = state.hand_components
    hand_components_len = state.hand_components_len
    hand_component_body_ids = state.hand_component_body_ids
    hand_contacts = state.hand_contacts
    object_contacts = state.object_contacts
    obj = state.object_mesh
    start_frame_count = state.start_frame_count
    contact_frame_count = state.contact_frame_count
    barriers = state.barriers
    boundary_radius = config.boundary_radius
    hand_boundary_radius = config.hand_boundary_radius
    waypts = state.waypts

    frame_pts = []
    obj_retarget_frame_pts = []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()

            if i % frames == 0:
                frame_pts = []
                obj_retarget_frame_pts = []
                i = 0

            d.qpos = qpos[:, i]
            d.mocap_pos = retargeted_spline_pos[i, :3]
            d.mocap_quat = retargeted_spline_pos[i, 3:]
            mujoco.mj_forward(m, d)

            local_hand_contacts = {}
            for hand_component_id in range(hand_components_len):
                contacts_this_frame = hand_contacts[hand_component_id][i]
                if contacts_this_frame is not None:
                    local_hand_contacts[hand_component_id] = []
                    for contact in contacts_this_frame:
                        face_id, bary_coords, object_contact_idx = contact
                        local_pos = get_local_pos(
                            face_id, bary_coords,
                            hand_components[hand_component_id][0],
                            hand_components[hand_component_id][1])
                        local_hand_contacts[hand_component_id].append(local_pos)
                    local_hand_contacts[hand_component_id] = np.array(local_hand_contacts[hand_component_id])

            quat_scipy = np.array([d.mocap_quat[0, 1], d.mocap_quat[0, 2], d.mocap_quat[0, 3], d.mocap_quat[0, 0]])
            rotation = R.from_quat(quat_scipy)
            rotation_matrix = rotation.as_matrix()

            frame_pts.append(qpos[:3, i])
            obj_retarget_frame_pts.append(retargeted_spline_pos[i, :3])

            geometry_count = 0

            if len(obj_retarget_frame_pts) <= start_frame_count:
                rgba = np.array([0, 0, 1, 1])
            elif len(obj_retarget_frame_pts) <= start_frame_count + contact_frame_count:
                rgba = np.array([0, 1, 0, 1])
            else:
                rgba = np.array([1, 0, 0, 1])

            for j in range(len(obj_retarget_frame_pts)):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[j + geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],
                    pos=np.array(obj_retarget_frame_pts[j]),
                    mat=np.eye(3).flatten(),
                    rgba=rgba,
                )
            geometry_count += len(obj_retarget_frame_pts)

            if barriers is not None:
                for j in range(len(barriers)):
                    if isinstance(barriers[j], str):
                        continue
                    elif barriers[j][0] == 'sphere':
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[j + geometry_count],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[barriers[j][1]['rad'], 0, 0],
                            pos=barriers[j][1]['pos'],
                            mat=np.eye(3).flatten(),
                            rgba=[0.5, 0.5, 0.5, 1])
                    elif barriers[j][0] == 'rect':
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[j + geometry_count],
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=np.array(barriers[j][1]['dims']) / 2,
                            pos=barriers[j][1]['pos'],
                            mat=np.eye(3).flatten(),
                            rgba=[0.5, 0.5, 0.5, 1])
                geometry_count += len(barriers)

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[boundary_radius, 0, 0],
                pos=np.array(obj_retarget_frame_pts[i]),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.0, 0.0, 0.5, 0.3]))
            geometry_count += 1

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[hand_boundary_radius, 0, 0],
                pos=np.array(frame_pts[i]),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.0, 0.0, 0.5, 0.3]))
            geometry_count += 1

            if waypts is not None:
                for j in range(len(waypts)):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[geometry_count + j],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=waypts[j][0],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 1]))
                geometry_count += len(waypts)

            if isinstance(object_contacts[i], np.ndarray):
                for j in range(len(object_contacts[i])):
                    local_vertex = obj.vertices[object_contacts[i][j]]
                    world_vertex = (rotation_matrix @ local_vertex + d.mocap_pos)[0]
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[j + geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.001, 0, 0],
                        pos=np.array(world_vertex),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1]),
                    )
                geometry_count += len(object_contacts[i])

            for j in local_hand_contacts:
                for k in range(len(local_hand_contacts[j])):
                    local_vertex = local_hand_contacts[j][k]
                    global_vertex = local_to_global(local_vertex, hand_component_body_ids[j], d)
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[k + geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.003, 0, 0],
                        pos=np.array(global_vertex),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 1, 1]),
                    )
                geometry_count += len(local_hand_contacts[j])

            viewer.user_scn.ngeom = geometry_count

            viewer.sync()
            i += 1

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def play_demonstration(config_path):
    """Replay the original (un-warped) demonstration referenced by ``config_path``.

    Loads exactly the same demonstration data the pipeline starts from, then
    animates the hand and object with the object path colored by contact time
    and the hand-object contact points overlaid.
    """
    from trajwarp.io.config import load_config
    from trajwarp.pipeline import load_demonstration

    config = load_config(config_path)
    config.config_path = config_path
    state = load_demonstration(config)

    m, d = state.model, state.data
    qpos = _ensure_quaternion_continuity(state.qpos)
    object_qpos = state.object_qpos
    frames = state.frames
    contact_start, contact_end = state.start_idx, state.end_idx

    print(f"Playing original demonstration: {state.config_name} "
          f"({frames} frames, contact {contact_start}-{contact_end})")

    obj_path = []
    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()
            if i % frames == 0:
                obj_path = []
                i = 0

            d.qpos = qpos[:, i]
            d.mocap_pos = object_qpos[:3, i]
            d.mocap_quat = object_qpos[3:, i]
            mujoco.mj_forward(m, d)

            obj_path.append(object_qpos[:3, i])
            count = _draw_object_path(viewer.user_scn, 0, obj_path, contact_start, contact_end)
            count += _draw_barriers(viewer.user_scn, count, state.barriers, alpha=0.3)
            count += _draw_hand_contacts(
                viewer.user_scn, count, state.hand_contacts, state.hand_components,
                state.hand_components_len, state.hand_component_body_ids, d, i)
            viewer.user_scn.ngeom = count

            viewer.sync()
            i += 1
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def play_saved(config_path, trajectory_dir="final_trajectories_0.0035"):
    """Replay a previously saved retargeted trajectory.

    Reads ``<trajectory_dir>/<config>_hand.npy`` and ``..._object.npy`` and
    animates them in the same scene the config describes, with the object path
    and barriers overlaid.
    """
    from trajwarp.io.config import load_config
    from trajwarp.io.scene_xml import write_object_xml
    from trajwarp.io.spline_io import parseSplines
    from trajwarp.io.contact_io import load_contacts_lcexp, get_contact_frame_range
    from trajwarp.utils.agents import get_hand_type
    from trajwarp.pipeline import _build_scene_xml

    config = load_config(config_path)
    config.config_path = config_path
    name = os.path.splitext(os.path.basename(config_path))[0]

    hand_path = os.path.join(trajectory_dir, f"{name}_hand.npy")
    object_path = os.path.join(trajectory_dir, f"{name}_object.npy")
    if not os.path.exists(hand_path) or not os.path.exists(object_path):
        raise FileNotFoundError(
            f"Saved trajectory not found in '{trajectory_dir}' for '{name}'. "
            f"Run retarget.py first or pass --trajectory-dir.")

    write_object_xml(config.object_mesh_file)
    xml_string, barriers = _build_scene_xml(config)
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    qpos = _ensure_quaternion_continuity(np.load(hand_path))
    object_traj = np.load(object_path)
    frames = qpos.shape[1]

    base = f"startingTrajectories/{config.agent}/{config.task}"
    _, seconds, _ = parseSplines(f"{base}/object.smexp")
    m.opt.timestep = 2 * seconds / frames

    contacts_lcexp = load_contacts_lcexp(f"{base}/contacts.lcexp", get_hand_type(config.agent))
    contact_start, contact_end = get_contact_frame_range(contacts_lcexp)

    print(f"Playing retargeted trajectory: {name} from {trajectory_dir} "
          f"({frames} frames)")

    obj_path = []
    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()
            if i % frames == 0:
                obj_path = []
                i = 0

            d.qpos = qpos[:, i]
            d.mocap_pos[0] = object_traj[i, :3]
            d.mocap_quat[0] = object_traj[i, 3:7]
            mujoco.mj_forward(m, d)

            obj_path.append(object_traj[i, :3])
            count = _draw_object_path(viewer.user_scn, 0, obj_path, contact_start, contact_end)
            count += _draw_barriers(viewer.user_scn, count, barriers, alpha=0.3)
            viewer.user_scn.ngeom = count

            viewer.sync()
            i += 1
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)
