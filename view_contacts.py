"""
Interactive MuJoCo viewer for finding the right camera angle for contact visualization.
Orbit with mouse, then close the window to print camera parameters.

Usage:
  python view_contacts.py <config.json> [--contact-frame N]
"""

import json
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import sys
import trimesh
import shutil
import re
import xml.etree.cElementTree as ET

from parse_splines import parseSplines
from trajectory import trajectoryConstraintsPolyline, apply_waypoint_rotations
from barrier import process_barriers, barrierConstraints, correct_barrier_axes, read_obj, save_obj, moveToEndPt
from generate_barrier import add_mesh_barriers_to_xml
from smoothspline import create_smoothing_bspline, create_quaternion_bspline
from handContacts import get_mesh_for_body, get_local_pos, local_to_global
from load_contacts import load_contacts_lcexp, get_contact_frame_range
from contacts import process_contacts


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def build_env_xml(agentName, taskName):
    root = ET.Element("mujoco", model=f"{agentName} {taskName}")
    ET.SubElement(root, "include", file=f"tasks/{taskName}.xml")
    ET.SubElement(root, "include", file=f"agents/{agentName}/assets.xml")
    ET.SubElement(root, "include", file=f"agents/{agentName}/actuators.xml")
    worldBody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldBody, "include", file=f"agents/{agentName}/body.xml")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write("env.xml")


def convert_to_quaternions_Allegro(qpos_spline_data):
    qpos = np.zeros((23, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]
    qpos[7:, :] = qpos_spline_data[6:, :]
    for j in range(qpos_spline_data.shape[1]):
        rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
        qpos[4:7, j] = rotation.as_quat()[:3]
        qpos[3, j] = rotation.as_quat()[3]
    return qpos


def convert_to_quaternions_MANO(qpos_spline_data):
    qpos = np.zeros((67, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]
    for j in range(qpos_spline_data.shape[1]):
        for i in range(3, 51, 3):
            rotation = R.from_euler('xyz', qpos_spline_data[i:i+3, j], degrees=False)
            qpos[int((i/3 - 1) * 4 + 4): int(i/3*4 + 3), j] = rotation.as_quat()[:3]
            qpos[int((i/3 - 1) * 4 + 3), j] = rotation.as_quat()[3]
    return qpos


def convert_to_quaternions_object(qpos_spline_data):
    qpos = np.zeros((7, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]
    for j in range(qpos_spline_data.shape[1]):
        rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
        qpos[4:7, j] = rotation.as_quat()[:3]
        qpos[3, j] = rotation.as_quat()[3]
    return qpos


def rotate_keyframe_angles(keyframes, rotation):
    keyframes_modified = keyframes.copy()
    additional_rot = R.from_euler('xyz', np.radians(rotation))
    for i in range(keyframes.shape[1]):
        existing_euler = keyframes[3:6, i]
        existing_rot = R.from_euler('xyz', existing_euler)
        combined_rot = additional_rot * existing_rot
        keyframes_modified[3:6, i] = combined_rot.as_euler('xyz')
    return keyframes_modified


def ensure_quaternion_continuity(qpos, quat_indices=(3, 7)):
    qpos_fixed = qpos.copy()
    start, end = quat_indices
    for i in range(1, qpos.shape[1]):
        prev_quat = qpos_fixed[start:end, i-1]
        curr_quat = qpos_fixed[start:end, i]
        if np.dot(prev_quat, curr_quat) < 0:
            qpos_fixed[start:end, i] = -curr_quat
    return qpos_fixed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_contacts.py <config.json> [--contact-frame N] [--mode hand|object|both]")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    AGENT = config['agent']
    TASK = config['task']

    contact_frame = None
    mode = 'hand'  # default to hand view
    for idx, arg in enumerate(sys.argv):
        if arg == '--contact-frame' and idx + 1 < len(sys.argv):
            contact_frame = int(sys.argv[idx + 1])
        if arg == '--mode' and idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    # === Setup scene (same as render_visualizations.py) ===
    scene_file = config['scene_file']
    object_mesh_file = config['object_mesh_file']
    barriers_raw = config.get('barriers', [])
    visuals_raw = config.get('visuals', [])
    rotation = config.get('rotation', [0, 0, 0])

    # Create object.xml
    object_xml_content = f"""<mujoco>
    <asset>
      <mesh name="object_mesh" file="{os.path.basename(object_mesh_file)}" scale="1 1 1"/>
    </asset>
    <worldbody>
      <body name="object" mocap="true">
        <geom type="mesh" mesh="object_mesh" group="6" rgba="0.8 0.6 0.4 1" mass="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  """
    with open('tasks/object.xml', 'w') as f:
        f.write(object_xml_content)

    os.makedirs('meshes', exist_ok=True)
    mesh_dest = f"meshes/{os.path.basename(object_mesh_file)}"
    if not os.path.exists(mesh_dest):
        shutil.copy(object_mesh_file, mesh_dest)

    build_env_xml(AGENT, TASK)

    barriers = [tuple(b) if isinstance(b, list) else b for b in barriers_raw]
    barriers = process_barriers(barriers)
    visuals = [tuple(v) if isinstance(v, list) else v for v in visuals_raw]
    for visual in visuals:
        if isinstance(visual, tuple) and len(visual) >= 2:
            visual_mesh_path = visual[0]
            visual_mesh_dest = f"meshes/{os.path.basename(visual_mesh_path)}"
            if not os.path.exists(visual_mesh_dest):
                shutil.copy(visual_mesh_path, visual_mesh_dest)

    xml_string = add_mesh_barriers_to_xml(scene_file, barriers, visuals)
    mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
    if mujoco_tag_match:
        insert_pos = mujoco_tag_match.end()
        visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
        xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]

    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    # === Load trajectories ===
    # Try final trajectories first, fall back to input
    hand_traj_path = f'final_trajectories_0.01/{config_name}_hand.npy'
    object_traj_path = f'final_trajectories_0.01/{config_name}_object.npy'

    if os.path.exists(hand_traj_path) and os.path.exists(object_traj_path):
        print(f"Using final trajectories from {hand_traj_path}")
        qpos = np.load(hand_traj_path)
        object_pos = np.load(object_traj_path)
        qpos = ensure_quaternion_continuity(qpos, quat_indices=(3, 7))
        total_frames = qpos.shape[1]
    else:
        print("Final trajectories not found, using input trajectories")
        splines, seconds, _ = parseSplines(f'startingTrajectories/{AGENT}/{TASK}/hand.smexp')
        objectSplines, objectSeconds, _ = parseSplines(f'startingTrajectories/{AGENT}/{TASK}/object.smexp')
        contacts_lcexp_tmp = load_contacts_lcexp(f'startingTrajectories/{AGENT}/{TASK}/contacts.lcexp',
                                                  'Allegro' if AGENT == 'Allegro_right' else 'MANO')
        frames = len(contacts_lcexp_tmp)
        sim_time = np.linspace(0, 1, frames)
        qpos_spline_data = np.array([spline(sim_time) for spline in splines])[:, :, 1]
        object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])[:, :, 1]
        qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
        object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)
        object_qpos = convert_to_quaternions_object(object_qpos_spline_data)
        if AGENT == 'Allegro_right':
            qpos = convert_to_quaternions_Allegro(qpos_spline_data)
            hand_shift = np.zeros([qpos.shape[0], 1])
            object_shift = np.zeros([object_qpos.shape[0], 1])
            pos_shift = [2.2, -2.1, .08]
            hand_shift[:3, 0] += pos_shift
            object_shift[:3, 0] += pos_shift
            qpos += hand_shift
            object_qpos += object_shift
        else:
            qpos = convert_to_quaternions_MANO(qpos_spline_data)
        object_pos = object_qpos.T
        total_frames = qpos.shape[1]

    # === Load contacts ===
    if AGENT == 'Allegro_right':
        hand = 'Allegro'
    else:
        hand = 'MANO'

    contacts_lcexp = load_contacts_lcexp(f'startingTrajectories/{AGENT}/{TASK}/contacts.lcexp', hand)
    contact_start, contact_end = get_contact_frame_range(contacts_lcexp)

    if contact_frame is None:
        contact_frame = (contact_start + contact_end) // 2

    if AGENT == 'Allegro_right':
        hand_body_names = [
            'allegro_palm',
            'allegro_th_base', 'allegro_th_proximal', 'allegro_th_medial', 'allegro_th_distal', 'allegro_th_tip',
            'allegro_ff_base', 'allegro_ff_proximal', 'allegro_ff_medial', 'allegro_ff_distal', 'allegro_ff_tip',
            'allegro_mf_base', 'allegro_mf_proximal', 'allegro_mf_medial', 'allegro_mf_distal', 'allegro_mf_tip',
            'allegro_rf_base', 'allegro_rf_proximal', 'allegro_rf_medial', 'allegro_rf_distal', 'allegro_rf_tip',
        ]
    else:
        hand_body_names = [
            'wrist',
            'thumb1', 'thumb2', 'thumb3',
            'ring1', 'ring2', 'ring3',
            'pinky1', 'pinky2', 'pinky3',
            'middle1', 'middle2', 'middle3',
            'index1', 'index2', 'index3',
        ]

    hand_components_len = len(hand_body_names)
    hand_component_body_ids = []
    for bname in hand_body_names:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        if body_id == -1:
            raise ValueError(f"Body '{bname}' not found in model")
        hand_component_body_ids.append(body_id)

    hand_contacts, object_contacts = process_contacts(contacts_lcexp, hand_components_len)

    hand_components = [None] * hand_components_len
    for j in range(hand_components_len):
        hand_components[j] = get_mesh_for_body(m, hand_component_body_ids[j])

    object_mesh = trimesh.load(config['object_mesh_file'], process=False)

    # === Compute contact positions ===
    frame_idx = min(contact_frame, total_frames - 1)
    print(f"Contact frame: {frame_idx}")

    # First set actual pose to compute contact positions
    d.qpos = qpos[:, frame_idx]
    d.mocap_pos[0] = object_pos[frame_idx, :3]
    d.mocap_quat[0] = object_pos[frame_idx, 3:]
    mujoco.mj_forward(m, d)

    # Object contacts
    obj_contact_world_positions = []
    obj_center = object_pos[frame_idx, :3].copy()
    quat_mujoco = object_pos[frame_idx, 3:]
    quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
    rotation_matrix = R.from_quat(quat_scipy).as_matrix()

    obj_verts = object_contacts[frame_idx]
    if obj_verts is not None and len(obj_verts) > 0:
        for vert_idx in obj_verts:
            local_vertex = object_mesh.vertices[vert_idx]
            world_vertex = rotation_matrix @ local_vertex + obj_center
            obj_contact_world_positions.append(world_vertex)

    # Hand contacts - compute with rest pose
    rest_qpos = qpos[:, frame_idx].copy()
    rest_qpos[7:] = 0  # open grip
    d.qpos = rest_qpos
    d.mocap_pos[0] = [0, 0, -10]  # hide object
    mujoco.mj_forward(m, d)

    hand_contact_world_positions = []
    for hand_comp_id in range(hand_components_len):
        contacts_this_frame = hand_contacts[hand_comp_id][frame_idx]
        if contacts_this_frame is not None:
            for contact in contacts_this_frame:
                face_id, bary_coords, object_contact_idx = contact
                local_pos = get_local_pos(face_id, bary_coords,
                                          hand_components[hand_comp_id][0],
                                          hand_components[hand_comp_id][1])
                world_pos = local_to_global(local_pos, hand_component_body_ids[hand_comp_id], d)
                hand_contact_world_positions.append(world_pos)

    print(f"Hand contacts: {len(hand_contact_world_positions)}, Object contacts: {len(obj_contact_world_positions)}")

    # === Set up the scene for viewing ===
    if mode == 'hand':
        # Rest pose, no object
        d.qpos = rest_qpos
        d.mocap_pos[0] = [0, 0, -10]
        mujoco.mj_forward(m, d)
        contact_positions = hand_contact_world_positions
        print("Mode: HAND (rest pose with contacts)")
    elif mode == 'object':
        # Object only
        d.qpos[:] = 0
        d.qpos[:3] = [0, 0, -10]  # hide hand
        d.mocap_pos[0] = object_pos[frame_idx, :3]
        d.mocap_quat[0] = object_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)
        contact_positions = obj_contact_world_positions
        print("Mode: OBJECT (with contacts)")
    else:
        # Both
        d.qpos = rest_qpos
        d.mocap_pos[0] = object_pos[frame_idx, :3]
        d.mocap_quat[0] = object_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)
        contact_positions = hand_contact_world_positions + obj_contact_world_positions
        print("Mode: BOTH (hand + object)")

    # Add contact spheres as sites to the model for visualization
    # We'll use the viewer's built-in perturbation geoms approach
    # Actually, we need to add geoms via the viewer callback

    def add_contact_geoms(viewer):
        """Add red contact spheres to the viewer scene."""
        scene = viewer.user_scn
        for pos in contact_positions:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.003, 0, 0],
                pos=np.array(pos, dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 1], dtype=np.float32))
            scene.ngeom += 1

    print("\n" + "=" * 60)
    print("INTERACTIVE VIEWER")
    print("=" * 60)
    print("  - Rotate: Left-click + drag")
    print("  - Pan: Right-click + drag")
    print("  - Zoom: Scroll wheel")
    print("  - Close window to print camera parameters")
    print("=" * 60 + "\n")

    # Launch interactive viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Set initial camera to look at hand
        if mode == 'hand' or mode == 'both':
            viewer.cam.lookat[:] = rest_qpos[:3]
        else:
            viewer.cam.lookat[:] = obj_center
        viewer.cam.distance = 0.35
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0

        # Add contact geoms
        add_contact_geoms(viewer)

        while viewer.is_running():
            viewer.sync()

        # Print camera parameters when window is closed
        print("\n" + "=" * 60)
        print("CAMERA PARAMETERS (copy these into render_7_contacts):")
        print("=" * 60)
        print(f"  cam.lookat[:] = [{viewer.cam.lookat[0]:.6f}, {viewer.cam.lookat[1]:.6f}, {viewer.cam.lookat[2]:.6f}]")
        print(f"  cam.distance = {viewer.cam.distance:.6f}")
        print(f"  cam.azimuth = {viewer.cam.azimuth:.1f}")
        print(f"  cam.elevation = {viewer.cam.elevation:.1f}")
        print("=" * 60)
