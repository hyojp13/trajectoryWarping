"""
Visualization pipeline: renders each stage of the trajectory retargeting process.
Produces 7 images showing: input trajectory, scene with waypoints/barriers,
object retargeting stages, final output, and contact visualization.
"""

import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.interpolate
import xml.etree.cElementTree as ET
import os
import re
import cv2
import trimesh
import shutil
import colorsys
import sys

from parse_splines import parseSplines
from trajectory import trajectoryConstraintsPolyline, apply_waypoint_rotations
from barrier import process_barriers, barrierConstraints, correct_barrier_axes, read_obj, save_obj, moveToEndPt
from generate_barrier import add_mesh_barriers_to_xml
from smoothspline import create_smoothing_bspline, create_quaternion_bspline
from handContacts import get_mesh_for_body, get_local_pos, local_to_global
from load_contacts import load_contacts_lcexp, get_contact_frame_range
from contacts import process_contacts


def time_to_rgb(t):
    """Convert t in [0,1] to RGB: green(0) -> red(1)."""
    hue = (1.0 - t) * 120.0 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return r, g, b


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


def setup_scene(config):
    """Set up MuJoCo scene from config. Returns (m, d, xml_string, barriers, visuals, waypts, AGENT, TASK)."""
    AGENT = config['agent']
    TASK = config['task']
    scene_file = config['scene_file']
    object_mesh_file = config['object_mesh_file']
    barriers_raw = config.get('barriers', [])
    visuals_raw = config.get('visuals', [])
    rotation = config.get('rotation', [0, 0, 0])

    waypts_raw = config.get('waypts', [])
    waypts = []
    for w in waypts_raw:
        if len(w) == 4:
            waypts.append((np.array(w[0]), w[1], w[2], w[3]))
        else:
            waypts.append((np.array(w[0]), w[1], w[2]))

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

    # Process barriers
    barriers = [tuple(b) if isinstance(b, list) else b for b in barriers_raw]
    barriers = process_barriers(barriers)

    # Process visuals
    visuals = [tuple(v) if isinstance(v, list) else v for v in visuals_raw]
    for visual in visuals:
        if isinstance(visual, tuple) and len(visual) >= 2:
            visual_mesh_path = visual[0]
            visual_mesh_dest = f"meshes/{os.path.basename(visual_mesh_path)}"
            if not os.path.exists(visual_mesh_dest):
                shutil.copy(visual_mesh_path, visual_mesh_dest)

    # Build XML
    xml_string = add_mesh_barriers_to_xml(scene_file, barriers, visuals)
    mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
    if mujoco_tag_match:
        insert_pos = mujoco_tag_match.end()
        visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
        xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]

    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    return m, d, xml_string, barriers, visuals, waypts, AGENT, TASK


def make_camera():
    # sink top 2
    cam = mujoco.MjvCamera()
    cam.azimuth = -178.9
    cam.elevation = -20.8
    cam.distance = 2.473
    cam.lookat[:] = [-0.228, -1.856, 0.907]
    return cam

    # sink top
    # cam = mujoco.MjvCamera()
    # cam.azimuth = -180.0
    # cam.elevation = -8.2
    # cam.distance = 3.276
    # cam.lookat[:] = [-0.660, -1.881, 1.084]
    # return cam

    # sink
    # cam = mujoco.MjvCamera()
    # cam.azimuth = 178.9
    # cam.elevation = -9.7
    # cam.distance = 3.005
    # cam.lookat[:] = [-1.030, -1.791, 1.059]
    # return cam

    # front view stove top
    # cam = mujoco.MjvCamera()
    # cam.azimuth = 90.0
    # cam.elevation = -11.2
    # cam.distance = 2.059
    # cam.lookat[:] = [2.418, 0.001, 1.456]
    # return cam

    # side view stove top
    # cam = mujoco.MjvCamera()
    # cam.azimuth = 178.8
    # cam.elevation = -11.4
    # cam.distance = 3.456
    # cam.lookat[:] = [-0.016, -0.378, 0.614]
    # return cam


def make_scene_options():
    """Returns (scene_option_full, scene_option_bg, scene_option_obj_only)."""
    # Full: hand + object + scene
    so_full = mujoco.MjvOption()
    so_full.geomgroup[0] = 0
    so_full.geomgroup[1] = 1  # scene
    so_full.geomgroup[2] = 1  # hand
    so_full.geomgroup[5] = 1  # object
    so_full.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        so_full.sitegroup[i] = 0

    # Background: scene only (no hand, no object)
    so_bg = mujoco.MjvOption()
    so_bg.geomgroup[0] = 0
    so_bg.geomgroup[1] = 1
    so_bg.geomgroup[2] = 0
    so_bg.geomgroup[5] = 0
    so_bg.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        so_bg.sitegroup[i] = 0

    # Object only: scene + object, no hand
    so_obj = mujoco.MjvOption()
    so_obj.geomgroup[0] = 0
    so_obj.geomgroup[1] = 1
    so_obj.geomgroup[2] = 0  # hand OFF
    so_obj.geomgroup[5] = 1  # object ON
    so_obj.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        so_obj.sitegroup[i] = 0

    return so_full, so_bg, so_obj


def add_barriers_geoms(renderer, geometry_count, barriers, highlight=False):
    """Add barrier geometries to the scene. Returns updated geometry_count."""
    if barriers is None:
        return geometry_count
    for j in range(len(barriers)):
        if isinstance(barriers[j], str):
            continue
        elif barriers[j][0] == 'sphere':
            rgba = [1, 0.3, 0.3, 0.6] if highlight else [0.5, 0.5, 0.5, 0.4]
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[barriers[j][1]['rad'], 0, 0],
                pos=barriers[j][1]['pos'],
                mat=np.eye(3).flatten(),
                rgba=np.array(rgba))
            geometry_count += 1
        elif barriers[j][0] == 'rect':
            rgba = [1, 0.3, 0.3, 0.6] if highlight else [0.5, 0.5, 0.5, 0.4]
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array(barriers[j][1]['dims']) / 2,
                pos=barriers[j][1]['pos'],
                mat=np.eye(3).flatten(),
                rgba=np.array(rgba))
            geometry_count += 1
    return geometry_count


def add_waypoints_geoms(renderer, geometry_count, waypts_processed, highlight=False, color=False):
    """Add waypoint geometries to the scene. Returns updated geometry_count."""
    if color:
        rgba_arr = np.array([[1, 0, 0.565, 1], [0, 0, 1, 1], [0.224, 1.0, 0.078, 1.0]])
    for j in range(len(waypts_processed)):
        size = 0.02 if highlight else 0.008
        rgba = [1, 0, 0, 1] if highlight else [0.5, 0.5, 0.5, 0.5]
        if color and j < 3:
            rgba = rgba_arr[j]
        mujoco.mjv_initGeom(
            renderer.scene.geoms[geometry_count],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0],
            pos=waypts_processed[j][0],
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba))
        if color and j < 3:
            renderer.scene.geoms[geometry_count].emission = 1.0
            renderer.scene.geoms[geometry_count].specular = 0.8
            renderer.scene.geoms[geometry_count].shininess = 1.0
        geometry_count += 1
    return geometry_count


def add_trajectory_dots(renderer, geometry_count, positions, contact_start, contact_end):
    """Add colored trajectory dots for contact region. Returns updated geometry_count."""
    contact_len = max(contact_end - contact_start, 1)
    for j in range(contact_start, contact_end + 1):
        if j >= len(positions):
            break
        t = (j - contact_start) / contact_len
        r, g, b = time_to_rgb(t)
        mujoco.mjv_initGeom(
            renderer.scene.geoms[geometry_count],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            pos=positions[j],
            mat=np.eye(3).flatten(),
            rgba=np.array([r, g, b, 1]))
        geometry_count += 1
    return geometry_count


def render_overlay_composite(renderer, cam, scene_option_full, scene_option_bg,
                              d, m, qpos, object_pos, overlay_frames, total_frames,
                              contact_start, contact_end):
    """Render overlay composite of hand+object at specified frames. Returns composite image."""
    last_frame = overlay_frames[-1]

    # Render background at last frame
    d.qpos = qpos[:, last_frame]
    d.mocap_pos[0] = object_pos[last_frame, :3]
    d.mocap_quat[0] = object_pos[last_frame, 3:]
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam, scene_option=scene_option_bg)
    bg_frame = renderer.render().copy()
    composite = bg_frame.astype(np.float64)

    contact_len = max(contact_end - contact_start, 1)

    for idx, frame_idx in enumerate(overlay_frames):
        if frame_idx >= total_frames:
            continue
        d.qpos = qpos[:, frame_idx]
        d.mocap_pos[0] = object_pos[frame_idx, :3]
        d.mocap_quat[0] = object_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, camera=cam, scene_option=scene_option_full)
        renderer.scene.ngeom = renderer.scene.ngeom
        fg_frame = renderer.render().copy()

        renderer.update_scene(d, camera=cam, scene_option=scene_option_bg)
        renderer.scene.ngeom = renderer.scene.ngeom
        fg_bg_frame = renderer.render().copy()

        diff = np.abs(fg_frame.astype(np.float64) - fg_bg_frame.astype(np.float64))
        mask = (diff.max(axis=2) > 5).astype(np.float64)

        t = (frame_idx - contact_start) / contact_len
        t = np.clip(t, 0, 1)
        tint = np.array(time_to_rgb(t))

        is_last = (idx == len(overlay_frames) - 1)
        if is_last:
            alpha = 1.0
            blended_fg = fg_frame.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg_frame.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    return np.clip(composite, 0, 255).astype(np.uint8)


def load_input_trajectory(config):
    """Load original input hand + object trajectories from startingTrajectories/."""
    AGENT = config['agent']
    TASK = config['task']
    rotation = config.get('rotation', [0, 0, 0])

    if AGENT == 'MANO_right' or AGENT == 'trajectories':
        hand = 'MANO'
    elif AGENT == 'Allegro_right':
        hand = 'Allegro'

    splines, seconds, _ = parseSplines(f'startingTrajectories/{AGENT}/{TASK}/hand.smexp')
    objectSplines, objectSeconds, _ = parseSplines(f'startingTrajectories/{AGENT}/{TASK}/object.smexp')

    contacts_lcexp = load_contacts_lcexp(f'startingTrajectories/{AGENT}/{TASK}/contacts.lcexp', hand)
    frames = len(contacts_lcexp)
    contact_start, contact_end = get_contact_frame_range(contacts_lcexp)

    sim_time = np.linspace(0, 1, frames)
    qpos_spline_data = np.array([spline(sim_time) for spline in splines])[:, :, 1]
    object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])[:, :, 1]

    qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
    object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)

    object_qpos = convert_to_quaternions_object(object_qpos_spline_data)

    if AGENT == 'MANO_right' or AGENT == 'trajectories':
        qpos = convert_to_quaternions_MANO(qpos_spline_data)
    elif AGENT == 'Allegro_right':
        qpos = convert_to_quaternions_Allegro(qpos_spline_data)

    if AGENT == 'Allegro_right':
        hand_shift = np.zeros([qpos.shape[0], 1])
        object_shift = np.zeros([object_qpos.shape[0], 1])
        pos_shift = [2.2, -2.1, .08]
        hand_shift[:3, 0] += pos_shift
        object_shift[:3, 0] += pos_shift
        qpos += hand_shift
        object_qpos += object_shift

    # object_qpos is (7, frames), transpose to (frames, 7) for consistency
    return qpos, object_qpos.T, frames, contact_start, contact_end, seconds, contacts_lcexp


def retarget_object_trajectory(config, object_qpos_input, contact_start, contact_end, frames,
                                use_barriers=False):
    """
    Re-run object trajectory retargeting inline.
    object_qpos_input: (7, frames) in MuJoCo format
    Returns: retargeted_spline_pos (frames, 7)
    """
    AGENT = config['agent']
    rotation = config.get('rotation', [0, 0, 0])
    new_start_pos_shift = np.array(config.get('new_start_pos_shift', [0, 0, 0]))
    end_final_pos_shift = np.array(config.get('end_final_pos_shift', [0, 0, 0]))
    end_obj_pos_shift = np.array(config.get('end_obj_pos_shift', [0, 0, 0]))
    boundary_radius = config.get('boundary_radius', 0.1)
    extra_pt_count = config.get('extra_pt_count', 0)
    barriers_raw = config.get('barriers', [])

    waypts_raw = config.get('waypts', [])
    waypts = []
    for w in waypts_raw:
        if len(w) == 4:
            waypts.append((np.array(w[0]), w[1], w[2], w[3]))
        else:
            waypts.append((np.array(w[0]), w[1], w[2]))

    object_qpos = object_qpos_input.copy()  # (7, frames)

    start_pos = object_qpos[:3, 0].copy()
    end_pos = object_qpos[:3, frames-1].copy()

    newStartPos = start_pos + new_start_pos_shift
    endFinalPos = end_pos + end_final_pos_shift
    endObjPos = endFinalPos + end_obj_pos_shift

    # Process waypoints to absolute positions
    waypts_processed = []
    for waypt in waypts:
        waypt_pos = newStartPos + waypt[0]
        if len(waypt) == 4:
            waypts_processed.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
        else:
            waypts_processed.append((waypt_pos, waypt[1], waypt[2]))

    startIdx = contact_start
    endIdx = contact_end

    # Waypoint constraints
    pos_waypt_constrained, _, _, waypts_idx = trajectoryConstraintsPolyline(
        object_qpos[:3, :], startPos=newStartPos, endPos=endObjPos,
        floor_height=start_pos[2], waypts=waypts_processed)

    # Apply waypoint rotations
    object_rotation_qpos = apply_waypoint_rotations(object_qpos[3:7, :], waypts_processed, waypts_idx, startIdx, frames)

    start_frame_count = startIdx
    contact_frame_count = endIdx - startIdx + 1
    end_frame_count = frames - start_frame_count - contact_frame_count

    # Save waypoint-constrained trajectory to file (clean state for each call)
    save_obj(pos_waypt_constrained.T, "scene/curve_positions.obj")

    # Barrier constraints (optional)
    if use_barriers and len(barriers_raw) > 0:
        barriers = [tuple(b) if isinstance(b, list) else b for b in barriers_raw]
        barriers = process_barriers(barriers)
        barrierConstraints(pos_waypt_constrained.T, boundary_radius, barriers)

    # Ensure above surface
    obj_pts = read_obj("scene/curve_positions.obj")
    obj_pts[:, 2] = np.maximum(obj_pts[:, 2], start_pos[2])
    save_obj(obj_pts, "scene/curve_positions.obj")

    trajectory = read_obj("scene/curve_positions.obj")

    # Ensure trajectory ends at end position
    if not np.isclose(trajectory[-1, :], endFinalPos, atol=1e-3).all():
        moveToEndPt("scene/curve_positions.obj", endFinalPos, extra_pt_count)
        trajectory = read_obj("scene/curve_positions.obj")
    else:
        extra_pt_count = 0

    # Build new object qpos
    new_object_qpos = np.zeros((object_qpos.shape[0], object_qpos.shape[1] + extra_pt_count))
    new_object_qpos[:3, :] = trajectory.T
    new_object_qpos[3:, :object_qpos.shape[1]] = object_rotation_qpos
    if extra_pt_count != 0:
        new_object_qpos[3:, -extra_pt_count:] = new_object_qpos[3:, -extra_pt_count-1].reshape(object_qpos.shape[0]-3, 1)

    # Convert to B-spline
    final_waypt_timesteps = np.array([w[2] for w in waypts_processed])
    pos_spline, contact_timewarp = create_smoothing_bspline(
        new_object_qpos[:3, :].T, parameterization="waypts",
        waypts_info=(waypts_idx, final_waypt_timesteps))

    # Quaternion spline
    quat_mujoco = new_object_qpos[3:7, :].T
    quat_scipy = np.column_stack([quat_mujoco[:, 1:4], quat_mujoco[:, 0]])
    quat_spline, _ = create_quaternion_bspline(quat_scipy, waypts_info=(waypts_idx, final_waypt_timesteps))

    retargeted_sim_time = np.linspace(0, 1, contact_frame_count)
    retargeted_pos = np.array(scipy.interpolate.splev(retargeted_sim_time, pos_spline)).T
    retargeted_quat_scipy = np.array(scipy.interpolate.splev(retargeted_sim_time, quat_spline)).T
    retargeted_quat_scipy_normalized = retargeted_quat_scipy / np.linalg.norm(retargeted_quat_scipy, axis=1, keepdims=True)
    retargeted_quat_mujoco = np.column_stack([retargeted_quat_scipy_normalized[:, 3], retargeted_quat_scipy_normalized[:, 0:3]])

    retargeted_spline_pos = np.hstack([retargeted_pos, retargeted_quat_mujoco])

    # Pad with start/end frames
    retargeted_start_pos = np.repeat(retargeted_spline_pos[0:1, :], start_frame_count, axis=0)
    retargeted_end_pos = np.repeat(retargeted_spline_pos[-1:, :], end_frame_count, axis=0)
    retargeted_spline_pos = np.vstack((retargeted_start_pos, retargeted_spline_pos, retargeted_end_pos))

    return retargeted_spline_pos  # (frames, 7)


# ============================================================
# Render functions
# ============================================================

def _make_white_bg_scene_option(show_hand=False, show_object=False):
    """Scene option that hides everything except optionally hand/object."""
    so = mujoco.MjvOption()
    so.geomgroup[0] = 0
    so.geomgroup[1] = 0  # scene OFF
    so.geomgroup[2] = 1 if show_hand else 0
    so.geomgroup[5] = 1 if show_object else 0
    so.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        so.sitegroup[i] = 0
    return so


def _render_on_white(renderer, cam, d, so_fg, so_empty, extra_geoms_fn=None):
    """Render foreground objects on a white background using diff-masking.

    Args:
        renderer: MuJoCo renderer
        cam: camera
        d: MjData
        so_fg: scene option showing the objects to render
        so_empty: scene option showing nothing (for background diff)
        extra_geoms_fn: optional callback(renderer, gc) -> gc to add extra geoms (dots, contacts)
    Returns:
        RGB image (numpy array) with white background
    """
    # Render with objects
    renderer.update_scene(d, camera=cam, scene_option=so_fg)
    gc = renderer.scene.ngeom
    if extra_geoms_fn:
        gc = extra_geoms_fn(renderer, gc)
    renderer.scene.ngeom = gc
    fg_frame = renderer.render().copy()

    # Render without objects (just background gradient)
    renderer.update_scene(d, camera=cam, scene_option=so_empty)
    bg_frame = renderer.render().copy()

    # Diff mask: where fg differs from bg, there's an object
    diff = np.abs(fg_frame.astype(np.float64) - bg_frame.astype(np.float64))
    mask = (diff.max(axis=2) > 5)

    # Composite onto white
    result = np.full_like(fg_frame, 255)
    result[mask] = fg_frame[mask]
    return result


def render_1a_input_object(renderer, cam, m, d, object_pos, overlay_frames,
                            total_frames, contact_start, contact_end, output_dir, name):
    """Render 1a: Input object trajectory on white background, no scene."""
    so_obj = _make_white_bg_scene_option(show_hand=False, show_object=True)
    so_empty = _make_white_bg_scene_option(show_hand=False, show_object=False)

    last_overlay = min(overlay_frames[-1], total_frames - 1)
    dot_end = min(last_overlay, contact_end)

    # Start with white canvas + trajectory dots
    d.qpos[:] = 0
    d.mocap_pos[0] = [0, 0, -10]  # hide object for dot rendering
    mujoco.mj_forward(m, d)

    def add_dots(renderer, gc):
        return add_trajectory_dots(renderer, gc, object_pos[:, :3], contact_start, dot_end)

    composite = _render_on_white(renderer, cam, d, so_empty, so_empty, add_dots).astype(np.float64)

    # Overlay object at each frame
    contact_len = max(contact_end - contact_start, 1)
    valid_frames = [f for f in overlay_frames if f < total_frames]
    for idx, frame_idx in enumerate(valid_frames):
        d.qpos[:] = 0
        d.mocap_pos[0] = object_pos[frame_idx, :3]
        d.mocap_quat[0] = object_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        fg = _render_on_white(renderer, cam, d, so_obj, so_empty)
        # Mask: where fg is not white
        mask = ~((fg[:,:,0] > 250) & (fg[:,:,1] > 250) & (fg[:,:,2] > 250))
        mask_f = mask.astype(np.float64)

        t = np.clip((frame_idx - contact_start) / contact_len, 0, 1)
        tint = np.array(time_to_rgb(t))

        is_last = (idx == len(valid_frames) - 1)
        if is_last:
            alpha = 1.0
            blended_fg = fg.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask_f[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    result = np.clip(composite, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_1a_input_object.png")
    cv2.imwrite(path, result_bgr)
    print(f"Saved: {path}")


def render_1b_input_hand(renderer, cam, m, d, qpos, object_pos, overlay_frames,
                          total_frames, contact_start, contact_end, output_dir, name):
    """Render 1b: Input hand trajectory on white background, no scene."""
    so_hand = _make_white_bg_scene_option(show_hand=True, show_object=False)
    so_empty = _make_white_bg_scene_option(show_hand=False, show_object=False)

    last_frame = min(overlay_frames[-1], total_frames - 1)
    dot_end = min(last_frame, contact_end)

    # Start with white canvas + hand wrist trajectory dots
    d.qpos = qpos[:, last_frame]
    d.mocap_pos[0] = [0, 0, -10]  # hide object
    mujoco.mj_forward(m, d)

    hand_positions = qpos[:3, :].T  # (frames, 3)
    def add_dots(renderer, gc):
        return add_trajectory_dots(renderer, gc, hand_positions, contact_start, dot_end)

    composite = _render_on_white(renderer, cam, d, so_empty, so_empty, add_dots).astype(np.float64)

    # Overlay hand at each frame
    contact_len = max(contact_end - contact_start, 1)
    valid_frames = [f for f in overlay_frames if f < total_frames]
    for idx, frame_idx in enumerate(valid_frames):
        d.qpos = qpos[:, frame_idx]
        d.mocap_pos[0] = [0, 0, -10]  # hide object
        mujoco.mj_forward(m, d)

        fg = _render_on_white(renderer, cam, d, so_hand, so_empty)
        mask = ~((fg[:,:,0] > 250) & (fg[:,:,1] > 250) & (fg[:,:,2] > 250))
        mask_f = mask.astype(np.float64)

        t = np.clip((frame_idx - contact_start) / contact_len, 0, 1)
        tint = np.array(time_to_rgb(t))

        is_last = (idx == len(valid_frames) - 1)
        if is_last:
            alpha = 1.0
            blended_fg = fg.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask_f[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    result = np.clip(composite, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_1b_input_hand.png")
    cv2.imwrite(path, result_bgr)
    print(f"Saved: {path}")


def render_1c_input_both(renderer, cam, m, d, qpos, object_pos, overlay_frames,
                          total_frames, contact_start, contact_end, output_dir, name):
    """Render 1c: Input hand + object together on white background, with object trajectory dots."""
    so_both = _make_white_bg_scene_option(show_hand=True, show_object=True)
    so_empty = _make_white_bg_scene_option(show_hand=False, show_object=False)

    last_overlay = min(overlay_frames[-1], total_frames - 1)
    dot_end = min(last_overlay, contact_end)

    # Start with white canvas + object trajectory dots
    d.qpos[:] = 0
    d.mocap_pos[0] = [0, 0, -10]  # hide object for dot rendering
    mujoco.mj_forward(m, d)

    def add_dots(renderer, gc):
        return add_trajectory_dots(renderer, gc, object_pos[:, :3], contact_start, dot_end)

    composite = _render_on_white(renderer, cam, d, so_empty, so_empty, add_dots).astype(np.float64)

    # Overlay hand + object at each frame
    contact_len = max(contact_end - contact_start, 1)
    valid_frames = [f for f in overlay_frames if f < total_frames]
    for idx, frame_idx in enumerate(valid_frames):
        d.qpos = qpos[:, frame_idx]
        d.mocap_pos[0] = object_pos[frame_idx, :3]
        d.mocap_quat[0] = object_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        fg = _render_on_white(renderer, cam, d, so_both, so_empty)
        mask = ~((fg[:,:,0] > 250) & (fg[:,:,1] > 250) & (fg[:,:,2] > 250))
        mask_f = mask.astype(np.float64)

        t = np.clip((frame_idx - contact_start) / contact_len, 0, 1)
        tint = np.array(time_to_rgb(t))

        is_last = (idx == len(valid_frames) - 1)
        if is_last and False:
            alpha = 1.0
            blended_fg = fg.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask_f[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    result = np.clip(composite, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_1c_input_both.png")
    cv2.imwrite(path, result_bgr)
    print(f"Saved: {path}")


def render_2_waypoints(renderer, cam, m, d, object_pos, last_frame,
                        barriers, waypts_processed, output_dir, name):
    """Render 2: Scene with waypoints highlighted."""
    _, so_bg, _ = make_scene_options()

    d.qpos[:] = 0
    d.mocap_pos[0] = object_pos[last_frame, :3]
    d.mocap_quat[0] = object_pos[last_frame, 3:]
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam, scene_option=so_bg)
    gc = renderer.scene.ngeom
    gc = add_barriers_geoms(renderer, gc, barriers, highlight=False)
    gc = add_waypoints_geoms(renderer, gc, waypts_processed, highlight=True)
    renderer.scene.ngeom = gc

    frame = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_2_waypoints.png")
    cv2.imwrite(path, frame_bgr)
    print(f"Saved: {path}")


def render_3_barriers(renderer, cam, m, d, object_pos, last_frame,
                       barriers, waypts_processed, output_dir, name):
    """Render 3: Scene with barriers highlighted."""
    _, so_bg, _ = make_scene_options()

    d.qpos[:] = 0
    d.mocap_pos[0] = object_pos[last_frame, :3]
    d.mocap_quat[0] = object_pos[last_frame, 3:]
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam, scene_option=so_bg)
    gc = renderer.scene.ngeom
    gc = add_barriers_geoms(renderer, gc, barriers, highlight=True)
    gc = add_waypoints_geoms(renderer, gc, waypts_processed, highlight=False)
    renderer.scene.ngeom = gc

    frame = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_3_barriers.png")
    cv2.imwrite(path, frame_bgr)
    print(f"Saved: {path}")


def render_2_3_combined(renderer, cam, m, d, object_pos, last_frame,
                        barriers, waypts_processed, output_dir, name):
    """Render 2_3: Scene with both waypoints and barriers highlighted."""
    _, so_bg, _ = make_scene_options()

    d.qpos[:] = 0
    d.mocap_pos[0] = object_pos[last_frame, :3]
    d.mocap_quat[0] = object_pos[last_frame, 3:]
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam, scene_option=so_bg)
    gc = renderer.scene.ngeom
    gc = add_barriers_geoms(renderer, gc, barriers, highlight=True)
    gc = add_waypoints_geoms(renderer, gc, waypts_processed, highlight=True)
    renderer.scene.ngeom = gc

    frame = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_2_3_waypoints_barriers.png")
    cv2.imwrite(path, frame_bgr)
    print(f"Saved: {path}")


def render_4_waypt_retarget(renderer, cam, m, d, retargeted_pos, contact_start, contact_end,
                             barriers, waypts_processed, overlay_frames, output_dir, name):
    """Render 4: Object retargeted with waypoints only (no barriers), with object posed at overlay frames."""
    _, so_bg, _ = make_scene_options()
    so_obj_white = _make_white_bg_scene_option(show_hand=False, show_object=True)
    so_empty = _make_white_bg_scene_option(show_hand=False, show_object=False)

    total_frames = len(retargeted_pos)

    # Render background: scene + dots + barriers + waypoints (no object)
    d.qpos[:] = 0
    d.mocap_pos[0] = [0, 0, -10]  # hide object
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam, scene_option=so_bg)
    gc = renderer.scene.ngeom
    gc = add_trajectory_dots(renderer, gc, retargeted_pos[:, :3], contact_start, contact_end)
    gc = add_barriers_geoms(renderer, gc, barriers, highlight=False)
    gc = add_waypoints_geoms(renderer, gc, waypts_processed, highlight=True)
    renderer.scene.ngeom = gc
    composite = renderer.render().copy().astype(np.float64)

    # Overlay object at each overlay frame
    contact_len = max(contact_end - contact_start, 1)
    valid_frames = [f for f in overlay_frames if f < total_frames]
    for idx, frame_idx in enumerate(valid_frames):
        d.qpos[:] = 0
        d.mocap_pos[0] = retargeted_pos[frame_idx, :3]
        d.mocap_quat[0] = retargeted_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        fg = _render_on_white(renderer, cam, d, so_obj_white, so_empty)
        mask = ~((fg[:,:,0] > 250) & (fg[:,:,1] > 250) & (fg[:,:,2] > 250))
        mask_f = mask.astype(np.float64)

        t = np.clip((frame_idx - contact_start) / contact_len, 0, 1)
        tint = np.array(time_to_rgb(t))

        is_last = (idx == len(valid_frames) - 1)
        if is_last:
            alpha = 1.0
            blended_fg = fg.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask_f[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    result = np.clip(composite, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_4_waypt_retarget.png")
    cv2.imwrite(path, result_bgr)
    print(f"Saved: {path}")


def render_5_full_retarget(renderer, cam, m, d, retargeted_pos, contact_start, contact_end,
                            barriers, waypts_processed, overlay_frames, output_dir, name):
    """Render 5: Object retargeted with waypoints + barriers, with object posed at overlay frames."""
    _, so_bg, _ = make_scene_options()
    so_obj_white = _make_white_bg_scene_option(show_hand=False, show_object=True)
    so_empty = _make_white_bg_scene_option(show_hand=False, show_object=False)

    total_frames = len(retargeted_pos)

    # Render background: scene + dots + barriers + waypoints (no object)
    d.qpos[:] = 0
    d.mocap_pos[0] = [0, 0, -10]  # hide object
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam, scene_option=so_bg)
    gc = renderer.scene.ngeom
    gc = add_trajectory_dots(renderer, gc, retargeted_pos[:, :3], contact_start, contact_end)
    gc = add_barriers_geoms(renderer, gc, barriers, highlight=True)
    gc = add_waypoints_geoms(renderer, gc, waypts_processed, highlight=True)
    renderer.scene.ngeom = gc
    composite = renderer.render().copy().astype(np.float64)

    # Overlay object at each overlay frame (render on white, mask non-white, composite)
    contact_len = max(contact_end - contact_start, 1)
    valid_frames = [f for f in overlay_frames if f < total_frames]
    for idx, frame_idx in enumerate(valid_frames):
        d.qpos[:] = 0
        d.mocap_pos[0] = retargeted_pos[frame_idx, :3]
        d.mocap_quat[0] = retargeted_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        fg = _render_on_white(renderer, cam, d, so_obj_white, so_empty)
        mask = ~((fg[:,:,0] > 250) & (fg[:,:,1] > 250) & (fg[:,:,2] > 250))
        mask_f = mask.astype(np.float64)

        t = np.clip((frame_idx - contact_start) / contact_len, 0, 1)
        tint = np.array(time_to_rgb(t))

        is_last = (idx == len(valid_frames) - 1)
        if is_last:
            alpha = 1.0
            blended_fg = fg.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask_f[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    result = np.clip(composite, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_5_full_retarget.png")
    cv2.imwrite(path, result_bgr)
    print(f"Saved: {path}")


def render_6_final(renderer, cam, m, d, qpos, object_pos, overlay_frames,
                    total_frames, contact_start, contact_end, barriers, waypts_processed,
                    output_dir, name, use_second_last=False):
    """Render 6: Final retargeted output with hand + object overlay at given frames."""
    so_full, so_bg, so_obj = make_scene_options()

    last_frame = min(overlay_frames[-1], total_frames - 1)

    # Render background: scene + dots + barriers + waypoints (no hand, no object)
    d.qpos[:] = 0
    d.mocap_pos[0] = [0, 0, -10]  # hide object
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam, scene_option=so_bg)
    gc = renderer.scene.ngeom
    gc = add_trajectory_dots(renderer, gc, object_pos[:, :3], contact_start, contact_end)
    gc = add_barriers_geoms(renderer, gc, barriers)
    gc = add_waypoints_geoms(renderer, gc, waypts_processed)
    renderer.scene.ngeom = gc
    bg_frame = renderer.render().copy()
    composite = bg_frame.astype(np.float64)

    contact_len = max(contact_end - contact_start, 1)
    valid_frames = [f for f in overlay_frames if f < total_frames]
    for idx, frame_idx in enumerate(valid_frames):
        is_last = (idx == len(valid_frames) - 1)

        if is_last and use_second_last and len(valid_frames) >= 2:
            # Use articulation from second-to-last overlay frame, position from last
            second_last_frame = valid_frames[-2]
            d.qpos = qpos[:, second_last_frame].copy()
            d.qpos[:3] = qpos[:3, frame_idx]
        else:
            d.qpos = qpos[:, frame_idx]
        d.mocap_pos[0] = object_pos[frame_idx, :3]
        d.mocap_quat[0] = object_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        # Render with hand + object
        renderer.update_scene(d, camera=cam, scene_option=so_full)
        fg_frame = renderer.render().copy()

        # Render background only (no hand, no object) for diff
        renderer.update_scene(d, camera=cam, scene_option=so_bg)
        fg_bg_frame = renderer.render().copy()

        diff = np.abs(fg_frame.astype(np.float64) - fg_bg_frame.astype(np.float64))
        mask = (diff.max(axis=2) > 5).astype(np.float64)

        t = np.clip((frame_idx - contact_start) / contact_len, 0, 1)
        tint = np.array(time_to_rgb(t))
        if is_last:
            alpha = 1.0
            blended_fg = fg_frame.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg_frame.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    result = np.clip(composite, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_6_final.png")
    cv2.imwrite(path, result_bgr)
    print(f"Saved: {path}")


def render_8_final_with_constraints(renderer, cam, m, d, qpos, object_pos, overlay_frames,
                                     total_frames, contact_start, contact_end,
                                     barriers, waypts_processed, output_dir, name,
                                     use_second_last=False):
    """Render 8: Final hand+object on white bg, with barriers and waypoints highlighted, no scene."""
    so_both = _make_white_bg_scene_option(show_hand=True, show_object=True)
    so_empty = _make_white_bg_scene_option(show_hand=False, show_object=False)

    last_overlay = min(overlay_frames[-1], total_frames - 1)
    dot_end = min(last_overlay, contact_end)

    # White canvas + object trajectory dots + barriers + waypoints
    d.qpos[:] = 0
    d.mocap_pos[0] = [0, 0, -10]
    mujoco.mj_forward(m, d)

    def add_extras(renderer, gc):
        gc = add_trajectory_dots(renderer, gc, object_pos[:, :3], contact_start, dot_end)
        gc = add_barriers_geoms(renderer, gc, barriers, highlight=True)
        gc = add_waypoints_geoms(renderer, gc, waypts_processed, highlight=True, color=False)
        return gc

    composite = _render_on_white(renderer, cam, d, so_empty, so_empty, add_extras).astype(np.float64)

    # Overlay hand + object at each overlay frame
    contact_len = max(contact_end - contact_start, 1)
    valid_frames = [f for f in overlay_frames if f < total_frames]
    for idx, frame_idx in enumerate(valid_frames):
        is_last = (idx == len(valid_frames) - 1)

        if is_last and use_second_last and len(valid_frames) >= 2:
            second_last_frame = valid_frames[-2]
            d.qpos = qpos[:, second_last_frame].copy()
            d.qpos[:3] = qpos[:3, frame_idx]
        else:
            d.qpos = qpos[:, frame_idx]
        d.mocap_pos[0] = object_pos[frame_idx, :3]
        d.mocap_quat[0] = object_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        fg = _render_on_white(renderer, cam, d, so_both, so_empty)
        mask = ~((fg[:,:,0] > 250) & (fg[:,:,1] > 250) & (fg[:,:,2] > 250))
        mask_f = mask.astype(np.float64)

        t = np.clip((frame_idx - contact_start) / contact_len, 0, 1)
        tint = np.array(time_to_rgb(t))
        if is_last and False:
            alpha = 1.0
            blended_fg = fg.astype(np.float64)
        else:
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        mask_3ch = mask_f[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    result = np.clip(composite, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    path = os.path.join(output_dir, f"{name}_8_final_constraints.png")
    cv2.imwrite(path, result_bgr)
    print(f"Saved: {path}")


def render_7_contacts(renderer, cam, m, d, qpos, object_pos, contact_frame,
                       total_frames, hand_contacts, object_contacts, object_mesh,
                       hand_components, hand_component_body_ids, output_dir, name):
    """Render 7: Contact visualization - two separate close-up images.
    Image 1: hand in rest pose (open grip) with contact points.
    Image 2: object with contact points.
    Matching contacts share the same color."""

    frame_idx = min(contact_frame, total_frames - 1)

    # Set up the frame to compute contact positions
    d.qpos = qpos[:, frame_idx]
    d.mocap_pos[0] = object_pos[frame_idx, :3]
    d.mocap_quat[0] = object_pos[frame_idx, 3:]
    mujoco.mj_forward(m, d)

    # Compute contact world positions for this frame
    obj_contact_world_positions = []
    hand_contact_world_positions = []

    # Object rotation
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

    # Hand contacts - compute using rest pose (open grip)
    # Set hand to rest pose: keep position + orientation, zero out finger joints
    rest_qpos = qpos[:, frame_idx].copy()
    rest_qpos[7:] = 0  # zero finger joints = open grip
    d.qpos = rest_qpos
    d.mocap_pos[0] = [0, 0, -10]  # hide object
    mujoco.mj_forward(m, d)

    hand_components_len = len(hand_component_body_ids)
    for hand_comp_id in range(hand_components_len):
        contacts_this_frame = hand_contacts[hand_comp_id][frame_idx]
        if contacts_this_frame is not None:
            for contact in contacts_this_frame:
                face_id, bary_coords, object_contact_idx = contact
                local_pos = get_local_pos(face_id, bary_coords,
                                          hand_components[hand_comp_id][0],
                                          hand_components[hand_comp_id][1])
                world_pos = local_to_global(local_pos, hand_component_body_ids[hand_comp_id], d)
                hand_contact_world_positions.append((world_pos, object_contact_idx))

    # Close-up camera for hand (palm facing camera)
    hand_cam = mujoco.MjvCamera()
    hand_center = rest_qpos[:3].copy()
    hand_cam.lookat[:] = [2.002125, -0.385242, 1.195536]
    hand_cam.distance = 0.350000
    hand_cam.azimuth = -79.9
    hand_cam.elevation = 37.5

    # Render hand image: rest pose with contacts
    so_hand = _make_white_bg_scene_option(show_hand=True, show_object=False)

    renderer.update_scene(d, camera=hand_cam, scene_option=so_hand)
    gc = renderer.scene.ngeom
    for world_pos, _ in hand_contact_world_positions:
        mujoco.mjv_initGeom(
            renderer.scene.geoms[gc],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.003, 0, 0],
            pos=np.array(world_pos),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 1]))
        gc += 1
    renderer.scene.ngeom = gc
    hand_frame = renderer.render().copy()

    hand_bgr = cv2.cvtColor(hand_frame, cv2.COLOR_RGB2BGR)
    hand_path = os.path.join(output_dir, f"{name}_7_contacts_hand.png")
    cv2.imwrite(hand_path, hand_bgr)
    print(f"Saved: {hand_path}")

    # Close-up camera for object
    obj_cam = mujoco.MjvCamera()
    obj_cam.lookat[:] = obj_center
    obj_cam.distance = 0.35
    obj_cam.azimuth = 90.0
    obj_cam.elevation = -20.0

    # Render object image: object with contacts
    so_obj = _make_white_bg_scene_option(show_hand=False, show_object=True)
    d.qpos[:] = 0
    d.mocap_pos[0] = object_pos[frame_idx, :3]
    d.mocap_quat[0] = object_pos[frame_idx, 3:]
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=obj_cam, scene_option=so_obj)
    gc = renderer.scene.ngeom
    for pos in obj_contact_world_positions:
        mujoco.mjv_initGeom(
            renderer.scene.geoms[gc],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.003, 0, 0],
            pos=np.array(pos),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 1]))
        gc += 1
    renderer.scene.ngeom = gc
    obj_frame = renderer.render().copy()

    obj_bgr = cv2.cvtColor(obj_frame, cv2.COLOR_RGB2BGR)
    obj_path = os.path.join(output_dir, f"{name}_7_contacts_object.png")
    cv2.imwrite(obj_path, obj_bgr)
    print(f"Saved: {obj_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_visualizations.py <config.json> --overlay 250,400,600")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    AGENT = config['agent']
    TASK = config['task']

    # Parse CLI args
    overlay_frames = None
    contact_frame = None
    output_dir = "visuals/pipeline"
    use_second_last = False
    for idx, arg in enumerate(sys.argv):
        if arg == '--overlay' and idx + 1 < len(sys.argv):
            overlay_frames = [int(f.strip()) for f in sys.argv[idx + 1].split(',')]
        if arg == '--contact-frame' and idx + 1 < len(sys.argv):
            contact_frame = int(sys.argv[idx + 1])
        if arg == '--output' and idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
        if arg == '--use-second-last':
            use_second_last = True

    if overlay_frames is None:
        print("Error: --overlay required (e.g., --overlay 250,400,600)")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Setup scene
    m, d, xml_string, barriers, visuals, waypts, AGENT, TASK = setup_scene(config)
    cam = make_camera()
    renderer = mujoco.Renderer(m, height=1080, width=1920)

    # Load input trajectory
    print("=" * 60)
    print("Loading input trajectory...")
    qpos_input, object_pos_input, frames, contact_start, contact_end, seconds, contacts_lcexp = load_input_trajectory(config)
    m.opt.timestep = 2 * seconds / frames
    print(f"  Frames: {frames}, Contact: [{contact_start}, {contact_end}]")

    # Determine hand type for contacts
    if AGENT == 'MANO_right' or AGENT == 'trajectories':
        hand = 'MANO'
    elif AGENT == 'Allegro_right':
        hand = 'Allegro'

    # Process contacts
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

    # Compute hand component meshes
    hand_components = [None] * hand_components_len
    for j in range(hand_components_len):
        hand_components[j] = get_mesh_for_body(m, hand_component_body_ids[j])

    object_mesh = trimesh.load(config['object_mesh_file'], process=False)

    # ---- Render 1a: Input object trajectory ----
    print("\n" + "=" * 60)
    print("Render 1a: Input object trajectory")
    print("=" * 60)
    render_1a_input_object(renderer, cam, m, d, object_pos_input,
                            overlay_frames, frames, contact_start, contact_end,
                            output_dir, config_name)

    # ---- Render 1b: Input hand trajectory ----
    print("\n" + "=" * 60)
    print("Render 1b: Input hand trajectory")
    print("=" * 60)
    render_1b_input_hand(renderer, cam, m, d, qpos_input, object_pos_input,
                          overlay_frames, frames, contact_start, contact_end,
                          output_dir, config_name)

    # ---- Render 1c: Input hand + object together ----
    print("\n" + "=" * 60)
    print("Render 1c: Input hand + object together")
    print("=" * 60)
    render_1c_input_both(renderer, cam, m, d, qpos_input, object_pos_input,
                          overlay_frames, frames, contact_start, contact_end,
                          output_dir, config_name)

    # ---- Render 4: Object retargeted with waypoints only ----
    print("\n" + "=" * 60)
    print("Render 4: Object retargeted (waypoints only)")
    print("=" * 60)
    retargeted_waypt_only = retarget_object_trajectory(
        config, object_pos_input.T, contact_start, contact_end, frames,
        use_barriers=False)

    # Compute waypoint positions for retargeted trajectory
    newStartPos_retarget = retargeted_waypt_only[0, :3]
    waypts_processed_retarget = []
    for waypt in waypts:
        waypt_pos = newStartPos_retarget + waypt[0]
        if len(waypt) == 4:
            waypts_processed_retarget.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
        else:
            waypts_processed_retarget.append((waypt_pos, waypt[1], waypt[2]))

    render_4_waypt_retarget(renderer, cam, m, d, retargeted_waypt_only,
                             contact_start, contact_end,
                             barriers, waypts_processed_retarget,
                             overlay_frames, output_dir, config_name)

    # ---- Render 5: Object retargeted with waypoints + barriers ----
    print("\n" + "=" * 60)
    print("Render 5: Object retargeted (waypoints + barriers)")
    print("=" * 60)
    retargeted_full = retarget_object_trajectory(
        config, object_pos_input.T, contact_start, contact_end, frames,
        use_barriers=True)

    newStartPos_full = retargeted_full[0, :3]
    waypts_processed_full = []
    for waypt in waypts:
        waypt_pos = newStartPos_full + waypt[0]
        if len(waypt) == 4:
            waypts_processed_full.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
        else:
            waypts_processed_full.append((waypt_pos, waypt[1], waypt[2]))

    render_5_full_retarget(renderer, cam, m, d, retargeted_full,
                            contact_start, contact_end,
                            barriers, waypts_processed_full,
                            overlay_frames, output_dir, config_name)

    # ---- Render 2: Waypoints highlighted (using retargeted positions) ----
    print("\n" + "=" * 60)
    print("Render 2: Scene with waypoints")
    print("=" * 60)
    render_2_waypoints(renderer, cam, m, d, retargeted_waypt_only,
                        min(overlay_frames[-1], retargeted_waypt_only.shape[0] - 1),
                        barriers, waypts_processed_retarget, output_dir, config_name)

    # ---- Render 3: Barriers highlighted (using retargeted positions) ----
    print("\n" + "=" * 60)
    print("Render 3: Scene with barriers")
    print("=" * 60)
    render_3_barriers(renderer, cam, m, d, retargeted_full,
                       min(overlay_frames[-1], retargeted_full.shape[0] - 1),
                       barriers, waypts_processed_full, output_dir, config_name)

    # ---- Render 2_3: Both waypoints and barriers highlighted ----
    print("\n" + "=" * 60)
    print("Render 2_3: Scene with waypoints + barriers")
    print("=" * 60)
    render_2_3_combined(renderer, cam, m, d, retargeted_full,
                         min(overlay_frames[-1], retargeted_full.shape[0] - 1),
                         barriers, waypts_processed_full, output_dir, config_name)

    # ---- Render 6: Final retargeted output ----
    print("\n" + "=" * 60)
    print("Render 6: Final retargeted output")
    print("=" * 60)
    trajectory_dir = "final_trajectories_delete"
    hand_traj_path = os.path.join(trajectory_dir, f"{config_name}_hand.npy")
    object_traj_path = os.path.join(trajectory_dir, f"{config_name}_object.npy")

    if os.path.exists(hand_traj_path) and os.path.exists(object_traj_path):
        qpos_final = np.load(hand_traj_path)
        object_pos_final = np.load(object_traj_path)
        qpos_final = ensure_quaternion_continuity(qpos_final, quat_indices=(3, 7))
        total_frames_final = qpos_final.shape[1]

        newStartPos_final = object_pos_final[0, :3]
        waypts_processed_final = []
        for waypt in waypts:
            waypt_pos = newStartPos_final + waypt[0]
            if len(waypt) == 4:
                waypts_processed_final.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
            else:
                waypts_processed_final.append((waypt_pos, waypt[1], waypt[2]))

        render_6_final(renderer, cam, m, d, qpos_final, object_pos_final,
                        overlay_frames, total_frames_final, contact_start, contact_end,
                        barriers, waypts_processed_final, output_dir, config_name,
                        use_second_last=use_second_last)

        # ---- Render 8: Final with constraints (white bg, no scene) ----
        print("\n" + "=" * 60)
        print("Render 8: Final with constraints (white bg)")
        print("=" * 60)
        render_8_final_with_constraints(renderer, cam, m, d, qpos_final, object_pos_final,
                                         overlay_frames, total_frames_final, contact_start, contact_end,
                                         barriers, waypts_processed_final, output_dir, config_name,
                                         use_second_last=use_second_last)
    else:
        print(f"  Skipped: final trajectories not found at {hand_traj_path}")

    # ---- Render 7: Contact frame exploded view ----
    print("\n" + "=" * 60)
    print("Render 7: Contact exploded view")
    print("=" * 60)

    if os.path.exists(hand_traj_path) and os.path.exists(object_traj_path):
        # Use final trajectory for contact visualization
        if contact_frame is None:
            contact_frame = (contact_start + contact_end) // 2
        print(f"  Contact frame: {contact_frame}")

        render_7_contacts(renderer, cam, m, d, qpos_final, object_pos_final,
                           contact_frame, total_frames_final,
                           hand_contacts, object_contacts, object_mesh,
                           hand_components, hand_component_body_ids,
                           output_dir, config_name)
    else:
        print(f"  Skipped: final trajectories not found")

    renderer.close()
    print(f"\nAll images saved to: {output_dir}/")
