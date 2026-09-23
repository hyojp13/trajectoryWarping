"""
Render scene-only images:
  1. Scene with barriers (no object or hand)
  2. Scene with waypoints (no object or hand)
"""

import argparse
import json
import mujoco
import numpy as np
import os
import re
import shutil
import cv2

from scipy.spatial.transform import Rotation as R

from trajwarp.object_warp.barriers import process_barriers
from trajwarp.io.scene_xml import add_mesh_barriers_to_xml
from trajwarp.io.spline_io import parseSplines


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


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def build_scene_xml(config, extra_barriers=None):
    """Build the scene XML with barriers and visuals, return xml_string and processed barriers."""
    scene_file = config['scene_file']
    barriers_raw = config.get('barriers', [])
    visuals = config.get('visuals', [])

    # Process barriers
    barriers_raw = [tuple(b) if isinstance(b, list) else b for b in barriers_raw]
    if extra_barriers:
        barriers_raw.extend(extra_barriers)
    barriers = process_barriers(barriers_raw)

    # Process visuals and copy mesh files
    visuals = [tuple(v) if isinstance(v, list) else v for v in visuals]
    for visual in visuals:
        if isinstance(visual, tuple) and len(visual) >= 2:
            visual_mesh_path = visual[0]
            visual_mesh_dest = f"meshes/{os.path.basename(visual_mesh_path)}"
            if not os.path.exists(visual_mesh_dest):
                shutil.copy(visual_mesh_path, visual_mesh_dest)

    # Build XML
    xml_string = add_mesh_barriers_to_xml(scene_file, barriers, visuals)

    # Hide the hood from the scene (it's shown as a barrier instead)
    xml_string = xml_string.replace('name="hood_main_group_g0" type="mesh" contype="0" conaffinity="0" group="1"',
                                    'name="hood_main_group_g0" type="mesh" contype="0" conaffinity="0" group="4"')

    # Hide the top shelf from the scene (it's shown as a barrier instead)
    xml_string = xml_string.replace('name="shelves_main_group_shelf_1_shelf_visual" size="0.5 0.2 0.015" type="box" contype="0" conaffinity="0" group="1"',
                                    'name="shelves_main_group_shelf_1_shelf_visual" size="0.5 0.2 0.015" type="box" contype="0" conaffinity="0" group="4"')

    # Make mesh barriers the same color as primitive barriers (bright orange)
    xml_string = xml_string.replace('rgba="0.2 0.8 0.2 1" group="2"',
                                    'rgba="1.0 0.5 0.0 1" group="2"')

    # Inject offscreen framebuffer
    mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
    if mujoco_tag_match:
        insert_pos = mujoco_tag_match.end()
        visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
        xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]

    return xml_string, barriers


def get_waypoints_absolute(config):
    """Load object trajectory and compute absolute waypoint positions."""
    AGENT = config['agent']
    TASK = config['task']
    new_start_pos_shift = np.array(config.get('new_start_pos_shift', [0, 0, 0]))
    rotation = config.get('rotation', [0, 0, 0])
    waypts_raw = config.get('waypts', [])

    if not waypts_raw:
        return []

    # Parse relative waypoints
    waypts = []
    for w in waypts_raw:
        if len(w) == 4:
            waypts.append((np.array(w[0]), w[1], w[2], w[3]))
        else:
            waypts.append((np.array(w[0]), w[1], w[2]))

    # Load object splines to get start position
    objectSplines, _, _ = parseSplines(f'startingTrajectories/{AGENT}/{TASK}/object.smexp')
    frames = 1000
    sim_time = np.linspace(0, 1, frames)
    object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])
    object_qpos_spline_data = object_qpos_spline_data[:, :, 1]
    object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)
    object_qpos = convert_to_quaternions_object(object_qpos_spline_data)

    start_pos = object_qpos[:3, 0]
    newStartPos = start_pos + new_start_pos_shift

    # Compute absolute waypoint positions
    waypts_abs = []
    for waypt in waypts:
        waypt_pos = newStartPos + waypt[0]
        waypts_abs.append(waypt_pos)

    return waypts_abs


def render_image(xml_string, output_path, extra_geoms_fn=None, cam=None):
    """Render a single image with optional extra geometry."""
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=1080, width=1920)

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1   # scene
    scene_option.geomgroup[2] = 1   # mesh barriers + hand
    scene_option.geomgroup[5] = 0   # object (hide)
    scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        scene_option.sitegroup[i] = 0

    if cam is None:
        cam = mujoco.MjvCamera()
        cam.azimuth = 89.9
        cam.elevation = -3.3
        cam.distance = 2.302
        cam.lookat[:] = [2.433, -0.003, 1.440]

    renderer.update_scene(d, camera=cam, scene_option=scene_option)

    if extra_geoms_fn is not None:
        extra_geoms_fn(renderer)

    frame = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, frame_bgr)
    print(f"Saved: {output_path}")

    renderer.close()


def main():
    parser = argparse.ArgumentParser(description='Render scene with barriers or waypoints')
    parser.add_argument('config', help='Path to retargeting config JSON')
    parser.add_argument('--output-dir', default='visuals/scene', help='Output directory')
    parser.add_argument('--azimuth', type=float, default=None)
    parser.add_argument('--elevation', type=float, default=None)
    parser.add_argument('--distance', type=float, default=None)
    parser.add_argument('--lookat', type=float, nargs=3, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    trajectory_name = os.path.splitext(os.path.basename(args.config))[0]

    # Add ventilator/hood as a mesh barrier
    hood_mesh = "meshes/robocasa/models/assets/fixtures/hoods/pack_2/visuals/model_0.obj"
    hood_barrier = (hood_mesh, {'scale': [1.15613, 1.06643, 1.15613], 'pos': [2.2, -0.3, 2.24807]})

    # Add top shelf as a rect barrier (shelf_1 at [3.2, -0.2, 1.85], size [0.5, 0.2, 0.015])
    shelf_barrier = ('rect', {'dims': [1.0, 0.4, 0.03], 'pos': [3.2, -0.2, 1.85]})

    xml_string, barriers = build_scene_xml(config, extra_barriers=[hood_barrier, shelf_barrier])

    # Build camera if custom args provided
    cam = None
    if any(v is not None for v in [args.azimuth, args.elevation, args.distance, args.lookat]):
        cam = mujoco.MjvCamera()
        cam.azimuth = args.azimuth if args.azimuth is not None else 89.9
        cam.elevation = args.elevation if args.elevation is not None else -3.3
        cam.distance = args.distance if args.distance is not None else 2.302
        cam.lookat[:] = args.lookat if args.lookat is not None else [2.433, -0.003, 1.440]

    # Render scene with barriers and waypoints in one image
    waypts_abs = get_waypoints_absolute(config)

    def add_barriers_and_waypoints(renderer):
        geometry_count = renderer.scene.ngeom

        # Barriers (bright orange)
        for j in range(len(barriers)):
            if isinstance(barriers[j], str):
                continue
            elif barriers[j][0] == 'sphere':
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[geometry_count + j],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[barriers[j][1]['rad'], 0, 0],
                    pos=barriers[j][1]['pos'],
                    mat=np.eye(3).flatten(),
                    rgba=[1.0, 0.5, 0.0, 1])
            elif barriers[j][0] == 'rect':
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[geometry_count + j],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=np.array(barriers[j][1]['dims']) / 2,
                    pos=barriers[j][1]['pos'],
                    mat=np.eye(3).flatten(),
                    rgba=[1.0, 0.5, 0.0, 1])
        geometry_count += len(barriers)

        # Waypoints (bright magenta)
        for j, waypt_pos in enumerate(waypts_abs):
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count + j],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=waypt_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.0, 1.0, 1]))
        geometry_count += len(waypts_abs)

        renderer.scene.ngeom = geometry_count

    output_path = os.path.join(args.output_dir, f"{trajectory_name}_scene.png")
    render_image(xml_string, output_path, extra_geoms_fn=add_barriers_and_waypoints, cam=cam)

    print(f"\nImage saved to: {output_path}")


if __name__ == "__main__":
    main()
