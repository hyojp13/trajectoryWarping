"""
Playback saved trajectories in MuJoCo viewer without any computation.
Loads pre-computed hand and object trajectories and displays them.
"""

import time
import argparse
import json
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.cElementTree as ET
import os
import re
import cv2
import trimesh

from handContacts import get_mesh_for_body
from load_contacts import load_contacts_lcexp
from parse_splines import parseSplines


def ensure_quaternion_continuity(qpos, quat_indices=(3, 7)):
    """
    Ensure quaternion sign continuity across frames to prevent spinning.

    Quaternions q and -q represent the same rotation, but interpolating between
    them causes the hand to spin. This function ensures consecutive quaternions
    are in the same hemisphere by checking the dot product.

    Args:
        qpos: (n_dof, n_frames) array containing position data with quaternions
        quat_indices: tuple (start, end) of quaternion indices in qpos (default: wrist at 3:7)

    Returns:
        qpos with quaternion signs corrected for continuity
    """
    qpos_fixed = qpos.copy()
    start, end = quat_indices

    for i in range(1, qpos.shape[1]):
        prev_quat = qpos_fixed[start:end, i-1]
        curr_quat = qpos_fixed[start:end, i]

        # If dot product is negative, quaternions are in opposite hemispheres
        # Flip current quaternion to same hemisphere as previous
        if np.dot(prev_quat, curr_quat) < 0:
            qpos_fixed[start:end, i] = -curr_quat

    return qpos_fixed


def build_env_xml(agentName, taskName):
    """Build environment XML for MuJoCo."""
    root = ET.Element("mujoco", model="{0} {1}".format(agentName, taskName))

    ET.SubElement(root, "include", file="tasks/{0}.xml".format(taskName))

    ET.SubElement(root, "include", file="agents/{0}/assets.xml".format(agentName))
    ET.SubElement(root, "include", file="agents/{0}/actuators.xml".format(agentName))

    worldBody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldBody, "include", file="agents/{0}/body.xml".format(agentName))

    tree = ET.ElementTree(root)

    ET.indent(tree, space="\t", level=0)

    tree.write("env.xml")


def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playback saved trajectory')
    parser.add_argument('trajectory_name', type=str,
                       help='Name of trajectory (e.g., "fryingpan_into_dishwasher")')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--record', action='store_true',
                       help='Record video instead of interactive playback')
    args = parser.parse_args()

    # Infer config path from trajectory name
    config_path = f"retargeting_configs/{args.trajectory_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    config = load_config(config_path)
    AGENT = config['agent']
    TASK = config['task']
    scene_file = config['scene_file']
    object_mesh_file = config['object_mesh_file']
    barriers = config.get('barriers', [])
    visuals = config.get('visuals', [])

    # Process waypoints
    waypts_raw = config.get('waypts', [])
    new_start_pos_shift = np.array(config.get('new_start_pos_shift', [0, 0, 0]))
    rotation = config.get('rotation', [0, 0, 0])

    # Create object.xml with the correct mesh
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

    # Copy the object mesh to the meshes directory if needed
    import shutil
    os.makedirs('meshes', exist_ok=True)
    mesh_dest = f"meshes/{os.path.basename(object_mesh_file)}"
    if not os.path.exists(mesh_dest):
        shutil.copy(object_mesh_file, mesh_dest)

    build_env_xml(AGENT, TASK)

    # Process barriers
    from barrier import process_barriers, correct_barrier_axes
    from generate_barrier import add_mesh_barriers_to_xml
    barriers = [tuple(b) if isinstance(b, list) else b for b in barriers]
    barriers = process_barriers(barriers)
    # barriers = correct_barrier_axes(barriers)

    # Process visuals and copy mesh files
    visuals = [tuple(v) if isinstance(v, list) else v for v in visuals]
    for visual in visuals:
        if isinstance(visual, tuple) and len(visual) >= 2:
            visual_mesh_path = visual[0]
            visual_mesh_dest = f"meshes/{os.path.basename(visual_mesh_path)}"
            if not os.path.exists(visual_mesh_dest):
                shutil.copy(visual_mesh_path, visual_mesh_dest)
            print(f"Visual: {visual_mesh_path} -> {visual_mesh_dest}")

    # Add mesh barriers and visuals to XML
    print(f"Adding {len(visuals)} visuals to XML")
    xml_string = add_mesh_barriers_to_xml(scene_file, barriers, visuals)

    # Add offscreen framebuffer size for recording
    if args.record:
        mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
        if mujoco_tag_match:
            insert_pos = mujoco_tag_match.end()
            visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
            xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]

    # Debug: save XML to see if visual was added
    with open('debug_scene.xml', 'w') as f:
        f.write(xml_string)
    print("XML written to debug_scene.xml for inspection")

    # Load saved trajectories
    trajectory_dir = "final_trajectories_0.0035"
    hand_traj_path = os.path.join(trajectory_dir, f"{args.trajectory_name}_hand.npy")
    object_traj_path = os.path.join(trajectory_dir, f"{args.trajectory_name}_object.npy")

    if not os.path.exists(hand_traj_path):
        raise FileNotFoundError(f"Hand trajectory not found: {hand_traj_path}")
    if not os.path.exists(object_traj_path):
        raise FileNotFoundError(f"Object trajectory not found: {object_traj_path}")

    qpos = np.load(hand_traj_path)
    retargeted_spline_pos = np.load(object_traj_path)

    # Fix quaternion sign discontinuity to prevent hand spinning
    # Wrist quaternion is at indices 3:7 (w,x,y,z format)
    qpos = ensure_quaternion_continuity(qpos, quat_indices=(3, 7))

    print(f"\nLoaded trajectories:")
    print(f"  Hand: {hand_traj_path} (shape: {qpos.shape})")
    print(f"  Object: {object_traj_path} (shape: {retargeted_spline_pos.shape})")

    frames = qpos.shape[1]

    # Process waypoints: convert relative positions to absolute positions
    # Need to get the ORIGINAL start position before retargeting to match main.py
    # Load original object splines to get the unmodified start position
    objectSplines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')
    sim_time = np.linspace(0, 1, frames)
    object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])
    object_qpos_spline_data = object_qpos_spline_data[:, :, 1]  # (6, frames)

    # Apply rotation to match what was done in main.py
    def rotate_keyframe_angles(keyframes, rotation):
        keyframes_modified = keyframes.copy()
        additional_rot = R.from_euler('xyz', np.radians(rotation))
        for i in range(keyframes.shape[1]):
            existing_euler = keyframes[3:6, i]
            existing_rot = R.from_euler('xyz', existing_euler)
            combined_rot = additional_rot * existing_rot
            keyframes_modified[3:6, i] = combined_rot.as_euler('xyz')
        return keyframes_modified

    object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)

    # Get original start position (before retargeting)
    start_pos = object_qpos_spline_data[:3, 0].copy()
    newStartPos = start_pos + new_start_pos_shift

    # Process waypoints like in main.py
    waypts = []
    for w in waypts_raw:
        if len(w) == 4:
            waypts.append((np.array(w[0]), w[1], w[2], w[3]))
        else:
            waypts.append((np.array(w[0]), w[1], w[2]))

    waypts_processed = []
    for waypt in waypts:
        waypt_pos = newStartPos + waypt[0]  # Add waypoint offset to newStartPos
        if len(waypt) == 4:
            waypts_processed.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
        else:
            waypts_processed.append((waypt_pos, waypt[1], waypt[2]))

    waypts = waypts_processed if waypts_processed else None

    # Load MuJoCo model (xml_string already includes barriers from earlier)
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    # Match main.py timestep calculation exactly
    # To speed up: decrease the coefficient (e.g., 0.5*seconds/frames for 4x faster)
    # To slow down: increase the coefficient (e.g., 4*seconds/frames for 2x slower)
    m.opt.timestep = 2*seconds/frames

    print(f"\nPlayback info:")
    print(f"  Frames: {frames}")
    print(f"  Timestep: {m.opt.timestep:.4f}s")
    print(f"  Speed multiplier: {args.speed}x")
    print(f"  Total duration: {frames * m.opt.timestep:.2f}s")
    print(f"  Waypoints: {len(waypts) if waypts else 0}")

    # Optionally load contacts for visualization
    contacts_file = f'startingTrajectories/{AGENT}/{TASK}/contacts.lcexp'
    hand_components = None
    hand_contacts = None
    object_contacts = None

    if os.path.exists(contacts_file):
        try:
            from contacts import process_contacts
            contacts_lcexp = load_contacts_lcexp(contacts_file)

            # Compute hand component meshes
            hand_components_len = 16
            hand_component_offset = 2
            hand_components = [None] * hand_components_len
            for j in range(hand_components_len):
                hand_components[j] = get_mesh_for_body(m, j + hand_component_offset)

            hand_contacts, object_contacts = process_contacts(contacts_lcexp, hand_components_len)
            print(f"  Contact visualization: enabled")
        except Exception as e:
            print(f"  Contact visualization: disabled (error: {e})")
    else:
        print(f"  Contact visualization: disabled (no contacts file)")

    # Recording or playback
    if args.record:
        print("\nRecording animation...")
        video_path = f"final_trajectories/{args.trajectory_name}.mp4"
        os.makedirs("final_trajectories", exist_ok=True)

        # Load object mesh for contact visualization
        object_mesh = trimesh.load(object_mesh_file, process=False)

        renderer = mujoco.Renderer(m, height=1080, width=1920)

        # Set up visualization options
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[0] = 0  # Disable group 0
        scene_option.geomgroup[1] = 0  # Disable group 0
        scene_option.geomgroup[5] = 1  # Enable group 5

        # Set up camera
        cam = mujoco.MjvCamera()
        cam.azimuth = 108.9
        cam.elevation = -23.4
        cam.distance = 5.430
        cam.lookat[:] = [0.983, 0.561, -0.635]

        # cam.azimuth = 168.7
        # cam.elevation = -40.5
        # cam.distance = 2.898
        # cam.lookat[:] = [-0.694, -1.136, 0.018]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(1.0 / m.opt.timestep)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1920, 1080))

        for i in range(frames):
            d.qpos = qpos[:, i]
            d.mocap_pos[0] = retargeted_spline_pos[i, :3]
            d.mocap_quat[0] = retargeted_spline_pos[i, 3:]
            mujoco.mj_forward(m, d)

            renderer.update_scene(d, camera=cam, scene_option=scene_option)

            # Add custom geometry to the scene
            geometry_count = renderer.scene.ngeom

            # Waypoints
            if waypts is not None:
                for j in range(len(waypts)):
                    mujoco.mjv_initGeom(
                        renderer.scene.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.01, 0, 0],
                        pos=waypts[j][0],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 0, 0, 1]))
                    geometry_count += 1

            # Barriers
            if barriers is not None:
                for barrier in barriers:
                    if isinstance(barrier, str):
                        continue
                    elif barrier[0] == 'sphere':
                        mujoco.mjv_initGeom(
                            renderer.scene.geoms[geometry_count],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[barrier[1]['rad'], 0, 0],
                            pos=barrier[1]['pos'],
                            mat=np.eye(3).flatten(),
                            rgba=[0.5, 0.5, 0.5, 0.3])
                        geometry_count += 1
                    elif barrier[0] == 'rect':
                        mujoco.mjv_initGeom(
                            renderer.scene.geoms[geometry_count],
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=np.array(barrier[1]['dims']) / 2,
                            pos=barrier[1]['pos'],
                            mat=np.eye(3).flatten(),
                            rgba=[0.5, 0.5, 0.5, 0.3])
                        geometry_count += 1

            # Object contacts
            if object_contacts is not None and isinstance(object_contacts[i], np.ndarray):
                # Convert object rotation for contacts
                quat_scipy = np.array([d.mocap_quat[0, 1], d.mocap_quat[0, 2], d.mocap_quat[0, 3], d.mocap_quat[0, 0]])
                rotation = R.from_quat(quat_scipy)
                rotation_matrix = rotation.as_matrix()

                for j in range(len(object_contacts[i])):
                    local_vertex = object_mesh.vertices[object_contacts[i][j]]
                    world_vertex = (rotation_matrix @ local_vertex + d.mocap_pos[0])

                    mujoco.mjv_initGeom(
                        renderer.scene.geoms[geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.001, 0, 0],
                        pos=np.array(world_vertex),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1]))
                    geometry_count += 1

            renderer.scene.ngeom = geometry_count

            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        renderer.close()
        print(f"Video saved to: {video_path}")

    else:
        # Playback loop
        print("\nStarting playback... (close viewer to exit)")

        with mujoco.viewer.launch_passive(m, d) as viewer:
            i = 0
            while viewer.is_running():
                step_start = time.time()

                # Loop trajectory
                if i >= frames:
                    i = 0

                # Set hand and object pose
                d.qpos = qpos[:, i]
                d.mocap_pos[0] = retargeted_spline_pos[i, :3]
                d.mocap_quat[0] = retargeted_spline_pos[i, 3:]
                mujoco.mj_forward(m, d)

                # Visualize contacts, barriers, and waypoints
                geometry_count = 0

                # Draw waypoints
                if waypts is not None:
                    for j in range(len(waypts)):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[geometry_count],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.01, 0, 0],
                            pos=waypts[j][0],
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0, 0, 0, 1]))
                        geometry_count += 1

                # Draw primitive barriers (mesh barriers are already in the MuJoCo model)
                if barriers is not None and len(barriers) > 0:
                    for j, barrier in enumerate(barriers):
                        if isinstance(barrier, str):
                            # Mesh barriers are already in the MuJoCo scene, skip
                            continue
                        elif barrier[0] == 'sphere':
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[geometry_count],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[barrier[1]['rad'], 0, 0],
                                pos=barrier[1]['pos'],
                                mat=np.eye(3).flatten(),
                                rgba=[0.5, 0.5, 0.5, 0.3])
                            geometry_count += 1
                        elif barrier[0] == 'rect':
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[geometry_count],
                                type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=np.array(barrier[1]['dims']) / 2,
                                pos=barrier[1]['pos'],
                                mat=np.eye(3).flatten(),
                                rgba=[0.5, 0.5, 0.5, 0.3])
                            geometry_count += 1

                viewer.user_scn.ngeom = geometry_count
                viewer.sync()
                i += 1

                elapsed = time.time() - step_start
                time_until_next_step = m.opt.timestep - elapsed

                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        print("\nPlayback finished.")