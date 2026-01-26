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

from handContacts import get_mesh_for_body
from load_contacts import load_contacts_lcexp
from parse_splines import parseSplines


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

    # Debug: save XML to see if visual was added
    with open('debug_scene.xml', 'w') as f:
        f.write(xml_string)
    print("XML written to debug_scene.xml for inspection")

    # Load saved trajectories
    trajectory_dir = "final_trajectories"
    hand_traj_path = os.path.join(trajectory_dir, f"{args.trajectory_name}_hand.npy")
    object_traj_path = os.path.join(trajectory_dir, f"{args.trajectory_name}_object.npy")

    if not os.path.exists(hand_traj_path):
        raise FileNotFoundError(f"Hand trajectory not found: {hand_traj_path}")
    if not os.path.exists(object_traj_path):
        raise FileNotFoundError(f"Object trajectory not found: {object_traj_path}")

    qpos = np.load(hand_traj_path)
    retargeted_spline_pos = np.load(object_traj_path)

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
    m.opt.timestep = 4*seconds/frames

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

            '''if hand_contacts is not None and object_contacts is not None:
                # Object contacts (red)
                if isinstance(object_contacts[i], np.ndarray):
                    import trimesh
                    object_mesh = trimesh.load(object_mesh_file, process=False)

                    quat_scipy = np.array([d.mocap_quat[0, 1], d.mocap_quat[0, 2],
                                          d.mocap_quat[0, 3], d.mocap_quat[0, 0]])
                    rotation = R.from_quat(quat_scipy)
                    rotation_matrix = rotation.as_matrix()

                    for j in range(len(object_contacts[i])):
                        local_vertex = object_mesh.vertices[object_contacts[i][j]]
                        world_vertex = (rotation_matrix @ local_vertex + d.mocap_pos)[0]

                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[geometry_count],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.001, 0, 0],
                            pos=np.array(world_vertex),
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1, 0, 0, 1])
                        )
                        geometry_count += 1

                # Hand contacts (blue)
                hand_component_offset = 2
                from handContacts import get_local_pos, local_to_global

                for component_id in range(len(hand_components)):
                    contacts_this_frame = hand_contacts[component_id][i]
                    if contacts_this_frame is not None:
                        for contact in contacts_this_frame:
                            face_id, bary_coords, object_contact_idx = contact
                            local_pos = get_local_pos(
                                face_id, bary_coords,
                                hand_components[component_id][0],
                                hand_components[component_id][1]
                            )
                            global_pos = local_to_global(local_pos, component_id + hand_component_offset, d)

                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[geometry_count],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.003, 0, 0],
                                pos=np.array(global_pos),
                                mat=np.eye(3).flatten(),
                                rgba=np.array([0, 0, 1, 1])
                            )
                            geometry_count += 1'''

            viewer.user_scn.ngeom = geometry_count
            viewer.sync()
            i += 1

            elapsed = time.time() - step_start
            time_until_next_step = m.opt.timestep - elapsed

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("\nPlayback finished.")