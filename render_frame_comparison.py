"""
Render comparison images of contact-based vs naive hand retargeting at specific frames.
Renders hand, object, and contacts (no waypoints).
"""

import argparse
import json
import mujoco
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
    """Ensure quaternion sign continuity across frames to prevent spinning."""
    qpos_fixed = qpos.copy()
    start, end = quat_indices

    for i in range(1, qpos.shape[1]):
        prev_quat = qpos_fixed[start:end, i-1]
        curr_quat = qpos_fixed[start:end, i]
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


def render_frames(config_path, trajectory_name, method, frames_to_render, output_dir):
    """
    Render specific frames for a given method.

    Args:
        config_path: Path to config JSON
        trajectory_name: Name of the trajectory
        method: 'contact' or 'naive'
        frames_to_render: List of frame indices to render
        output_dir: Directory to save images
    """
    config = load_config(config_path)
    AGENT = config['agent']
    TASK = config['task']
    scene_file = config['scene_file']
    object_mesh_file = config['object_mesh_file']
    barriers = config.get('barriers', [])
    visuals = config.get('visuals', [])
    rotation = config.get('rotation', [0, 0, 0])
    new_start_pos_shift = np.array(config.get('new_start_pos_shift', [0, 0, 0]))

    # Process waypoints from config (relative offsets)
    waypts_raw = config.get('waypts', [])
    waypts = []
    for w in waypts_raw:
        if len(w) == 4:
            waypts.append((np.array(w[0]), w[1], w[2], w[3]))
        else:
            waypts.append((np.array(w[0]), w[1], w[2]))

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
    from barrier import process_barriers
    from generate_barrier import add_mesh_barriers_to_xml
    barriers = [tuple(b) if isinstance(b, list) else b for b in barriers]
    barriers = process_barriers(barriers)

    # Process visuals and copy mesh files
    visuals = [tuple(v) if isinstance(v, list) else v for v in visuals]
    for visual in visuals:
        if isinstance(visual, tuple) and len(visual) >= 2:
            visual_mesh_path = visual[0]
            visual_mesh_dest = f"meshes/{os.path.basename(visual_mesh_path)}"
            if not os.path.exists(visual_mesh_dest):
                shutil.copy(visual_mesh_path, visual_mesh_dest)

    # Add mesh barriers and visuals to XML
    xml_string = add_mesh_barriers_to_xml(scene_file, barriers, visuals)

    # Add offscreen framebuffer size for recording
    mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
    if mujoco_tag_match:
        insert_pos = mujoco_tag_match.end()
        visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
        xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]

    # Load saved trajectories based on method
    if method == 'contact':
        trajectory_dir = "final_trajectories_delete"
        hand_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_hand.npy")
        object_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_object.npy")
    else:  # naive
        trajectory_dir = "final_trajectories_delete"
        hand_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_hand_naive.npy")
        object_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_object_naive.npy")

    if not os.path.exists(hand_traj_path):
        raise FileNotFoundError(f"Hand trajectory not found: {hand_traj_path}")
    if not os.path.exists(object_traj_path):
        raise FileNotFoundError(f"Object trajectory not found: {object_traj_path}")

    qpos = np.load(hand_traj_path)
    retargeted_spline_pos = np.load(object_traj_path)

    # Fix quaternion sign discontinuity
    qpos = ensure_quaternion_continuity(qpos, quat_indices=(3, 7))

    # Compute absolute waypoint positions
    newStartPos = retargeted_spline_pos[0, :3]
    waypts_processed = []
    for waypt in waypts:
        waypt_pos = newStartPos + waypt[0]
        if len(waypt) == 4:
            waypts_processed.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
        else:
            waypts_processed.append((waypt_pos, waypt[1], waypt[2]))

    print(f"\nLoaded trajectories ({method}):")
    print(f"  Hand: {hand_traj_path} (shape: {qpos.shape})")
    print(f"  Object: {object_traj_path} (shape: {retargeted_spline_pos.shape})")

    total_frames = qpos.shape[1]

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    # Load contacts for visualization
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

    # Load object mesh for contact visualization
    object_mesh = trimesh.load(object_mesh_file, process=False)

    renderer = mujoco.Renderer(m, height=1080, width=1920)

    # Set up visualization options
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 0   # scene
    scene_option.geomgroup[5] = 1   # object
    scene_option.geomgroup[2] = 0   # hand
    # Disable frame visualizations
    scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE
    # Disable all site groups (removes blue discs)
    for i in range(6):
        scene_option.sitegroup[i] = 0

    # Set up camera
    cam = mujoco.MjvCamera()
    cam.azimuth = 104.3
    cam.elevation = -26.3
    cam.distance = 4.064
    cam.lookat[:] = [0.015, -0.020, 0.230]

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in frames_to_render:
        if frame_idx >= total_frames:
            print(f"Skipping frame {frame_idx} (out of range, max: {total_frames-1})")
            continue

        d.qpos = qpos[:, frame_idx]
        d.mocap_pos[0] = retargeted_spline_pos[frame_idx, :3]
        d.mocap_quat[0] = retargeted_spline_pos[frame_idx, 3:]
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, camera=cam, scene_option=scene_option)

        # Add custom geometry to the scene
        geometry_count = renderer.scene.ngeom

        # Object path (green spheres up to current frame)
        for j in range(frame_idx + 1):
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count + j],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=retargeted_spline_pos[j, :3],
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 1, 0, 1]))
        geometry_count += frame_idx + 1

        # barrier
        if barriers is not None:
            for j in range(len(barriers)):
                if isinstance(barriers[j], str):
                    continue
                elif barriers[j][0] == 'sphere':
                    mujoco.mjv_initGeom(
                        renderer.scene.geoms[j + geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[barriers[j][1]['rad'], 0, 0],
                        pos=barriers[j][1]['pos'],
                        mat=np.eye(3).flatten(),
                        rgba=[0.5, 0.5, 0.5, 1])
                elif barriers[j][0] == 'rect':
                    mujoco.mjv_initGeom(
                    renderer.scene.geoms[j + geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=np.array(barriers[j][1]['dims']) / 2,
                    pos=barriers[j][1]['pos'],
                    mat=np.eye(3).flatten(),
                    rgba=[0.5, 0.5, 0.5, 1])
            geometry_count += len(barriers)

        # # Waypoints (black spheres)
        # if waypts_processed is not None:
        #     for j in range(len(waypts_processed)):
        #         mujoco.mjv_initGeom(
        #             renderer.scene.geoms[geometry_count + j],
        #             type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #             size=[0.01, 0, 0],
        #             pos=waypts_processed[j][0],
        #             mat=np.eye(3).flatten(),
        #             rgba=np.array([0, 0, 0, 1]))
        #     geometry_count += len(waypts_processed)

        # # Object contacts (red spheres)
        # if object_contacts is not None and frame_idx < len(object_contacts) and isinstance(object_contacts[frame_idx], np.ndarray):
        #     quat_scipy = np.array([d.mocap_quat[0, 1], d.mocap_quat[0, 2], d.mocap_quat[0, 3], d.mocap_quat[0, 0]])
        #     rotation_mat = R.from_quat(quat_scipy).as_matrix()

        #     for j in range(len(object_contacts[frame_idx])):
        #         local_vertex = object_mesh.vertices[object_contacts[frame_idx][j]]
        #         world_vertex = (rotation_mat @ local_vertex + d.mocap_pos[0])

        #         mujoco.mjv_initGeom(
        #             renderer.scene.geoms[geometry_count],
        #             type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #             size=[0.002, 0, 0],
        #             pos=np.array(world_vertex),
        #             mat=np.eye(3).flatten(),
        #             rgba=np.array([1, 0, 0, 1]))
        #         geometry_count += 1

        # # Hand contacts (blue spheres) - computed from hand_contacts
        # if hand_contacts is not None and frame_idx < len(hand_contacts) and hand_contacts[frame_idx] is not None:
        #     for contact in hand_contacts[frame_idx]:
        #         hand_link_idx, face_idx, bary = contact
        #         if hand_components is not None and hand_link_idx < len(hand_components) and hand_components[hand_link_idx] is not None:
        #             mesh_verts, mesh_faces = hand_components[hand_link_idx]
        #             if face_idx < len(mesh_faces):
        #                 face = mesh_faces[face_idx]
        #                 # Compute contact point in local coords using barycentric
        #                 local_point = (bary[0] * mesh_verts[face[0]] +
        #                               bary[1] * mesh_verts[face[1]] +
        #                               bary[2] * mesh_verts[face[2]])

        #                 # Get body transform to convert to world coords
        #                 body_id = hand_link_idx + 2  # hand_component_offset = 2
        #                 body_pos = d.xpos[body_id]
        #                 body_quat = d.xquat[body_id]
        #                 body_rot = R.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
        #                 world_point = body_rot.apply(local_point) + body_pos

        #                 mujoco.mjv_initGeom(
        #                     renderer.scene.geoms[geometry_count],
        #                     type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #                     size=[0.002, 0, 0],
        #                     pos=np.array(world_point),
        #                     mat=np.eye(3).flatten(),
        #                     rgba=np.array([0, 0, 1, 1]))
        #                 geometry_count += 1

        renderer.scene.ngeom = geometry_count

        frame = renderer.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_dir, f"{trajectory_name}_{method}_frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, frame_bgr)
        print(f"Saved: {output_path}")

    renderer.close()


if __name__ == "__main__":
    trajectory_name = "mug_pass_narrow"
    config_path = f"retargeting_configs/{trajectory_name}.json"

    # Frames between 602 to 787 inclusive, every ~50 frames
    frames_to_render = [i for i in range(50, 800, 10)]

    output_dir = "visuals/frame_comparison"

    print("=" * 60)
    print("Rendering CONTACT-OPTIMIZED method frames")
    print("=" * 60)
    render_frames(config_path, trajectory_name, 'contact', frames_to_render, output_dir)

    print("\n" + "=" * 60)
    print("Rendering NAIVE method frames")
    print("=" * 60)
    render_frames(config_path, trajectory_name, 'naive', frames_to_render, output_dir)

    print(f"\nAll images saved to: {output_dir}/")
