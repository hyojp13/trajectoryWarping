"""
Render comparison images of contact-based vs naive hand retargeting at specific frames.
Renders hand, object, and contacts (no waypoints).
"""

import argparse
import json
import mujoco
import numpy as np
import scipy.interpolate
import xml.etree.cElementTree as ET
import os
import re
import cv2
import trimesh

def time_to_rgb(t):
    """Convert t in [0,1] to RGB using HSV hue: green(0) -> red(1)."""
    # Hue: 0.333 (green) -> 0.0 (red)
    hue = (1.0 - t) * 120.0 / 360.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return r, g, b


from handContacts import get_mesh_for_body
from load_contacts import load_contacts_lcexp


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
    scene_option.geomgroup[1] = 1   # scene
    scene_option.geomgroup[5] = 1   # object
    scene_option.geomgroup[2] = 1   # hand
    # Disable frame visualizations
    scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE
    # Disable all site groups (removes blue discs)
    for i in range(6):
        scene_option.sitegroup[i] = 0

    # Set up camera
    cam = mujoco.MjvCamera()
    cam.azimuth = 179.7
    cam.elevation = -26.3
    cam.distance = 3.568
    cam.lookat[:] = [-1.257, -1.840, 0.516]

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
                    rgba=[0.5, 0.5, 0.5, 0.4])
            geometry_count += len(barriers)

        # Waypoints (black spheres)
        if waypts_processed is not None:
            for j in range(len(waypts_processed)):
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[geometry_count + j],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.01, 0, 0],
                    pos=waypts_processed[j][0],
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1, 0, 0, 1]))
            geometry_count += len(waypts_processed)

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


def render_overlay(config_path, trajectory_name, method, overlay_frames, output_dir, use_timewarp=False):
    """
    Render a single composite image showing multiple hand/object poses overlaid.

    The last frame in overlay_frames is rendered fully opaque. Earlier frames are
    overlaid as semi-transparent ghosts. The object trajectory path uses time-varying
    colors (blue -> green -> red), and overlaid hand/objects are tinted by time.

    Args:
        config_path: Path to config JSON
        trajectory_name: Name of the trajectory
        method: 'contact' or 'naive'
        overlay_frames: Sorted list of frame indices (e.g., [150, 250, 490])
        output_dir: Directory to save the composite image
    """
    config = load_config(config_path)
    AGENT = config['agent']
    TASK = config['task']
    scene_file = config['scene_file']
    object_mesh_file = config['object_mesh_file']
    barriers = config.get('barriers', [])
    visuals = config.get('visuals', [])
    # Process waypoints from config
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

    import shutil
    os.makedirs('meshes', exist_ok=True)
    mesh_dest = f"meshes/{os.path.basename(object_mesh_file)}"
    if not os.path.exists(mesh_dest):
        shutil.copy(object_mesh_file, mesh_dest)

    build_env_xml(AGENT, TASK)

    from barrier import process_barriers
    from generate_barrier import add_mesh_barriers_to_xml
    barriers = [tuple(b) if isinstance(b, list) else b for b in barriers]
    barriers = process_barriers(barriers)

    visuals = [tuple(v) if isinstance(v, list) else v for v in visuals]
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

    # Load trajectories
    if method == 'contact':
        trajectory_dir = "final_trajectories_delete"
        hand_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_hand.npy")
        object_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_object.npy")
    else:
        trajectory_dir = "final_trajectories_delete"
        hand_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_hand_naive.npy")
        object_traj_path = os.path.join(trajectory_dir, f"{trajectory_name}_object_naive.npy")

    qpos = np.load(hand_traj_path)
    retargeted_spline_pos = np.load(object_traj_path)
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

    total_frames = qpos.shape[1]
    overlay_frames = sorted([f for f in overlay_frames if f < total_frames])
    if len(overlay_frames) == 0:
        print("No valid overlay frames.")
        return

    print(f"\nOverlay render ({method}): frames {overlay_frames}")

    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    renderer = mujoco.Renderer(m, height=1080, width=1920)

    # Scene options with hand + object visible
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1   # scene
    scene_option.geomgroup[2] = 1   # hand
    scene_option.geomgroup[5] = 1   # object
    scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        scene_option.sitegroup[i] = 0

    # Scene options without hand + object (background only)
    scene_option_bg = mujoco.MjvOption()
    scene_option_bg.geomgroup[0] = 0
    scene_option_bg.geomgroup[1] = 1   # scene
    scene_option_bg.geomgroup[2] = 0   # hand OFF
    scene_option_bg.geomgroup[5] = 0   # object OFF
    scene_option_bg.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        scene_option_bg.sitegroup[i] = 0

    cam = mujoco.MjvCamera()
    cam.azimuth = 88.3
    cam.elevation = -10.9
    cam.distance = 3.250
    cam.lookat[:] = [2.473, 0.007, 0.871]

    os.makedirs(output_dir, exist_ok=True)

    last_input_frame = overlay_frames[-1]

    # Load timewarp data if available (for mapping input frames to output positions)
    timewarp_path = os.path.join(
        "final_trajectories_0.01" if method == 'contact' else "final_trajectories_0.01",
        f"{trajectory_name}_timewarp.npy"
    )
    timewarp_data = None
    if os.path.exists(timewarp_path):
        timewarp_data = np.load(timewarp_path, allow_pickle=True).item()
        contact_timewarp = timewarp_data['contact_timewarp']
        start_frame_count = timewarp_data['start_frame_count']
        contact_frame_count = timewarp_data['contact_frame_count']
        pos_spline = timewarp_data.get('pos_spline', None)
        print(f"  Timewarp loaded: {len(contact_timewarp)} input frames -> {contact_frame_count} contact frames (start offset: {start_frame_count})")
        if pos_spline is not None:
            print(f"  Position spline loaded for exact evaluation")
    else:
        print(f"  Timewarp not found at {timewarp_path}, skipping timewarp visualization")

    # Map last frame through timewarp if enabled, otherwise use directly as output frame
    if use_timewarp and timewarp_data is not None and last_input_frame < len(contact_timewarp):
        t_out = contact_timewarp[last_input_frame]
        last_out_frame = int(np.clip(np.round(start_frame_count + t_out * (contact_frame_count - 1)), 0, total_frames - 1))
    else:
        last_out_frame = min(last_input_frame, total_frames - 1)

    # Render background (no hand/object) at the last frame's pose
    d.qpos = qpos[:, last_out_frame]
    d.mocap_pos[0] = retargeted_spline_pos[last_out_frame, :3]
    d.mocap_quat[0] = retargeted_spline_pos[last_out_frame, 3:]
    mujoco.mj_forward(m, d)

    # Add trajectory path and barriers/waypoints to background
    renderer.update_scene(d, camera=cam, scene_option=scene_option_bg)
    geometry_count = renderer.scene.ngeom

    if use_timewarp and timewarp_data is not None:
        # Timewarp trajectory: show ALL input frames mapped onto the retargeted trajectory.
        # Where points cluster = time compression, spread = time stretching.
        # Only sample contact frames (skip pre-contact and post-contact which pile at start/end)
        contact_start = start_frame_count
        contact_end = min(start_frame_count + contact_frame_count, len(contact_timewarp))
        tw_contact_vals = contact_timewarp[contact_start:contact_end]
        print(f"  Timewarp debug: sampling contact input frames [{contact_start}, {contact_end}) = {contact_end - contact_start} frames")
        print(f"    t_out range: [{tw_contact_vals.min():.4f}, {tw_contact_vals.max():.4f}]")
        print(f"    out_frame range: [{start_frame_count + tw_contact_vals.min() * (contact_frame_count - 1):.1f}, {start_frame_count + tw_contact_vals.max() * (contact_frame_count - 1):.1f}]")
        # Show per-segment info based on waypoint timesteps
        waypts_timesteps = [w[2] for w in waypts]
        prev_t = 0
        for wi, wt in enumerate(waypts_timesteps):
            in_segment = np.sum((tw_contact_vals >= prev_t) & (tw_contact_vals < wt))
            out_range = f"[{start_frame_count + prev_t * (contact_frame_count - 1):.0f}, {start_frame_count + wt * (contact_frame_count - 1):.0f}]"
            print(f"    Segment {wi}: {in_segment} contact input frames -> output time [{prev_t:.3f}, {wt:.3f}] -> output frames {out_range}")
            prev_t = wt
        in_segment = np.sum(tw_contact_vals >= prev_t)
        print(f"    Segment {len(waypts_timesteps)}: {in_segment} contact input frames -> output time [{prev_t:.3f}, 1.000] -> output frames [{start_frame_count + prev_t * (contact_frame_count - 1):.0f}, {start_frame_count + contact_frame_count - 1}]")

        n_contact = contact_end - contact_start

        for i in range(n_contact):
            frame_idx = contact_start + i
            t_out = contact_timewarp[frame_idx]

            # Evaluate spline directly at exact t_out for unique position per input frame
            if pos_spline is not None:
                mapped_pos = np.array(scipy.interpolate.splev(t_out, pos_spline))
            else:
                # Fallback: interpolate from pre-sampled trajectory
                out_frame_float = start_frame_count + t_out * (contact_frame_count - 1)
                out_frame_float = np.clip(out_frame_float, 0, total_frames - 1)
                f_lo = int(np.floor(out_frame_float))
                f_hi = min(f_lo + 1, total_frames - 1)
                frac = out_frame_float - f_lo
                mapped_pos = retargeted_spline_pos[f_lo, :3] * (1 - frac) + retargeted_spline_pos[f_hi, :3] * frac

            t = i / max(n_contact - 1, 1)
            r, g, b = time_to_rgb(t)
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=mapped_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([r, g, b, 0.85]))
            geometry_count += 1
    else:
        # Linear output trajectory: uniformly spaced points along retargeted path
        for j in range(last_out_frame + 1):
            t = j / max(last_out_frame, 1)
            r, g, b = time_to_rgb(t)
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=retargeted_spline_pos[j, :3],
                mat=np.eye(3).flatten(),
                rgba=np.array([r, g, b, 1]))
            geometry_count += 1

    # Barriers
    if barriers is not None:
        for j in range(len(barriers)):
            if isinstance(barriers[j], str):
                continue
            elif barriers[j][0] == 'sphere':
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[barriers[j][1]['rad'], 0, 0],
                    pos=barriers[j][1]['pos'],
                    mat=np.eye(3).flatten(),
                    rgba=[0.5, 0.5, 0.5, 1])
                geometry_count += 1
            elif barriers[j][0] == 'rect':
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=np.array(barriers[j][1]['dims']) / 2,
                    pos=barriers[j][1]['pos'],
                    mat=np.eye(3).flatten(),
                    rgba=[0.5, 0.5, 0.5, 1])
                geometry_count += 1

    # Waypoints
    if waypts_processed:
        for j in range(len(waypts_processed)):
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=waypts_processed[j][0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 1]))
            geometry_count += 1

    renderer.scene.ngeom = geometry_count
    bg_frame = renderer.render().copy()

    # Composite image starts as the background
    composite = bg_frame.astype(np.float64)

    # Map overlay frames through timewarp if enabled, otherwise treat as output frame indices
    if use_timewarp and timewarp_data is not None:
        output_frames = []
        for input_frame in overlay_frames:
            if input_frame < len(contact_timewarp):
                t_out = contact_timewarp[input_frame]
                out_f = start_frame_count + t_out * (contact_frame_count - 1)
                output_frames.append(int(np.clip(np.round(out_f), 0, total_frames - 1)))
            else:
                output_frames.append(min(input_frame, total_frames - 1))
        print(f"  Overlay input frames {overlay_frames} -> output frames {output_frames}")
    else:
        output_frames = [min(f, total_frames - 1) for f in overlay_frames]

    # Render each overlay frame and alpha-blend the hand/object onto composite
    for idx, (input_frame, out_frame) in enumerate(zip(overlay_frames, output_frames)):
        d.qpos = qpos[:, out_frame]
        d.mocap_pos[0] = retargeted_spline_pos[out_frame, :3]
        d.mocap_quat[0] = retargeted_spline_pos[out_frame, 3:]
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, camera=cam, scene_option=scene_option)
        renderer.scene.ngeom = renderer.scene.ngeom  # keep default geoms only
        fg_frame = renderer.render().copy()

        # Also render this frame without hand/object to get per-frame background
        renderer.update_scene(d, camera=cam, scene_option=scene_option_bg)
        renderer.scene.ngeom = renderer.scene.ngeom
        fg_bg_frame = renderer.render().copy()

        # Mask: pixels where hand/object are visible (differ from this frame's background)
        diff = np.abs(fg_frame.astype(np.float64) - fg_bg_frame.astype(np.float64))
        mask = (diff.max(axis=2) > 5).astype(np.float64)  # threshold to handle anti-aliasing

        # Time-based tint matching trajectory dot colors
        if use_timewarp and timewarp_data is not None:
            # Match the dot coloring: position within contact frames
            contact_start = start_frame_count
            contact_end = min(start_frame_count + contact_frame_count, len(contact_timewarp))
            t = (input_frame - contact_start) / max(contact_end - contact_start - 1, 1)
            t = np.clip(t, 0, 1)
        else:
            t = input_frame / max(last_input_frame, 1)
        tint = np.array(time_to_rgb(t))

        is_last = (idx == len(overlay_frames) - 1)

        if is_last:
            # Last frame: fully opaque, original colors (no tint)
            alpha = 1.0
            blended_fg = fg_frame.astype(np.float64)
        else:
            # Earlier frames: semi-transparent, strongly colorized
            alpha = 0.7
            tint_strength = 0.7
            fg_f = fg_frame.astype(np.float64)
            luminance = 0.299 * fg_f[:,:,0] + 0.587 * fg_f[:,:,1] + 0.114 * fg_f[:,:,2]
            colorized = np.stack([luminance * tint[0], luminance * tint[1], luminance * tint[2]], axis=2)
            blended_fg = fg_f * (1.0 - tint_strength) + colorized * tint_strength

        # Apply masked alpha blend onto composite
        mask_3ch = mask[:, :, np.newaxis]
        composite = composite * (1.0 - mask_3ch * alpha) + blended_fg * mask_3ch * alpha

    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # Save separate timewarp visualizer image
    if use_timewarp and timewarp_data is not None:
        img_w = composite.shape[1]
        img_h = composite.shape[0]
        bar = np.full((img_h, img_w, 3), 40, dtype=np.uint8)

        contact_start_tw = start_frame_count
        contact_end_tw = min(start_frame_count + contact_frame_count, len(contact_timewarp))
        n_contact_tw = contact_end_tw - contact_start_tw

        # Each input frame draws a 1px vertical line spanning the full image height
        for i in range(n_contact_tw):
            frame_idx = contact_start_tw + i
            t_in = i / max(n_contact_tw - 1, 1)
            t_out = contact_timewarp[frame_idx]
            r, g, b = time_to_rgb(t_in)
            x = int(np.clip(t_out * (img_w - 1), 0, img_w - 1))
            bar[:, x] = [int(r*255), int(g*255), int(b*255)]

        bar_bgr = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
        frames_str = "_".join(str(f) for f in overlay_frames)
        bar_path = os.path.join(output_dir, f"{trajectory_name}_{method}_timewarp_{frames_str}.png")
        cv2.imwrite(bar_path, bar_bgr)
        print(f"Saved timewarp bar: {bar_path}")

    composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    frames_str = "_".join(str(f) for f in overlay_frames)
    output_path = os.path.join(output_dir, f"{trajectory_name}_{method}_overlay_{frames_str}.png")
    cv2.imwrite(output_path, composite_bgr)
    print(f"Saved overlay: {output_path}")

    renderer.close()


if __name__ == "__main__":
    trajectory_name = "mug_pass_wall_2d"
    config_path = f"retargeting_configs/{trajectory_name}.json"

    # Frames between 602 to 787 inclusive, every ~50 frames
    frames_to_render = [i for i in range(50, 790, 10)]

    output_dir = "visuals/frame_comparison"

    import sys
    # Check for --overlay flag: e.g. --overlay 150,250,490
    # Check for --timewarp flag to use timewarp-mapped trajectory instead of linear output
    overlay_frames = None
    use_timewarp = False
    for idx, arg in enumerate(sys.argv):
        if arg == '--overlay' and idx + 1 < len(sys.argv):
            overlay_frames = [int(f.strip()) for f in sys.argv[idx + 1].split(',')]
        if arg == '--timewarp':
            use_timewarp = True

    if overlay_frames is not None:
        print("=" * 60)
        mode_str = "TIMEWARP" if use_timewarp else "LINEAR"
        print(f"Rendering OVERLAY image at frames: {overlay_frames} (trajectory: {mode_str})")
        print("=" * 60)
        render_overlay(config_path, trajectory_name, 'contact', overlay_frames, output_dir, use_timewarp=use_timewarp)
    else:
        print("=" * 60)
        print("Rendering CONTACT-OPTIMIZED method frames")
        print("=" * 60)
        render_frames(config_path, trajectory_name, 'contact', frames_to_render, output_dir)

        print("\n" + "=" * 60)
        print("Rendering NAIVE method frames")
        print("=" * 60)
        render_frames(config_path, trajectory_name, 'naive', frames_to_render, output_dir)

    print(f"\nAll images saved to: {output_dir}/")
