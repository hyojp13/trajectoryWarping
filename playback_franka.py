"""
Playback Franka hand trajectories from roboverse pickle file.
Loads object states and hand states and displays them in MuJoCo.

Expected data format:
- Pickle file contains tuple: (robot_states, object_states)
- robot_states: (n_frames, 16) = [pos(3), quat(4), 9 joint angles]
- object_states: (n_frames, 7) = [pos(3), quat/rot(4)]
"""

import time
import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import torch
import cv2
import os


def load_franka_trajectory(pkl_path):
    """
    Load Franka trajectory from roboverse torch file.

    Returns:
        hand_qpos: (n_dofs, n_frames) - Franka hand joint positions
        object_qpos: (7, n_frames) - Object position and quaternion (w,x,y,z)
        n_frames: Number of frames
    """
    data = torch.load(pkl_path, map_location='cpu', weights_only=False)

    # Handle both tuple and dict formats
    if isinstance(data, dict):
        robot_states = data['robot_states']
        object_states = data['object_poses']
    else:
        robot_states, object_states = data

    # Convert from torch tensors if needed
    if torch.is_tensor(robot_states):
        robot_states = robot_states.cpu().numpy()
    if torch.is_tensor(object_states):
        object_states = object_states.cpu().numpy()

    n_frames = robot_states.shape[0]

    print(f"Loaded trajectory with {n_frames} frames")
    print(f"  Robot states shape: {robot_states.shape}")
    print(f"  Object states shape: {object_states.shape}")

    # Parse robot states: [pos(3), quat(4), 9 joints]
    # Total: 16 values per frame
    robot_pos = robot_states[:, :3]  # (n_frames, 3)

    robot_shift = [3, -4, 0.95]
    robot_pos += robot_shift

    robot_quat = robot_states[:, 3:7]  # (n_frames, 4) - assumed format
    robot_joints = robot_states[:, 7:16]  # (n_frames, 9)

    print(f"  Robot position range: {robot_pos.min(axis=0)} to {robot_pos.max(axis=0)}")
    print(f"  Robot joints shape: {robot_joints.shape}")

    # Parse object states: [pos(3), rot(4)]
    object_pos = object_states[:, :3]  # (n_frames, 3)

    # Apply same shift to object as robot
    object_pos += robot_shift
    # object_shift = [0, -0.02, -0.02]
    # object_pos += object_shift

    object_rot = object_states[:, 3:7]  # (n_frames, 4)

    print(f"  Object position range: {object_pos.min(axis=0)} to {object_pos.max(axis=0)}")

    # Build hand_qpos: [pos(3), quat(4), joints(9)] = 16 DOFs
    hand_qpos = np.zeros((16, n_frames))
    hand_qpos[:3, :] = robot_pos.T

    # Normalize quaternions
    for frame_idx in range(n_frames):
        quat = robot_quat[frame_idx]
        quat_norm = quat / np.linalg.norm(quat)
        hand_qpos[3:7, frame_idx] = quat_norm

    hand_qpos[7:16, :] = robot_joints.T

    # Build object_qpos: [pos(3), quat(4)] = 7 DOFs
    object_qpos = np.zeros((7, n_frames))
    object_qpos[:3, :] = object_pos.T

    # Normalize quaternions
    for frame_idx in range(n_frames):
        quat = object_rot[frame_idx]
        quat_norm = quat / np.linalg.norm(quat)
        object_qpos[3:7, frame_idx] = quat_norm

    return hand_qpos, object_qpos, n_frames


def create_franka_xml_with_menagerie(object_mesh_path=None, use_kitchen=True):
    """
    Create a MuJoCo XML using official Franka Panda model from MuJoCo Menagerie.
    Modifies the model to have a free-floating base and optionally includes kitchen scene.
    """
    import os

    if use_kitchen:
        # Use unified kitchen + Franka XML
        import xml.etree.ElementTree as ET

        tree = ET.parse('scenes/franka_in_kitchen.xml')
        root = tree.getroot()

        # Add object mesh if provided
        if object_mesh_path:
            asset = root.find('asset')
            if asset is None:
                asset = ET.SubElement(root, 'asset')

            mesh_name = os.path.basename(object_mesh_path).replace('.obj', '')
            # Use relative path from meshdir or absolute path
            mesh_file = object_mesh_path if object_mesh_path.startswith('/') or object_mesh_path.startswith('../') else os.path.basename(object_mesh_path)
            ET.SubElement(asset, 'mesh', name=mesh_name, file=mesh_file)

            # Add object to worldbody (find the worldbody after includes are processed)
            worldbody = root.find('worldbody')
            object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', mocap='true')
            ET.SubElement(object_body, 'geom', type='mesh', mesh=mesh_name, rgba='0.8 0.6 0.4 1', mass='0.1')
        else:
            # Add simple sphere object
            worldbody = root.find('worldbody')
            object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', mocap='true')
            ET.SubElement(object_body, 'geom', type='sphere', size='0.03', rgba='0.8 0.6 0.4 1', mass='0.1')

        xml_string = ET.tostring(root, encoding='unicode')
        return xml_string

    else:
        # Use standalone Franka model
        import xml.etree.ElementTree as ET

        tree = ET.parse('franka_model/panda.xml')
        root = tree.getroot()

        # Update meshdir to point to correct location
        compiler = root.find('compiler')
        if compiler is not None:
            compiler.set('meshdir', 'franka_model/assets')

        # Remove keyframes (they won't be valid with freejoint)
        keyframe = root.find('keyframe')
        if keyframe is not None:
            root.remove(keyframe)

        # Find worldbody and link0
        worldbody = root.find('worldbody')
        link0 = worldbody.find(".//body[@name='link0']")

        # Insert freejoint as first child of link0
        freejoint = ET.Element('freejoint', name='panda_root')
        link0.insert(0, freejoint)

        # Add object mesh if provided
        asset = root.find('asset')

        if object_mesh_path:
            if asset is None:
                asset = ET.SubElement(root, 'asset')

            mesh_name = os.path.basename(object_mesh_path).replace('.obj', '')
            # Use relative path from meshdir or absolute path
            mesh_file = object_mesh_path if object_mesh_path.startswith('/') or object_mesh_path.startswith('../') else os.path.basename(object_mesh_path)
            ET.SubElement(asset, 'mesh', name=mesh_name, file=mesh_file)

            # Add object to worldbody
            object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', mocap='true')
            ET.SubElement(object_body, 'geom', type='mesh', mesh=mesh_name, rgba='0.8 0.6 0.4 1', mass='0.1')
        else:
            # Add simple sphere object
            object_body = ET.SubElement(worldbody, 'body', name='manipulated_object', mocap='true')
            ET.SubElement(object_body, 'geom', type='sphere', size='0.03', rgba='0.8 0.6 0.4 1', mass='0.1')

        # Convert back to string
        xml_string = ET.tostring(root, encoding='unicode')

        return xml_string


def render_frames(m, d, hand_qpos, object_qpos, n_frames, frames_to_render, output_dir):
    """Render specific frames to images using offscreen renderer."""
    # Add offscreen framebuffer size
    renderer = mujoco.Renderer(m, height=1080, width=1920)

    # Set up visualization options
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1   # scene
    scene_option.geomgroup[2] = 1   # hand
    scene_option.geomgroup[5] = 1   # object
    scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE
    for i in range(6):
        scene_option.sitegroup[i] = 0

    # Set up camera
    cam = mujoco.MjvCamera()
    cam.azimuth = 163.4
    cam.elevation = -16.9
    cam.distance = 6.853
    cam.lookat[:] = [-1.451, -2.681, -0.052]

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in frames_to_render:
        if frame_idx >= n_frames:
            print(f"Skipping frame {frame_idx} (out of range, max: {n_frames-1})")
            continue

        d.qpos[:] = hand_qpos[:, frame_idx]
        if m.nmocap > 0:
            d.mocap_pos[0] = object_qpos[:3, frame_idx]
            d.mocap_quat[0] = object_qpos[3:7, frame_idx]
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, camera=cam, scene_option=scene_option)

        geometry_count = renderer.scene.ngeom

        # # Object path (green spheres up to current frame)
        # for j in range(frame_idx + 1):
        #     mujoco.mjv_initGeom(
        #         renderer.scene.geoms[geometry_count + j],
        #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #         size=[0.005, 0, 0],
        #         pos=object_qpos[:3, j],
        #         mat=np.eye(3).flatten(),
        #         rgba=np.array([0, 1, 0, 1]))
        # geometry_count += frame_idx + 1

        # End effector position (larger green sphere)
        ee_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        if ee_body_id >= 0:
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.008, 0, 0],
                pos=d.xpos[ee_body_id],
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 1, 0, 0.8]))
            geometry_count += 1

        renderer.scene.ngeom = geometry_count

        frame = renderer.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_dir, f"franka_frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, frame_bgr)
        print(f"Saved: {output_path}")

    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playback Franka trajectory')
    parser.add_argument('pkl_path', type=str, nargs='?', default=None,
                       help='Path to pickle file with trajectory data')
    parser.add_argument('--hand-npy', type=str, default=None,
                       help='Path to hand trajectory .npy file (shape: n_dofs x n_frames)')
    parser.add_argument('--object-npy', type=str, default=None,
                       help='Path to object trajectory .npy file (shape: 7 x n_frames)')
    parser.add_argument('--object-mesh', type=str, default='meshes/box.obj',
                       help='Path to object mesh file (.obj) (default: meshes/box.obj)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--no-kitchen', action='store_true',
                       help='Disable kitchen scene (show only robot)')
    parser.add_argument('--record', action='store_true',
                       help='Record video instead of interactive playback')
    parser.add_argument('--render', action='store_true',
                       help='Render frames to images instead of interactive playback')
    parser.add_argument('--render-output', type=str, default='visuals/franka_frames',
                       help='Output directory for rendered frames (default: visuals/franka_frames)')
    parser.add_argument('--render-start', type=int, default=0,
                       help='First frame to render (default: 0)')
    parser.add_argument('--render-end', type=int, default=None,
                       help='Last frame to render (default: all frames)')
    parser.add_argument('--render-step', type=int, default=10,
                       help='Frame step for rendering (default: 10)')
    args = parser.parse_args()

    # Load trajectory from either npy files or pickle
    if args.hand_npy and args.object_npy:
        print(f"Loading trajectories from .npy files:")
        print(f"  Hand:   {args.hand_npy}")
        print(f"  Object: {args.object_npy}")
        hand_qpos = np.load(args.hand_npy)
        object_qpos = np.load(args.object_npy)
        n_frames = hand_qpos.shape[1]
        print(f"  Frames: {n_frames}")
        print(f"  Hand shape: {hand_qpos.shape}")
        print(f"  Object shape: {object_qpos.shape}")
    elif args.pkl_path:
        print(f"Loading trajectory from: {args.pkl_path}")
        hand_qpos, object_qpos, n_frames = load_franka_trajectory(args.pkl_path)
    else:
        parser.error("Provide either a pkl_path or both --hand-npy and --object-npy")

    # Create object.xml with the correct mesh (similar to main.py)
    import os
    object_xml_content = f"""<mujoco>
    <asset>
      <mesh name="object_mesh" file="{os.path.basename(args.object_mesh)}" scale="1 1 1"/>
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
    mesh_dest = f"meshes/{os.path.basename(args.object_mesh)}"
    if not os.path.exists(mesh_dest) and not args.object_mesh.startswith('meshes/'):
        import shutil
        os.makedirs('meshes', exist_ok=True)
        shutil.copy(args.object_mesh, mesh_dest)

    # Create MuJoCo model
    use_kitchen = not args.no_kitchen
    print(f"Using object mesh: {args.object_mesh}")
    # Don't pass object_mesh_path since object.xml now has it
    xml_string = create_franka_xml_with_menagerie(None, use_kitchen=use_kitchen)

    if use_kitchen:
        print("Using kitchen scene")

    # Add offscreen framebuffer size for rendering
    import re
    mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
    if mujoco_tag_match:
        insert_pos = mujoco_tag_match.end()
        visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
        xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]

    # Load model
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)

    # Set timestep for visualization
    dt = 0.03  # 100 Hz
    m.opt.timestep = dt / args.speed

    print(f"\nPlayback info:")
    print(f"  Frames: {n_frames}")
    print(f"  Hand DOFs: {hand_qpos.shape[0]}")
    print(f"  Model DOFs: {m.nq}")
    print(f"  Model actuators: {m.nu}")
    print(f"  Timestep: {m.opt.timestep:.4f}s")
    print(f"  Speed: {args.speed}x")
    print(f"  Total duration: {n_frames * m.opt.timestep:.2f}s")

    print("\nJoint names:")
    for i in range(m.njnt):
        print(f"  {i}: {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)}")

    if args.render:
        render_end = args.render_end if args.render_end is not None else n_frames
        frames_to_render = list(range(args.render_start, render_end, args.render_step))
        print(f"\nRendering {len(frames_to_render)} frames to {args.render_output}/")
        render_frames(m, d, hand_qpos, object_qpos, n_frames, frames_to_render, args.render_output)
        print(f"\nAll images saved to: {args.render_output}/")

    if args.record:
        print("\nRecording animation...")
        video_dir = "final_trajectories"
        os.makedirs(video_dir, exist_ok=True)
        if args.hand_npy:
            traj_name = os.path.splitext(os.path.basename(args.hand_npy))[0].replace('_hand', '')
        else:
            traj_name = os.path.splitext(os.path.basename(args.pkl_path))[0]
        video_path = os.path.join(video_dir, f"{traj_name}.mp4")

        renderer = mujoco.Renderer(m, height=1080, width=1920)

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[0] = 0
        scene_option.geomgroup[1] = 1
        scene_option.geomgroup[2] = 1
        scene_option.geomgroup[5] = 1
        scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE
        for idx in range(6):
            scene_option.sitegroup[idx] = 0

        cam = mujoco.MjvCamera()
        cam.azimuth = 163.4
        cam.elevation = -16.9
        cam.distance = 6.853
        cam.lookat[:] = [-1.451, -2.681, -0.052]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(1.0 / m.opt.timestep)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1920, 1080))

        print(f"  Recording {n_frames} frames")

        for i in range(n_frames):
            d.qpos[:] = hand_qpos[:, i]
            if m.nmocap > 0:
                d.mocap_pos[0] = object_qpos[:3, i]
                d.mocap_quat[0] = object_qpos[3:7, i]
            mujoco.mj_forward(m, d)

            renderer.update_scene(d, camera=cam, scene_option=scene_option)
            geometry_count = renderer.scene.ngeom

            # Object path (green spheres up to current frame)
            # for j in range(i + 1):
            #     mujoco.mjv_initGeom(
            #         renderer.scene.geoms[geometry_count],
            #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
            #         size=[0.005, 0, 0],
            #         pos=object_qpos[:3, j],
            #         mat=np.eye(3).flatten(),
            #         rgba=np.array([0, 1, 0, 1]))
            #     geometry_count += 1

            renderer.scene.ngeom = geometry_count

            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        renderer.close()
        print(f"Video saved to: {video_path}")
        import sys
        sys.exit(0)

    print("\nStarting playback... (close viewer to exit)")

    # Playback loop
    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()

            # Loop trajectory
            if i >= n_frames:
                i = 0

            # Set hand pose
            # hand_qpos format: [pos(3), quat(4), 7 arm joints, 2 finger joints] = 16 DOFs
            # MuJoCo qpos format: [pos(3), quat(4) for freejoint, 7 arm joints, 2 finger joints] = 16 DOFs

            # Simply copy all qpos since the layouts match
            d.qpos[:] = hand_qpos[:, i]

            # Set object pose (mocap)
            if m.nmocap > 0:
                d.mocap_pos[0] = object_qpos[:3, i]
                d.mocap_quat[0] = object_qpos[3:7, i]

            mujoco.mj_forward(m, d)

            # Visualize trajectory
            geometry_count = 0

            # Draw object trajectory (blue spheres)
            max_geoms = viewer.user_scn.maxgeom
            trajectory_skip = max(1, n_frames // 100)  # Show ~100 points
            for j in range(0, min(i + 1, n_frames), trajectory_skip):
                if geometry_count >= max_geoms:
                    break

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],
                    pos=object_qpos[:3, j],
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 1, 0.5])
                )
                geometry_count += 1

            # Draw end effector trajectory (green spheres)
            # Get end effector position from MuJoCo
            ee_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
            if ee_body_id >= 0 and geometry_count < max_geoms:
                ee_pos = d.xpos[ee_body_id]

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.008, 0, 0],
                    pos=ee_pos,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 1, 0, 0.8])
                )
                geometry_count += 1

            viewer.user_scn.ngeom = geometry_count
            viewer.sync()
            i += 1

            if i % 30 == 0:
                print(f"Camera: azimuth={viewer.cam.azimuth:.1f}, elevation={viewer.cam.elevation:.1f}, "
                      f"distance={viewer.cam.distance:.3f}, lookat=[{viewer.cam.lookat[0]:.3f}, {viewer.cam.lookat[1]:.3f}, {viewer.cam.lookat[2]:.3f}]")

            # Timing
            elapsed = time.time() - step_start
            time_until_next_step = m.opt.timestep - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("\nPlayback finished.")
