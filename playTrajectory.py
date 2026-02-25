import time
import os
import re

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

import torch
import trimesh

import xml.etree.cElementTree as ET

from parse_splines import *
from trajectory import *
from generate_barrier import *
from barrier import *
from generateBinObj import *
from smoothspline import *
from handContacts import *
from differentiable_fk import *
from optimize_contacts import *
from load_contacts import load_contacts_lcexp, get_contact_frame_range

AGENT="Allegro_right"
TASK="apple_pass"

def build_env_xml(agentName, taskName):
  root = ET.Element("mujoco", model="{0} {1}".format(agentName, taskName))
  
  ET.SubElement(root, "include", file="tasks/{0}.xml".format(taskName))
  
  ET.SubElement(root, "include", file="agents/{0}/assets.xml".format(agentName))
  ET.SubElement(root, "include", file="agents/{0}/actuators.xml".format(agentName))
  
  worldBody = ET.SubElement(root, "worldbody")
  ET.SubElement(worldBody, "include", file="agents/{0}/body.xml".format(agentName))
  
  tree = ET.ElementTree(root)
  
  ET.indent(tree, space="\t", level=0)
  
  tree.write("env.xml")


def convert_to_quaternions_MANO(qpos_spline_data):
  # 51 dofs, [3:] are for rotations
  qpos = np.zeros((67, qpos_spline_data.shape[1]))
  qpos[:3, :] = qpos_spline_data[:3, :]

  for j in range(qpos_spline_data.shape[1]):
    for i in range(3, 51, 3):
      rotation = R.from_euler('xyz', qpos_spline_data[i:i+3, j], degrees=False)
      qpos[int((i/3 - 1) * 4 + 4): int(i/3*4 + 3), j] = rotation.as_quat()[:3]
      qpos[int((i/3 - 1) * 4 + 3), j] = rotation.as_quat()[3]

  return qpos

def convert_to_quaternions_Allegro(qpos_spline_data):
  # 22 dofs, [3:6] need to be converted to quaternions
  qpos = np.zeros((23, qpos_spline_data.shape[1]))
  qpos[:3, :] = qpos_spline_data[:3, :]
  qpos[7:, :] = qpos_spline_data[6:, :]

  for j in range(qpos_spline_data.shape[1]):
    rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
    qpos[4:7, j] = rotation.as_quat()[:3]
    qpos[3, j] = rotation.as_quat()[3]

  return qpos

def convert_to_quaternions_object(qpos_spline_data):
  # 6 dofs, [3:6] need to be converted to quaternions
  qpos = np.zeros((7, qpos_spline_data.shape[1]))
  qpos[:3, :] = qpos_spline_data[:3, :]

  for j in range(qpos_spline_data.shape[1]):
    rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
    qpos[4:7, j] = rotation.as_quat()[:3]
    qpos[3, j] = rotation.as_quat()[3]

  return qpos


def rotate_keyframe_angles(keyframes, rotation):
  """
  Add rotation to existing orientations in keyframes
  keyframes: (6, n_frames) array 
  rotation: [rx, ry, rz] to add to existing rotations
  """
  keyframes_modified = keyframes.copy()
  
  additional_rot = R.from_euler('xyz', np.radians(rotation))
  
  for i in range(keyframes.shape[1]):
      # Get existing rotation
      existing_euler = keyframes[3:6, i]
      existing_rot = R.from_euler('xyz', existing_euler)
      
      combined_rot = additional_rot * existing_rot
      
      # Convert back to euler and store
      keyframes_modified[3:6, i] = combined_rot.as_euler('xyz')
  
  return keyframes_modified


def render_frames(m, d, qpos, object_qpos, frames, frames_to_render, output_dir,
                   contact_start=0, contact_end=None):
    """Render specific frames to images using offscreen renderer."""
    if contact_end is None:
        contact_end = frames

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
    cam.azimuth = -179.7
    cam.elevation = -6.0
    cam.distance = 0.703
    cam.lookat[:] = [2.427, -0.396, 1.124]

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in frames_to_render:
        if frame_idx >= frames:
            print(f"Skipping frame {frame_idx} (out of range, max: {frames-1})")
            continue

        d.qpos = qpos[:, frame_idx]
        d.mocap_pos = object_qpos[:3, frame_idx]
        d.mocap_quat = object_qpos[3:, frame_idx]
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, camera=cam, scene_option=scene_option)

        geometry_count = renderer.scene.ngeom

        # Object path (matte green spheres, contact region only, up to current frame)
        draw_end = min(frame_idx + 1, contact_end + 1)
        for j in range(contact_start, draw_end, 1):
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=object_qpos[:3, j],
                mat=np.eye(3).flatten(),
                rgba=np.array([0.2, 0.8, 0.2, 1]))
            geometry_count += 1

        # Hand path (matte blue spheres, contact region only, up to current frame)
        for j in range(contact_start, draw_end, 1):
            mujoco.mjv_initGeom(
                renderer.scene.geoms[geometry_count],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=qpos[:3, j],
                mat=np.eye(3).flatten(),
                rgba=np.array([0.2, 0.2, 0.8, 1]))
            geometry_count += 1

        renderer.scene.ngeom = geometry_count

        frame = renderer.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_dir, f"trajectory_frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, frame_bgr)
        print(f"Saved: {output_path}")

    renderer.close()


if __name__ == "__main__":
  build_env_xml(AGENT, TASK)

  splines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/hand.smexp')
  objectSplines, objectSeconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')


  frames = 1000
  frames = 710
  
  # Load model with offscreen framebuffer for rendering
  with open('kitchen2.xml', 'r') as f:
    xml_string = f.read()
  mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
  if mujoco_tag_match:
    insert_pos = mujoco_tag_match.end()
    visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
    xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]
  m = mujoco.MjModel.from_xml_string(xml_string)
  d = mujoco.MjData(m)
  m.opt.timestep = 2*seconds/frames


  # with open('/Users/hjp/desktop/exports/s1/fryingpan_cook_2_full_export_objectmesh.obj', 'r') as f:
  #   num_vertices = sum(1 for line in f if line.startswith('v '))
  # print("Number of vertices:", num_vertices)

  if AGENT == "Allegro_right":
    object = trimesh.load('/Users/hjp/desktop/trajectoryRetargeting/tasks/apple.obj', process=False)
    
    # Load contacts from .lcexp file
    hand = 'Allegro'
    contacts_lcexp = load_contacts_lcexp('startingTrajectories/' + AGENT + '/' + TASK + '/contacts.lcexp', hand)
  else:
    object = trimesh.load('/Users/hjp/desktop/exports/s1/cubemedium_inspect_1_full_export_objectmesh.obj', process=False)
  
    # Load contacts from .lcexp file
    contacts_lcexp = load_contacts_lcexp('startingTrajectories/' + AGENT + '/' + TASK + '/contacts.lcexp')
  
  contact_start, contact_end = get_contact_frame_range(contacts_lcexp)

  if len(contacts_lcexp) != frames:
    raise Exception("Number of contacts in .lcexp file does not match number of frames")

  sim_time = np.linspace(0, 1, frames)


  qpos_spline_data = np.array([spline(sim_time) for spline in splines]) # (51, frames, 2)
  object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines]) # (6, frames, 2)

  qpos_frames = qpos_spline_data[0, :, 0] # frames are same for all dof
  qpos_spline_data = qpos_spline_data[:, :, 1] # (51, frames)

  object_qpos_spline_data = object_qpos_spline_data[:, :, 1] # (6, frames)

  rotation = [0, 0, 146.441]
  rotation = [0, 0, 0]
  qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
  object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)

  if AGENT == 'MANO_right' or AGENT == 'trajectories':
    qpos = convert_to_quaternions_MANO(qpos_spline_data)
  if AGENT == 'Allegro_right':
    qpos = convert_to_quaternions_Allegro(qpos_spline_data)

  object_qpos = convert_to_quaternions_object(object_qpos_spline_data)




  #### LOAD HAND AND OBJECT CONTACTS FROM .lcexp FILE ####
  hand_contacts = {}  # per hand component, each contains a list of contacts per frame
  hand_components_len = 21
  hand_component_offset = 2  # starting index of body_id for hand components in mujoco model

  if AGENT == 'Allegro_right':
    # Look up body IDs by name since the kitchen model has many bodies before the hand.
    allegro_body_names = [
        'allegro_palm',
        'allegro_th_base', 'allegro_th_proximal', 'allegro_th_medial', 'allegro_th_distal', 'allegro_th_tip',
        'allegro_ff_base', 'allegro_ff_proximal', 'allegro_ff_medial', 'allegro_ff_distal', 'allegro_ff_tip',
        'allegro_mf_base', 'allegro_mf_proximal', 'allegro_mf_medial', 'allegro_mf_distal', 'allegro_mf_tip',
        'allegro_rf_base', 'allegro_rf_proximal', 'allegro_rf_medial', 'allegro_rf_distal', 'allegro_rf_tip',
    ]
    hand_component_body_ids = []
    for name in allegro_body_names:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found in model")
        hand_component_body_ids.append(body_id)

  for i in range(hand_components_len):
    hand_contacts[i] = [None] * frames
  
  object_contacts = [None] * frames

  # Convert contacts from .lcexp format to playTrajectory format
  for frame_idx in range(frames):
    frame_contacts = contacts_lcexp[frame_idx]  # List of (obj_vertex, hand_link_idx, (face_idx, bary1, bary2, bary3))

    if len(frame_contacts) == 0:
      object_contacts[frame_idx] = np.array([], dtype=np.int64)
      continue
    
    # Extract object vertex indices for this frame
    obj_vertex_indices = []
    for contact in frame_contacts:
      obj_vertex_idx = contact[0]
      obj_vertex_indices.append(obj_vertex_idx)
    
    object_contacts[frame_idx] = np.array(obj_vertex_indices, dtype=np.int64)
    
    # Create mapping from object vertex index to index in object_contacts array
    obj_vertex_to_idx = {v: i for i, v in enumerate(obj_vertex_indices)}
    
    # Group contacts by hand component
    contacts_by_component = {}
    for contact in frame_contacts:
      obj_vertex_idx, hand_link_idx, hand_contact_info = contact
      face_idx, bary1, bary2, bary3 = hand_contact_info

      hand_component_id = hand_link_idx

      # print(hand_component_id, hand_components_len)

      if hand_component_id < 0 or hand_component_id >= hand_components_len:
        raise Exception("Hand component id is out of range at frame " + str(frame_idx))
      
      # Get object_contact_idx (index into object_contacts[frame_idx])
      object_contact_idx = obj_vertex_to_idx[obj_vertex_idx]
      
      contact_tuple = (face_idx, (bary1, bary2, bary3), object_contact_idx)
      
      if hand_component_id not in contacts_by_component:
        contacts_by_component[hand_component_id] = []
      contacts_by_component[hand_component_id].append(contact_tuple)
    
    # Assign to hand_contacts
    for hand_component_id in contacts_by_component:
      hand_contacts[hand_component_id][frame_idx] = contacts_by_component[hand_component_id]
  
  contacts = object_contacts.copy()

  # check that all frames have all object contacts covered
  for i in range(frames):
    contact_count = len(object_contacts[i])
    contact_checker = [False] * contact_count

    hand_contact_count = 0
    for j in range(hand_components_len):
      component_contacts = hand_contacts[j][i]
      if component_contacts is None:
        continue
      for k in component_contacts:
        contact_checker[k[2]] = True
        hand_contact_count += 1
    
    if False in contact_checker:
      raise Exception("Missing contact at frame " + str(i))
    if hand_contact_count > contact_count:
      raise Exception("More hand contact count than object contact count at frame " + str(i))

  
  # Render frames if --render flag is passed
  import sys
  if '--render' in sys.argv:
    render_start = 0
    render_end = frames
    render_step = 10
    render_output = 'visuals/trajectory_frames'
    for idx, arg in enumerate(sys.argv):
      if arg == '--render-output' and idx + 1 < len(sys.argv):
        render_output = sys.argv[idx + 1]
      if arg == '--render-start' and idx + 1 < len(sys.argv):
        render_start = int(sys.argv[idx + 1])
      if arg == '--render-end' and idx + 1 < len(sys.argv):
        render_end = int(sys.argv[idx + 1])
      if arg == '--render-step' and idx + 1 < len(sys.argv):
        render_step = int(sys.argv[idx + 1])
    frames_to_render = list(range(render_start, render_end, render_step))
    print(f"Rendering {len(frames_to_render)} frames to {render_output}/")
    render_frames(m, d, qpos, object_qpos, frames, frames_to_render, render_output,
                  contact_start=contact_start, contact_end=contact_end)
    print(f"All images saved to: {render_output}/")

  frame_pts = []
  obj_frame_pts = []
  obj_orig_frame_pts = []
  obj_retarget_frame_pts = []


  with mujoco.viewer.launch_passive(m, d) as viewer:
    # compute hand component meshes
    hand_components = [None] * hand_components_len
    for j in range(hand_components_len):
      if AGENT == 'Allegro_right':
        hand_components[j] = get_mesh_for_body(m, hand_component_body_ids[j])
      else:
        hand_components[j] = get_mesh_for_body(m, j+hand_component_offset)
    
    # qpos_optimized = qpos.copy()
    # qpos = optimize_trajectory(
    #     qpos_optimized, object_qpos, m, d, hand_contacts, object_contacts,
    #     hand_components, hand_component_offset, object,
    #     agent_type=AGENT, optimize_wrist=True, optimize_joints=True,
    #     lr=0.01, n_iter=10
    # )

    i = 0
    while viewer.is_running():
      step_start = time.time()

      # update using only translation
      #d.qpos[:3] = qpos[:3, i % frames]

      d.qpos = qpos[:, i % frames]
      d.mocap_pos = object_qpos[:3, i % frames]
      d.mocap_quat = object_qpos[3:, i % frames]
      mujoco.mj_forward(m, d)

      # process local hand contacts
      # local_hand_contacts: dict of hand component id -> list of local contact positions at frame i
      local_hand_contacts = {}
      for hand_component_id in range(hand_components_len):
        contacts_this_frame = hand_contacts[hand_component_id][i % frames]
        if contacts_this_frame is not None:
          local_hand_contacts[hand_component_id] = []
          for contact in contacts_this_frame:
            face_id, bary_coords, object_contact_idx = contact
            local_pos = get_local_pos(face_id, bary_coords, hand_components[hand_component_id][0], hand_components[hand_component_id][1])
            local_hand_contacts[hand_component_id].append(local_pos)
          local_hand_contacts[hand_component_id] = np.array(local_hand_contacts[hand_component_id])


      # convert local vertex coordinates to world coordinates
      quat_scipy = np.array([d.mocap_quat[0, 1], d.mocap_quat[0, 2], d.mocap_quat[0, 3], d.mocap_quat[0, 0]])
      rotation = R.from_quat(quat_scipy)
      rotation_matrix = rotation.as_matrix()


      if i % frames == 0:
        frame_pts = []
        obj_frame_pts = []
        i = 0


      frame_pts.append(qpos[:3, i % frames])
      obj_frame_pts.append(object_qpos[:3, i % frames])    # uncomment to visualize

      geometry_count = 0

      for j in range(len(obj_orig_frame_pts)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[j + geometry_count],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            pos=np.array(obj_orig_frame_pts[j]),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, j % frames / frames, 0, 1])
        )
      geometry_count += len(obj_orig_frame_pts)
      
      for j in range(len(obj_frame_pts)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[j + geometry_count],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            pos=np.array(obj_frame_pts[j]),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0 if j % frames / frames < 0.72 else 1, 1, 1])
        )
      geometry_count += len(obj_frame_pts)

      # if isinstance(contacts[i], np.ndarray):
      #   for j in range(len(contacts[i])):
      #     local_vertex = object.vertices[contacts[i][j]]
      #     world_vertex = (rotation_matrix @ local_vertex + d.mocap_pos)[0]

      #     mujoco.mjv_initGeom(
      #         viewer.user_scn.geoms[j + geometry_count],
      #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
      #         size=[0.001, 0, 0],
      #         pos=np.array(world_vertex),
      #         mat=np.eye(3).flatten(),
      #         rgba=np.array([1, 0, 0, 1])
      #     )
      #     # print(object.vertices[contacts[i][j]])
      #   geometry_count += len(contacts[i])


      for j in local_hand_contacts: # iterate over hand components
        for k in range(len(local_hand_contacts[j])):  # iterate over contacts in this component
          local_vertex = local_hand_contacts[j][k]
          if AGENT == 'Allegro_right':
            global_vertex = local_to_global(local_vertex, hand_component_body_ids[j], d)
          else:
            global_vertex = local_to_global(local_vertex, j+hand_component_offset, d)

          mujoco.mjv_initGeom(
              viewer.user_scn.geoms[k + geometry_count],
              type=mujoco.mjtGeom.mjGEOM_SPHERE,
              size=[0.001, 0, 0],
              pos=np.array(global_vertex),
              mat=np.eye(3).flatten(),
              rgba=np.array([0, 1, 0, 1])
          )
        geometry_count += len(local_hand_contacts[j])
      
      viewer.user_scn.ngeom = geometry_count


      viewer.sync()
      i += 1

      if i % 30 == 0:
        print(f"Camera: azimuth={viewer.cam.azimuth:.1f}, elevation={viewer.cam.elevation:.1f}, "
              f"distance={viewer.cam.distance:.3f}, lookat=[{viewer.cam.lookat[0]:.3f}, {viewer.cam.lookat[1]:.3f}, {viewer.cam.lookat[2]:.3f}]")

      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)