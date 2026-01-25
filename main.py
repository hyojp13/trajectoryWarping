import time
import argparse
import json
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
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
from load_contacts import *
from differentiable_fk import *
from contacts import *

def build_env_xml(agentName, taskName):
  root = ET.Element("mujoco", model="{0} {1}".format(agentName, taskName))
  
  ET.SubElement(root, "include", file="tasks/{0}.xml".format(taskName))
  
  ET.SubElement(root, "include", file="agents/{0}/assets.xml".format(agentName))
  ET.SubElement(root, "include", file="agents/{0}/actuators.xml".format(agentName))
  
  worldBody = ET.SubElement(root, "worldbody")
  ET.SubElement(worldBody, "include", file="agents/{0}/body.xml".format(agentName))
  
  #ET.SubElement(worldBody, "include", file="agents/{0}/mocap_bodies.xml".format(agentName))

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


def load_config(config_path):
  with open(config_path, 'r') as f:
    config = json.load(f)

  required_fields = [
    'agent', 'task', 'scene_file', 'barriers',
    'learning_rate', 'n_iter', 'first_frame_iter',
    'boundary_radius', 'hand_boundary_radius',
    'new_start_pos_shift', 'end_final_pos_shift', 'end_obj_pos_shift',
    'waypts', 'extra_pt_count', 'optimization_device',
    'object_mesh_file', 'rotation'
  ]
  for field in required_fields:
    if field not in config:
      raise ValueError(f"Configuration file must contain '{field}' field")

  # Optional barrier optimization parameters with defaults
  config.setdefault('barrier_weight', 0.6)  # Relative weight vs contact loss
  config.setdefault('barrier_margin', 0.01)  # cm safety margin
  config.setdefault('barrier_n', 2.0)  # Quadratic penalty

  return config


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run trajectory retargeting with configuration file')
  parser.add_argument('config', type=str, help='Path to configuration JSON file')
  args = parser.parse_args()

  # Load configuration
  config = load_config(args.config)
  AGENT = config['agent']
  TASK = config['task']
  scene_file = config['scene_file']
  barriers = config['barriers']
  learning_rate = config['learning_rate']
  n_iter = config['n_iter']
  first_frame_iter = config['first_frame_iter']
  boundary_radius = config['boundary_radius']
  hand_boundary_radius = config['hand_boundary_radius']
  new_start_pos_shift = np.array(config['new_start_pos_shift'])
  end_final_pos_shift = np.array(config['end_final_pos_shift'])
  end_obj_pos_shift = np.array(config['end_obj_pos_shift'])
  barrier_weight = config['barrier_weight']
  barrier_margin = config['barrier_margin']
  barrier_n = config['barrier_n']
  # Process waypoints, preserving rotation if present
  waypts = []
  for w in config['waypts']:
    if len(w) == 4:
      waypts.append((np.array(w[0]), w[1], w[2], w[3]))
    else:
      waypts.append((np.array(w[0]), w[1], w[2]))
  extra_pt_count = config['extra_pt_count']
  optimization_device = config['optimization_device']
  object_mesh_file = config['object_mesh_file']
  rotation = config['rotation']

  # Create object.xml with the correct mesh
  import os
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
  mesh_dest = f"meshes/{os.path.basename(object_mesh_file)}"
  if not os.path.exists(mesh_dest):
    import shutil
    os.makedirs('meshes', exist_ok=True)
    shutil.copy(object_mesh_file, mesh_dest)

  build_env_xml(AGENT, TASK)

  # retrieve splines
  splines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/hand.smexp')
  objectSplines, objectSeconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')

  objectSplinesOrig, _, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')

  # Convert barriers from config format to tuples
  barriers = [tuple(b) if isinstance(b, list) else b for b in barriers]
  barriers = process_barriers(barriers) # apply scale and pos to obj barriers

  # add mesh barriers to MuJoCo
  xml_string = add_mesh_barriers_to_xml(scene_file, barriers)

  object = trimesh.load(object_mesh_file, process=False)

  # Load contacts from .lcexp file
  contacts_lcexp = load_contacts_lcexp('startingTrajectories/' + AGENT + '/' + TASK + '/contacts.lcexp')
  startIdx, endIdx = get_contact_frame_range(contacts_lcexp)

  frames = len(contacts_lcexp)
  m = mujoco.MjModel.from_xml_string(xml_string)
  d = mujoco.MjData(m)
  m.opt.timestep = 2*seconds/frames

  # load hand and object contacts from contacts_lcexp
  hand_components_len = 16
  hand_component_offset = 2  # starting index of body_id for hand components in mujoco model
  hand_contacts, object_contacts = process_contacts(contacts_lcexp, hand_components_len)

  # correct barrier axes (issue due to using repulsive curves previously)
  barrier = correct_barrier_axes(barriers)

  sim_time = np.linspace(0, 1, frames)

  # read splines to numpy
  qpos_spline_data = np.array([spline(sim_time) for spline in splines]) # (51, frames, 2)
  object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines]) # (6, frames, 2)

  qpos_frames = qpos_spline_data[0, :, 0] # frames are same for all dof
  qpos_spline_data = qpos_spline_data[:, :, 1] # (51, frames)

  object_qpos_spline_data = object_qpos_spline_data[:, :, 1] # (6, frames)

  qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
  object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)

  if AGENT == 'MANO_right' or AGENT == 'trajectories':
    qpos = convert_to_quaternions_MANO(qpos_spline_data)
  if AGENT == 'Allegro_right':
    qpos = convert_to_quaternions_Allegro(qpos_spline_data)
  qpos_copy = qpos.copy()

  object_qpos = convert_to_quaternions_object(object_qpos_spline_data)  # with waypoint
  # object_qpos_copy = object_qpos.copy()



  start_pos = object_qpos[:3, 0].copy()
  end_pos = object_qpos[:3, frames-1].copy()

  newStartPos = start_pos + new_start_pos_shift
  endFinalPos = end_pos + end_final_pos_shift  # Actual end destination
  endObjPos = endFinalPos + end_obj_pos_shift  # Motion should end here, points from here to endFinalPos is linear interpolation

  # generate desired end object
  # basket_mesh = create_basket(radius=0.1, height=0.16, wall_thickness=0.01)
  # basket_mesh.export('scene/basket.obj')
  # shift_obj('scene/basket.obj', endObjPos)

  # determine entrance to object
  # endWayPt = getEntrance('scene/basket.obj')

  # endObj = 'scene/basket.obj'
  # waypts = [start_pos + [-0.2, 0, 0.5],
            # start_pos + [0, -0.5, 0.2]]

  # Process waypoints: convert relative positions to absolute positions
  waypts_processed = []
  for waypt in waypts:
    waypt_pos = newStartPos + waypt[0]  # Add waypoint offset to newStartPos
    if len(waypt) == 4:
      # Include rotation if provided
      waypts_processed.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
    else:
      waypts_processed.append((waypt_pos, waypt[1], waypt[2]))
  waypts = waypts_processed


  # pos_waypt_constrained: (3, frames)
  pos_waypt_constrained, _, _, waypts_idx = trajectoryConstraintsPolyline(object_qpos[:3, :], startPos = newStartPos, endPos = endObjPos, floor_height = start_pos[2], waypts = waypts)

  # Apply rotations at waypoints if specified
  # Rotations are applied gradually from the previous waypoint (or contact start) to current waypoint
  # object_rotation_qpos: (4, frames)
  object_rotation_qpos = apply_waypoint_rotations(object_qpos[3:7, :], waypts, waypts_idx, startIdx, frames)


  start_frame_count = startIdx
  contact_frame_count = endIdx-startIdx+1
  end_frame_count = frames - start_frame_count - contact_frame_count

  print("Frames before contact:", start_frame_count)
  print("Frames of contact:", contact_frame_count)
  print("Frames after contact:", end_frame_count)

  # ensure that waypoints are not too close to a barrier
  # if not barrierWayptsCheck(barriers, waypts, boundary_radius):
  #   raise Exception("waypoints are closer to the barrier than the object radius")

  barrierConstraints(pos_waypt_constrained.T, boundary_radius, barriers)
  

  # ensure all points are above the surface
  obj = read_obj("scene/curve_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/curve_positions.obj")


  trajectory = read_obj("scene/curve_positions.obj")  # (frames, 3)

  # ensure trajectory ends in end position
  if not np.isclose(trajectory[-1, :], endFinalPos, atol=1e-3).all():
    print("trajectory does not end in end position, adding extra points")
    print(trajectory[-1, :], endFinalPos)
    moveToEndPt("scene/curve_positions.obj", endFinalPos, extra_pt_count)
    trajectory = read_obj("scene/curve_positions.obj")
  else:
    extra_pt_count = 0
  

  # new object polyline after barrier and waypoints constraints (entire polyline considered "contact" trajectory)
  # new_object_qpos: # (7, n+extra)
  new_object_qpos = np.zeros((object_qpos.shape[0], object_qpos.shape[1] + extra_pt_count))
  new_object_qpos[:3, :] = trajectory.T
  new_object_qpos[3:, :object_qpos.shape[1]] = object_rotation_qpos
  if extra_pt_count != 0:
    new_object_qpos[3:, -extra_pt_count:] = new_object_qpos[3:, -extra_pt_count-1].reshape(object_qpos.shape[0]-3, 1)


  # convert to bspline
  final_waypt_timesteps = np.array([w[2] for w in waypts])  # (waypts, )

  # Create separate splines for position and quaternion to avoid coupling
  pos_spline, contact_timewarp = create_smoothing_bspline(new_object_qpos[:3, :].T, parameterization="waypts", waypts_info=(waypts_idx, final_waypt_timesteps))

  # contact_timewarp: (frames + extra_pt_count,): [0, 1] -> [0, 1]
  # all points in timewarp are DURING contact only, ie. 0 -> first contact

  # Quaternion spline
  # Convert from MuJoCo format [w,x,y,z] to scipy format [x,y,z,w] for processing
  quat_mujoco = new_object_qpos[3:7, :].T  # (n, 4) in [w,x,y,z] format
  quat_scipy = np.column_stack([quat_mujoco[:, 1:4], quat_mujoco[:, 0]])  # Convert to [x,y,z,w]
  quat_spline, _ = create_quaternion_bspline(quat_scipy, waypts_info=(waypts_idx, final_waypt_timesteps))


  retargeted_sim_time = np.linspace(0, 1, contact_frame_count)

  retargeted_pos = np.array(scipy.interpolate.splev(retargeted_sim_time, pos_spline)).T
  retargeted_quat_scipy = np.array(scipy.interpolate.splev(retargeted_sim_time, quat_spline)).T  # (n, 4) in [x,y,z,w]

  # Normalize quaternions
  retargeted_quat_scipy_normalized = retargeted_quat_scipy / np.linalg.norm(retargeted_quat_scipy, axis=1, keepdims=True)
  # Convert back to MuJoCo format [w,x,y,z]
  retargeted_quat_mujoco = np.column_stack([retargeted_quat_scipy_normalized[:, 3], retargeted_quat_scipy_normalized[:, 0:3]])

  # Combine position and rotation splines
  retargeted_spline_pos = np.hstack([retargeted_pos, retargeted_quat_mujoco])

  # add points for the position before and after contact
  retargeted_start_pos = retargeted_spline_pos[0, :].reshape(1, -1)
  retargeted_start_pos = np.repeat(retargeted_start_pos, start_frame_count, axis=0)
  retargeted_end_pos = retargeted_spline_pos[-1, :].reshape(1, -1)
  retargeted_end_pos = np.repeat(retargeted_end_pos, end_frame_count, axis=0)

  retargeted_spline_pos = np.vstack((retargeted_start_pos, retargeted_spline_pos, retargeted_end_pos))

  save_obj(retargeted_spline_pos[:, :3], "scene/curve_positions.obj")
  assert(len(retargeted_spline_pos) == frames)


  ##### hand retargeting #####
  # retarget hand during contact
  # compute hand component meshes
  hand_components = [None] * hand_components_len
  for j in range(hand_components_len):
    hand_components[j] = get_mesh_for_body(m, j+hand_component_offset)

  # contact_timewarp: (frames + extra_pt_count,)
  # [0, 1] -> [0, 1] where the indices are progressing linearly from 0 to 1, and each index contains an output in [0,1]
  # all points in timewarp are during "contact" only, but this actually includes the non contact ranges as all frames are taken as contact

  # for each frame in the contact range, we find the closest original frame using the timewarp.
  # the closest original frame is in [0, frames+extra_pt_count] (length of contact_timewarp)
  # this is clipped to [0, frames] to match the contact data that we have
  # the closest original frame more precisely in [start_frame_count, frames+extra_pt_count]
  # since we are marking all frames as contact, which actually only begin at start_frame_count
  print(f"Using device for optimization: {optimization_device}")

  kinematic_tree = extract_kinematic_tree(m, hand_component_offset, hand_components_len)
  kinematic_tree_torch = precompute_kinematic_tree_tensors(
    kinematic_tree, hand_component_offset, hand_components_len, optimization_device
  )

  for frame in range(contact_frame_count):
    closet_original_frame = get_closest_original_frame(contact_timewarp, (frame+1)/contact_frame_count)

    print(frame, "/", contact_frame_count, ":", closet_original_frame, start_frame_count, frames+extra_pt_count)
    if closet_original_frame > endIdx:  # maps to final entering frames
      closet_original_frame = endIdx

    # Pre-compute local hand contacts for this frame (cached across optimization iterations)
    local_contacts_cache = precompute_local_hand_contacts(
      hand_contacts, hand_components, hand_components_len,
      closet_original_frame, optimization_device
    )

    # initial hand frame starts with actuation at closest_original_frame
    if frame == 0:
      wrist_init_pos = retargeted_spline_pos[start_frame_count, :3].T
      wrist_init_quat = qpos_copy[3:7, closet_original_frame]
    else:
      wrist_init_pos = qpos[:3, start_frame_count + frame - 1]  # previous optimized

      prev_obj_quat = retargeted_spline_pos[start_frame_count+frame-1, 3:7]  # w,x,y,z
      curr_obj_quat = retargeted_spline_pos[start_frame_count+frame, 3:7]  # w,x,y,z

      # Compute relative rotation from previous to current frame
      prev_obj_rot = R.from_quat(prev_obj_quat)
      curr_obj_rot = R.from_quat(curr_obj_quat)
      delta_rotation = curr_obj_rot * prev_obj_rot.inv()

      # Apply same delta rotation to previous hand wrist rotation
      prev_wrist_quat = qpos[3:7, start_frame_count + frame - 1]  # w,x,y,z
      prev_wrist_rot = R.from_quat(prev_wrist_quat)
      new_wrist_rot = delta_rotation * prev_wrist_rot
      wrist_init_quat = new_wrist_rot.as_quat()  # w,x,y,z

    initial_qpos = np.concatenate((wrist_init_pos, wrist_init_quat, qpos_copy[7:, closet_original_frame]))


    qpos[:, start_frame_count+frame] = optimize_frame(
      initial_qpos,
      retargeted_spline_pos[start_frame_count+frame, :],
      m, d, hand_contacts, object_contacts,
      hand_components, hand_component_offset, object, closet_original_frame,
      kinematic_tree, lr=learning_rate, n_iter=n_iter + (first_frame_iter-n_iter)*(frame==0), optimize_wrist=True,
      optimize_joints=True, agent_type=AGENT,
      print_logs=True, device=optimization_device,
      kinematic_tree_torch=kinematic_tree_torch,
      local_contacts_cache=local_contacts_cache,
      # barriers=barriers, barrier_weight=barrier_weight,
      # barrier_margin=barrier_margin, barrier_n=barrier_n
    )
  
  # shift hand translation for before and after contact
  hand_shift_start = qpos[:3, startIdx] - qpos_copy[:3, startIdx]
  hand_shift_end = qpos[:3, endIdx] - qpos_copy[:3, endIdx]
  qpos[:3, :startIdx] += hand_shift_start[:, np.newaxis]
  qpos[:3, endIdx+1:] += hand_shift_end[:, np.newaxis]

  # Compute rotation difference at contact boundaries
  wrist_rot_end_orig = R.from_quat(qpos_copy[3:7, endIdx])
  wrist_rot_end_new = R.from_quat(qpos[3:7, endIdx])
  delta_rot_end = wrist_rot_end_new * wrist_rot_end_orig.inv()

  # Apply rotation difference to frames after contact
  for i in range(endIdx+1, frames):
    orig_rot = R.from_quat(qpos_copy[3:7, i])
    new_rot = delta_rot_end * orig_rot
    qpos[3:7, i] = new_rot.as_quat()


  # retarget hand before and after contact
  qpos_start = qpos[:3, :start_frame_count]
  qpos_end = qpos[:3, -end_frame_count:]

  # retarget hand before contact
  barrierConstraints(qpos_start.T, hand_boundary_radius, barriers = barriers, traj_path = "scene/hand_start_positions.obj")
  
  # ensure all points are above the surface
  obj = read_obj("scene/hand_start_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/hand_start_positions.obj")

  retargeted_start_trajectory = read_obj("scene/hand_start_positions.obj")
  retargeted_start_spline, _ = create_smoothing_bspline(retargeted_start_trajectory, parameterization="uniform")
  retargeted_sim_time = np.linspace(0, 1, start_frame_count)
  qpos[:3, :start_frame_count] = np.array(scipy.interpolate.splev(retargeted_sim_time, retargeted_start_spline))


  # retarget hand after contact
  barrierConstraints(qpos_end.T, hand_boundary_radius, barriers = barriers, traj_path = "scene/hand_end_positions.obj")
  
  # ensure all points are above the surface
  obj = read_obj("scene/hand_end_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/hand_end_positions.obj")

  retargeted_end_trajectory = read_obj("scene/hand_end_positions.obj")
  retargeted_end_spline, _ = create_smoothing_bspline(retargeted_end_trajectory, parameterization="uniform")
  retargeted_sim_time = np.linspace(0, 1, end_frame_count)
  qpos[:3, -end_frame_count:] = np.array(scipy.interpolate.splev(retargeted_sim_time, retargeted_end_spline))

  # Smooth hand trajectory
  qpos = smooth_hand_trajectory(qpos, frames, AGENT, window_length=21, polyorder=3, visualize=False)

  # Save final trajectories
  import os
  trajectory_dir = "final_trajectories"
  os.makedirs(trajectory_dir, exist_ok=True)

  # Extract trajectory name from config file
  config_name = os.path.splitext(os.path.basename(args.config))[0]

  hand_traj_path = os.path.join(trajectory_dir, f"{config_name}_hand.npy")
  object_traj_path = os.path.join(trajectory_dir, f"{config_name}_object.npy")

  np.save(hand_traj_path, qpos)
  np.save(object_traj_path, retargeted_spline_pos)

  print(f"\nTrajectories saved:")
  print(f"  Hand: {hand_traj_path}")
  print(f"  Object: {object_traj_path}")

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

      # process local hand contacts
      # local_hand_contacts: dict of hand component id -> list of local contact positions at frame i
      local_hand_contacts = {}
      for hand_component_id in range(hand_components_len):
        contacts_this_frame = hand_contacts[hand_component_id][i]
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


      frame_pts.append(qpos[:3, i])
      obj_retarget_frame_pts.append(retargeted_spline_pos[i, :3])

      geometry_count = 0

      # object path color
      if len(obj_retarget_frame_pts) <= start_frame_count:
        rgba = np.array([0, 0, 1, 1])
      elif len(obj_retarget_frame_pts) <= start_frame_count+contact_frame_count:
        rgba = np.array([0, 1, 0, 1])
      else:
        rgba = np.array([1, 0, 0, 1])

      # object path
      for j in range(len(obj_retarget_frame_pts)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[j + geometry_count],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            pos=np.array(obj_retarget_frame_pts[j]),
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
      geometry_count += len(obj_retarget_frame_pts)

      # barrier
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
      
      # bounding sphere
      mujoco.mjv_initGeom(
          viewer.user_scn.geoms[geometry_count],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[boundary_radius, 0, 0],
          pos=np.array(obj_retarget_frame_pts[i]),
          mat=np.eye(3).flatten(),
          rgba=np.array([0.0, 0.0, 0.5, 0.3]))
      geometry_count += 1

      # hand bounding sphere
      mujoco.mjv_initGeom(
          viewer.user_scn.geoms[geometry_count],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[hand_boundary_radius, 0, 0],
          pos=np.array(frame_pts[i]),
          mat=np.eye(3).flatten(),
          rgba=np.array([0.0, 0.0, 0.5, 0.3]))
      geometry_count += 1
      
      if waypts is not None:
      # waypoints
        for j in range(len(waypts)):
          mujoco.mjv_initGeom(
              viewer.user_scn.geoms[geometry_count + j],
              type=mujoco.mjtGeom.mjGEOM_SPHERE,
              size=[0.01, 0, 0],
              pos=waypts[j][0],
              mat=np.eye(3).flatten(),
              rgba=np.array([0, 0, 0, 1]))
        geometry_count += len(waypts)

      # object contacts
      if isinstance(object_contacts[i], np.ndarray):
        for j in range(len(object_contacts[i])):
          local_vertex = object.vertices[object_contacts[i][j]]
          world_vertex = (rotation_matrix @ local_vertex + d.mocap_pos)[0]

          mujoco.mjv_initGeom(
              viewer.user_scn.geoms[j + geometry_count],
              type=mujoco.mjtGeom.mjGEOM_SPHERE,
              size=[0.001, 0, 0],
              pos=np.array(world_vertex),
              mat=np.eye(3).flatten(),
              rgba=np.array([1, 0, 0, 1])
          )
        geometry_count += len(object_contacts[i])

      # hand contacts
      for j in local_hand_contacts: # iterate over hand components
        for k in range(len(local_hand_contacts[j])):  # iterate over contacts in this component
          local_vertex = local_hand_contacts[j][k]
          global_vertex = local_to_global(local_vertex, j+hand_component_offset, d)

          mujoco.mjv_initGeom(
              viewer.user_scn.geoms[k + geometry_count],
              type=mujoco.mjtGeom.mjGEOM_SPHERE,
              size=[0.003, 0, 0],
              pos=np.array(global_vertex),
              mat=np.eye(3).flatten(),
              rgba=np.array([0, 0, 1, 1])
          )
        geometry_count += len(local_hand_contacts[j])
      
      viewer.user_scn.ngeom = geometry_count


      viewer.sync()
      i += 1

      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)