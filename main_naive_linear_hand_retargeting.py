import time
import argparse
import json
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import xml.etree.cElementTree as ET
import cv2

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
      existing_euler = keyframes[3:6, i]
      existing_rot = R.from_euler('xyz', existing_euler)

      combined_rot = additional_rot * existing_rot

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

  config.setdefault('barrier_weight', 0.6)
  config.setdefault('barrier_margin', 0.01)
  config.setdefault('barrier_n', 2.0)

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
  visuals = config.get('visuals', [])
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

  if AGENT == 'Allegro_right':
    hand = 'Allegro'
  else:
    hand = 'MANO'

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

  # Convert visuals from config format to tuples and copy mesh files
  import os
  import shutil
  os.makedirs('meshes', exist_ok=True)
  visuals = [tuple(v) if isinstance(v, list) else v for v in visuals]
  for visual in visuals:
    if isinstance(visual, tuple) and len(visual) >= 2:
      visual_mesh_path = visual[0]
      visual_mesh_dest = f"meshes/{os.path.basename(visual_mesh_path)}"
      if not os.path.exists(visual_mesh_dest):
        shutil.copy(visual_mesh_path, visual_mesh_dest)

  # add mesh barriers and visuals to MuJoCo
  xml_string = add_mesh_barriers_to_xml(scene_file, barriers, visuals)

  # Add offscreen framebuffer size for recording
  xml_string = xml_string.replace(
    '<mujoco',
    '<mujoco',
    1
  )
  import re
  mujoco_tag_match = re.search(r'<mujoco[^>]*>', xml_string)
  if mujoco_tag_match:
    insert_pos = mujoco_tag_match.end()
    visual_settings = '\n  <visual>\n    <global offwidth="1920" offheight="1080"/>\n  </visual>'
    xml_string = xml_string[:insert_pos] + visual_settings + xml_string[insert_pos:]

  object = trimesh.load(object_mesh_file, process=False)

  # Load contacts from .lcexp file
  contacts_lcexp = load_contacts_lcexp('startingTrajectories/' + AGENT + '/' + TASK + '/contacts.lcexp', hand)
  startIdx, endIdx = get_contact_frame_range(contacts_lcexp)

  frames = len(contacts_lcexp)
  m = mujoco.MjModel.from_xml_string(xml_string)
  d = mujoco.MjData(m)
  m.opt.timestep = 2*seconds/frames

  # Look up body IDs by name since the kitchen model has many bodies before the hand.
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
  for name in hand_body_names:
      body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
      if body_id == -1:
          raise ValueError(f"Body '{name}' not found in model")
      hand_component_body_ids.append(body_id)

  # load hand and object contacts from contacts_lcexp
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
  object_qpos_copy = object_qpos.copy()



  start_pos = object_qpos[:3, 0].copy()
  end_pos = object_qpos[:3, frames-1].copy()

  newStartPos = start_pos + new_start_pos_shift
  endFinalPos = end_pos + end_final_pos_shift  # Actual end destination
  endObjPos = endFinalPos + end_obj_pos_shift  # Motion should end here, points from here to endFinalPos is linear interpolation

  # Process waypoints: convert relative positions to absolute positions
  waypts_processed = []
  for waypt in waypts:
    waypt_pos = newStartPos + waypt[0]  # Add waypoint offset to newStartPos
    if len(waypt) == 4:
      waypts_processed.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
    else:
      waypts_processed.append((waypt_pos, waypt[1], waypt[2]))
  waypts = waypts_processed


  # pos_waypt_constrained: (3, frames)
  pos_waypt_constrained, _, _, waypts_idx = trajectoryConstraintsPolyline(object_qpos[:3, :], startPos = newStartPos, endPos = endObjPos, floor_height = start_pos[2], waypts = waypts)

  # Apply rotations at waypoints if specified
  object_rotation_qpos = apply_waypoint_rotations(object_qpos[3:7, :], waypts, waypts_idx, startIdx, frames)


  start_frame_count = startIdx
  contact_frame_count = endIdx-startIdx+1
  end_frame_count = frames - start_frame_count - contact_frame_count

  print("Frames before contact:", start_frame_count)
  print("Frames of contact:", contact_frame_count)
  print("Frames after contact:", end_frame_count)

  barrierConstraints(pos_waypt_constrained.T, boundary_radius, barriers)


  # ensure all points are above the surface
  obj = read_obj("scene/curve_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/curve_positions.obj")


  trajectory = read_obj("scene/curve_positions.obj")  # (frames, 3)

  extra_pt_count = 0


  # new object polyline after barrier and waypoints constraints
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
  # Compute hand component meshes (needed for metrics calculation and visualization)
  hand_components = [None] * hand_components_len
  for j in range(hand_components_len):
    hand_components[j] = get_mesh_for_body(m, hand_component_body_ids[j])

  # Input-frame-iterated interpolation approach:
  # 1. Iterate over INPUT contact frames
  # 2. For each input frame, get its output time from the timewarp
  # 3. Evaluate the retargeted object spline at that output time
  # 4. Compute the exact rigid body hand transform (wrist + fingers)
  # 5. Store as keyframes at non-uniform output times
  # 6. Interpolate all hand DOFs between keyframes to get the uniform output trajectory
  #
  # This exposes the SE(3) interpolation non-equivariance: keyframes are exact,
  # but interpolated frames between them don't preserve contact geometry.

  contact_timewarp_subset = contact_timewarp[startIdx:endIdx+1]
  n_input_frames = len(contact_timewarp_subset)

  # For each input frame, compute the output time and the rigid body hand transform
  keyframe_times = np.zeros(n_input_frames)  # output times (non-uniform)
  keyframe_hand_pos = np.zeros((n_input_frames, 3))
  keyframe_hand_quat_scipy = np.zeros((n_input_frames, 4))  # scipy [x,y,z,w]
  keyframe_fingers = np.zeros((n_input_frames, qpos_copy.shape[0] - 7))  # all finger DOFs

  for i in range(n_input_frames):
    orig_frame = startIdx + i

    # Get the output time for this input frame from the timewarp
    output_time = contact_timewarp_subset[i]
    keyframe_times[i] = output_time

    # Get original object pose at this input frame
    original_obj_pos = object_qpos_copy[:3, orig_frame]
    original_obj_quat_mj = object_qpos_copy[3:7, orig_frame]  # MuJoCo [w,x,y,z]
    original_obj_rot = R.from_quat([original_obj_quat_mj[1], original_obj_quat_mj[2], original_obj_quat_mj[3], original_obj_quat_mj[0]])

    # Get original hand pose at this input frame
    original_hand_pos = qpos_copy[:3, orig_frame]
    original_hand_quat_mj = qpos_copy[3:7, orig_frame]  # MuJoCo [w,x,y,z]
    original_hand_rot = R.from_quat([original_hand_quat_mj[1], original_hand_quat_mj[2], original_hand_quat_mj[3], original_hand_quat_mj[0]])

    # Evaluate the retargeted object spline at this output time
    retargeted_obj_pos_at_t = np.array(scipy.interpolate.splev(output_time, pos_spline)).flatten()
    retargeted_obj_quat_scipy_at_t = np.array(scipy.interpolate.splev(output_time, quat_spline)).flatten()
    retargeted_obj_quat_scipy_at_t = retargeted_obj_quat_scipy_at_t / np.linalg.norm(retargeted_obj_quat_scipy_at_t)
    retargeted_obj_rot = R.from_quat(retargeted_obj_quat_scipy_at_t)  # already in scipy [x,y,z,w]

    # Compute hand pose in object's local frame (constant relative pose)
    hand_pos_local = original_obj_rot.inv().apply(original_hand_pos - original_obj_pos)
    hand_rot_local = original_obj_rot.inv() * original_hand_rot

    # Apply relative pose to retargeted object
    new_hand_pos = retargeted_obj_pos_at_t + retargeted_obj_rot.apply(hand_pos_local)
    new_hand_rot = retargeted_obj_rot * hand_rot_local

    keyframe_hand_pos[i] = new_hand_pos
    keyframe_hand_quat_scipy[i] = new_hand_rot.as_quat()  # scipy [x,y,z,w]

    # Copy finger joints from this input frame
    keyframe_fingers[i] = qpos_copy[7:, orig_frame]

  # Now interpolate from non-uniform keyframe_times to uniform output times
  # using cubic B-spline fitting via smoothspline (same as object trajectory)
  # Ensure keyframe_times is strictly increasing (required by splprep)
  kf_times = keyframe_times.copy()
  for i in range(1, len(kf_times)):
    if kf_times[i] <= kf_times[i-1]:
      kf_times[i] = kf_times[i-1] + 1e-10

  # Sample within the range of kf_times to avoid cubic B-spline extrapolation
  output_times = np.linspace(kf_times[0], kf_times[-1], contact_frame_count)

  # Helper: fit cubic B-spline with timewarp parameterization and sample at output_times
  # splprep has a 10-dimension limit, so we batch data into groups of <=10
  def fit_and_sample_bspline(data, times, sample_times, smoothing_factor=0.001):
    """Fit cubic B-spline to data (n_points, n_dims) parameterized by times, sample at sample_times."""
    n_dims = data.shape[1]
    result = np.zeros((len(sample_times), n_dims))
    batch_size = 10  # splprep max dimension
    for start in range(0, n_dims, batch_size):
      end = min(start + batch_size, n_dims)
      batch = data[:, start:end]
      batch = break_adjacent_duplicates(batch)
      batch_list = [*batch.T]
      spline, _ = scipy.interpolate.splprep(batch_list, u=times, s=smoothing_factor, k=3)
      sampled = np.array(scipy.interpolate.splev(sample_times, spline)).T
      result[:, start:end] = sampled
    return result

  # Fit hand position (3 dims) with timewarp parameterization, sample at uniform output times
  interp_hand_pos = fit_and_sample_bspline(keyframe_hand_pos, kf_times, output_times)

  # Fit finger joints (batched in groups of 10) with timewarp parameterization
  interp_fingers = fit_and_sample_bspline(keyframe_fingers, kf_times, output_times)

  # Fit hand quaternion with timewarp parameterization
  # Ensure quaternion hemisphere consistency
  aligned_quats = keyframe_hand_quat_scipy.copy()
  for i in range(1, n_input_frames):
    if np.dot(aligned_quats[i], aligned_quats[i-1]) < 0:
      aligned_quats[i] = -aligned_quats[i]

  # Break identical adjacent quaternions
  eps = 1e-8
  for i in range(1, len(aligned_quats)):
    if np.allclose(aligned_quats[i], aligned_quats[i-1], atol=eps):
      aligned_quats[i] += eps * np.array([1.0, 0.0, 0.0, 0.0])
      aligned_quats[i] /= np.linalg.norm(aligned_quats[i])

  interp_hand_quat_scipy = fit_and_sample_bspline(aligned_quats, kf_times, output_times)
  # Normalize quaternions
  interp_hand_quat_scipy = interp_hand_quat_scipy / np.linalg.norm(interp_hand_quat_scipy, axis=1, keepdims=True)

  # Convert to MuJoCo [w,x,y,z]
  interp_hand_quat_mujoco = np.column_stack([interp_hand_quat_scipy[:, 3], interp_hand_quat_scipy[:, 0:3]])

  # Update qpos for contact frames
  qpos[:3, start_frame_count:start_frame_count+contact_frame_count] = interp_hand_pos.T
  qpos[3:7, start_frame_count:start_frame_count+contact_frame_count] = interp_hand_quat_mujoco.T
  qpos[7:, start_frame_count:start_frame_count+contact_frame_count] = interp_fingers.T


  # shift hand translation for before and after contact
  hand_shift_start = qpos[:3, start_frame_count] - qpos_copy[:3, startIdx]
  hand_shift_end = qpos[:3, start_frame_count + contact_frame_count - 1] - qpos_copy[:3, endIdx]
  qpos[:3, :start_frame_count] += hand_shift_start[:, np.newaxis]
  qpos[:3, start_frame_count+contact_frame_count:] += hand_shift_end[:, np.newaxis]

  # Compute rotation difference at contact boundaries
  wrist_rot_end_orig_quat = qpos_copy[3:7, endIdx]
  wrist_rot_end_orig = R.from_quat([wrist_rot_end_orig_quat[1], wrist_rot_end_orig_quat[2], wrist_rot_end_orig_quat[3], wrist_rot_end_orig_quat[0]])

  wrist_rot_end_new_quat = qpos[3:7, start_frame_count + contact_frame_count - 1]
  wrist_rot_end_new = R.from_quat([wrist_rot_end_new_quat[1], wrist_rot_end_new_quat[2], wrist_rot_end_new_quat[3], wrist_rot_end_new_quat[0]])

  delta_rot_end = wrist_rot_end_new * wrist_rot_end_orig.inv()

  # Apply rotation difference to frames after contact
  for i in range(start_frame_count+contact_frame_count, frames):
    orig_quat_mujoco = qpos_copy[3:7, endIdx + 1 + (i - start_frame_count - contact_frame_count)]
    orig_rot = R.from_quat([orig_quat_mujoco[1], orig_quat_mujoco[2], orig_quat_mujoco[3], orig_quat_mujoco[0]])
    new_rot = delta_rot_end * orig_rot
    new_quat_scipy = new_rot.as_quat()
    qpos[3:7, i] = np.array([new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]])

  # # retarget hand before and after contact
  qpos_start = qpos[:3, :start_frame_count]
  qpos_end = qpos[:3, -end_frame_count:]
  # qpos[:3, -end_frame_count-20:] = np.ones((3, end_frame_count+20))
  # qpos[:3, :start_frame_count+20] = np.ones((3, start_frame_count+20))
  
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

  # Compute contact distance metrics after full pipeline
  print("\nComputing contact distance metrics...")
  from optimize_contacts import compute_contact_metrics
  contact_metrics = compute_contact_metrics(
    qpos, retargeted_spline_pos, m, d, hand_contacts, object_contacts,
    hand_components, hand_component_body_ids, object, startIdx, endIdx
  )
  print(f"  Contact frames: {startIdx} to {endIdx} ({endIdx - startIdx + 1} frames)")
  print(f"  Overall average distance: {np.mean(contact_metrics['average_distances']):.6f}")
  print(f"  Max average distance: {np.max(contact_metrics['average_distances']):.6f}")

  # Save final trajectories
  import os
  trajectory_dir = "final_trajectories_delete"
  os.makedirs(trajectory_dir, exist_ok=True)

  # Extract trajectory name from config file
  config_name = os.path.splitext(os.path.basename(args.config))[0]

  hand_traj_path = os.path.join(trajectory_dir, f"{config_name}_hand_naive.npy")
  object_traj_path = os.path.join(trajectory_dir, f"{config_name}_object_naive.npy")
  metrics_path = os.path.join(trajectory_dir, f"{config_name}_metrics_naive.npy")

  np.save(hand_traj_path, qpos)
  np.save(object_traj_path, retargeted_spline_pos)
  np.save(metrics_path, contact_metrics, allow_pickle=True)

  print(f"\nTrajectories saved:")
  print(f"  Hand: {hand_traj_path}")
  print(f"  Object: {object_traj_path}")
  print(f"  Metrics: {metrics_path}")

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

      viewer.user_scn.ngeom = geometry_count


      viewer.sync()
      i += 1

      if i % 30 == 0:
        print(f"Camera: azimuth={viewer.cam.azimuth:.1f}, elevation={viewer.cam.elevation:.1f}, "
              f"distance={viewer.cam.distance:.3f}, lookat=[{viewer.cam.lookat[0]:.3f}, {viewer.cam.lookat[1]:.3f}, {viewer.cam.lookat[2]:.3f}]")

      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
