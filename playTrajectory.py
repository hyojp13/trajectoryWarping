import time

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

AGENT="trajectories"
TASK="fryingpan_cook_more_ctrlpts"

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


if __name__ == "__main__":
  build_env_xml(AGENT, TASK)

  splines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/hand.smexp')
  objectSplines, objectSeconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')


  frames = 1000
  
  m = mujoco.MjModel.from_xml_path('kitchen2.xml')
  d = mujoco.MjData(m)
  m.opt.timestep = 2*seconds/frames

  sim_time = np.linspace(0, 1, frames)


  qpos_spline_data = np.array([spline(sim_time) for spline in splines]) # (51, frames, 2)
  object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines]) # (6, frames, 2)

  qpos_frames = qpos_spline_data[0, :, 0] # frames are same for all dof
  qpos_spline_data = qpos_spline_data[:, :, 1] # (51, frames)

  object_qpos_spline_data = object_qpos_spline_data[:, :, 1] # (6, frames)

  rotation = [0, 0, 146.441]
  qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
  object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)

  if AGENT == 'MANO_right' or AGENT == 'trajectories':
    qpos = convert_to_quaternions_MANO(qpos_spline_data)
  if AGENT == 'Allegro_right':
    qpos = convert_to_quaternions_Allegro(qpos_spline_data)

  object_qpos = convert_to_quaternions_object(object_qpos_spline_data)


  frame_pts = []
  obj_frame_pts = []
  obj_orig_frame_pts = []
  obj_retarget_frame_pts = []
  

  with mujoco.viewer.launch_passive(m, d) as viewer:
    i = 0
    while viewer.is_running():

      step_start = time.time()

      # update using only translation
      #d.qpos[:3] = qpos[:3, i % frames]

      d.qpos = qpos[:, i % frames]
      d.mocap_pos = object_qpos[:3, i % frames]
      d.mocap_quat = object_qpos[3:, i % frames]


      if i % frames == 0:
        frame_pts = []
        obj_frame_pts = []


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
            rgba=np.array([0.5, j % frames / frames, 1, 1])
        )
      geometry_count += len(obj_frame_pts)
      
      viewer.user_scn.ngeom = geometry_count


      mujoco.mj_forward(m, d)
      viewer.sync()
      i += 1

      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)