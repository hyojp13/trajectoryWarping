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

AGENT="MANO_right"
TASK="flashlight_on"

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



if __name__ == "__main__":
  build_env_xml(AGENT, TASK)

  # apple
  '''start_pos = np.array([0.10897509, -0.00805262, 0.88921297])
  boundary_radius = 0.07

  barrier_pos1 = start_pos + [-0.1, -0.35, 0.6]
  barrier_radius1 = 0.08
  barrier_pos2 = start_pos + [0.05, 0.1, 0.2]
  barrier_radius2 = 0.05

  barriers = [[barrier_pos1, barrier_radius1], [barrier_pos2, barrier_radius2]]

  waypts = [start_pos + [0, 0.2, 0.5],
            start_pos + [0, -0.5, 0.2],
            start_pos + [-0.3, 0, 0.7]]

  startShift = start_pos + np.array([0.5, -0.5, 0.0])'''

  #flashlight
  '''start_pos = np.array([0.11113561, -0.26607215, 0.8973825])
  end_pos = np.array([0.10741476, -0.23793504, 0.89747757])
  boundary_radius = 0.09
  barriers = [[start_pos + [-0.3, 0, 0.2], 0.1]]
  endShift = end_pos + ([-0.2, -0.2, 0.0])
  waypts = []'''

  start_pos = np.array([0.11113561, -0.26607215, 0.8973825])
  end_pos = np.array([0.10741476, -0.23793504, 0.89747757])
  boundary_radius = 0.07

  #barriers = [[start_pos + [0, -0.3, 0.2], 0.05],
  #            [start_pos + [0, -0.1, 0.18], 0.05],
  #            [start_pos + [0, -0.35, 0.15], 0.05],
  #            [start_pos + [0.1, -0.35, 0.23], 0.05],
  #            [start_pos + [0.05, -0.2, 0.15], 0.05],
  #            [start_pos + [-0.05, -0.4, 0.25], 0.05],
  #            [start_pos + [-0.05, -0.2, 0.25], 0.05]]
  #barriers = [[start_pos + [-0.2, 0, 0.4], 0.05]]
  #barriers = None


  endObjPos = end_pos + ([0.2, -0.4, 0.08])

  # generate desired end object
  basket_mesh = create_basket(radius=0.1, height=0.16, wall_thickness=0.01)
  basket_mesh.export('scene/basket.obj')
  shift_obj('scene/basket.obj', endObjPos)

  # determine entrance to object
  endWayPt = getEntrance('scene/basket.obj')

  endObj = 'scene/basket.obj'
  waypts = [start_pos + [-0.2, 0, 0.5],
            start_pos + [0, -0.5, 0.2]]

  barriers = []
  barriers.append(('rect', {"dims": [0.12, 0.04, 0.14], "pos": [0.05, -0.46607215, 1.12]}))
  barriers.append(('sphere', {"rad": 0.05, "pos": [-0.08886439, -0.30607215, 1.1973825]}))
  barriers.append(('sphere', {"rad": 0.05, "pos": [-0.15, -0.35, 1.1973825]}))
  barriers.append(('sphere', {"rad": 0.05, "pos": [-0.15, -0.23, 1.1973825]}))
  barriers.append(('rect', {"dims": [0.25, 0.04, 0.16], "pos": [-0.15, -0.45, 1.15]}))

  # startShift = start_pos + ([-0.1, -0.1, 0.0])
  # endWayPt = end_pos + ([0, -0.4, 0.0])
  # waypts = [start_pos + [-0.25, 0.1, 0.3], start_pos + [0.1, -0.5, 0.1]]
  # barriers = []
  # # barriers.append(('sphere', {"rad": 0.2, "pos": [0.05, -0.46607215, 1.12]}))
  # barriers.append(('rect', {"dims": [0.4, 0.2, 0.5], "pos": [0.05, -0.46607215, 1.12]}))
  

  splines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/hand.smexp')
  objectSplines, objectSeconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')

  objectSplinesOrig, _, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')

  objectSplines, startTime, endTime, wayPointIdx = trajectoryConstraints(objectSplines, endPos = endWayPt, waypts = waypts,
                                       floor_height = start_pos[2])
                                      #  bounding_sphere_radius=boundary_radius,
                                      #  barriers=barriers)

  #generate_barrier('rect', {"dims": [0.05, 0.05, 0.05], "pos": [-0.08886439, -0.26607215, 1.2973825]}, path="/Users/hjp/desktop/repulsive-curves/scenes/retargeting/barrierRect.obj")
  #generate_barrier('sphere', {"rad": 0.05, "pos": [-0.08886439, -0.26607215, 1.2973825]}, path="/Users/hjp/desktop/repulsive-curves/scenes/retargeting/barrierSphere.obj")
  

  frames = 702
  startFrame = startTime * frames

  # ensure that waypoints are not too close to a barrier
  if not barrierWayptsCheck(barriers, waypts, boundary_radius):
    raise Exception("waypoints are closer to the barrier than the object radius")

  barrierConstraints(objectSplines, boundary_radius, waypts,
                                    endWayPt,
                                    barriers = barriers, resolution = frames, endObj=endObj)
  

  # ensure all points are above the surface
  obj = read_obj("scene/curve_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/curve_positions.obj")


  trajectory = read_obj("scene/curve_positions.obj")
  retargeted_spline = create_smoothing_bspline(trajectory)
  
  # (outdated, for repulsive curves) prevent motion before and after contact
  # (outdated, for repulsive curves) NOTE: assumes that frames = resolution in barrierConstraints
  # cleanTrajectory(start_pos[2], int(startTime*frames), int(endTime*frames), objectSplines[:3], frames)

  m = mujoco.MjModel.from_xml_path('env.xml')
  d = mujoco.MjData(m)


  m.opt.timestep = 2*seconds/frames

  sim_time = np.linspace(0, 1, frames)
  qpos_spline_data = np.array([spline(sim_time) for spline in splines]) # (51, frames, 2)
  object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines]) # (6, frames, 2)
  object_orig_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplinesOrig]) # (6, frames, 2)

  qpos_frames = qpos_spline_data[0, :, 0] # frames are same for all dof
  qpos_spline_data = qpos_spline_data[:, :, 1] # (51, frames)

  object_qpos_spline_data = object_qpos_spline_data[:, :, 1] # (6, frames)
  object_orig_qpos_spline_data = object_orig_qpos_spline_data[:, :, 1] # (6, frames)

  if AGENT == 'MANO_right':
    qpos = convert_to_quaternions_MANO(qpos_spline_data)
  if AGENT == 'Allegro_right':
    qpos = convert_to_quaternions_Allegro(qpos_spline_data)

  object_qpos = convert_to_quaternions_object(object_qpos_spline_data)  # with waypoint
  object_orig_qpos = convert_to_quaternions_object(object_orig_qpos_spline_data)  # without waypoint

  # object_repul_pos = read_obj("scene/curve_positions.obj")

  retargeted_spline_pos = np.array(scipy.interpolate.splev(sim_time, retargeted_spline)).T

  object_shift = retargeted_spline_pos.T - object_orig_qpos[:3, :] # (3, frames)
  qpos[:3, :] += object_shift
  
  save_obj(retargeted_spline_pos, "scene/curve_positions.obj")

  assert(len(retargeted_spline_pos) == frames)
  
  '''for i in range(100):
    object_qpos[:3, i] += starting_loc_shift

  print(object_qpos[:3, 0])'''

  #print(barrier_pos)

  frame_pts = []
  obj_frame_pts = []
  obj_repul_frame_pts = []
  obj_orig_frame_pts = []
  

  with mujoco.viewer.launch_passive(m, d) as viewer:
    i = 0
    while viewer.is_running():

      step_start = time.time()

      # update using only translation
      #d.qpos[:3] = qpos[:3, i % frames]

      d.qpos = qpos[:, i % frames]
      d.mocap_pos = retargeted_spline_pos[i % frames]
      d.mocap_quat = object_qpos[3:, i % frames]


      if i % frames == 0:
        frame_pts = []
        obj_frame_pts = []
        obj_repul_frame_pts = []
        obj_orig_frame_pts = []


      frame_pts.append(qpos[:3, i % frames])
      obj_frame_pts.append(object_qpos[:3, i % frames])
      obj_repul_frame_pts.append(retargeted_spline_pos[i % frames])
      obj_orig_frame_pts.append(object_orig_qpos[:3, i % frames])

      geometry_count = 0

      '''for j in range(len(frame_pts)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[j],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            pos=np.array(frame_pts[j]),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 1])
        )
      geometry_count += len(frame_pts)'''

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

      for j in range(len(obj_repul_frame_pts)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[j + geometry_count],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            pos=np.array(obj_repul_frame_pts[j]),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 1, j % frames / frames, 1])
        )
      geometry_count += len(obj_repul_frame_pts)

      # barrier
      if barriers is not None:
        for j in range(len(barriers)):
          if barriers[j][0] == 'sphere':
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
          pos=np.array(obj_repul_frame_pts[i % frames]),
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
              pos=waypts[j],
              mat=np.eye(3).flatten(),
              rgba=np.array([0, 0, 0, 1]))
        geometry_count += len(waypts)
      
      viewer.user_scn.ngeom = geometry_count


      mujoco.mj_forward(m, d)
      viewer.sync()
      i += 1

      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)