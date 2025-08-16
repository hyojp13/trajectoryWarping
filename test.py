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
TASK="fryingpan_cook"

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


if __name__ == "__main__":
  build_env_xml(AGENT, TASK)

  # retrieve splines
  splines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/hand.smexp')
  objectSplines, objectSeconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')

  objectSplinesOrig, _, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')

  # add mesh barriers to MuJoCo
  scene_file = "kitchen2.xml"
  with open(scene_file) as f:
    xml_string = f.read()
  
  barriers = []
  barriers.append(("/Users/hjp/Desktop/robocasa/robocasa/models/assets/fixtures/hoods/pack_2/visuals/model_0.obj", {"scale":(1.15613, 1.06643, 1.15613), "pos":(2.2-0.15, -0.3, 2.24807-0.3)}))
  barriers.append(('rect', {"dims": [1, 0.4, 0.03], "pos": [3.2, -0.2, 1.85-0.445]}))
  barriers = process_barriers(barriers) # apply scale and pos to obj barriers

  for i, barrier in enumerate(barriers):
    if isinstance(barrier, str):
      # Path to your .obj file
      mesh_file = barrier.split('/')[1]
      import os
      print(os.path.abspath(mesh_file))


      mesh_asset = f"""
          <mesh name="mesh_{i}" file="{mesh_file}"/>
      """

      mesh_body = f"""
          <body name="mesh_body" pos="0 0 0">
              <geom type="mesh" mesh="mesh_{i}" rgba="0.2 0.8 0.2 1" group="2"/>
          </body>
      """

      if "</asset>" in xml_string:
        xml_string = xml_string.replace("</asset>", mesh_asset + "\n</asset>")
      else:
          raise ValueError("No <asset> block found in XML.")

      # Inject body into <worldbody> block (before </worldbody>)
      if "</worldbody>" in xml_string:
          xml_string = xml_string.replace("</worldbody>", mesh_body + "\n</worldbody>")
      else:
          raise ValueError("No <worldbody> block found in XML.")

  # Load model from modified XML string
  m = mujoco.MjModel.from_xml_string(xml_string)
  d = mujoco.MjData(m)


  # correct barrier axes (issue due to using repulsive curves previously)
  barrier = correct_barrier_axes(barriers)

  # configurations
  frames = 1000
  # m = mujoco.MjModel.from_xml_path(scene_file)
  # d = mujoco.MjData(m)
  m.opt.timestep = 2*seconds/frames
  sim_time = np.linspace(0, 1, frames)


  # read splines to numpy
  qpos_spline_data = np.array([spline(sim_time) for spline in splines]) # (51, frames, 2)
  object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines]) # (6, frames, 2)
  object_orig_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplinesOrig]) # (6, frames, 2)

  qpos_frames = qpos_spline_data[0, :, 0] # frames are same for all dof
  qpos_spline_data = qpos_spline_data[:, :, 1] # (51, frames)

  object_qpos_spline_data = object_qpos_spline_data[:, :, 1] # (6, frames)
  object_orig_qpos_spline_data = object_orig_qpos_spline_data[:, :, 1] # (6, frames)

  rotation = [0, 0, 146.441]
  qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
  object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)
  object_orig_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)

  if AGENT == 'MANO_right' or AGENT == 'trajectories':
    qpos = convert_to_quaternions_MANO(qpos_spline_data)
  if AGENT == 'Allegro_right':
    qpos = convert_to_quaternions_Allegro(qpos_spline_data)

  object_qpos = convert_to_quaternions_object(object_qpos_spline_data)  # with waypoint
  object_orig_qpos = convert_to_quaternions_object(object_orig_qpos_spline_data)  # without waypoint

  start_pos = object_qpos[:3, 0]
  end_pos = object_qpos[:3, frames-1]


  boundary_radius = 0.1
  hand_boundary_radius = 0.1

  newStartPos = start_pos + ([-0.23, 0.18, 0])
  endObjPos = end_pos + ([0.9, 0.18, 0.48])

  # generate desired end object
  # basket_mesh = create_basket(radius=0.1, height=0.16, wall_thickness=0.01)
  # basket_mesh.export('scene/basket.obj')
  # shift_obj('scene/basket.obj', endObjPos)

  # determine entrance to object
  # endWayPt = getEntrance('scene/basket.obj')

  # endObj = 'scene/basket.obj'
  # waypts = [start_pos + [-0.2, 0, 0.5],
            # start_pos + [0, -0.5, 0.2]]
  waypts = []

  # barriers.append(('rect', {"dims": [0.12, 0.04, 0.14], "pos": [0.05, -0.40607215, 1.12]}))
  # barriers.append(('sphere', {"rad": 0.05, "pos": [-0.08886439, -0.30607215, 1.1973825]}))
  # barriers.append(('sphere', {"rad": 0.05, "pos": [-0.15, -0.35, 1.1973825]}))
  # barriers.append(('sphere', {"rad": 0.05, "pos": [-0.15, -0.23, 1.1973825]}))
  # barriers.append(('sphere', {"rad": 0.09, "pos": [0.12, 0.05, 1]})) # intersection before/after contact
  # barriers.append(('rect', {"dims": [0.25, 0.04, 0.16], "pos": [-0.15, -0.45, 1.15]}))
  


  # print(objectSplines)


  objectSplines, startTime, endTime, wayPointIdx = trajectoryConstraints(objectSplines, startPos = newStartPos, endPos = endObjPos, floor_height = start_pos[2])
                                      #  bounding_sphere_radius=boundary_radius,
                                      #  barriers=barriers)

  #generate_barrier('rect', {"dims": [0.05, 0.05, 0.05], "pos": [-0.08886439, -0.26607215, 1.2973825]}, path="/Users/hjp/desktop/repulsive-curves/scenes/retargeting/barrierRect.obj")
  #generate_barrier('sphere', {"rad": 0.05, "pos": [-0.08886439, -0.26607215, 1.2973825]}, path="/Users/hjp/desktop/repulsive-curves/scenes/retargeting/barrierSphere.obj")
  

  # calculate number of frames for each segment of the trajectory
  start_frame_count = int(frames * startTime)
  contact_frame_count = int(frames * (endTime - startTime))
  end_frame_count = frames - start_frame_count - contact_frame_count

  print("Frames before contact:", start_frame_count)
  print("Frames of contact:", contact_frame_count)
  print("Frames after contact:", end_frame_count)

  # print(start_frame_count, contact_frame_count, end_frame_count)

  # startFrame = startTime * frames

  # ensure that waypoints are not too close to a barrier
  if not barrierWayptsCheck(barriers, waypts, boundary_radius):
    raise Exception("waypoints are closer to the barrier than the object radius")

  barrierConstraints(objectSplines, boundary_radius, barriers = barriers, resolution = frames)
  

  # ensure all points are above the surface
  obj = read_obj("scene/curve_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/curve_positions.obj")


  trajectory = read_obj("scene/curve_positions.obj")
  retargeted_spline = create_smoothing_bspline(trajectory, parameterization="chord")
  



  retargeted_sim_time = np.linspace(0, 1, contact_frame_count)
  retargeted_spline_pos = np.array(scipy.interpolate.splev(retargeted_sim_time, retargeted_spline)).T

  # add points for the position before and after contact
  retargeted_start_pos = retargeted_spline_pos[0, :].reshape(1, -1)
  retargeted_start_pos = np.repeat(retargeted_start_pos, start_frame_count, axis=0)
  retargeted_end_pos = retargeted_spline_pos[-1, :].reshape(1, -1)
  retargeted_end_pos = np.repeat(retargeted_end_pos, end_frame_count, axis=0)

  # print(retargeted_spline_pos.shape, retargeted_start_pos.shape, retargeted_end_pos.shape)

  retargeted_spline_pos = np.vstack((retargeted_start_pos, retargeted_spline_pos, retargeted_end_pos))

  object_shift = retargeted_spline_pos.T - object_orig_qpos[:3, :] # (3, frames)
  qpos[:3, :] += object_shift

  save_obj(retargeted_spline_pos, "scene/curve_positions.obj")

  assert(len(retargeted_spline_pos) == frames)



  # # retarget hand before and after contact
  qpos_start = qpos[:3, :start_frame_count]
  x1 = np.linspace(0, 1, start_frame_count)
  start_splines = []
  for i in range(3):
      spl = scipy.interpolate.make_interp_spline(x1, qpos_start[i, :], k=3)
      start_splines.append(spl)


  
  qpos_end = qpos[:3, -end_frame_count:]
  x2 = np.linspace(0, 1, end_frame_count)
  end_splines = []
  for i in range(3):
      spl = scipy.interpolate.make_interp_spline(x2, qpos_end[i, :], k=3)
      end_splines.append(spl)

  # print(qpos_end)

  # retarget hand before contact
  barrierConstraints(start_splines, hand_boundary_radius, barriers = barriers,
                    resolution = start_frame_count, traj_path = "scene/hand_start_positions.obj")
  
  # ensure all points are above the surface
  obj = read_obj("scene/hand_start_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/hand_start_positions.obj")

  retargeted_start_trajectory = read_obj("scene/hand_start_positions.obj")
  retargeted_start_spline = create_smoothing_bspline(retargeted_start_trajectory, parameterization="uniform")
  retargeted_sim_time = np.linspace(0, 1, start_frame_count)
  qpos[:3, :start_frame_count] = np.array(scipy.interpolate.splev(retargeted_sim_time, retargeted_start_spline))
  # qpos[:3, :start_frame_count] = retargeted_start_trajectory.T


  # retarget hand after contact
  barrierConstraints(end_splines, hand_boundary_radius, barriers = barriers,
                    resolution = end_frame_count, traj_path = "scene/hand_end_positions.obj")
  
  # ensure all points are above the surface
  obj = read_obj("scene/hand_end_positions.obj")
  obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
  save_obj(obj, "scene/hand_end_positions.obj")


  retargeted_end_trajectory = read_obj("scene/hand_end_positions.obj")
  retargeted_end_spline = create_smoothing_bspline(retargeted_end_trajectory, parameterization="uniform")
  retargeted_sim_time = np.linspace(0, 1, end_frame_count)
  qpos[:3, -end_frame_count:] = np.array(scipy.interpolate.splev(retargeted_sim_time, retargeted_end_spline))
  # qpos[:3, -end_frame_count:] = retargeted_end_trajectory.T

  

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
      d.mocap_pos = retargeted_spline_pos[i % frames]
      d.mocap_quat = object_qpos[3:, i % frames]


      if i % frames == 0:
        frame_pts = []
        obj_frame_pts = []
        obj_retarget_frame_pts = []
        obj_orig_frame_pts = []


      frame_pts.append(qpos[:3, i % frames])
      # obj_frame_pts.append(object_qpos[:3, i % frames])    # uncomment to visualize
      obj_retarget_frame_pts.append(retargeted_spline_pos[i % frames])
      # obj_orig_frame_pts.append(object_orig_qpos[:3, i % frames])

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

      if len(obj_retarget_frame_pts) <= start_frame_count:
        rgba = np.array([0, 0, 1, 1])
      elif len(obj_retarget_frame_pts) <= start_frame_count+contact_frame_count:
        rgba = np.array([0, 1, 0, 1])
      else:
        rgba = np.array([1, 0, 0, 1])

      if len(obj_retarget_frame_pts) == start_frame_count+1:
        print("Contact start point: ", frame_pts[-1])
      if len(obj_retarget_frame_pts) == start_frame_count+contact_frame_count+1:
        print("End start point: ", frame_pts[-1])

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
          pos=np.array(obj_retarget_frame_pts[i % frames]),
          mat=np.eye(3).flatten(),
          rgba=np.array([0.0, 0.0, 0.5, 0.3]))
      geometry_count += 1

      # hand bounding sphere
      mujoco.mjv_initGeom(
          viewer.user_scn.geoms[geometry_count],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[hand_boundary_radius, 0, 0],
          pos=np.array(frame_pts[i % frames]),
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