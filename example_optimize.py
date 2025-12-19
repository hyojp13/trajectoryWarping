"""
Example script showing how to use the contact optimization in playTrajectory.py
This demonstrates how to optimize hand wrist position and joint rotations to minimize
contact correspondence error.
Optimizes first 200 frames with more training iterations and displays in MuJoCo viewer.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import trimesh
from scipy.spatial.transform import Rotation as R
from optimize_contacts import optimize_trajectory
from playTrajectory import (
    build_env_xml, parseSplines, convert_to_quaternions_MANO, 
    convert_to_quaternions_Allegro, convert_to_quaternions_object,
    get_contacts_per_frame, rotate_keyframe_angles
)
from handContacts import get_mesh_for_body, get_local_pos, local_to_global

# Configuration (same as in playTrajectory.py)
AGENT = "trajectories"
TASK = "fryingpan_cook"
CONTACT_FILE = "fryingpan_cook_2_right_full_export_motion.npz"
frames = 1000
OPTIMIZE_FRAMES = 200  # Optimize first N frames
TRAINING_ITERATIONS = 150  # More iterations for better optimization

def main():
    # Build environment
    build_env_xml(AGENT, TASK)
    
    # Load model
    m = mujoco.MjModel.from_xml_path('kitchen2.xml')
    d = mujoco.MjData(m)
    
    # Load splines and convert to qpos
    splines, seconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/hand.smexp')
    objectSplines, objectSeconds, _ = parseSplines('startingTrajectories/' + AGENT + '/' + TASK + '/object.smexp')
    
    m.opt.timestep = 2*seconds/frames
    sim_time = np.linspace(0, 1, frames)
    
    qpos_spline_data = np.array([spline(sim_time) for spline in splines])[:, :, 1]  # (51, frames)
    object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])[:, :, 1]  # (6, frames)
    
    rotation = [0, 0, 146.441]
    qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, rotation)
    object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, rotation)
    
    if AGENT == 'MANO_right' or AGENT == 'trajectories':
        qpos = convert_to_quaternions_MANO(qpos_spline_data)
    elif AGENT == 'Allegro_right':
        qpos = convert_to_quaternions_Allegro(qpos_spline_data)
    
    object_qpos = convert_to_quaternions_object(object_qpos_spline_data)
    
    # Load contact data
    dumpMotion = np.load('startingTrajectories/' + AGENT + '/' + TASK + '/' + CONTACT_FILE, allow_pickle=True)
    contactFrames = dumpMotion['contactFrames']
    contactFrameCounts = dumpMotion['objectContactFrameCounts']
    contactLocations = dumpMotion['objectContactLocations']
    
    object_mesh = trimesh.load('/Users/hjp/desktop/exports/s1/fryingpan_cook_2_full_export_objectmesh.obj', process=False)
    object_contacts = get_contacts_per_frame(contactFrames, contactFrameCounts, contactLocations, frames)

    object_contacts = [None] * frames
    for i in range(frames):
        object_contacts[i] = np.array([100, 150])
    
    # Set up hand contacts (you would load these from your actual contact data)
    hand_contacts = {}  # per hand component, each contains a list of contacts per frame
    hand_components_len = 16
    hand_component_offset = 2
    for i in range(hand_components_len):
        hand_contacts[i] = [None] * frames
    
    # Example: populate hand_contacts with your actual contact data
    # For now, using dummy data as in playTrajectory.py
    for j in range(frames):
        # hand_contacts[0][j] = [(0, (0.2, 0.2, 0.6), 0)]
        hand_contacts[3][j] = [(0, (0.2, 0.2, 0.6), 0)]
        hand_contacts[9][j] = [(0, (0.2, 0.2, 0.6), 1)]
    
    # Compute hand component meshes
    hand_components = [None] * hand_components_len
    for j in range(hand_components_len):
        hand_components[j] = get_mesh_for_body(m, j + hand_component_offset)
    
    print("Starting optimization...")
    print(f"Qpos shape: {qpos.shape}")
    print(f"Object qpos shape: {object_qpos.shape}")
    print(f"Optimizing first {OPTIMIZE_FRAMES} frames with {TRAINING_ITERATIONS} iterations each")
    
    # Optimize first N frames
    qpos_optimized = optimize_trajectory(
        qpos, object_qpos, m, d, hand_contacts, object_contacts,
        hand_components, hand_component_offset, object_mesh,
        agent_type=AGENT, optimize_wrist=True, optimize_joints=True,
        lr=0.005, n_iter=TRAINING_ITERATIONS,
        start_frame=0, end_frame=OPTIMIZE_FRAMES - 1
    )
    
    # Save optimized trajectory
    np.save('qpos_optimized.npy', qpos_optimized)
    print(f"\nSaved optimized qpos to qpos_optimized.npy")
    
    # Visualize in MuJoCo
    print("\nDisplaying optimized motion in MuJoCo viewer...")
    visualize_trajectory(m, d, qpos_optimized, object_qpos, hand_contacts, 
                        hand_components, hand_component_offset, 
                        object_mesh, object_contacts, OPTIMIZE_FRAMES)
    
    return qpos_optimized


def visualize_trajectory(m, d, qpos, object_qpos, hand_contacts, hand_components,
                        hand_component_offset, object_mesh, object_contacts, max_frames):
    """
    Visualize the trajectory in MuJoCo viewer, similar to playTrajectory.py
    """
    hand_components_len = len(hand_components)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        i = 0
        frame_pts = []
        obj_frame_pts = []
        
        while viewer.is_running() and i < max_frames:
            step_start = time.time()
            
            # Update qpos and object pose
            d.qpos = qpos[:, i]
            d.mocap_pos = object_qpos[:3, i]
            d.mocap_quat = object_qpos[3:, i]
            mujoco.mj_forward(m, d)
            
            # Process local hand contacts
            local_hand_contacts = {}
            for hand_component_id in range(hand_components_len):
                contacts_this_frame = hand_contacts[hand_component_id][i]
                if contacts_this_frame is not None:
                    local_hand_contacts[hand_component_id] = []
                    for contact in contacts_this_frame:
                        face_id, bary_coords, object_contact_idx = contact
                        local_pos = get_local_pos(
                            face_id, bary_coords,
                            hand_components[hand_component_id][0],
                            hand_components[hand_component_id][1]
                        )
                        local_hand_contacts[hand_component_id].append(local_pos)
                    local_hand_contacts[hand_component_id] = np.array(local_hand_contacts[hand_component_id])
            
            # Convert object rotation for contact visualization
            quat_scipy = np.array([d.mocap_quat[0, 1], d.mocap_quat[0, 2], 
                                   d.mocap_quat[0, 3], d.mocap_quat[0, 0]])
            rotation = R.from_quat(quat_scipy)
            rotation_matrix = rotation.as_matrix()
            
            # Track trajectory points
            if i % max_frames == 0:
                frame_pts = []
                obj_frame_pts = []
            
            frame_pts.append(qpos[:3, i])
            obj_frame_pts.append(object_qpos[:3, i])
            
            geometry_count = 0
            
            # Draw trajectory points (hand wrist)
            for j in range(len(frame_pts)):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[j + geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],
                    pos=np.array(frame_pts[j]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 1, 0, 0.5])  # Green for hand trajectory
                )
            geometry_count += len(frame_pts)
            
            # Draw trajectory points (object)
            for j in range(len(obj_frame_pts)):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[j + geometry_count],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.005, 0, 0],
                    pos=np.array(obj_frame_pts[j]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 0, 1, 0.5])  # Blue for object trajectory
                )
            geometry_count += len(obj_frame_pts)
            
            # Draw object contact points (red)
            if isinstance(object_contacts[i], np.ndarray):
                for j in range(len(object_contacts[i])):
                    local_vertex = object_mesh.vertices[object_contacts[i][j]]
                    world_vertex = (rotation_matrix @ local_vertex + d.mocap_pos)[0]
                    
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[j + geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.003, 0, 0],
                        pos=np.array(world_vertex),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1])  # Red for object contacts
                    )
                geometry_count += len(object_contacts[i])
            
            # Draw hand contact points (cyan)
            for j in local_hand_contacts:
                for k in range(len(local_hand_contacts[j])):
                    local_vertex = local_hand_contacts[j][k]
                    global_vertex = local_to_global(local_vertex, j + hand_component_offset, d)
                    
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[k + geometry_count],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.003, 0, 0],
                        pos=np.array(global_vertex),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0, 1, 1, 1])  # Cyan for hand contacts
                    )
                geometry_count += len(local_hand_contacts[j])
            
            viewer.user_scn.ngeom = geometry_count
            viewer.sync()
            
            i += 1
            if i >= max_frames:
                i = 0  # Loop back to start
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    qpos_opt = main()
