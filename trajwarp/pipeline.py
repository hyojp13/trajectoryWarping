"""Top-level retargeting pipeline.

Strings together the four warping stages from the paper:

    1. load the demonstration, contacts, and scene            (load_demonstration)
    2. warp the object trajectory                              (warp_object)
         - spatial waypoint-constrained warp   (Sec. III-A)
         - barrier-constrained warp            (Sec. III-B)
         - temporal waypoint-constrained warp  (Sec. III-C)
    3. recover the hand trajectory                             (a hand strategy)
         - contact-based IK (ours) or a naive SE(3) baseline (Sec. III-D)
    4. score, save, and optionally visualize the result

``run`` is the single entry point used by ``retarget.py`` and the naive baseline
scripts; they differ only in the ``strategy`` they pass.
"""
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.interpolate
import trimesh
import mujoco
from scipy.spatial.transform import Rotation as R

from trajwarp.io.config import load_config
from trajwarp.io.spline_io import parseSplines
from trajwarp.io.contact_io import load_contacts_lcexp, get_contact_frame_range
from trajwarp.io.scene_xml import write_object_xml, add_mesh_barriers_to_xml
from trajwarp.object_warp.spatial import (
    trajectoryConstraintsPolyline,
    apply_waypoint_rotations,
)
from trajwarp.object_warp.barriers import (
    process_barriers,
    correct_barrier_axes,
    barrierConstraints,
    read_obj,
    save_obj,
    moveToEndPt,
)
from trajwarp.object_warp.smoothing import (
    create_smoothing_bspline,
    create_quaternion_bspline,
)
from trajwarp.hand_warp.correspondences import process_contacts
from trajwarp.hand_warp.hand_mesh import get_mesh_for_body
from trajwarp.hand_warp.optimizer import compute_contact_metrics
from trajwarp.utils.quaternions import (
    convert_to_quaternions_MANO,
    convert_to_quaternions_Allegro,
    convert_to_quaternions_object,
    rotate_keyframe_angles,
)
from trajwarp.utils.agents import get_hand_type, get_hand_body_names

ALLEGRO_KITCHEN_SHIFT = [2.2, -2.1, 0.08]


@dataclass
class WarpState:
    """Mutable container for everything the warping stages share."""

    config: object
    config_name: str
    agent: str
    hand: str

    # MuJoCo model + hand component bookkeeping
    model: object
    data: object
    hand_components: list
    hand_component_body_ids: list
    hand_components_len: int

    # Demonstration data
    qpos: np.ndarray
    qpos_copy: np.ndarray
    object_qpos: np.ndarray
    object_qpos_copy: np.ndarray
    object_mesh: object
    hand_contacts: list
    object_contacts: list
    frames: int
    start_idx: int
    end_idx: int

    # Scene
    barriers: list

    # Object-warp outputs (filled by warp_object)
    retargeted_spline_pos: np.ndarray = None
    contact_timewarp: np.ndarray = None
    pos_spline: object = None
    quat_spline: object = None
    waypts: list = field(default_factory=list)
    start_frame_count: int = 0
    contact_frame_count: int = 0
    end_frame_count: int = 0
    extra_pt_count: int = 0
    start_pos: np.ndarray = None

    # Timing
    retargeting_elapsed_time: float = 0.0


def _build_scene_xml(config):
    """Build the full MuJoCo scene string: base scene with the correct hand
    injected, plus barrier and visual meshes added."""
    barriers = [tuple(b) if isinstance(b, list) else b for b in config.barriers]
    barriers = process_barriers(barriers)

    visuals = [tuple(v) if isinstance(v, list) else v for v in config.visuals]
    os.makedirs('meshes', exist_ok=True)
    for visual in visuals:
        if isinstance(visual, tuple) and len(visual) >= 2:
            visual_mesh_dest = f"meshes/{os.path.basename(visual[0])}"
            if not os.path.exists(visual_mesh_dest):
                import shutil
                shutil.copy(visual[0], visual_mesh_dest)

    xml_string = add_mesh_barriers_to_xml(config.scene_file, barriers, visuals)
    xml_string = _inject_agent(xml_string, config.agent)
    return xml_string, barriers


def _inject_agent(xml_string, agent):
    """Swap the base scene's hand includes for the requested agent's hand.

    The base kitchen scene includes the MANO hand; for Allegro configs we
    substitute the Allegro assets/actuators/body so the scene is agent-aware
    without manual XML editing.
    """
    if agent == 'Allegro_right':
        for kind in ('assets', 'actuators', 'body'):
            xml_string = xml_string.replace(
                f'agents/MANO_right/{kind}.xml',
                f'agents/Allegro_right/{kind}.xml',
            )
    return xml_string


def load_demonstration(config, initial=False):
    """Load splines, contacts, scene, and the MuJoCo model; return a WarpState."""
    agent = config.agent
    hand = get_hand_type(agent)

    # Object mocap include + mesh copy.
    write_object_xml(config.object_mesh_file)

    # Demonstration splines.
    base = f"startingTrajectories/{agent}/{config.task}"
    splines, seconds, _ = parseSplines(f"{base}/hand.smexp")
    objectSplines, objectSeconds, _ = parseSplines(f"{base}/object.smexp")

    # Scene (agent-aware) + barriers + object mesh.
    xml_string, barriers = _build_scene_xml(config)
    object_mesh = trimesh.load(config.object_mesh_file, process=False)

    # Contacts.
    contacts_lcexp = load_contacts_lcexp(f"{base}/contacts.lcexp", hand)
    start_idx, end_idx = get_contact_frame_range(contacts_lcexp)
    frames = len(contacts_lcexp)

    # MuJoCo model.
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)
    m.opt.timestep = 2 * seconds / frames

    # Hand component bodies (looked up by name to skip the kitchen geometry).
    hand_body_names = get_hand_body_names(agent)
    hand_components_len = len(hand_body_names)
    hand_component_body_ids = []
    for name in hand_body_names:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found in model")
        hand_component_body_ids.append(body_id)

    hand_contacts, object_contacts = process_contacts(contacts_lcexp, hand_components_len)

    # Correct mesh-barrier axes (no-op for primitive barriers).
    correct_barrier_axes(barriers)

    sim_time = np.linspace(0, 1, frames)

    if initial:
        initial_hand_path = f"initial_trajectories/{config.task}_hand.npy"
        initial_object_path = f"initial_trajectories/{config.task}_object.npy"
        print(f"Loading initial trajectories from: {initial_hand_path}, {initial_object_path}")
        qpos = np.load(initial_hand_path)
        object_qpos = np.load(initial_object_path).T
        qpos_copy = qpos.copy()
    else:
        qpos_spline_data = np.array([spline(sim_time) for spline in splines])
        object_qpos_spline_data = np.array([spline(sim_time) for spline in objectSplines])

        qpos_spline_data = qpos_spline_data[:, :, 1]
        object_qpos_spline_data = object_qpos_spline_data[:, :, 1]

        qpos_spline_data = rotate_keyframe_angles(qpos_spline_data, config.rotation)
        object_qpos_spline_data = rotate_keyframe_angles(object_qpos_spline_data, config.rotation)

        object_qpos = convert_to_quaternions_object(object_qpos_spline_data)

        if hand == 'MANO':
            qpos = convert_to_quaternions_MANO(qpos_spline_data)
        else:
            qpos = convert_to_quaternions_Allegro(qpos_spline_data)

            hand_shift = np.zeros([qpos.shape[0], 1])
            object_shift = np.zeros([object_qpos.shape[0], 1])
            hand_shift[:3, 0] += ALLEGRO_KITCHEN_SHIFT
            object_shift[:3, 0] += ALLEGRO_KITCHEN_SHIFT
            qpos += hand_shift
            object_qpos += object_shift

        qpos_copy = qpos.copy()

    hand_components = [get_mesh_for_body(m, bid) for bid in hand_component_body_ids]

    config_name = os.path.splitext(os.path.basename(config.config_path))[0]
    return WarpState(
        config=config,
        config_name=config_name,
        agent=agent,
        hand=hand,
        model=m,
        data=d,
        hand_components=hand_components,
        hand_component_body_ids=hand_component_body_ids,
        hand_components_len=hand_components_len,
        qpos=qpos,
        qpos_copy=qpos_copy,
        object_qpos=object_qpos,
        object_qpos_copy=object_qpos.copy(),
        object_mesh=object_mesh,
        hand_contacts=hand_contacts,
        object_contacts=object_contacts,
        frames=frames,
        start_idx=start_idx,
        end_idx=end_idx,
        barriers=barriers,
    )


def warp_object(state):
    """Warp the object trajectory through the spatial, barrier, and temporal
    stages (paper Sec. III-A/B/C). Fills the object-warp fields of ``state``."""
    config = state.config
    object_qpos = state.object_qpos
    frames = state.frames
    start_idx, end_idx = state.start_idx, state.end_idx

    start_pos = object_qpos[:3, 0].copy()
    end_pos = object_qpos[:3, frames - 1].copy()

    newStartPos = start_pos + config.new_start_pos_shift
    endFinalPos = end_pos + config.end_final_pos_shift
    endObjPos = endFinalPos + config.end_obj_pos_shift

    # Convert relative waypoint offsets to absolute positions.
    waypts = []
    for waypt in config.waypts:
        waypt_pos = newStartPos + waypt[0]
        if len(waypt) == 4:
            waypts.append((waypt_pos, waypt[1], waypt[2], waypt[3]))
        else:
            waypts.append((waypt_pos, waypt[1], waypt[2]))

    # Sec. III-A: spatial waypoint-constrained warp.
    pos_waypt_constrained, _, _, waypts_idx = trajectoryConstraintsPolyline(
        object_qpos[:3, :], startPos=newStartPos, endPos=endObjPos,
        floor_height=start_pos[2], waypts=waypts,
    )
    object_rotation_qpos = apply_waypoint_rotations(
        object_qpos[3:7, :], waypts, waypts_idx, start_idx, frames,
    )

    start_frame_count = start_idx
    contact_frame_count = end_idx - start_idx + 1
    end_frame_count = frames - start_frame_count - contact_frame_count

    print("Frames before contact:", start_frame_count)
    print("Frames of contact:", contact_frame_count)
    print("Frames after contact:", end_frame_count)

    # Sec. III-B: barrier-constrained warp.
    barrierConstraints(pos_waypt_constrained.T, config.boundary_radius, state.barriers)

    obj = read_obj("scene/curve_positions.obj")
    obj[:, 2] = np.maximum(obj[:, 2], start_pos[2])
    save_obj(obj, "scene/curve_positions.obj")

    trajectory = read_obj("scene/curve_positions.obj")

    extra_pt_count = config.extra_pt_count
    if not np.isclose(trajectory[-1, :], endFinalPos, atol=1e-3).all():
        print("trajectory does not end in end position, adding extra points")
        print(trajectory[-1, :], endFinalPos)
        moveToEndPt("scene/curve_positions.obj", endFinalPos, extra_pt_count)
        trajectory = read_obj("scene/curve_positions.obj")
    else:
        extra_pt_count = 0

    new_object_qpos = np.zeros((object_qpos.shape[0], object_qpos.shape[1] + extra_pt_count))
    new_object_qpos[:3, :] = trajectory.T
    new_object_qpos[3:, :object_qpos.shape[1]] = object_rotation_qpos
    if extra_pt_count != 0:
        new_object_qpos[3:, -extra_pt_count:] = new_object_qpos[3:, -extra_pt_count - 1].reshape(
            object_qpos.shape[0] - 3, 1)

    # Sec. III-C: temporal waypoint-constrained warp (B-spline retiming).
    final_waypt_timesteps = np.array([w[2] for w in waypts])
    pos_spline, contact_timewarp = create_smoothing_bspline(
        new_object_qpos[:3, :].T, parameterization="waypts",
        waypts_info=(waypts_idx, final_waypt_timesteps),
    )

    quat_mujoco = new_object_qpos[3:7, :].T
    quat_scipy = np.column_stack([quat_mujoco[:, 1:4], quat_mujoco[:, 0]])
    quat_spline, _ = create_quaternion_bspline(
        quat_scipy, waypts_info=(waypts_idx, final_waypt_timesteps))

    retargeted_sim_time = np.linspace(0, 1, contact_frame_count)
    retargeted_pos = np.array(scipy.interpolate.splev(retargeted_sim_time, pos_spline)).T
    retargeted_quat_scipy = np.array(scipy.interpolate.splev(retargeted_sim_time, quat_spline)).T

    retargeted_quat_scipy_normalized = retargeted_quat_scipy / np.linalg.norm(
        retargeted_quat_scipy, axis=1, keepdims=True)
    retargeted_quat_mujoco = np.column_stack([
        retargeted_quat_scipy_normalized[:, 3], retargeted_quat_scipy_normalized[:, 0:3]])

    retargeted_spline_pos = np.hstack([retargeted_pos, retargeted_quat_mujoco])

    retargeted_start_pos = retargeted_spline_pos[0, :].reshape(1, -1)
    retargeted_start_pos = np.repeat(retargeted_start_pos, start_frame_count, axis=0)
    retargeted_end_pos = retargeted_spline_pos[-1, :].reshape(1, -1)
    retargeted_end_pos = np.repeat(retargeted_end_pos, end_frame_count, axis=0)

    retargeted_spline_pos = np.vstack((retargeted_start_pos, retargeted_spline_pos, retargeted_end_pos))

    save_obj(retargeted_spline_pos[:, :3], "scene/curve_positions.obj")
    assert len(retargeted_spline_pos) == frames

    state.retargeted_spline_pos = retargeted_spline_pos
    state.contact_timewarp = contact_timewarp
    state.pos_spline = pos_spline
    state.quat_spline = quat_spline
    state.waypts = waypts
    state.start_frame_count = start_frame_count
    state.contact_frame_count = contact_frame_count
    state.end_frame_count = end_frame_count
    state.extra_pt_count = extra_pt_count
    state.start_pos = start_pos
    return state


def compute_metrics(state):
    """Compute hand-object contact distance metrics over the contact window."""
    print("\nComputing contact distance metrics...")
    metrics = compute_contact_metrics(
        state.qpos, state.retargeted_spline_pos, state.model, state.data,
        state.hand_contacts, state.object_contacts, state.hand_components,
        state.hand_component_body_ids, state.object_mesh, state.start_idx, state.end_idx,
    )
    print(f"  Contact frames: {state.start_idx} to {state.end_idx} "
          f"({state.end_idx - state.start_idx + 1} frames)")
    print(f"  Overall average distance: {np.mean(metrics['average_distances']):.6f}")
    print(f"  Max average distance: {np.max(metrics['average_distances']):.6f}")
    return metrics


def save_outputs(state, metrics, suffix=""):
    """Save hand/object trajectories, metrics, timewarp data, and a run summary."""
    config = state.config
    trajectory_dir = f"final_trajectories_{config.loss_threshold}"
    os.makedirs(trajectory_dir, exist_ok=True)
    name = state.config_name

    hand_traj_path = os.path.join(trajectory_dir, f"{name}{suffix}_hand.npy")
    object_traj_path = os.path.join(trajectory_dir, f"{name}{suffix}_object.npy")
    metrics_path = os.path.join(trajectory_dir, f"{name}{suffix}_metrics.npy")
    timewarp_path = os.path.join(trajectory_dir, f"{name}{suffix}_timewarp.npy")

    np.save(hand_traj_path, state.qpos)
    np.save(object_traj_path, state.retargeted_spline_pos)
    np.save(metrics_path, metrics, allow_pickle=True)
    np.save(timewarp_path, {
        'contact_timewarp': state.contact_timewarp,
        'start_frame_count': state.start_frame_count,
        'contact_frame_count': state.contact_frame_count,
        'end_frame_count': state.end_frame_count,
        'pos_spline': state.pos_spline,
    }, allow_pickle=True)

    print("\nTrajectories saved:")
    print(f"  Hand: {hand_traj_path}")
    print(f"  Object: {object_traj_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Timewarp: {timewarp_path}")
    print(f"\nRetargeting time: {state.retargeting_elapsed_time:.2f} seconds")

    _write_run_summary(state, os.path.join(trajectory_dir, f"{name}{suffix}_data.txt"))


def _write_run_summary(state, path):
    config = state.config
    with open(path, 'w') as f:
        f.write("Retargeting Timing and Hyperparameters\n")
        f.write(f"{'=' * 40}\n\n")
        f.write(f"Config: {config.config_path}\n")
        f.write(f"Agent: {state.agent}\n")
        f.write(f"Task: {config.task}\n\n")
        f.write("Timing\n")
        f.write(f"{'-' * 40}\n")
        f.write(f"Retargeting time: {state.retargeting_elapsed_time:.2f} seconds\n\n")
        f.write("Hyperparameters\n")
        f.write(f"{'-' * 40}\n")
        f.write(f"learning_rate: {config.learning_rate}\n")
        f.write(f"n_iter: {config.n_iter}\n")
        f.write(f"first_frame_iter: {config.first_frame_iter}\n")
        f.write(f"boundary_radius: {config.boundary_radius}\n")
        f.write(f"hand_boundary_radius: {config.hand_boundary_radius}\n")
        f.write(f"barrier_weight: {config.barrier_weight}\n")
        f.write(f"barrier_margin: {config.barrier_margin}\n")
        f.write(f"barrier_n: {config.barrier_n}\n")
        f.write(f"loss_threshold: {config.loss_threshold}\n")
        f.write(f"extra_pt_count: {config.extra_pt_count}\n")
        f.write(f"optimization_device: {config.optimization_device}\n")
        f.write(f"frames: {state.frames}\n")
        f.write(f"rotation: {config.rotation}\n")


# Hand-warp strategies are resolved lazily to keep optional heavy imports local.
def _get_strategy(name):
    if name == "contact_based":
        from trajwarp.hand_warp.strategies.contact_based import warp_hand
    elif name == "naive_spline":
        from trajwarp.hand_warp.strategies.naive_spline import warp_hand
    else:
        raise ValueError(f"Unknown hand-warp strategy '{name}'")
    return warp_hand


def run(config_path, strategy="contact_based", initial=False, view=True,
        loss_threshold=None):
    """Run the full retargeting pipeline for a single config file.

    ``loss_threshold`` overrides the config's contact loss threshold (which also
    selects the ``final_trajectories_<threshold>/`` output directory), letting a
    sweep reproduce results at multiple thresholds.
    """
    config = load_config(config_path)
    config.config_path = config_path
    if loss_threshold is not None:
        config.loss_threshold = loss_threshold

    warp_hand = _get_strategy(strategy)
    suffix = "" if strategy == "contact_based" else "_naive"

    state = load_demonstration(config, initial=initial)

    start_time = time.time()
    warp_object(state)
    print(f"Using device for optimization: {config.optimization_device}")
    warp_hand(state)
    state.retargeting_elapsed_time = time.time() - start_time

    metrics = compute_metrics(state)
    save_outputs(state, metrics, suffix=suffix)

    if view:
        from trajwarp.viz.viewer import view_result
        view_result(state)

    return state
