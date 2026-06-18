"""
Signed distance functions for barrier collision avoidance during hand optimization.
Provides distance computation from hand components to various barrier primitives.
"""

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R


def sdf_sphere(points, center, radius):
    """
    Signed distance function for a sphere.

    Args:
        points: torch tensor (N, 3) - points to query
        center: torch tensor (3,) - sphere center
        radius: float - sphere radius

    Returns:
        torch tensor (N,) - signed distances (negative = inside, positive = outside)
    """
    distances_from_center = torch.norm(points - center, dim=1)
    return distances_from_center - radius


def sdf_box(points, center, dims):
    """
    Signed distance function for an axis-aligned box.

    Args:
        points: torch tensor (N, 3) - points to query
        center: torch tensor (3,) - box center
        dims: torch tensor (3,) - box dimensions (full width, height, depth)

    Returns:
        torch tensor (N,) - signed distances (negative = inside, positive = outside)
    """
    half_dims = dims / 2.0

    # Transform points to box-local coordinates
    q = torch.abs(points - center) - half_dims

    # Distance outside the box
    outside_distance = torch.norm(torch.maximum(q, torch.zeros_like(q)), dim=1)

    # Distance inside the box (negative)
    inside_distance = torch.min(torch.max(q, dim=1)[0], torch.zeros_like(q[:, 0]))

    return outside_distance + inside_distance


def sdf_mesh(points, mesh_vertices, mesh_faces, device='cpu'):
    """
    Approximate signed distance function for a triangle mesh.
    Uses closest point on mesh surface.

    Args:
        points: torch tensor (N, 3) - points to query
        mesh_vertices: numpy array (V, 3) - mesh vertices
        mesh_faces: numpy array (F, 3) - mesh face indices
        device: torch device

    Returns:
        torch tensor (N,) - approximate signed distances
    """
    points_np = points.detach().cpu().numpy()

    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)

    closest_points, distances, _ = mesh.nearest.on_surface(points_np)

    if mesh.is_watertight:
        is_inside = mesh.contains(points_np)
        distances = np.where(is_inside, -distances, distances)
    else:
        raise Exception("barrier mesh must be watertight")

    return torch.tensor(distances, dtype=torch.float32, device=device)


def load_mesh_barrier(barrier_path):
    """
    Load a mesh barrier from file.

    Args:
        barrier_path: str - path to mesh file (.obj, .stl, etc.)

    Returns:
        tuple: (vertices, faces) as numpy arrays
    """
    mesh = trimesh.load(barrier_path)
    return mesh.vertices, mesh.faces


def compute_barrier_distance(points, barrier, device='cpu'):
    """
    Compute signed distance from points to a single barrier.

    Args:
        points: torch tensor (N, 3) - points to query
        barrier: tuple - barrier specification
                 ('sphere', {'pos': [x,y,z], 'rad': r}) or
                 ('rect', {'pos': [x,y,z], 'dims': [l,w,h]}) or
                 ('mesh', {'vertices': ndarray, 'faces': ndarray, 'qpos': ndarray}) or
                 str (path to mesh file)
        device: torch device

    Returns:
        torch tensor (N,) - signed distances to barrier
    """
    if isinstance(barrier, str):
        # Mesh barrier from file (static)
        vertices, faces = load_mesh_barrier(barrier)
        return sdf_mesh(points, vertices, faces, device)
    else:
        barrier_type, config = barrier

        if barrier_type == 'sphere':
            center = torch.tensor(config['pos'], dtype=torch.float32, device=device)
            radius = config['rad']
            return sdf_sphere(points, center, radius)

        elif barrier_type == 'rect':
            center = torch.tensor(config['pos'], dtype=torch.float32, device=device)
            dims = torch.tensor(config['dims'], dtype=torch.float32, device=device)
            return sdf_box(points, center, dims)

        # used for hand-object collision where object is a non-static "barrier"
        elif barrier_type == 'mesh':
            # Dynamic mesh barrier with transformation
            vertices = config['vertices']
            faces = config['faces']
            qpos = config['qpos']  # [pos(3), quat(4) in w,x,y,z format]

            # Transform mesh vertices to global frame
            obj_pos = qpos[:3]
            obj_quat = qpos[3:]  # [w, x, y, z]

            # Convert MuJoCo quat [w,x,y,z] to SciPy quat [x,y,z,w]
            quat_scipy = np.array([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])
            rotation = R.from_quat(quat_scipy)
            rotation_matrix = rotation.as_matrix()

            # Transform vertices to global frame
            global_vertices = (rotation_matrix @ vertices.T).T + obj_pos

            return sdf_mesh(points, global_vertices, faces, device)

        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")


def compute_hand_component_vertices_torch(qpos_torch, kinematic_tree, hand_components,
                                         hand_component_body_ids,
                                         root_body_id, device):
    """
    Compute global positions of hand component vertices for barrier collision checking.

    Args:
        qpos_torch: (nq,) torch tensor with requires_grad=True
        kinematic_tree: dict from extract_kinematic_tree (with pre-converted torch tensors)
        hand_components: list of (faces, verts) tuples
        hand_component_body_ids: list of body IDs for hand components
        root_body_id: root body ID
        device: torch device

    Returns:
        dict mapping hand_component_id -> torch tensor (V, 3) of global vertex positions
    """
    from differentiable_fk import compute_forward_kinematics_torch

    # Compute forward kinematics
    transforms = compute_forward_kinematics_torch(
        qpos_torch, kinematic_tree, hand_component_body_ids, root_body_id
    )

    global_vertices = {}

    for hand_component_id in range(len(hand_component_body_ids)):
        body_id = hand_component_body_ids[hand_component_id]

        if body_id not in transforms:
            continue

        body_transform = transforms[body_id]
        body_pos = body_transform['pos']  # (3,)
        body_rot = body_transform['rot']  # (3, 3)

        # Get mesh vertices for this component
        faces, verts = hand_components[hand_component_id]

        # Convert to torch tensor
        local_vertices = torch.tensor(verts, dtype=torch.float32, device=device)

        # Transform to global: global = R @ local.T + pos
        global_verts = (body_rot @ local_vertices.T).T + body_pos.unsqueeze(0)  # (V, 3)
        global_vertices[hand_component_id] = global_verts

    return global_vertices


def compute_hand_barrier_loss(hand_component_positions, barriers, n=2.0, margin=0.05, device='cpu'):
    """
    Compute loss term to prevent hand from intersecting barriers.

    The loss function f(d, n) = (1 - d/margin)^n penalizes points close to or inside barriers:
    - When d >= margin: penalty = 0 (hand is far enough)
    - When 0 < d < margin: penalty increases smoothly from 0 to 1
    - When d = 0: penalty = 1 (hand touching barrier)
    - When d < 0: penalty > 1 and grows rapidly (hand penetrating barrier)

    Args:
        hand_component_positions: dict mapping hand_component_id -> torch tensor (M, 3)
                                 Global positions of hand segments/vertices
        barriers: list of barrier specifications (from config)
        n: float - hyperparameter controlling penalty function steepness (default: 2.0)
                  n=1: linear, n=2: quadratic, n>2: sharper near barrier
        margin: float - distance threshold in meters (default: 0.05m = 5cm)
                       Penalty applied when hand is within this distance
        device: torch device

    Returns:
        torch tensor - scalar loss value (sum of penalties across all vertices and barriers)
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Collect all hand points
    all_hand_points = []
    for component_id in hand_component_positions:
        points = hand_component_positions[component_id]
        if points.shape[0] > 0:
            all_hand_points.append(points)

    if len(all_hand_points) == 0:
        return total_loss

    all_hand_points = torch.cat(all_hand_points, dim=0)  # (N, 3)

    # Convert margin to tensor for consistent torch operations
    margin_tensor = torch.tensor(margin, dtype=torch.float32, device=device)
    n_tensor = torch.tensor(n, dtype=torch.float32, device=device)

    # Compute penalty for each barrier
    for barrier in barriers:
        # Compute signed distances from all hand points to this barrier
        distances = compute_barrier_distance(all_hand_points, barrier, device)

        # Apply penalty function f(d, n) = (1 - d/margin)^n for d < margin
        # This creates a smooth, differentiable penalty that:
        # - Is 0 at d = margin (hand is far enough from barrier)
        # - Is 1 at d = 0 (hand is touching barrier surface)
        # - Is > 1 for d < 0 (hand penetrates barrier) - increases rapidly
        # - Higher n makes penalty steeper near the barrier
        #
        # Example with n=2, margin=0.01:
        # d = 0.01: penalty = 0
        # d = 0.005: penalty = 0.25
        # d = 0: penalty = 1
        # d = -0.005: penalty = 2.25 (penetration!)
        # d = -0.01: penalty = 4

        # penalize proximity within margin
        mask = distances < margin_tensor

        # Compute normalized distance: (margin - d) / margin
        # This ranges from 0 (at margin) to 1 (at surface) to >1 (inside)
        normalized_dist = 1.0 - distances / margin_tensor

        # Apply power function - creates smooth penalty curve
        # Clamp to ensure non-negative for numerical stability
        penalties = torch.pow(torch.clamp(normalized_dist, min=0.0), n_tensor)

        # Mask out points beyond margin
        masked_penalties = penalties * mask.float()

        # Average penalty across all points (normalized by number of points)
        # Use mean instead of sum to avoid loss scaling with number of vertices
        barrier_loss = masked_penalties.sum()
        total_loss = total_loss + barrier_loss

        # print("barrier loss for", barrier, masked_penalties.sum())

    return total_loss
