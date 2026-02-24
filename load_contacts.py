"""
Load contact data from .lcexp files.
Returns an array of length n_frames, where each element is a list of contacts.
Each contact is a 3-tuple: (object_vertex_index, hand_link_index, (face_index, bary1, bary2, bary3))
"""

import json
import numpy as np
import trimesh
import os


# Cache for hand link meshes to avoid reloading
_hand_mesh_cache = {}


def get_hand_link_mesh(hand_link_idx):
    """
    Load the mesh for a given hand link index.

    Args:
        hand_link_idx: Index of the hand link in the contact file format:
            0-2: index1-3, 3-5: middle1-3, 6-8: pinky1-3,
            9-11: ring1-3, 12-14: thumb1-3, 15: wrist

    Returns:
        trimesh.Trimesh object
    """
    # For Allegro hand, use the following
    # hand_link_names = [
    #     'base_link', 'link_12_right', 'link_15_tip',
    #     'link_0', 'link_0', 'link_0',
    #     'link_1', 'link_1', 'link_1',
    #     'link_2', 'link_2', 'link_2',
    #     'link_3', 'link_3', 'link_3',
    #     'link_13', 'link_14', 'link_15',
    #     'link_3_tip', 'link_3_tip', 'link_3_tip'
    # ]
    
    # Map MANO hand link indices to mesh names
    # Based on the contact file ordering:
    # 0: index1, 1: index2, 2: index3
    # 3: middle1, 4: middle2, 5: middle3
    # 6: pinky1, 7: pinky2, 8: pinky3
    # 9: ring1, 10: ring2, 11: ring3
    # 12: thumb1, 13: thumb2, 14: thumb3
    # 15: wrist
    hand_link_names = [
        'right_index1', 'right_index2', 'right_index3',
        'right_middle1', 'right_middle2', 'right_middle3',
        'right_pinky1', 'right_pinky2', 'right_pinky3',
        'right_ring1', 'right_ring2', 'right_ring3',
        'right_thumb1', 'right_thumb2', 'right_thumb3',
        'right_wrist'
    ]

    if hand_link_idx < 0 or hand_link_idx >= len(hand_link_names):
        raise ValueError(f"Invalid hand_link_idx: {hand_link_idx}")

    mesh_name = hand_link_names[hand_link_idx]

    # Check cache first
    if hand_link_idx in _hand_mesh_cache:
        return _hand_mesh_cache[hand_link_idx]

    # Load mesh (agent type is always MANO_right)
    mesh_path = f'agents/MANO_right/geom_assets/{mesh_name}.obj'
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, process=False)
    _hand_mesh_cache[hand_link_idx] = mesh

    return mesh


def vertex_to_face_bary(vertex_idx, mesh):
    """
    Convert a vertex index to a face index with extreme barycentric coordinates.
    Finds the first face that contains the vertex and returns barycentric coordinates
    with 1.0 for the vertex and 0.0 for others.

    Args:
        vertex_idx: Index of the vertex
        mesh: trimesh.Trimesh object

    Returns:
        Tuple of (face_idx, bary1, bary2, bary3)
    """
    # Find the first face that contains this vertex
    face_idx = None
    face = None
    for idx, f in enumerate(mesh.faces):
        if vertex_idx in f:
            face_idx = idx
            face = f
            break

    if face_idx is None:
        raise ValueError(f"Vertex {vertex_idx} not found in any face (mesh has {len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")

    # Sanity check
    if face_idx >= len(mesh.faces):
        raise ValueError(f"BUG: face_idx {face_idx} is out of bounds for mesh with {len(mesh.faces)} faces")

    # Create barycentric coordinates with 1.0 at the vertex position
    vertex_position_in_face = np.where(face == vertex_idx)[0][0]
    bary_coords = [0.0, 0.0, 0.0]
    bary_coords[vertex_position_in_face] = 1.0

    return (face_idx, bary_coords[0], bary_coords[1], bary_coords[2])


def edge_to_face_bary(vertex_idx1, vertex_idx2, mesh, weight1=0.5, weight2=0.5):
    """
    Convert an edge (two vertices) to a face index with edge barycentric coordinates.
    Finds the first face that contains both vertices and returns barycentric coordinates
    with weights for the two edge vertices and 0.0 for the third.

    Args:
        vertex_idx1: Index of the first vertex
        vertex_idx2: Index of the second vertex
        mesh: trimesh.Trimesh object
        weight1: Weight for first vertex (default 0.5)
        weight2: Weight for second vertex (default 0.5)

    Returns:
        Tuple of (face_idx, bary1, bary2, bary3)
    """
    # Find faces that contain both vertices
    faces_with_edge = []
    for face_idx, face in enumerate(mesh.faces):
        if vertex_idx1 in face and vertex_idx2 in face:
            faces_with_edge.append((face_idx, face))

    if not faces_with_edge:
        raise ValueError(f"Edge ({vertex_idx1}, {vertex_idx2}) not found in any face")

    # Use the first face containing this edge
    face_idx, face = faces_with_edge[0]

    # Create barycentric coordinates
    pos1 = np.where(face == vertex_idx1)[0][0]
    pos2 = np.where(face == vertex_idx2)[0][0]

    bary_coords = [0.0, 0.0, 0.0]
    bary_coords[pos1] = weight1
    bary_coords[pos2] = weight2

    return (face_idx, bary_coords[0], bary_coords[1], bary_coords[2])


def get_contact_frame_range(contacts_by_frame):
    """
    Find the first and last frames that contain contacts.

    Args:
        contacts_by_frame: List of length n_frames, where each element is a list of contacts
                          (as returned by load_contacts_lcexp)

    Returns:
        Tuple of (first_frame, last_frame) where frames are 0-indexed.
        Returns (None, None) if no contacts are found in any frame.
    """
    first_frame = None
    last_frame = None

    for frame_idx, contacts in enumerate(contacts_by_frame):
        if contacts:  # If this frame has at least one contact
            if first_frame is None:
                first_frame = frame_idx
            last_frame = frame_idx

    return (first_frame, last_frame)


def load_contacts_lcexp(filepath):
    """
    Load contact data from a .lcexp JSON file.

    Args:
        filepath: Path to the .lcexp file

    Returns:
        contacts_by_frame: List of length n_frames, where each element is a list of contacts.
                          Each contact is a 3-tuple:
                          (object_vertex_index, hand_link_index, (face_index, bary1, bary2, bary3))
                          Empty list if no contacts for that frame.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Find the maximum frame number to determine array size
    max_frame = 0
    for frame_data in data['data']:
        if 'frame' in frame_data:
            max_frame = max(max_frame, frame_data['frame'])

    # Initialize array with empty lists
    contacts_by_frame = [[] for _ in range(max_frame)]

    # Parse each frame's contact data
    for frame_data in data['data']:
        if 'frame' not in frame_data:
            continue

        frame_num = frame_data['frame']  # 1-indexed in JSON
        frame_idx = frame_num - 1  # Convert to 0-indexed

        assert not (frame_idx < 0 or frame_idx >= len(contacts_by_frame))

        # Get contact arrays (all should have same length)
        contact_points_object = frame_data.get('contactPointsObject', [])
        hand_link_indices = frame_data.get('handLinkIndices', [])
        contact_points_hand_links = frame_data.get('contactPointsHandLinks', [])

        # Verify they all have the same length
        n_contacts = len(contact_points_object)
        if len(hand_link_indices) != n_contacts or len(contact_points_hand_links) != n_contacts:
            raise Exception(f"Warning: Frame {frame_num} has mismatched contact array lengths")

        # Parse each contact
        contacts = []
        for i in range(n_contacts):
            # Parse object vertex: "v 2155" -> 2155
            obj_vertex_str = contact_points_object[i]
            if obj_vertex_str.startswith('v '):
                obj_vertex_idx = int(obj_vertex_str[2:].strip())
            else:
                raise Exception(f"Warning: Could not parse object vertex '{obj_vertex_str}' in frame {frame_num}")

            # Get hand link index (in contact file format)
            hand_link_idx_contact_format = hand_link_indices[i]

            # Convert from contact file format to MuJoCo hand_component format
            # Contact file: 0-2: index, 3-5: middle, 6-8: pinky, 9-11: ring, 12-14: thumb, 15: wrist
            # MuJoCo hand_component: 0: wrist, 1-3: thumb, 4-6: ring, 7-9: pinky, 10-12: middle, 13-15: index
            contact_to_mujoco = {
                0: 13,  # index1
                1: 14,  # index2
                2: 15,  # index3
                3: 10,  # middle1
                4: 11,  # middle2
                5: 12,  # middle3
                6: 7,   # pinky1
                7: 8,   # pinky2
                8: 9,   # pinky3
                9: 4,   # ring1
                10: 5,  # ring2
                11: 6,  # ring3
                12: 1,  # thumb1
                13: 2,  # thumb2
                14: 3,  # thumb3
                15: 0   # wrist
            }
            hand_link_idx = contact_to_mujoco[hand_link_idx_contact_format]

            # Parse hand link contact:
            # Format 1: "f 21 0.862154 0.032906 0.104941" -> (21, 0.862154, 0.032906, 0.104941)
            # Format 2: "v 14" -> vertex index (convert to face format with extreme barycentrics)
            # Format 3: "e 14 15" -> edge indices (convert to face format with edge barycentrics)
            hand_link_str = contact_points_hand_links[i]
            if hand_link_str.startswith('f '):
                parts = hand_link_str[2:].strip().split()
                if len(parts) >= 4:
                    face_idx = int(parts[0])
                    bary1 = float(parts[1])
                    bary2 = float(parts[2])
                    bary3 = float(parts[3])
                    hand_contact_info = (face_idx, bary1, bary2, bary3)
                else:
                    raise Exception(f"Warning: Could not parse hand link contact '{hand_link_str}' in frame {frame_num}")
            elif hand_link_str.startswith('v '):
                # Vertex-only format: "v 14" means contact is on vertex 14 (0-indexed)
                # Load the hand mesh and find a face containing this vertex
                vertex_idx = int(hand_link_str[2:].strip())
                # Use contact format index to load the correct mesh
                mesh = get_hand_link_mesh(hand_link_idx_contact_format)
                hand_contact_info = vertex_to_face_bary(vertex_idx, mesh)
            elif hand_link_str.startswith('e '):
                # Edge format: "e 14 15" means contact is on edge between vertices 14 and 15 (0-indexed)
                # Load the hand mesh and find a face containing this edge
                parts = hand_link_str[2:].strip().split()
                if len(parts) >= 2:
                    vertex_idx1 = int(parts[0])
                    vertex_idx2 = int(parts[1])
                    # Parse optional weights if provided
                    weight1 = float(parts[2]) if len(parts) > 2 else 0.5
                    weight2 = float(parts[3]) if len(parts) > 3 else 0.5
                    # Use contact format index to load the correct mesh
                    mesh = get_hand_link_mesh(hand_link_idx_contact_format)
                    hand_contact_info = edge_to_face_bary(vertex_idx1, vertex_idx2, mesh, weight1, weight2)
                else:
                    raise Exception(f"Warning: Could not parse hand link contact '{hand_link_str}' in frame {frame_num}")
            else:
                raise Exception(f"Warning: Could not parse hand link contact '{hand_link_str}' in frame {frame_num}")

            # Create contact tuple
            contact = (obj_vertex_idx, hand_link_idx, hand_contact_info)
            contacts.append(contact)

        contacts_by_frame[frame_idx] = contacts

    return contacts_by_frame