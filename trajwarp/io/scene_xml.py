import os
import shutil
import xml.etree.cElementTree as ET

import numpy as np
import trimesh


def build_env_xml(agent_name, task_name, out_path="env.xml"):
    """Write a minimal hand+task MuJoCo model that includes the task object and
    the agent's assets/actuators/body. Used by the metrics pipeline, which needs
    a lightweight model without the full kitchen scene."""
    root = ET.Element("mujoco", model=f"{agent_name} {task_name}")
    ET.SubElement(root, "compiler", meshdir="meshes/", texturedir="meshes/")

    ET.SubElement(root, "include", file=f"tasks/{task_name}.xml")
    ET.SubElement(root, "include", file=f"agents/{agent_name}/assets.xml")
    ET.SubElement(root, "include", file=f"agents/{agent_name}/actuators.xml")

    world_body = ET.SubElement(root, "worldbody")
    ET.SubElement(world_body, "include", file=f"agents/{agent_name}/body.xml")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(out_path)


def write_object_xml(object_mesh_file, out_path="tasks/object.xml", mesh_dir="meshes"):
    """Write the mocap object include (``tasks/object.xml``) pointing at the given
    object mesh, copying the mesh into ``meshes/`` if it is not already there."""
    mesh_basename = os.path.basename(object_mesh_file)
    object_xml_content = f"""<mujoco>
    <asset>
      <mesh name="object_mesh" file="{mesh_basename}" scale="1 1 1"/>
    </asset>

    <worldbody>
      <body name="object" mocap="true">
        <geom type="mesh" mesh="object_mesh" group="6" rgba="0.8 0.6 0.4 1" mass="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(object_xml_content)

    mesh_dest = os.path.join(mesh_dir, mesh_basename)
    if not os.path.exists(mesh_dest):
        os.makedirs(mesh_dir, exist_ok=True)
        shutil.copy(object_mesh_file, mesh_dest)

# def offset_mesh(obj_filepath, radius):

#     mesh = o3d.io.read_triangle_mesh(obj_filepath)
#     mesh.compute_vertex_normals()

#     # Compute vertex normals
#     mesh.compute_vertex_normals()
#     vertices = np.asarray(mesh.vertices)
#     normals = np.asarray(mesh.vertex_normals)

#     # Move each vertex outward by distance r
#     new_vertices = vertices + radius * normals
#     new_mesh = o3d.geometry.TriangleMesh()
#     new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
#     new_mesh.triangles = mesh.triangles
#     new_mesh.compute_vertex_normals()

#     o3d.io.write_triangle_mesh(obj_filepath, new_mesh)

def scale_obj(input_file, object_radius):
    mesh = trimesh.load(input_file)
    center = mesh.centroid

    vertices_centered = mesh.vertices - center
    distances = np.linalg.norm(vertices_centered, axis=1)
    
    # find the maximum distance from center
    max_distance = np.max(distances)
    
    scale = 1 + (object_radius / max_distance)
    
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    mesh.apply_translation(center)
    
    mesh.export(input_file)

def shift_obj(input_file, offset):
    x, y, z = offset
    new_offset = (-x, z, y)
    
    mesh = trimesh.load(input_file)
    mesh.apply_translation(new_offset)
    mesh.export(input_file)

def soften_box(box_mesh, radius):
    """
    Take an extruded box mesh and round its edges/corners in-place.
    Subdivides the mesh, then projects each vertex onto the rounded-box surface.

    Args:
        box_mesh: trimesh.Trimesh of a box (centered at origin)
        radius:   rounding radius
    
    Returns:
        trimesh.Trimesh with softened edges
    """
    r = float(radius)
    if r < 1e-8:
        return box_mesh

    # Half-extents of the extruded box
    H = np.array(box_mesh.extents) / 2.0
    # Inner half-extents (the flat region before rounding starts)
    h = H - r

    # Subdivide so we have enough vertices at edges/corners
    max_edge = r * 0.5
    verts, faces = trimesh.remesh.subdivide_to_size(
        box_mesh.vertices, box_mesh.faces, max_edge=max_edge
    )

    # For each vertex, project onto the rounded-box surface:
    #   clamped = clamp(v, -h, h)   (nearest point on inner box)
    #   delta   = v - clamped
    #   if |delta| > 0: v_new = clamped + normalize(delta) * r
    #   else:           v_new = v      (on a flat face, keep as-is)
    clamped = np.clip(verts, -h, h)
    delta = verts - clamped
    dist = np.linalg.norm(delta, axis=1, keepdims=True)

    # Mask for vertices near an edge or corner (not on a flat face)
    mask = (dist > 1e-10).flatten()
    verts[mask] = clamped[mask] + (delta[mask] / dist[mask]) * r

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()
    return mesh

def add_mesh_barriers_to_xml(scene_file, barriers, visuals=None):
    """
    Add mesh barriers and visuals to MuJoCo XML string.

    Args:
        scene_file: Path to the XML scene file
        barriers: List of barriers (already processed with process_barriers)
        visuals: List of visual-only objects (same format as barriers but no collision)

    Returns:
        xml_string: Modified XML string with mesh barriers and visuals added
    """
    import os
    from scipy.spatial.transform import Rotation as R

    with open(scene_file) as f:
        xml_string = f.read()

    # Add barriers
    for i, barrier in enumerate(barriers):
        if isinstance(barrier, str):
            # Path to your .obj file
            mesh_file = barrier.split('/')[1]
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

    # Add visuals
    if visuals:
        for i, visual in enumerate(visuals):
            if isinstance(visual, tuple) and len(visual) >= 2:
                mesh_path = visual[0]
                config = visual[1]

                # Extract mesh file name
                mesh_file = os.path.basename(mesh_path)
                mesh_name = f"visual_mesh_{i}"

                # Get position and scale
                pos = config.get('pos', [0, 0, 0])
                scale = config.get('scale', [1, 1, 1])
                rotation = config.get('rotation', [0, 0, 0])  # euler angles in degrees
                rgba = config.get('rgba', [0.8, 0.8, 0.8, 1])

                # Convert euler angles to quaternion
                if rotation != [0, 0, 0]:
                    rot = R.from_euler('xyz', np.radians(rotation))
                    quat = rot.as_quat()  # [x, y, z, w]
                    quat_str = f'quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}"'
                else:
                    quat_str = ""

                mesh_asset = f"""
          <mesh name="{mesh_name}" file="{mesh_file}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      """

                mesh_body = f"""
          <body name="visual_body_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" {quat_str}>
              <geom type="mesh" mesh="{mesh_name}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" contype="0" conaffinity="0" group="1"/>
          </body>
      """

                if "</asset>" in xml_string:
                    xml_string = xml_string.replace("</asset>", mesh_asset + "\n</asset>")
                else:
                    raise ValueError("No <asset> block found in XML.")

                if "</worldbody>" in xml_string:
                    xml_string = xml_string.replace("</worldbody>", mesh_body + "\n</worldbody>")
                else:
                    raise ValueError("No <worldbody> block found in XML.")

    return xml_string

def generate_barrier(object_radius, shape, config={}, path=None):
    if path is None:
        raise ValueError("generate_barrier requires an output 'path'")

    if shape == 'sphere':
        assert(len(config) == 2) # pos + radius
        c = config['pos']

        sphere = trimesh.creation.icosphere(radius=object_radius + config['rad'], subdivisions=4)
        # sphere = trimesh.creation.icosphere(radius=config['rad'], subdivisions=4)

        sphere.apply_translation([-c[0], c[2], c[1]])

        sphere.export(path)

    if shape == 'rect':
        assert(len(config) == 2) # center + (l, w, h)

        dims = [dim + object_radius*2 for dim in config['dims']]  # (l, w, h)
        c = config['pos']

        box = trimesh.creation.box(extents=[dims[0], dims[2], dims[1]])
        box = soften_box(box, object_radius * 0.5)

        box.apply_translation([-c[0], c[2], c[1]])
        box.export(path)