import numpy as np
import trimesh

def create_bin(outer_width=1.0, outer_length=1.0, outer_height=1.0, wall_thickness=0.05):
    # Create outer box - Trimesh default has origin at center of box
    outer_box = trimesh.creation.box(
        extents=[outer_width, outer_height, outer_length]  # Swap height to Y-axis
    )
    
    # Create inner box (smaller to create wall thickness)
    inner_width = outer_width - 2 * wall_thickness
    inner_height = outer_height  # Full height for Y direction
    inner_length = outer_length - 2 * wall_thickness
    
    inner_box = trimesh.creation.box(
        extents=[inner_width, inner_height, inner_length]
    )
    
    # Move inner box up in Y direction to leave solid bottom
    inner_box.apply_translation([0, wall_thickness/2, 0])
    
    # Boolean difference to create hollow bin with walls
    bin_mesh = outer_box.difference(inner_box)
    
    # Ensure the mesh is watertight
    bin_mesh.fill_holes()
    bin_mesh.fix_normals()
    
    # Check if mesh is watertight
    if bin_mesh.is_watertight:
        print("Successfully created watertight bin!")
    else:
        print("Warning: Bin mesh is not watertight!")
    
    return bin_mesh

def create_basket(radius=0.5, height=0.7, wall_thickness=0.05, segments=32):
    # Create outer cylinder with Y-axis orientation
    # Default is Z-axis, so rotate 90 degrees around X-axis
    rotation = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    outer_cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=segments,
        transform=rotation
    )
    
    # Create inner cylinder (smaller to create wall thickness)
    inner_radius = radius - wall_thickness
    inner_height = height
    
    inner_cylinder = trimesh.creation.cylinder(
        radius=inner_radius,
        height=inner_height,
        sections=segments,
        transform=rotation
    )
    
    # Move inner cylinder up in Y direction to leave solid bottom
    inner_cylinder.apply_translation([0, wall_thickness, 0])
    
    # Boolean difference to create hollow basket with walls
    basket_mesh = outer_cylinder.difference(inner_cylinder)
    
    # Ensure the mesh is watertight
    basket_mesh.fill_holes()
    basket_mesh.fix_normals()
    
    # Check if mesh is watertight
    if basket_mesh.is_watertight:
        print("Successfully created watertight basket!")
    else:
        print("Warning: Basket mesh is not watertight!")
        
    return basket_mesh

# # Create and export a rectangular bin
# bin_mesh = create_bin(
#     outer_width=1.0, 
#     outer_length=1.0, 
#     outer_height=0.7, 
#     wall_thickness=0.05
# )
# bin_mesh.export('scene/bin.obj')

# # Create and export a cylindrical basket
# basket_mesh = create_basket(
#     radius=0.1,
#     height=0.16,
#     wall_thickness=0.01
# )
# basket_mesh.export('scene/basket.obj')