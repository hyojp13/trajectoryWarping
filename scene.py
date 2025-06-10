# import os
# import numpy as np
# from robosuite.models.arenas import TableArena

# def create_and_export_simple_scene():
#     arena = TableArena(
#         table_full_size=(0.8, 0.8, 0.05),
#         table_friction=(1.0, 0.005, 0.0001)
#     )
#     xml_content = arena.get_xml()
    

#     with open("scene.xml", "w") as f:
#         f.write(xml_content)


# if __name__ == "__main__":
#     create_and_export_simple_scene()

from robocasa.utils.env_utils import create_env
from robocasa.models.scenes.scene_registry import LayoutType, StyleType

env = create_env(
    env_name="PnPCounterToCab",
    layout_ids=[LayoutType.ONE_WALL_SMALL, LayoutType.L_SHAPED_LARGE, LayoutType.WRAPAROUND],
    style_ids=[StyleType.COASTAL, StyleType.FARMHOUSE, StyleType.RUSTIC],
)

model = env.sim.model
xml_content = model.get_xml()

with open("scene.xml", "w") as f:
    f.write(xml_content)
