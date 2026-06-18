import mujoco
import mujoco.viewer
import numpy as np
import time

def load_and_visualize_xml(xml_file_path):
    model = mujoco.MjModel.from_xml_path(xml_file_path)
    data = mujoco.MjData(model)
    
    print(f"Successfully loaded model from: {xml_file_path}")
    print(f"Model has {model.nbody} bodies")
    print(f"Model has {model.njnt} joints") 
    print(f"Model has {model.ngeom} geometries")
    
    mujoco.mj_resetData(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            
            time.sleep(0.01)

if __name__ == "__main__":
    xml_file = "scenes/kitchen.xml"
    xml_file = "scene.xml"
    # xml_file = "env.xml"
    
    print("Launching interactive viewer...")
    load_and_visualize_xml(xml_file)