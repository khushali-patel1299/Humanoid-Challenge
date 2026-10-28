import mujoco
import mujoco.viewer

# Load the robot model and physics data
# scene.xml includes the robot, a floor, and lighting
model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml") 
data = mujoco.MjData(model)

# Launch the interactive 3D viewer
mujoco.viewer.launch(model, data)