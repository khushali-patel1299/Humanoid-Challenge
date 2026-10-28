import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("test_paper.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)