import mujoco

# Load the robot model
model = mujoco.MjModel.from_xml_path("unitree_g1/scene_with_hands.xml")

print("--- Unitree G1 Joint Names ---")
# Loop through all joints in the robot and print their names
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    # We only want to print joints that sound like they belong to a hand/arm
    if joint_name and ("hand" in joint_name.lower() or "finger" in joint_name.lower() or "wrist" in joint_name.lower() or "thumb" in joint_name.lower() or "index" in joint_name.lower() or "middle" in joint_name.lower() or "ring" in joint_name.lower() or "pinky" in joint_name.lower()):
        print(f"Joint ID {i}: {joint_name}")