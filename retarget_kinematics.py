import mujoco
import mujoco.viewer
import csv
import numpy as np
import time

# Helper function to calculate the bend angle of a finger
def get_bend_angle(wrist, base, tip):
    v1 = np.array(wrist) - np.array(base)
    v2 = np.array(tip) - np.array(base)
    # Calculate the angle between the two vectors
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle

# 1. Load the model with hands
model = mujoco.MjModel.from_xml_path("unitree_g1/grasp_scene.xml")
data = mujoco.MjData(model)

# 2. Find the IDs for the right hand joints
# (Assuming you recorded with your right hand; if left, we can easily swap these names)
joints = {
    "thumb_0": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_thumb_0_joint"),
    "thumb_1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_thumb_1_joint"),
    "thumb_2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_thumb_2_joint"),
    "index_0": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_index_0_joint"),
    "index_1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_index_1_joint"),
    "middle_0": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_middle_0_joint"),
    "middle_1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_middle_1_joint"),
}

# 3. Read the CSV data
frames = []
with open('hand_coordinates.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader) # Skip headers
    for row in reader:
        coords = np.array([float(x) for x in row[1:]]).reshape(21, 3)
        frames.append(coords)

print(f"Loaded {len(frames)} frames. Launching MuJoCo...")

# 4. Launch Viewer and Animate
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    print("Simulation starting in 3 seconds... Get ready!")
    time.sleep(10) # Delay before the animation starts
    
    for frame_coords in frames:
        
        # Extract MediaPipe points
        wrist = frame_coords[0]
        thumb_base, thumb_tip = frame_coords[2], frame_coords[4]
        index_base, index_tip = frame_coords[5], frame_coords[8]
        middle_base, middle_tip = frame_coords[9], frame_coords[12]

        # Calculate heuristic bend angles
        thumb_bend = get_bend_angle(wrist, thumb_base, thumb_tip)
        index_bend = get_bend_angle(wrist, index_base, index_tip)
        middle_bend = get_bend_angle(wrist, middle_base, middle_tip)
        
        # Apply angles to the robot's joints (with tuned gains)
        data.qpos[model.jnt_qposadr[joints["thumb_0"]]] = -thumb_bend * 0.1
        data.qpos[model.jnt_qposadr[joints["thumb_1"]]] = -thumb_bend * 0.4
        data.qpos[model.jnt_qposadr[joints["thumb_2"]]] = -thumb_bend * 0.3
        
        data.qpos[model.jnt_qposadr[joints["index_0"]]] = index_bend * 0.05
        data.qpos[model.jnt_qposadr[joints["index_1"]]] = index_bend * 0.6
        
        data.qpos[model.jnt_qposadr[joints["middle_0"]]] = middle_bend * 0.05
        data.qpos[model.jnt_qposadr[joints["middle_1"]]] = middle_bend * 0.6

        # Update the graphics without running physics
        mujoco.mj_kinematics(model, data)
        viewer.sync()
        
        time.sleep(0.03) # Playback at roughly 30 FPS

    print("Playback finished! You can now inspect the hand. Close the window when ready.")
    
    # Keep the window open until you manually close it
    while viewer.is_running():
        time.sleep(0.1) # Prevents the CPU from maxing out while idling