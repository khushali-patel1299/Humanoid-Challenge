import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the new Tasks API Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Open the video and prepare the CSV file
cap = cv2.VideoCapture('test_video.mp4')
csv_filename = 'hand_coordinates.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Create headers for 21 joints (x, y, z for each)
    headers = ['frame']
    for i in range(21):
        headers.extend([f'j{i}_x', f'j{i}_y', f'j{i}_z'])
    writer.writerow(headers)

    frame_count = 0
    print(f"Extracting coordinates to {csv_filename}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to MediaPipe's format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run the detection
        detection_result = detector.detect(mp_image)

        # Save coordinates if a hand is detected
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                row = [frame_count]
                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z]) # 3D math!
                writer.writerow(row)
        
        frame_count += 1

cap.release()
print(f"Extraction complete! Saved to {csv_filename}.")