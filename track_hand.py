import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the new Tasks API Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 2. Hand connections map for drawing the skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

# 3. Process the video
cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to MediaPipe's Image object format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run the detection
    detection_result = detector.detect(mp_image)

    # 4. Draw the results manually
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, c = frame.shape
            points = []
            
            # Convert normalized coordinates (0 to 1) to actual pixel coordinates
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Draw joints
            
            # Draw the bones
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    cv2.imshow('Hand Tracking (Latest API)', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()