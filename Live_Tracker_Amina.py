import cv2
import mediapipe as mp
import csv
import numpy as np

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")


# Set up the output CSV file path
output_csv = 'C:/Users/User/OneDrive/Desktop/Uni/Uni HW/pose_landmarks.csv'

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, smooth_segmentation=True)

# Open the default camera
cap = cv2.VideoCapture(0)

frame_number = 0
csv_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)


    # Overlay the segmentation mask on the frame
    if result.segmentation_mask is not None:
        condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
        overlay = frame.copy()
       # Create an overlay with green color for the segmentation mask
        overlay[..., 1] = np.where(condition[..., 1], 255, overlay[..., 1])  # Green channel
        overlay[..., 0] = np.where(condition[..., 0], 0, overlay[..., 0])    # Blue channel
        overlay[..., 2] = np.where(condition[..., 2], 0, overlay[..., 2])    # Red channel


    # Blend the original frame and the overlay
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)


    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data)

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

# Save the CSV data to a file
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
    csv_writer.writerows(csv_data)
