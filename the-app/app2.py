import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Array of MP4 file paths
VIDEO_FILES = ["/", "/"]

BG_COLOR = (192, 192, 192)  # gray

# Initialize Mediapipe Pose model
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    for video_idx, video_file in enumerate(VIDEO_FILES):
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_height, image_width, _ = frame.shape

            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue

            # Track the specified landmark indices
            landmarks_11_12_13 = [results.pose_landmarks.landmark[11],
                                   results.pose_landmarks.landmark[12],
                                   results.pose_landmarks.landmark[13]]

            for landmark in landmarks_11_12_13:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)

                # Draw a circle at the landmark position
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Red circle with a radius of 5

            # Draw segmentation on the image.
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, frame, bg_image)

            # Display the annotated frame
            cv2.imshow('Annotated Frame', annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
