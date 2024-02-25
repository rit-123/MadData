import cv2
import numpy as np
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Array of MP4 file paths
VIDEO_FILES = ["the-app/Lunge1.mp4"]

BG_COLOR = (192, 192, 192)  # gray

output_file_name = "lunge_video_1.mp4"
output_width = 640
output_height = 480
output_fps = 15

# l1, l2 and l3 are the POINTS of which the angle is to be calculated
def getAngle(l1, l2, l3):
    l1 = (results.pose_landmarks.landmark[l1].x * image_width, results.pose_landmarks.landmark[l1].y * image_height)
    l2 = (results.pose_landmarks.landmark[l2].x * image_width, results.pose_landmarks.landmark[l2].y * image_height)
    l3 = (results.pose_landmarks.landmark[l3].x * image_width, results.pose_landmarks.landmark[l3].y * image_height)
    numerator = (l3[0] - l2[0]) * (l1[0] - l2[0]) + (l3[1] - l2[1]) * (l1[1] - l2[1])
    denominator = math.sqrt((l3[0] - l2[0])**2 + (l3[1] - l2[1])**2) * math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
    angle = math.acos(numerator / denominator)
    return math.degrees(angle)

# Initialize Mediapipe Pose model
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_name, fourcc, output_fps, (output_width, output_height))

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
            # landmarks_11_12_13 = [results.pose_landmarks.landmark[11],
            #                        results.pose_landmarks.landmark[12],
            #                        results.pose_landmarks.landmark[13]]
            landmarks = [results.pose_landmarks.landmark[i] for i in range(1, 32)]
            angles_to_calculate = [(23,25,27), (24,26,28), (11,23,25), (12,24,26), (25,27,31), (26,28,32)]
            POINTS = []

            k = 0
            for landmark in landmarks:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)

                # Draw a circle at the landmark position
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Red circle with a radius of 5
                POINTS.append((x, y))
                if (k < 3):
                    k += 1
                    continue
                cv2.line(frame, (x,y), POINTS[len(POINTS)-3], (0, 0, 255), 2)
            
            for i in angles_to_calculate:
                angle = getAngle(i[0], i[1], i[2])
                #print(angle)
                cv2.putText(frame, str(int(angle)), (int(results.pose_landmarks.landmark[i[1]].x * image_width), int(results.pose_landmarks.landmark[i[1]].y * image_height)), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA)

            extra_points = [(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12]), (results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[24])]
            for i in extra_points:
                x1 = int(i[0].x * image_width)
                y1 = int(i[0].y * image_height)
                x2 = int(i[1].x * image_width)
                y2 = int(i[1].y * image_height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # for i in POINTS:
            #     for j in POINTS:
            #         if (i != j):
            #             cv2.line(frame, i, j, (0, 0, 255), 2)
                
            out.write(cv2.resize(frame, (output_width, output_height)))

            # Draw segmentation on the image.
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # bg_image = np.zeros(frame.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, frame, bg_image)

            # Display the annotated frame
            cv2.imshow('Annotated Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()