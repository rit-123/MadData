'''
    PROCESSING OF LUNGE VIDEOS
'''

import cv2
import numpy as np
import mediapipe as mp
import math
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192)  # gray

output_file_name = "lunge_video_1.mp4"
output_width = 750
output_height = 480
output_fps = 15

prev_knee_1 = 999.9
prev_knee_2 = 999.9
down_count = 0
CRITICAL_DOWN_COUNT = 7
is_going_down = True
critical_angles = []
critical_hit_counter = 1

# 1 - left up
# 2 - left down
# 3 - right up
# 4 - right down
def getCategory(leftX, rightX, leftY, rightY):
    # right up
    if (leftX < rightX and abs(leftY - rightY) < 0.1):
        return 3
    # right down
    elif (leftX < rightX and abs(leftY - rightY) >= 0.1):
        return 4

    # left up
    elif (leftX > rightX and abs(leftY - rightY) < 0.1):
        return 1
    elif (leftX > rightX and abs(leftY - rightY) >= 0.1):
        return 2

def critical(knee_1, knee_2):
    global prev_knee_1, prev_knee_2, CRITICAL_DOWN_COUNT, down_count, is_going_down
    if (knee_1 - prev_knee_1 > 0) and (knee_2 - prev_knee_2 > 0):
        prev_knee_1 = knee_1
        prev_knee_2 = knee_2
        down_count += 1
        if down_count == CRITICAL_DOWN_COUNT:
            down_count = 0
            is_going_down = not is_going_down
            return True
        return False
    else:
        prev_knee_1 = knee_1
        prev_knee_2 = knee_2
        return False



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
# with mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=2,
#     enable_segmentation=True,
#     min_detection_confidence=0.5) as pose:

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_file_name, fourcc, output_fps, (output_width, output_height))

#     for video_idx, video_file in enumerate(VIDEO_FILES):
#         cap = cv2.VideoCapture(video_file)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             image_height, image_width, _ = frame.shape

#             # Convert the BGR image to RGB before processing.
#             results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             if not results.pose_landmarks:
#                 continue

#             # Checks to see if it is a critical point
#             if not critical(results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[26].y):
#                 continue
#             print("critical point")
#             # Track the specified landmark indices
#             landmarks_11_12_13 = [results.pose_landmarks.landmark[11],
#                                    results.pose_landmarks.landmark[12],
#                                    results.pose_landmarks.landmark[13]]
#             landmarks = [results.pose_landmarks.landmark[i] for i in range(1, 32)]
#             angles_to_calculate = [(23,25,27), (24,26,28), (11,23,25), (12,24,26), (25,27,31), (26,28,32)]
#             POINTS = []

#             k = 0
#             for landmark in landmarks:
#                 x = int(landmark.x * image_width)
#                 y = int(landmark.y * image_height)

#                 # Draw a circle at the landmark position
#                 cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Red circle with a radius of 5
#                 POINTS.append((x, y))
#                 if (k < 3):
#                     k += 1
#                     continue
#                 cv2.line(frame, (x,y), POINTS[len(POINTS)-3], (0, 0, 255), 2)

#             for i in angles_to_calculate:
#                 angle = getAngle(i[0], i[1], i[2])
#                 critical_angles.append(angle)
#                 cv2.putText(frame, str(round(angle, 2)), (int(results.pose_landmarks.landmark[i[1]].x * image_width), int(results.pose_landmarks.landmark[i[1]].y * image_height)), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             category = getCategory(results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[26].y)
#             critical_angles.append(category)
#             critical_angles = []
#             extra_points = [(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12]), (results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[24])]
#             for i in extra_points:
#                 x1 = int(i[0].x * image_width)
#                 y1 = int(i[0].y * image_height)
#                 x2 = int(i[1].x * image_width)
#                 y2 = int(i[1].y * image_height)
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             # for i in POINTS:
#             #     for j in POINTS:
#             #         if (i != j):
#             #             cv2.line(frame, i, j, (0, 0, 255), 2)
                
#             out.write(cv2.resize(frame, (output_width, output_height)))
#             cv2.putText(frame, "Critical Hit: " + str(critical_hit_counter), (50, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             critical_hit_counter += 1
#             # Draw segmentation on the image.
#             # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#             # bg_image = np.zeros(frame.shape, dtype=np.uint8)
#             # bg_image[:] = BG_COLOR
#             # annotated_image = np.where(condition, frame, bg_image)

#             # Display the annotated frame
#             cv2.imshow('Annotated Frame', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#####################################################################################################
#####################################################################################################
        ############################################################################################
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image_height, image_width, _ = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks is None:
        print("No person detected")
        continue

    # Checks to see if it is a critical point
    if not critical(results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[26].y):
        continue
    print("critical point")

    # Track the specified landmark indices
    landmarks = [results.pose_landmarks.landmark[i] for i in range(1, 32)]
    angles_to_calculate = [(23,25,27), (24,26,28), (11,23,25), (12,24,26), (25,27,31), (26,28,32)]
    POINTS = []
    k = 0
    # for landmark in landmarks:
    #     x = int(landmark.x * image_width)
    #     y = int(landmark.y * image_height)

    #     # Draw a circle at the landmark position
    #     cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Red circle with a radius of 5
    #     POINTS.append((x, y))
    #     if (k < 3):
    #         k += 1
    #         continue
    #     cv2.line(image, (x,y), POINTS[len(POINTS)-3], (0, 0, 255), 2)


    #     
    for i in angles_to_calculate:
            angle = getAngle(i[0], i[1], i[2])
            critical_angles.append(angle)
            cv2.putText(image, str(round(angle, 2)), (int(results.pose_landmarks.landmark[i[1]].x * image_width), int(results.pose_landmarks.landmark[i[1]].y * image_height)), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()