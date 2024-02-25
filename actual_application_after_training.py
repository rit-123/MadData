'''
    PROCESSING OF LUNGE VIDEOS
'''

import cv2
import numpy as np
import mediapipe as mp
import math
import csv
import pandas as pd

class exercise_model:
    def __init__(self, fileName) -> None:
        self.df = pd.read_csv(fileName)
        self.df = self.df.iloc[1:]
        self.averages = self.df.mean(axis=1)
        self.std_devs = self.df.std(axis=1)
    
    def getError(self,dfToCheck):
        error_rates = []
        for column in range(len(self.df.columns)):
            error_rate = self.std_devs.iloc[int(column)] / self.averages.iloc[int(column)]
            error_rates.append(error_rate)
        return error_rates
    
    def classifyStates(self, dfToCheck):
        # array1 = self.df[self.df[:6] == 1].iloc[:, :-1].values
        # array2 = self.df[self.df[:6] == 2].iloc[:, :-1].values
        # array3 = self.df[self.df[:6] == 3].iloc[:, :-1].values
        # array4 = self.df[self.df[:6] == 4].iloc[:, :-1].values

        array1 = []
        array2 = []
        array3 = []
        array4 = []


        with open("lungesClassified.csv", mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                for i in range(len(row) - 1):
                    row[i] = float(row[i])
                    row[6] = int(row[6])
            if row[6] == 1:
                array1.append(row)
            elif row[6] == 2:
                array2.append(row)
            elif row[6] == 3:
                array3.append(row)
            elif row[6] == 4:
                array4.append(row)

        
        MEANS = {1: [0,0,0,0,0,0], 2:[0,0,0,0,0,0], 3:[0,0,0,0,0,0], 4:[0,0,0,0,0,0]}
        STD_DEVS = {1: [[],[],[],[],[],[]], 2:[[],[],[],[],[],[]], 3:[[],[],[],[],[],[]], 4:[[],[],[],[],[],[]]}

        no_of_readings  = len(array1) + len(array2) + len(array3) + len(array4)
        # category = 1

        for i in range(len(array1)):
            for j in range(6):
            # print(i)
            # print(len(array1))
            # print(array1)
                MEANS[1][j] += array1[i][j]
                STD_DEVS[1][j] = array1[i][j]

        for i in range(len(array2)):
            for j in range(6):
            # print(i)
            # print(len(array1))
            # print(array1)
                MEANS[2][j] += array2[i][j]
                STD_DEVS[2][j] = array2[i][j]

        for i in range(len(array3)):
            for j in range(6):
            # print(i)
            # print(len(array1))
            # print(array1)
                MEANS[3][j] += array3[i][j]
                STD_DEVS[3][j] = array3[i][j]
        
        for i in range(len(array4)):
            for j in range(6):
            # print(i)
            # print(len(array1))
            # print(array1)
                MEANS[4][j] += array4[i][j]
                STD_DEVS[1][j] = array4[i][j]
        # for i in range(6):
        #     MEANS[2][i] += array1[i]
        # for i in range(6):
        #     MEANS[3][i] += array1[i]
        # for i in range(6):
        #     MEANS[4][i] += array4[i]
            
        for category in range(1, len(MEANS) + 1):
            for i in range(len(MEANS[category])):
                MEANS[category][i] = MEANS[category][i] / no_of_readings

        for category in range(1, len(STD_DEVS)+1):
            for i in range(len(MEANS[category])):
                STD_DEVS[category][i] = np.std(STD_DEVS[category][i])
        


        mean1, std_dev1 = np.mean(array1, axis=0), np.std(array1, axis=0)
        mean2, std_dev2 = np.mean(array2, axis=0), np.std(array2, axis=0)
        mean3, std_dev3 = np.mean(array3, axis=0), np.std(array3, axis=0)
        mean4, std_dev4 = np.mean(array4, axis=0), np.std(array4, axis=0)

        return MEANS, STD_DEVS

    # def evaluateData(self, means, std_devs, data):
    #     for i, (mean, std_dev, d) in enumerate(means, std_devs, data):
    #         if np.any(np.abs(d - mean) > std_dev/2):
    #             return f"Critical angle at index {i} is not in good form"
    #     return "good form"
    
    def evaluateData(self, means, std_devs, data):
        category = data[6]
        print(category)
        print(data)
        for i, d in enumerate(data):
            if i == 6:
                continue
            if abs(d - means[category][i]) > std_devs[category][i]/2:
                return f"Critical angle at index {i} is not in good form"
        return "good form"

        

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

classifier = exercise_model("lungesClassified.csv")
averages, std_devs = classifier.classifyStates(classifier.df)
print("DATA EVALUATED")

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
    global results, image_width, image_height
    l1 = (results.pose_landmarks.landmark[l1].x * image_width, results.pose_landmarks.landmark[l1].y * image_height)
    l2 = (results.pose_landmarks.landmark[l2].x * image_width, results.pose_landmarks.landmark[l2].y * image_height)
    l3 = (results.pose_landmarks.landmark[l3].x * image_width, results.pose_landmarks.landmark[l3].y * image_height)
    numerator = (l3[0] - l2[0]) * (l1[0] - l2[0]) + (l3[1] - l2[1]) * (l1[1] - l2[1])
    denominator = math.sqrt((l3[0] - l2[0])**2 + (l3[1] - l2[1])**2) * math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
    angle = math.acos(numerator / denominator)
    return math.degrees(angle)
#####################################################################################################
#####################################################################################################
############################################################################################
VIDEO_FILES = ["the-app\WIN_20240225_10_23_24_Pro.mp4"]
for video_idx, video_file in enumerate(VIDEO_FILES):
    cap = cv2.VideoCapture(video_file)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
            # If loading a video, use 'break' instead of 'continue'.
                break
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
            for landmark in landmarks:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)

                # Draw a circle at the landmark position
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Red circle with a radius of 5
                POINTS.append((x, y))
                if (k < 3):
                    k += 1
                    continue
                cv2.line(image, (x,y), POINTS[len(POINTS)-3], (0, 0, 255), 2)


            # calculating the angles
            for i in angles_to_calculate:
                    angle = getAngle(i[0], i[1], i[2])
                    critical_angles.append(angle)
                    cv2.putText(image, str(round(angle, 2)), (int(results.pose_landmarks.landmark[i[1]].x * image_width), int(results.pose_landmarks.landmark[i[1]].y * image_height)), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA)

            category = getCategory(results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[26].y)
            critical_angles.append(category)
            print(classifier.evaluateData(averages, std_devs, critical_angles))


            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    

    
cap.release()