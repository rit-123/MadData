import cv2
import numpy

input_path= "./videos/lunges/lunge1.mp4"
cap = cv2.VideoCapture(input_path)

while (cap.isOpened()):
    ret, frame = cap.read()
    print(frame, ret)
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        break


cap.release()
cv2.destroyAllWindows()


