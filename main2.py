import numpy as np
import cv2
from collections import deque
import HandTrackingModule as htm
from detection import CharacterDetector
from tensorflow import keras

# Default trackbar function
def setValues(x):
    print('')

tipIds = [4, 8, 12, 16, 20]

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
vpoints = [deque(maxlen=1024)]

black_index = 0
green_index = 0
red_index = 0
voilet_index = 0

kernel = np.ones((5, 5), np.uint8)

colors = [(0, 0, 0), (255,0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

paintWindow = np.zeros((471, 636, 3)) + 0xFF

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.75)
det = CharacterDetector(loadFile="model_eng_alphabets.h5")

while True:
    (ret, frame) = cap.read(0)
    frame = cv2.flip(frame, 1)
    img = detector.findHands(frame)
    lmList = detector.findPosition(img, draw=False)
    fingers = []

    if len(lmList) != 0:
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

    frame = cv2.circle(frame,(40,90), 20, (255,255,255),-1)
    frame = cv2.circle(frame,(40,140), 20, (0,0,0),-1)
    frame = cv2.circle(frame,(40,190),20,(255,0,0),-1)
    frame = cv2.circle(frame,(40,240), 20, (0,255,0),-1)
    frame = cv2.circle(frame,(40,290), 20, (0,0,255),-1)
    frame = cv2.rectangle(frame, (520,1), (630,65), (0,0,0), -1)

    cv2.putText(
        frame,
        'C',
        (32,94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,0,0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "Recognise", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

    center = None

    if len(lmList) != 0 and totalFingers==1:
        lst = lmList[tipIds[fingers.index(1)]]
        x,y = lst[1],lst[2]

        cv2.circle(frame, (x,y), int(20), (0, 0xFF, 0xFF), 2)
        center = (x, y)

        if center[0] <= 60:
            if 70 <= center[1] <= 110:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                vpoints = [deque(maxlen=512)]

                black_index = 0
                green_index = 0
                red_index = 0
                voilet_index = 0

                paintWindow[:, :, :] = 0xFF
            elif 120 <= center[1] <= 160:
                colorIndex = 0  # Black
            elif 170 <= center[1] <= 210:
                colorIndex = 1  # Voilet
            elif 220 <= center[1] <= 260:
                colorIndex = 2  # Green
            elif 270 <= center[1] <= 310:
                colorIndex = 3  # Red
        elif 520< center[0] < 630 and 1 < center[1] < 65:
            if bpoints[0]:  # Check if bpoints[0] is not empty
                x_min = min([point[0] for point in bpoints[0]])
                x_max = max([point[0] for point in bpoints[0]])
                y_min = min([point[1] for point in bpoints[0]])
                y_max = max([point[1] for point in bpoints[0]])

                cropped_character = paintWindow[y_min:y_max, x_min:x_max]

                cv2.imwrite("cropped_character.jpg", cropped_character)

                print(det.predict("cropped_character.jpg"))
            else:
                print("No points to crop.")

        else:
            if colorIndex == 0:
                bpoints[black_index].appendleft(center)
            elif colorIndex == 1:
                vpoints[voilet_index].appendleft(center)
            elif colorIndex == 2:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 3:
                rpoints[red_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        black_index += 1
        vpoints.append(deque(maxlen=512))
        voilet_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1

    points = [bpoints, vpoints, gpoints, rpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] \
                    is None:
                    continue

                cv2.line(frame, points[i][j][k - 1], points[i][j][k],
                         colors[i], 20)
                cv2.line(paintWindow, points[i][j][k - 1],
                         points[i][j][k], colors[i], 20)

    cv2.imshow('Tracking', frame)
    cv2.imshow('Paint', paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
