import cv2
import os
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

# =======================
# SETTINGS
# =======================
width, height = 1280, 720
folderPath = "Presentation"

gestureThreshold = int(height * 0.6)  # LOWER LINE (change if needed)

# =======================
# CAMERA
# =======================
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# =======================
# HAND DETECTOR
# =======================
detector = HandDetector(detectionCon=0.8, maxHands=1)

# =======================
# LOAD SLIDES
# =======================
if not os.path.exists(folderPath):
    print("❌ Presentation folder not found!")
    exit()

pathImages = sorted(os.listdir(folderPath))
print("Loaded slides:", pathImages)

# =======================
# VARIABLES
# =======================
imgNumber = 0
buttonPressed = False
counter = 0
delay = 20

annotations = [[]]
annotationNumber = -1
annotationStart = False

# smoothing
prevX, prevY = 0, 0
smoothening = 7

# FPS
pTime = 0

# webcam overlay size
hs, ws = 150, 250

# =======================
# LOOP
# =======================
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # =======================
    # LOAD & FIT SLIDE
    # =======================
    pathFull = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFull)

    if imgCurrent is None:
        continue

    # resize + center fit (no distortion)
    h, w = imgCurrent.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    imgCurrent = cv2.resize(imgCurrent, (new_w, new_h))
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = imgCurrent
    imgCurrent = bg

    # =======================
    # HAND DETECTION
    # =======================
    hands, img = detector.findHands(img)

    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 3)

    if hands and not buttonPressed:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)

        x, y = lmList[8][0], lmList[8][1]

        # smoother movement
        xVal = np.interp(x, [width // 2, width], [0, width])
        yVal = np.interp(y, [150, height - 150], [0, height])

        currX = prevX + (xVal - prevX) // smoothening
        currY = prevY + (yVal - prevY) // smoothening

        indexFinger = (int(currX), int(currY))

        prevX, prevY = currX, currY

        # =======================
        # SLIDE NAVIGATION (SWIPE STYLE)
        # =======================
        if hand["center"][1] < gestureThreshold:

            # LEFT
            if fingers == [1, 0, 0, 0, 0]:
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    buttonPressed = True

            # RIGHT
            elif fingers == [0, 0, 0, 0, 1]:
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    buttonPressed = True

        # =======================
        # DRAW MODE (index finger)
        # =======================
        if fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])

            annotations[annotationNumber].append(indexFinger)

            cv2.circle(imgCurrent, indexFinger, 8, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        # =======================
        # UNDO (2 fingers)
        # =======================
        if fingers == [0, 1, 1, 0, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

        # =======================
        # CLEAR (open hand)
        # =======================
        if sum(fingers) == 5:
            annotations = [[]]
            annotationNumber = -1
            buttonPressed = True

    # =======================
    # BUTTON DELAY
    # =======================
    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    # =======================
    # DRAW ANNOTATIONS
    # =======================
    for annotation in annotations:
        for i in range(1, len(annotation)):
            cv2.line(imgCurrent, annotation[i-1], annotation[i], (0, 0, 255), 5)

    # =======================
    # WEBCAM OVERLAY
    # =======================
    imgSmall = cv2.resize(img, (ws, hs))
    imgCurrent[0:hs, width-ws:width] = imgSmall

    # =======================
    # FPS
    # =======================
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(imgCurrent, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # =======================
    # SLIDE NUMBER
    # =======================
    cv2.putText(imgCurrent, f'{imgNumber+1}/{len(pathImages)}',
                (width - 150, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # =======================
    # SHOW
    # =======================
    cv2.imshow("Presentation", imgCurrent)
    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()