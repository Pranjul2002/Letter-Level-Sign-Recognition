
# For Data Creation

import math
import cv2 as cv
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

cap=cv.VideoCapture(1)

detector = HandDetector(maxHands=1)

OFF_SET=20
IMG_SIZE=300

Dataset_path= "Train_Data/X"
counter=0

while True:
    ret, frame=cap.read()
    hands, frame=detector.findHands(frame)

    if hands:
        hand=hands[0]
        x, y, w, h = hand['bbox']

        imgWhite=np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8)*255

        imgCrop=frame[y-OFF_SET:y+h+OFF_SET, x-OFF_SET:x+w+OFF_SET]



        aspectRatio=h/w

        if aspectRatio>1:
            k=IMG_SIZE/h
            wCal=math.ceil(k*w)
            imgResize=cv.resize(imgCrop, (wCal, IMG_SIZE))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((IMG_SIZE-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize


        else:
            k=IMG_SIZE/w
            hCal=math.ceil(k*h)
            imgResize=cv.resize(imgCrop, (IMG_SIZE, hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((IMG_SIZE-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize


        cv.imshow("img crop", imgWhite)

    cv.imshow("Web cam", frame)

    key=cv.waitKey(1)

    if key==ord("s"):
        counter+=1
        cv.imwrite(f'{Dataset_path}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
