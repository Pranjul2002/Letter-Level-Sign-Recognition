
# real time implementation of letter level sign recognition

import math
import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("sign_classifier_model4.h5")

# Define index-to-label map
index_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'L', 9: 'O', 10: 'P', 11: 'Q', 12: 'R', 13: 'S', 14: 'T', 15: 'U', 16: 'V', 17: 'W', 18: 'X', 19: 'Y'}
# update based on your training classes

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)

OFFSET = 20
CAP_IMG_SIZE = 300       # For display
MODEL_INPUT_SIZE = 64    # For prediction input

while True:
    ret, frame = cap.read()
    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((CAP_IMG_SIZE, CAP_IMG_SIZE, 3), np.uint8) * 255

        # Crop with boundary check
        y1, y2 = max(y - OFFSET, 0), min(y + h + OFFSET, frame.shape[0])
        x1, x2 = max(x - OFFSET, 0), min(x + w + OFFSET, frame.shape[1])
        imgCrop = frame[y1:y2, x1:x2]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = CAP_IMG_SIZE / h
            wCal = math.ceil(k * w)
            imgResize = cv.resize(imgCrop, (wCal, CAP_IMG_SIZE))
            wGap = math.ceil((CAP_IMG_SIZE - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = CAP_IMG_SIZE / w
            hCal = math.ceil(k * h)
            imgResize = cv.resize(imgCrop, (CAP_IMG_SIZE, hCal))
            hGap = math.ceil((CAP_IMG_SIZE - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Resize to MODEL_INPUT_SIZE for prediction
        img_input = cv.resize(imgWhite, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        img_input = img_input.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Prediction
        prediction = model.predict(img_input)
        predicted_index = np.argmax(prediction)
        predicted_label = index_labels[predicted_index]

        # Show prediction on webcam
        cv.putText(frame, predicted_label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)

        #cv.imshow("Cropped Input 300x300", imgWhite)
        #cv.imshow("Model Input 30x30", cv.resize(imgWhite, (150, 150)))  # Optional preview

    cv.imshow("Webcam Feed", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
