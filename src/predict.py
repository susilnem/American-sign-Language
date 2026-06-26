import ctypes
import ctypes.util

_libX11 = ctypes.util.find_library("X11")
if _libX11:
    ctypes.cdll.LoadLibrary(_libX11).XInitThreads()

import traceback
from pathlib import Path

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tf_keras.models import load_model

from predictor import draw_skeleton, predict_letter

ASSETS = Path(__file__).parent.parent / "assets"
OFFSET = 29

model = load_model(str(ASSETS / "cnn8grps_rad1_model.h5"))
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            crop = np.array(
                frame[y - OFFSET : y + h + OFFSET, x - OFFSET : x + w + OFFSET]
            )
            canvas = np.ones((400, 400, 3), np.uint8) * 255
            handz, _ = hd2.findHands(crop, draw=False, flipType=True)
            if handz:
                pts = handz[0]["lmList"]
                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15
                draw_skeleton(canvas, pts, os_x, os_y)
                cv2.imshow("skeleton", canvas)

                prob = model.predict(canvas.reshape(1, 400, 400, 3), verbose=0)[0]
                ch1 = predict_letter(pts, prob)
                print(f"Predicted: {ch1}")
                frame = cv2.putText(
                    frame,
                    f"Predicted: {ch1}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        print(traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
