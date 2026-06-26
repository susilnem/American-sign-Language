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

from predictor import draw_skeleton

ASSETS = Path(__file__).parent.parent / "assets"
DATA_DIR = Path(__file__).parent.parent / "data"

OFFSET = 15

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

letter = "A"
(DATA_DIR / letter).mkdir(parents=True, exist_ok=True)
saved = len(list((DATA_DIR / letter).iterdir()))

step = 0
collecting = False
collection_count = 0

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        canvas = np.ones((400, 400, 3), np.uint8) * 255
        skeleton = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            crop = np.array(
                frame[y - OFFSET : y + h + OFFSET, x - OFFSET : x + w + OFFSET]
            )
            handz, _ = hd2.findHands(crop, draw=False, flipType=True)
            if handz:
                pts = handz[0]["lmList"]
                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15
                draw_skeleton(canvas, pts, os_x, os_y)
                skeleton = canvas.copy()
                cv2.imshow("skeleton", canvas)

        frame = cv2.putText(
            frame,
            f"letter={letter}  saved={saved}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord("n"):
            next_letter = chr(ord(letter) + 1)
            letter = "A" if next_letter > "Z" else next_letter
            collecting = False
            (DATA_DIR / letter).mkdir(parents=True, exist_ok=True)
            saved = len(list((DATA_DIR / letter).iterdir()))
        elif key == ord("a"):
            collecting = not collecting
            collection_count = 0

        if collecting and skeleton is not None:
            if collection_count >= 180:
                collecting = False
            elif step % 3 == 0:
                cv2.imwrite(str(DATA_DIR / letter / f"{saved}.jpg"), skeleton)
                saved += 1
                collection_count += 1
            step += 1

    except Exception:
        print(traceback.format_exc())

capture.release()
cv2.destroyAllWindows()
