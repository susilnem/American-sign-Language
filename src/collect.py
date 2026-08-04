import ctypes
import ctypes.util

_libX11 = ctypes.util.find_library("X11")
if _libX11:
    ctypes.cdll.LoadLibrary(_libX11).XInitThreads()

import traceback
from pathlib import Path

import cv2
import numpy as np

from detector_worker import _HandLandmarker, locate_hand
from predictor import draw_skeleton

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    capture = cv2.VideoCapture(0)
    # single instance — two race mediapipe's GL/EGL init
    # crop OFFSET now matches predict.py/app.py (was smaller, a train/inference skew)
    hd = _HandLandmarker(max_hands=1)

    letter = "A"
    (DATA_DIR / letter).mkdir(parents=True, exist_ok=True)
    saved = len(list((DATA_DIR / letter).iterdir()))

    step = 0
    collecting = False
    collection_count = 0

    try:
        while True:
            try:
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                canvas = np.full((400, 400, 3), 255, np.uint8)
                skeleton = None

                located = locate_hand(hd, frame)
                if located is not None:
                    pts, w, h = located
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
    finally:
        hd.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
