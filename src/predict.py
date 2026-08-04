import ctypes
import ctypes.util

_libX11 = ctypes.util.find_library("X11")
if _libX11:
    ctypes.cdll.LoadLibrary(_libX11).XInitThreads()

import traceback
from pathlib import Path

import cv2
import numpy as np
from tf_keras.models import load_model

from detector_worker import _HandLandmarker, locate_hand
from predictor import draw_skeleton, predict_letter

ASSETS = Path(__file__).parent.parent / "assets"


def main():
    model = load_model(str(ASSETS / "cnn8grps_rad1_model.h5"))
    capture = cv2.VideoCapture(0)
    # single instance — two race mediapipe's GL/EGL init
    hd = _HandLandmarker(max_hands=1)

    try:
        while True:
            try:
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame = cv2.flip(frame, 1)

                located = locate_hand(hd, frame)
                canvas = np.full((400, 400, 3), 255, np.uint8)
                if located is not None:
                    pts, w, h = located
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
    finally:
        hd.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
