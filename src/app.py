import ctypes
import ctypes.util

# Must be called before any X11/XCB usage — MediaPipe 0.10+ EGL threads call
# XInitThreads() themselves, which aborts XCB if it was opened without thread-safety.
_libX11 = ctypes.util.find_library("X11")
if _libX11:
    ctypes.cdll.LoadLibrary(_libX11).XInitThreads()

import numpy as np
import cv2
import traceback
import tkinter as tk

cv2.setNumThreads(1)
from pathlib import Path
from string import ascii_uppercase
from PIL import Image, ImageTk
from tf_keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from predictor import predict_letter, draw_skeleton

ASSETS = Path(__file__).parent.parent / "assets"

try:
    import enchant

    _spell = enchant.Dict(enchant.get_default_language())
except Exception:
    _spell = None

try:
    import pyttsx3

    _tts = pyttsx3.init()
    _tts.setProperty("rate", 100)
    _voices = _tts.getProperty("voices")
    _tts.setProperty("voice", _voices[0].id)
except Exception:
    _tts = None

OFFSET = 29


def get_suggestions(word):
    if not _spell or not word.strip():
        return [], [], [], []
    suggestions = _spell.suggest(word)
    return (
        suggestions[0] if len(suggestions) > 0 else " ",
        suggestions[1] if len(suggestions) > 1 else " ",
        suggestions[2] if len(suggestions) > 2 else " ",
        suggestions[3] if len(suggestions) > 3 else " ",
    )


class Application:
    def __init__(self):
        self.current_image = None
        self.ct = {"blank": 0}
        self.blank_flag = 0
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10

        for i in ascii_uppercase:
            self.ct[i] = 0

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1300x700")

        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.model = load_model(str(ASSETS / "cnn8grps_rad1_model.h5"))
        self.vs = cv2.VideoCapture(0)

        self.panel = tk.Label(self.root)
        self.panel.place(x=40, y=3, width=480, height=640)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=550, y=115, width=400, height=400)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(
            text="Sign Language To Text Conversion",
            font=("Times New Roman", 30, "bold"),
        )

        image1 = Image.open(str(ASSETS / "signs.png"))
        image1 = image1.resize((500, 400), Image.LANCZOS)
        test = ImageTk.PhotoImage(image1)
        label1 = tk.Label(image=test)
        label1.image = test
        label1.place(x=1000, y=110)

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=280, y=585)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=580)
        self.T1.config(text="Character :", font=("Times New Roman", 30, "bold"))

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=260, y=632)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=632)
        self.T3.config(text="Sentence :", font=("Times New Roman", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=700)
        self.T4.config(
            text="Suggestions :", fg="red", font=("Times New Roman", 30, "bold")
        )

        self.b1 = tk.Button(self.root)
        self.b1.place(x=390, y=700)
        self.b2 = tk.Button(self.root)
        self.b2.place(x=590, y=700)
        self.b3 = tk.Button(self.root)
        self.b3.place(x=790, y=700)
        self.b4 = tk.Button(self.root)
        self.b4.place(x=990, y=700)

        self.speak_btn = tk.Button(self.root)
        self.speak_btn.place(x=1305, y=630)
        self.speak_btn.config(
            text="Speak",
            font=("Times New Roman", 20),
            wraplength=100,
            command=self.speak_fun,
        )

        self.clear_btn = tk.Button(self.root)
        self.clear_btn.place(x=1205, y=630)
        self.clear_btn.config(
            text="Clear",
            font=("Times New Roman", 20),
            wraplength=100,
            command=self.clear_fun,
        )

        self.str = " "
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"
        self.word1 = self.word2 = self.word3 = self.word4 = " "

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            hands = self.hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy = np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                hand = hands[0]
                x, y, w, h = hand["bbox"]
                image = cv2image_copy[
                    y - OFFSET : y + h + OFFSET, x - OFFSET : x + w + OFFSET
                ]
                white = np.ones((400, 400, 3), np.uint8) * 255

                handz, _ = self.hd2.findHands(image, draw=False, flipType=True)
                if handz:
                    hand = handz[0]
                    self.pts = hand["lmList"]

                    os_x = ((400 - w) // 2) - 15
                    os_y = ((400 - h) // 2) - 15
                    draw_skeleton(white, self.pts, os_x, os_y)
                    self.predict(white)

                    self.current_image2 = Image.fromarray(white)
                    imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)
                    self.panel3.config(
                        text=self.current_symbol, font=("Times New Roman", 30)
                    )

                    self.b1.config(
                        text=self.word1,
                        font=("Times New Roman", 20),
                        wraplength=825,
                        command=self.action1,
                    )
                    self.b2.config(
                        text=self.word2,
                        font=("Times New Roman", 20),
                        wraplength=825,
                        command=self.action2,
                    )
                    self.b3.config(
                        text=self.word3,
                        font=("Times New Roman", 20),
                        wraplength=825,
                        command=self.action3,
                    )
                    self.b4.config(
                        text=self.word4,
                        font=("Times New Roman", 20),
                        wraplength=825,
                        command=self.action4,
                    )

            self.panel5.config(
                text=self.str, font=("Times New Roman", 30), wraplength=1025
            )
        except Exception:
            print(traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def predict(self, skeleton):
        image = skeleton.reshape(1, 400, 400, 3)
        prob = self.model.predict(image, verbose=0)[0]
        ch1 = predict_letter(self.pts, prob)

        if ch1 == "next" and self.prev_char != "next":
            prev = self.ten_prev_char[(self.count - 2) % 10]
            if prev != "next":
                if prev == "Backspace":
                    self.str = self.str[:-1]
                else:
                    self.str += prev
            else:
                fallback = self.ten_prev_char[self.count % 10]
                if fallback != "Backspace":
                    self.str += fallback

        if ch1 == " " and self.prev_char != " ":
            self.str += " "

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1

        if self.str.strip():
            st = self.str.rfind(" ")
            word = self.str[st + 1 :]
            self.word = word
            self.word1, self.word2, self.word3, self.word4 = get_suggestions(word)

    def _replace_current_word(self, replacement):
        idx = self.str.rfind(" ")
        self.str = self.str[: idx + 1] + replacement.upper()

    def action1(self):
        self._replace_current_word(self.word1)

    def action2(self):
        self._replace_current_word(self.word2)

    def action3(self):
        self._replace_current_word(self.word3)

    def action4(self):
        self._replace_current_word(self.word4)

    def speak_fun(self):
        if _tts:
            _tts.say(self.str)
            _tts.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "

    def destructor(self):
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")
Application().root.mainloop()
