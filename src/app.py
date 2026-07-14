import ctypes
import ctypes.util

# Must be called before any X11/XCB usage — MediaPipe 0.10+ EGL threads call
# XInitThreads() themselves, which aborts XCB if it was opened without thread-safety.
_libX11 = ctypes.util.find_library("X11")
if _libX11:
    ctypes.cdll.LoadLibrary(_libX11).XInitThreads()

import os
import shutil
import subprocess
import time
import tkinter as tk
from tkinter import ttk

import cv2
from loguru import logger

cv2.setNumThreads(1)
from pathlib import Path

from logging_config import configure_logging

configure_logging()

# Skip TensorFlow's CUDA probing on machines with no NVIDIA driver — it still
# runs fine on CPU, but without this it wastes time and prints scary-looking
# "Failed to determine cuDNN version" errors that aren't actually failures.
if not shutil.which("nvidia-smi") and not Path("/proc/driver/nvidia").exists():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
from PIL import Image, ImageTk

from detector_worker import DetectorWorker

ASSETS = Path(__file__).parent.parent / "assets"
DEFAULT_MODEL = ASSETS / "cnn8grps_rad1_model.h5"

_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    logger.info("Using GPU: {}", _gpus[0].name)
else:
    logger.info("No GPU detected — using CPU")


def resolve_model_path(cli_value=None):
    """Default to the shipped model; pass a path (e.g. from train.py) to
    use a custom one instead. Never overwrites — just picks which to load."""
    return Path(cli_value) if cli_value else DEFAULT_MODEL


# See src/README.md's troubleshooting table: change this if the wrong camera
# gets picked. A single constant so the exposure-fix's /dev/videoN target
# (see _disable_exposure_dynamic_framerate) can't drift out of sync with it.
CAMERA_INDEX = 0

FONT_FAMILY = "Helvetica"
CAM_W, CAM_H = 400, 300
SKELETON_SIZE = 260

BG = "#1e1f2b"
CARD_BG = "#262837"
CARD_BORDER = "#383b52"
CARD_BORDER_HOVER = "#4d5178"
TEXT = "#eef0fa"
MUTED = "#9498b3"
MUTED_DIM = "#686c8a"
ACCENT = "#6d8bff"
ACCENT_DARK = "#5670d6"
DANGER = "#e0576b"
DANGER_DARK = "#c14458"
SUCCESS = "#4fd699"

# Consistent spacing rhythm — every card/section pads to one of these
# instead of ad-hoc pixel values, so the whole layout reads as one system.
SP_XS, SP_SM, SP_MD, SP_LG, SP_XL = 4, 8, 14, 20, 28

# How long the skeleton panel keeps showing the last hand it saw before it
# gives up and shows the "no hand detected" empty state instead of a stale
# frame. Worker results only arrive when a hand is actually found (see
# detector_worker.py), so silence here means "no hand", not "worker stuck".
HAND_IDLE_TIMEOUT = 1.0

# The CNN classifies every single processed frame independently — a noisy
# single-frame misread (or a hand mid-transition between two letters) used
# to be accepted immediately, both flickering the on-screen character and
# feeding garbage into the "next"-gesture commit history. Requiring the same
# letter to repeat this many results in a row before it's accepted damps
# that out; only stable, deliberately-held letters ever reach the sentence
# logic below.
STABLE_FRAMES = 4

# Reference content for the "Instructions" dialog — mirrors workflow.md's
# gesture/group tables so this is documented in-app, not just in a repo file
# a user running the built app will never open.
GESTURES = [
    ("Index + pinky extended, others curled", "Space"),
    ("Flat open hand facing the camera", "Next — commits the current letter"),
    ("Thumb up, fingers pointing inward", "Backspace"),
]
LETTER_GROUPS = [
    ("A, E, M, N, S, T", "fingers curled into a fist"),
    ("B, D, F, I, K, R, U, V, W", "index finger (+others) extended"),
    ("C, O", "curved hand, thumb-to-middle-finger distance"),
    ("G, H", "index+middle fingers pointing sideways"),
    ("L", "thumb + index at a right angle"),
    ("P, Q, Z", "thumb crossed over, fingers down"),
    ("X", "index finger hooked"),
    ("Y, J", "thumb + pinky extended"),
]


def _tracked(text, gap=" "):
    """Letter-space an all-caps label (e.g. "SENTENCE" -> "S E N T E N C E")
    — a cheap typographic trick that makes small uppercase section labels
    read as deliberate eyebrow text instead of default-cramped tk type."""
    return gap.join(text)


try:
    import enchant

    _spell = enchant.Dict(enchant.get_default_language())
except Exception:
    _spell = None


try:
    from tts_worker import TTSWorker

    _tts = TTSWorker()
except Exception:
    _tts = None


def get_suggestions(word):
    if not _spell or not word.strip():
        return " ", " ", " ", " "
    suggestions = _spell.suggest(word)
    return (
        suggestions[0] if len(suggestions) > 0 else " ",
        suggestions[1] if len(suggestions) > 1 else " ",
        suggestions[2] if len(suggestions) > 2 else " ",
        suggestions[3] if len(suggestions) > 3 else " ",
    )


class Application:
    def __init__(self, model_path=None):
        model_path = resolve_model_path(model_path)
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self._pending_char = None
        self._pending_count = 0

        self._fps_frame_count = 0
        self._fps_window_start = time.perf_counter()
        self._last_hand_seen = None
        self._skeleton_showing_hand = False
        self._instructions_win = None

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.configure(bg=BG)
        self.root.geometry("1360x820")
        self.root.minsize(1200, 760)

        self._build_style()

        # Hand detection + CNN classification (~90ms combined, see the note
        # in video_loop) runs on a single dedicated background thread instead
        # of inline here on the Tk main thread — see detector_worker.py for
        # the full design rationale. Camera capture stays on the main thread:
        # it's cheap (~5ms) and only the main thread may touch Tk widgets.
        self.worker = DetectorWorker(model_path)
        self.vs = cv2.VideoCapture(CAMERA_INDEX)
        self._disable_exposure_dynamic_framerate()

        # Header
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=SP_XL, pady=(SP_LG, SP_SM))
        header.columnconfigure(0, weight=1)

        title_box = tk.Frame(header, bg=BG)
        title_box.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_box,
            text="🤟  Sign Language To Text",
            font=(FONT_FAMILY, 25, "bold"),
            fg=TEXT,
            bg=BG,
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="Real-time ASL gesture recognition",
            font=(FONT_FAMILY, 12),
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(2, 0))

        status_box = tk.Frame(header, bg=BG)
        status_box.grid(row=0, column=1, sticky="e")
        self.status_dot = tk.Label(
            status_box, text="●", font=(FONT_FAMILY, 11), fg=MUTED, bg=BG
        )
        self.status_dot.pack(side="left", padx=(0, 6))
        self.status_label = tk.Label(
            status_box,
            text="Starting…",
            font=(FONT_FAMILY, 12, "bold"),
            fg=MUTED,
            bg=BG,
        )
        self.status_label.pack(side="left")
        self.fps_label = tk.Label(
            status_box,
            text="",
            font=(FONT_FAMILY, 11),
            fg=MUTED_DIM,
            bg=BG,
        )
        self.fps_label.pack(side="left", padx=(10, 0))

        self.help_btn = ttk.Button(
            header,
            text="❔  Instructions",
            style="Chip.TButton",
            cursor="hand2",
            command=self._show_instructions,
        )
        self.help_btn.grid(row=0, column=2, sticky="e", padx=(SP_LG, 0))

        # Main Content and Skeleton
        main = tk.Frame(self.root, bg=BG)
        main.grid(row=1, column=0, sticky="nsew", padx=24)
        # `uniform` (see the note on the `side` rows below) makes the 50/50
        # split exact instead of approximate.
        main.columnconfigure(0, weight=1, uniform="main_cols")
        main.columnconfigure(1, weight=1, uniform="main_cols")
        main.rowconfigure(0, weight=1)

        self.cam_card = cam_card = self._card(main, "Live Camera", icon="📷")
        cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        cam_card.grid_propagate(False)
        self._cam_blank = self._blank_photo(CAM_W, CAM_H, "black")
        self.panel = tk.Label(cam_card, bg="black", image=self._cam_blank)
        self.panel.grid(row=1, column=0, padx=14, pady=(0, 14))

        side = tk.Frame(main, bg=BG)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(0, weight=1, uniform="side_rows")
        side.rowconfigure(1, weight=1, uniform="side_rows")

        self.skeleton_card = skeleton_card = self._card(side, "Hand Skeleton", icon="🖐")
        skeleton_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        skeleton_card.grid_propagate(False)  # see note above cam_card
        self._skeleton_blank = self._blank_photo(SKELETON_SIZE, SKELETON_SIZE, CARD_BG)
        self.panel2 = tk.Label(skeleton_card, bg=CARD_BG, image=self._skeleton_blank)
        self.panel2.grid(row=1, column=0, padx=14, pady=(0, 14))
        self._show_empty_skeleton_state()

        self.reference_card = reference_card = self._card(
            side, "ASL Reference Chart", icon="📖"
        )
        reference_card.grid(row=1, column=0, sticky="nsew")
        reference_card.grid_propagate(False)  # see note above cam_card
        self._reference_src = Image.open(str(ASSETS / "signs.png"))
        self.reference_label = tk.Label(reference_card, bg=CARD_BG)
        self.reference_label.grid(row=1, column=0, padx=14, pady=(0, 14))
        reference_card.bind("<Configure>", self._render_reference)

        # Output Card: Character / Sentence / Suggestions / Actions
        output_card = self._card(self.root, "Output", icon="⌨")
        output_card.grid(row=2, column=0, sticky="ew", padx=24, pady=(16, 24))
        output_card.columnconfigure(1, weight=1)

        tk.Label(
            output_card,
            text=_tracked("CHARACTER"),
            font=(FONT_FAMILY, 11, "bold"),
            fg=MUTED,
            bg=CARD_BG,
        ).grid(row=1, column=0, sticky="w", pady=(0, 2))
        self.panel3 = tk.Label(
            output_card, font=(FONT_FAMILY, 32, "bold"), fg=ACCENT, bg=CARD_BG
        )
        self.panel3.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 14))

        tk.Label(
            output_card,
            text=_tracked("SENTENCE"),
            font=(FONT_FAMILY, 11, "bold"),
            fg=MUTED,
            bg=CARD_BG,
        ).grid(row=3, column=0, sticky="w", pady=(0, 2))
        self.panel5 = tk.Label(
            output_card,
            font=(FONT_FAMILY, 18),
            fg=TEXT,
            bg=CARD_BG,
            justify="left",
            anchor="w",
            wraplength=1000,
        )
        self.panel5.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        tk.Label(
            output_card,
            text=_tracked("SUGGESTIONS"),
            font=(FONT_FAMILY, 11, "bold"),
            fg=MUTED,
            bg=CARD_BG,
        ).grid(row=5, column=0, sticky="w", pady=(0, 6))

        suggestions_row = tk.Frame(output_card, bg=CARD_BG)
        suggestions_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        self.b1 = ttk.Button(suggestions_row, command=self.action1)
        self.b2 = ttk.Button(suggestions_row, command=self.action2)
        self.b3 = ttk.Button(suggestions_row, command=self.action3)
        self.b4 = ttk.Button(suggestions_row, command=self.action4)
        for btn in (self.b1, self.b2, self.b3, self.b4):
            btn.pack(side="left", padx=(0, 10))
            self._set_suggestion(btn, " ")

        actions_row = tk.Frame(output_card, bg=CARD_BG)
        actions_row.grid(row=7, column=0, columnspan=2, sticky="e", pady=(0, 14))
        self.clear_btn = ttk.Button(
            actions_row,
            text="🗑  Clear",
            style="Danger.TButton",
            cursor="hand2",
            command=self.clear_fun,
        )
        self.clear_btn.pack(side="left", padx=(0, 10))
        self.speak_btn = ttk.Button(
            actions_row,
            text="🔊  Speak" if _tts else "🔇  Speak unavailable",
            style="Accent.TButton",
            cursor="hand2" if _tts else "arrow",
            command=self.speak_fun,
        )
        if not _tts:
            self.speak_btn.state(["disabled"])
        self.speak_btn.pack(side="left")

        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.str = " "
        self.current_symbol = "C"
        self.word1 = self.word2 = self.word3 = self.word4 = " "

        for card in (cam_card, skeleton_card, reference_card, output_card):
            self._enable_card_hover(card)

        self.root.update_idletasks()
        self._render_reference()

        self.video_loop()

    def _disable_exposure_dynamic_framerate(self):
        """Best-effort fix for the camera capping itself at ~15fps.

        Measured on this machine's webcam: `cv2.VideoCapture.read()` took a
        steady ~66ms (15fps) at the default 640x480 under normal room
        lighting, and ~33ms (30fps) — a hard 2x — the instant the driver's
        `exposure_dynamic_framerate` UVC control was turned off. That control
        (on by default on many UVC webcams) lets the "Aperture Priority"
        auto-exposure mode lengthen the frame's exposure time — and therefore
        halve the delivered framerate — to keep brightness up under normal
        indoor light. It's driver/hardware behavior, invisible to and
        unfixable by anything downstream (the threaded detector, frame
        skipping, etc. all still pay the 66ms/frame camera-read tax first).

        `exposure_dynamic_framerate` isn't one of OpenCV's standard
        `cv2.CAP_PROP_*` properties (it's a UVC extension control), so there's
        no `cap.set(...)` for it — shelling out to `v4l2-ctl`, the standard
        Linux tool for exactly this, is the only way to reach it. Silently
        no-ops if `v4l2-ctl` isn't installed or the camera doesn't expose this
        control — worst case we're back to the pre-fix framerate, not broken.
        """
        if not shutil.which("v4l2-ctl"):
            return
        try:
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    f"/dev/video{CAMERA_INDEX}",
                    "--set-ctrl=exposure_dynamic_framerate=0",
                ],
                capture_output=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def _build_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        def button_style(name, bg, active_bg, fg=TEXT, font_size=13, padding=(16, 10)):
            style.configure(
                name,
                background=bg,
                foreground=fg,
                font=(FONT_FAMILY, font_size, "bold"),
                borderwidth=0,
                focuscolor=bg,
                padding=padding,
            )
            style.map(
                name,
                background=[
                    ("disabled", CARD_BORDER),
                    ("active", active_bg),
                    ("pressed", active_bg),
                ],
                foreground=[("disabled", MUTED_DIM)],
            )

        button_style("Chip.TButton", CARD_BORDER, ACCENT, font_size=13, padding=(14, 8))
        button_style(
            "Chip.Disabled.TButton",
            CARD_BG,
            CARD_BG,
            fg=MUTED_DIM,
            font_size=13,
            padding=(14, 8),
        )
        button_style("Accent.TButton", ACCENT, ACCENT_DARK)
        button_style("Danger.TButton", DANGER, DANGER_DARK)
        button_style("Progress.TButton", CARD_BORDER, CARD_BORDER, fg=ACCENT)

    def _blank_photo(self, w, h, color):
        return ImageTk.PhotoImage(Image.new("RGB", (w, h), color))

    def _fit_size(self, card, native_w, native_h, pad_x=28, pad_y=14):
        """Largest (w, h) that preserves native_w:native_h and fits inside
        the card's content row (row=1), given the card's *actual* allocated
        size from the grid/window manager (not the fixed pixel constants
        the images were originally designed around). Falls back to the
        native size before the window has been laid out."""
        bbox = card.grid_bbox(0, 1)
        if not bbox:
            return native_w, native_h
        _, _, avail_w, avail_h = bbox
        avail_w = max(1, avail_w - pad_x)
        avail_h = max(1, avail_h - pad_y)
        scale = min(avail_w / native_w, avail_h / native_h)
        if scale <= 0:
            return native_w, native_h
        return max(1, round(native_w * scale)), max(1, round(native_h * scale))

    def _render_reference(self, event=None):
        w, h = self._fit_size(
            self.reference_card, self._reference_src.width, self._reference_src.height
        )
        resized = self._reference_src.resize((w, h), Image.LANCZOS)
        self._reference_photo = ImageTk.PhotoImage(resized)
        self.reference_label.config(image=self._reference_photo)

    def _show_instructions(self):
        """Open the gesture/letter-group reference dialog. Re-focuses the
        existing window instead of stacking duplicates if already open."""
        if self._instructions_win is not None and self._instructions_win.winfo_exists():
            self._instructions_win.lift()
            self._instructions_win.focus_set()
            return

        win = tk.Toplevel(self.root)
        self._instructions_win = win
        win.title("Instructions")
        win.configure(bg=BG)
        win.transient(self.root)
        win.geometry("560x640")
        win.minsize(480, 480)

        def close():
            win.destroy()
            self._instructions_win = None

        win.protocol("WM_DELETE_WINDOW", close)

        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True, padx=SP_XL, pady=SP_LG)

        tk.Label(
            body, text="How to use", font=(FONT_FAMILY, 18, "bold"), fg=TEXT, bg=BG
        ).pack(anchor="w")
        tk.Label(
            body,
            text=(
                "Hold your hand roughly arm's length from the camera, well lit, "
                "filling a good part of the frame — detection gets unreliable if "
                "your hand is small/far away or the room is dim."
            ),
            font=(FONT_FAMILY, 11),
            fg=MUTED,
            bg=BG,
            justify="left",
            wraplength=500,
        ).pack(anchor="w", pady=(SP_XS, SP_LG))

        tk.Label(
            body,
            text=_tracked("SPECIAL GESTURES"),
            font=(FONT_FAMILY, 11, "bold"),
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(0, SP_SM))
        for gesture, action in GESTURES:
            row = tk.Frame(body, bg=BG)
            row.pack(fill="x", pady=(0, SP_SM))
            tk.Label(
                row,
                text=action,
                font=(FONT_FAMILY, 12, "bold"),
                fg=ACCENT,
                bg=BG,
                width=22,
                anchor="w",
                justify="left",
            ).pack(side="left")
            tk.Label(
                row,
                text=gesture,
                font=(FONT_FAMILY, 11),
                fg=TEXT,
                bg=BG,
                anchor="w",
                justify="left",
                wraplength=280,
            ).pack(side="left", fill="x", expand=True)

        tk.Frame(body, bg=CARD_BORDER, height=1).pack(fill="x", pady=SP_LG)

        tk.Label(
            body,
            text=_tracked("LETTER GROUPS"),
            font=(FONT_FAMILY, 11, "bold"),
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(0, SP_SM))
        for letters, hint in LETTER_GROUPS:
            row = tk.Frame(body, bg=BG)
            row.pack(fill="x", pady=(0, 4))
            tk.Label(
                row,
                text=letters,
                font=(FONT_FAMILY, 12, "bold"),
                fg=TEXT,
                bg=BG,
                width=22,
                anchor="w",
            ).pack(side="left")
            tk.Label(
                row, text=hint, font=(FONT_FAMILY, 11), fg=MUTED, bg=BG, anchor="w"
            ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            body,
            text="Got it",
            style="Accent.TButton",
            cursor="hand2",
            command=close,
        ).pack(anchor="e", pady=(SP_LG, 0))

    def _show_empty_skeleton_state(self):
        """Swap the skeleton panel from an image to a centered placeholder
        message. Called once at startup and again whenever the worker goes
        HAND_IDLE_TIMEOUT seconds without a result — without this the panel
        would otherwise just freeze on the last hand it ever saw, which reads
        as broken rather than "no hand in frame right now"."""
        self.panel2.imgtk = None
        self.panel2.config(
            image="",
            text="🖐\nNo hand detected\nShow your hand to the camera",
            font=(FONT_FAMILY, 12),
            fg=MUTED,
            bg=CARD_BG,
            justify="center",
        )

    def _set_suggestion(self, btn, word):
        """Render one suggestion chip. Blank suggestions (word is just " ",
        the sentinel get_suggestions() returns for "no suggestion here") get
        a flat, non-interactive style instead of an empty-but-clickable
        button, so the row doesn't invite clicks that do nothing."""
        has_word = bool(word.strip())
        btn.config(
            text=word if has_word else "···",
            style="Chip.TButton" if has_word else "Chip.Disabled.TButton",
            cursor="hand2" if has_word else "arrow",
        )
        btn.state(["!disabled"] if has_word else ["disabled"])

    def _card(self, parent, title, icon=None):
        card = tk.Frame(
            parent, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1
        )
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        heading = f"{icon}  {title}" if icon else title
        tk.Label(
            card,
            text=heading,
            font=(FONT_FAMILY, 13, "bold"),
            fg=TEXT,
            bg=CARD_BG,
        ).grid(row=0, column=0, sticky="w", padx=SP_MD, pady=(SP_MD, SP_SM))

        # Thin divider under the title — a cheap way to separate "chrome"
        # from content without a second nested frame per card.
        tk.Frame(card, bg=CARD_BORDER, height=1).grid(
            row=0, column=0, sticky="sew", padx=SP_MD
        )

        return card

    def _enable_card_hover(self, card):
        def hover_on(_e):
            card.config(highlightbackground=CARD_BORDER_HOVER)

        def hover_off(_e):
            card.config(highlightbackground=CARD_BORDER)

        def bind_recursive(widget):
            widget.bind("<Enter>", hover_on, add="+")
            widget.bind("<Leave>", hover_off, add="+")
            for child in widget.winfo_children():
                bind_recursive(child)

        bind_recursive(card)

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                return
            cv2image = cv2.flip(frame, 1)
            display_rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            current_image = Image.fromarray(display_rgb)
            cam_w, cam_h = self._fit_size(self.cam_card, CAM_W, CAM_H)
            display_image = current_image.resize((cam_w, cam_h), Image.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=display_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Rolling camera FPS, recomputed twice a second. This increments
            # every tick regardless of whether the worker returns a result,
            # so — unlike the skeleton/character panels, which only update
            # when a hand is found — it's a heartbeat that keeps moving even
            # through stretches with no hand in frame, proving the app is
            # still alive rather than frozen.
            now = time.perf_counter()
            self._fps_frame_count += 1
            fps_elapsed = now - self._fps_window_start
            if fps_elapsed >= 0.5:
                self.fps_label.config(
                    text=f"{self._fps_frame_count / fps_elapsed:.0f} fps"
                )
                self._fps_frame_count = 0
                self._fps_window_start = now

            if _tts is not None:
                speak_state = _tts.latest_state()
                if speak_state is not None:
                    self._apply_speak_state(speak_state)

            # Hand detection + CNN classification (~25-40ms full-frame
            # findHands + ~10-16ms crop findHands + ~30ms CNN, profiled) runs
            # on DetectorWorker's own background thread, not here — see
            # detector_worker.py. `submit` never blocks: it just swaps in the
            # newest frame for the worker to pick up whenever it's free,
            # dropping the previous one if the worker hadn't gotten to it
            # yet. `latest_result` never blocks either: it returns the newest
            # finished prediction, or None if nothing new is ready. Because
            # the worker naturally paces itself against how long detection
            # actually takes, there's no need for the old fixed
            # every-other-frame throttle — the worker is inherently always
            # working on the latest frame rather than queuing up a backlog.
            #
            # `cv2image` is the raw BGR frame (only flipped, never
            # color-converted) — the worker does its own single BGR->RGB
            # conversion internally right before it touches mediapipe; see
            # detector_worker.py's module docstring for why feeding it an
            # already-RGB frame was a real, confirmed bug. cv2.flip() always
            # returns a fresh array (never mutates `frame` in place) and
            # nothing here mutates `cv2image` afterward, so handing the
            # worker this same object (instead of a defensive np.array()
            # copy) is safe — no thread ever writes to it again.
            self.worker.submit(cv2image)

            result = self.worker.latest_result()
            if result is not None:
                self._apply_prediction(result.letter)
                self._last_hand_seen = now
                self._skeleton_showing_hand = True

                skel_w, skel_h = self._fit_size(self.skeleton_card, 400, 400)
                current_image2 = Image.fromarray(result.skeleton).resize(
                    (skel_w, skel_h), Image.LANCZOS
                )
                imgtk2 = ImageTk.PhotoImage(image=current_image2)
                self.panel2.imgtk = imgtk2
                self.panel2.config(image=imgtk2, text="")
                self.panel3.config(text=self.current_symbol)

                self._set_suggestion(self.b1, self.word1)
                self._set_suggestion(self.b2, self.word2)
                self._set_suggestion(self.b3, self.word3)
                self._set_suggestion(self.b4, self.word4)

                self.status_dot.config(fg=SUCCESS)
                self.status_label.config(text="Tracking hand", fg=SUCCESS)
            elif (
                self._last_hand_seen is None
                or now - self._last_hand_seen > HAND_IDLE_TIMEOUT
            ):
                if self._skeleton_showing_hand:
                    self._show_empty_skeleton_state()
                    self._skeleton_showing_hand = False
                    # The hand was just lost — clear the debounce state too.
                    # Without this, _pending_count stays at whatever it last
                    # reached (often well above STABLE_FRAMES) across the
                    # gap, so the very first frame once a hand is shown again
                    # gets accepted immediately instead of having to hold
                    # STABLE_FRAMES times again — silently defeating the
                    # anti-noise debounce exactly when it matters most (a
                    # fresh hand entering frame is the noisiest moment).
                    self._pending_char = None
                    self._pending_count = 0
                    # Also reset prev_char: if the user's last committed
                    # letter before lowering their hand was itself "next",
                    # leaving prev_char == "next" here means the *next* time
                    # they sign "next" again (a completely normal sign ->
                    # next -> relax -> sign -> next flow), _apply_prediction's
                    # `self.prev_char != "next"` guard is false and that
                    # commit is silently dropped.
                    self.prev_char = ""
                self.status_dot.config(fg=MUTED)
                self.status_label.config(text="No hand detected", fg=MUTED)

            self.panel5.config(text=self.str)
        except Exception:
            logger.exception("video_loop failed")
        finally:
            self.root.after(1, self.video_loop)

    def _apply_prediction(self, ch1):
        """Update the running sentence/word-suggestion state from a symbol
        the worker predicted. This is plain bookkeeping (no CV/TF calls) so
        it's cheap enough to run on the main thread; keeping it here (rather
        than in the worker) also means all of self.str/self.count/etc. are
        only ever mutated from the main thread, with no cross-thread
        synchronization needed for them."""
        if ch1 == self._pending_char:
            self._pending_count += 1
        else:
            self._pending_char = ch1
            self._pending_count = 1

        if self._pending_count < STABLE_FRAMES:
            return  # not held long enough yet — treat as noise, ignore

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
        if not _tts or not self.str.strip():
            return
        self.speak_btn.state(["disabled"])
        self.speak_btn.config(text="Loading voice…", style="Progress.TButton")
        _tts.speak(self.str)

    def _apply_speak_state(self, state):
        """Applied once per video_loop tick if _tts reports a new state —
        see TTSWorker.latest_state's docstring for why this is a poll, not
        a callback that pokes Tk from the worker's background thread."""
        if state == "loading":
            self.speak_btn.config(text="⏳Loading voice…", style="Progress.TButton")
        elif state == "speaking":
            self.speak_btn.config(text="🔊Speaking…", style="Progress.TButton")
        elif state == "error":
            self.speak_btn.config(
                text="⚠  Speak failed — click to retry", style="Danger.TButton"
            )
            self.speak_btn.state(["!disabled"])
        else:  # "done"
            self.speak_btn.config(text="🔊Speak", style="Accent.TButton")
            self.speak_btn.state(["!disabled"])

    def clear_fun(self):
        self.str = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        for btn in (self.b1, self.b2, self.b3, self.b4):
            self._set_suggestion(btn, " ")

    def destructor(self):
        self.worker.stop()
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to a .h5 model (default: shipped model)")
    args = parser.parse_args()

    logger.info("Starting Application...")
    Application(model_path=args.model).root.mainloop()
