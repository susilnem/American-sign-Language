"""Background worker owning the single HandDetector + CNN model instance.

Runs detection + CNN inference (~90ms/frame) on its own thread instead of
inline on the Tk main thread, which used to freeze the whole UI (resize,
clicks, repaint) every time it ran.

Exactly one HandLandmarker + one loaded model, both constructed inside
`DetectorWorker._run` (the worker thread itself, never the main thread, never
more than one instance) — mediapipe's GL/EGL context init crashes (xcb
"Unknown sequence number") if two HandLandmarker instances, or one instance
driven from two threads, race to initialize it. The pre-rewrite app.py hit
this exact crash with two cvzone `HandDetector`s.

Both queues are maxsize=1, "get-then-put" to replace rather than accumulate:
the worker always works the newest frame, the main thread always reads the
newest result, and a slower worker just drops frames instead of falling
behind or blocking the UI.

Talks to mediapipe's Tasks API directly instead of through
`cvzone.HandTrackingModule.HandDetector`, replaced after finding two bugs in
it: (1) its `findHands()` always does its own BGR->RGB conversion, but a
caller here was already feeding it a pre-converted RGB frame, double
converting and handing mediapipe wrong-colored images; (2) its
`detectionCon`/`minTrackCon` constructor args are silently never passed into
`HandLandmarkerOptions` — dead config. `_HandLandmarker` below does exactly
one conversion and wires the confidence thresholds for real.

Uses IMAGE mode, not VIDEO: this module calls `findHands` twice per logical
frame (full frame, then an unrelated crop) — VIDEO mode's temporal tracking
assumes a continuous single viewpoint, so alternating two unrelated views
through it is a semantic mismatch. IMAGE mode is stateless, so one shared
instance handles both calls safely.

No GPU delegate: tested on this machine's AMD iGPU (Mesa/EGL) and it
segfaults on `.detect()` ("Required pass not found ... Corruption of the
global PassRegistry"). CPU/XNNPACK measured ~14ms/call, fast enough.
"""

import queue
import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tf_keras.models import load_model

from predictor import draw_skeleton, predict_letter

OFFSET = 29

# mediapipe silently loses the hand once its crop shrinks below roughly this
# many pixels on its short side (hand too far from camera) — upscale first,
# then scale detected landmarks back down so nothing downstream changes for
# hands already close enough to work.
MIN_DETECT_SIDE = 200

# Real, wired-in confidence thresholds (cvzone accepted these as constructor
# args but never passed them to mediapipe — dead config). Left at mediapipe's
# own default (0.5) so this doesn't change detection behavior; tune here.
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Smooths the *drawn* skeleton only — never the pts predict_letter reasons
# about. Its geometry rules are hardcoded pixel-distance thresholds
# calibrated on raw landmark positions; blending those toward previous frames
# lags mid-transition and can push borderline cases across a threshold
# wrong. The rendered skeleton has no such constraint (it's just line
# drawing), and without damping it visibly shakes frame to frame even when
# the hand is dead still, since IMAGE mode has no cross-frame smoothing of
# its own. STABLE_FRAMES in app.py separately handles output-letter jitter.
SMOOTHING_ALPHA = 0.7


def smooth_landmarks(prev_pts, new_pts, alpha=SMOOTHING_ALPHA):
    """Blend new_pts toward prev_pts to damp per-frame jitter. Pass-through
    on the first frame (prev_pts is None) or if point count changed."""
    if prev_pts is None or len(prev_pts) != len(new_pts):
        return new_pts
    return [
        [round(n * alpha + p * (1 - alpha)) for n, p in zip(new, prev)]
        for new, prev in zip(new_pts, prev_pts)
    ]


_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_LANDMARKER_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / ".downloaded_cache"
    / "hand_landmarker.task"
)


def _ensure_landmarker_model():
    """Download mediapipe's hand_landmarker.task once, if not already
    present at `_LANDMARKER_MODEL_PATH`."""
    if _LANDMARKER_MODEL_PATH.exists():
        return
    _LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_LANDMARKER_MODEL_URL, str(_LANDMARKER_MODEL_PATH))


class _HandLandmarker:
    """Minimal direct wrapper around
    `mediapipe.tasks.python.vision.HandLandmarker`, replacing
    `cvzone.HandTrackingModule.HandDetector`. See the module docstring for
    why: cvzone silently double-converts colors on the full-frame call and
    silently drops the confidence-threshold args it claims to accept.

    Stateless (IMAGE running mode): `find_hands` can be called any number of
    times, on any image content (full frames, unrelated crops, any order),
    with no shared timestamp/tracking state between calls to corrupt — see
    the module docstring's "Running mode" section for why that matters here.
    """

    def __init__(self, max_hands=1):
        _ensure_landmarker_model()
        base_options = mp_python.BaseOptions(
            model_asset_path=str(_LANDMARKER_MODEL_PATH)
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def find_hands(self, img_bgr):
        """`img_bgr` must be a raw BGR image — e.g. straight off
        `cv2.VideoCapture.read()`, or a BGR crop of one — never pre-converted
        to RGB (see the module docstring's "Double color conversion" point).
        Does exactly one BGR->RGB conversion, right here, before handing
        mediapipe its expected SRGB `mp.Image`.

        Returns a list of hands (empty if none found). Each hand is a dict
        with `lmList` (21 `[x, y, 0]` int pixel-coordinate landmark triples,
        in `img_bgr`'s coordinate space — the exact shape cvzone produced,
        which `predictor.py`/`draw_skeleton` depend on) and `bbox`
        (x, y, w, h), matching cvzone's format.
        """
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._detector.detect(mp_image)

        hands = []
        for hand_landmarks in result.hand_landmarks:
            lm_list = [[int(lm.x * w), int(lm.y * h), 0] for lm in hand_landmarks]
            xs = [p[0] for p in lm_list]
            ys = [p[1] for p in lm_list]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            hands.append(
                {
                    "lmList": lm_list,
                    "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
                }
            )
        return hands

    def close(self):
        self._detector.close()


def locate_hand(hd, full_frame_bgr):
    """Two-pass detection (full frame, then a tight crop) -> (pts, w, h),
    or None if no hand found. `w, h` are the full-frame bbox size, used
    by _build_result to center the skeleton on canvas.

    Module-level (not tied to DetectorWorker) so predict.py/collect.py can
    reuse the exact same detection path the live app uses instead of
    hand-rolling their own — that duplication previously carried two live
    bugs (unclamped crop bounds, and two competing HandDetector instances)
    that had already been fixed once, here, and nowhere else."""
    hands = hd.find_hands(full_frame_bgr)
    if not hands:
        return None

    hand = hands[0]
    x, y, w, h = hand["bbox"]
    frame_h, frame_w = full_frame_bgr.shape[:2]
    x0 = max(0, x - OFFSET)
    y0 = max(0, y - OFFSET)
    x1 = min(frame_w, x + w + OFFSET)
    y1 = min(frame_h, y + h + OFFSET)
    cropped = full_frame_bgr[y0:y1, x0:x1]
    if not cropped.size:
        return None

    # only upscale crops smaller than MIN_DETECT_SIDE
    scale = 1.0
    short_side = min(cropped.shape[:2])
    detect_crop = cropped
    if short_side < MIN_DETECT_SIDE:
        scale = MIN_DETECT_SIDE / short_side
        detect_crop = cv2.resize(
            cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    handz = hd.find_hands(detect_crop)
    if not handz:
        return None

    hand = handz[0]
    pts = hand["lmList"]
    if scale != 1.0:
        pts = [[round(px / scale), round(py / scale), pz] for px, py, pz in pts]

    return pts, w, h


class DetectionResult:
    """One completed detection: the rendered skeleton image and the raw
    predicted symbol (letter / ' ' / 'next' / 'Backspace')."""

    __slots__ = ("skeleton", "letter")

    def __init__(self, skeleton, letter):
        self.skeleton = skeleton
        self.letter = letter


class DetectorWorker:
    """Runs hand-detection + CNN classification on a single dedicated
    background thread. See module docstring for why it must be exactly one
    thread owning exactly one HandLandmarker/model for the process lifetime."""

    def __init__(self, model_path):
        self._model_path = model_path
        self._in_q = queue.Queue(maxsize=1)
        self._out_q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="detector-worker", daemon=True
        )
        self._thread.start()

    def submit(self, frame_bgr):
        """Hand a new camera frame to the worker. Never blocks: drops
        whatever frame was previously queued (if the worker hadn't picked it
        up yet) in favor of this newer one.

        `frame_bgr` must be a raw BGR frame (e.g. straight from
        `cv2.VideoCapture.read()`, only flipped) — never pre-converted to
        RGB. It's used both as the full-frame detection input and as the
        source the hand crop is cut from."""
        try:
            self._in_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._in_q.put_nowait(frame_bgr)
        except queue.Full:
            pass  # worker grabbed the slot between our get/put; fine, skip.

    def latest_result(self):
        """Non-blocking poll for the newest finished detection. Returns a
        DetectionResult, or None if nothing new has finished since the last
        call (including: no hand was found in the newest processed frame)."""
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self, timeout=2):
        """Signal the worker to exit and wait for it to actually stop.
        Safe to call from the main thread during app shutdown."""
        self._stop.set()
        try:
            self._in_q.put_nowait(None)  # wake the worker if it's blocked
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    @staticmethod
    def _init_engine(model_path):
        """Construct the HandLandmarker + load the model. If the model
        fails to load after the landmarker was already constructed, close
        the landmarker before re-raising — otherwise its native GL/EGL
        context leaks for the rest of the process even though the worker
        thread gives up and exits."""
        hd = _HandLandmarker(max_hands=1)
        try:
            model = load_model(str(model_path))
        except Exception:
            hd.close()
            raise
        return hd, model

    def _run(self):
        # Constructed here, on this thread, once, for the lifetime of the
        # process — never on the main thread, never more than one instance.
        # See the module docstring for why that matters.
        try:
            hd, model = self._init_engine(self._model_path)
        except Exception:
            logger.exception("DetectorWorker failed to start")
            return  # thread exits; app stays up, just gets no hand results

        smoothed_pts = None  # reset whenever the hand is lost — see below

        try:
            while not self._stop.is_set():
                item = self._in_q.get()
                if item is None or self._stop.is_set():
                    break

                result, smoothed_pts = self._process_one(hd, model, item, smoothed_pts)
                if result is None:
                    continue

                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._out_q.put_nowait(result)
                except queue.Full:
                    pass
        finally:
            hd.close()

    def _process_one(self, hd, model, frame_bgr, smoothed_pts):
        """Locate + classify one frame. Returns (result, new_smoothed_pts):
        `result` is None if no hand was found, or if detection/
        classification raised — a bad frame (e.g. a shape mismatch from a
        custom --model) must never be allowed to propagate out of _run and
        silently kill the worker thread for the rest of the process."""
        try:
            located = locate_hand(hd, frame_bgr)
        except Exception:
            logger.exception("Hand detection failed for a frame")
            return None, None  # don't smooth into a stale position

        if located is None:
            return None, None  # don't smooth into a stale position

        pts, w, h = located
        new_smoothed_pts = smooth_landmarks(smoothed_pts, pts)
        try:
            result = self._build_result(model, pts, new_smoothed_pts, w, h)
        except Exception:
            logger.exception("Building detection result failed for a frame")
            return None, new_smoothed_pts

        return result, new_smoothed_pts

    @staticmethod
    def _build_result(model, pts, smoothed_pts, w, h):
        """`smoothed_pts` draws the skeleton (and so is what the CNN's group
        classification sees); `pts` — raw, unsmoothed — is what
        `predict_letter`'s geometry rules reason about. See SMOOTHING_ALPHA's
        comment for why those two must not be the same array."""
        white = np.full((400, 400, 3), 255, np.uint8)
        os_x = ((400 - w) // 2) - 15
        os_y = ((400 - h) // 2) - 15
        draw_skeleton(white, smoothed_pts, os_x, os_y)

        image = white.reshape(1, 400, 400, 3)
        prob = model(image, training=False).numpy()[0]
        letter = predict_letter(pts, prob)

        return DetectionResult(skeleton=white, letter=letter)
