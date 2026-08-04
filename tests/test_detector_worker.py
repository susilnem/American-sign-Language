import numpy as np
import pytest

import src.detector_worker as dw
from src.detector_worker import DetectorWorker, locate_hand, smooth_landmarks


def test_smooth_landmarks_passes_through_on_first_frame():
    """No previous frame to blend with — use the raw detection as-is."""
    pts = [[10, 20, 0], [30, 40, 0]]
    assert smooth_landmarks(None, pts) == pts


def test_smooth_landmarks_damps_a_sudden_jump():
    """A single noisy frame shouldn't move the drawn point all the way to
    the new (possibly jittery) position — it should land partway there."""
    prev = [[0, 0, 0]]
    new = [[100, 100, 0]]
    smoothed = smooth_landmarks(prev, new)
    assert 0 < smoothed[0][0] < 100
    assert 0 < smoothed[0][1] < 100


def test_smooth_landmarks_converges_to_a_steady_position():
    """If the hand actually holds still, repeated smoothing should settle
    on (not drift away from) that position."""
    pts = [[50, 50, 0]]
    smoothed = None
    for _ in range(50):
        smoothed = smooth_landmarks(smoothed, pts)
    assert smoothed[0][0] == 50
    assert smoothed[0][1] == 50


class _FakeHandLandmarker:
    """Stand-in for _HandLandmarker: first call sees the full frame and
    reports a hand bbox: second call (on the crop) reports a hand filling
    the crop it was given, recording the crop's shape so the test can
    assert the crop was clamped rather than wrapped around."""

    def __init__(self, first_bbox):
        self._first_bbox = first_bbox
        self.crops_seen = []
        self._calls = 0

    def find_hands(self, img_bgr):
        self._calls += 1
        if self._calls == 1:
            x, y, w, h = self._first_bbox
            return [{"lmList": [[x, y, 0]] * 21, "bbox": (x, y, w, h)}]
        self.crops_seen.append(img_bgr.shape)
        h, w = img_bgr.shape[:2]
        return [{"lmList": [[0, 0, 0]] * 21, "bbox": (0, 0, w, h)}]


def test_locate_hand_clamps_crop_near_top_left_edge_instead_of_wrapping():
    """A hand bbox within OFFSET pixels of the frame's top-left corner used
    to produce x - OFFSET / y - OFFSET < 0, which numpy slicing silently
    wraps to "from the end of the array" instead of clamping to 0 —
    corrupting the crop. locate_hand must clamp instead."""
    frame = np.zeros((480, 640, 3), dtype="uint8")
    # bbox starts at (5, 5) — well inside dw.OFFSET (29) of the (0, 0) edge.
    hd = _FakeHandLandmarker(first_bbox=(5, 5, 50, 50))

    result = locate_hand(hd, frame)

    assert result is not None
    # The crop handed to the second find_hands call must have non-negative,
    # in-bounds shape — not a numpy negative-index wraparound artifact.
    crop_h, crop_w = hd.crops_seen[0][:2]
    assert 0 < crop_h <= frame.shape[0]
    assert 0 < crop_w <= frame.shape[1]


def test_locate_hand_returns_none_when_no_hand_found():
    class _NoHand:
        def find_hands(self, img_bgr):
            return []

    assert locate_hand(_NoHand(), np.zeros((10, 10, 3), dtype="uint8")) is None


# --- DetectorWorker._init_engine: must not leak the landmarker if the
# model fails to load after it was already constructed ---


def test_init_engine_closes_landmarker_when_model_load_fails(monkeypatch):
    closed = []

    class _FakeLandmarker:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(dw, "_HandLandmarker", lambda max_hands=1: _FakeLandmarker())

    def _boom(path):
        raise RuntimeError("bad model file")

    monkeypatch.setattr(dw, "load_model", _boom)

    with pytest.raises(RuntimeError, match="bad model file"):
        DetectorWorker._init_engine("does-not-matter.h5")

    assert closed == [True], "landmarker must be closed if model loading fails"


# --- DetectorWorker._process_one: a bad frame must never kill the worker ---


def _bare_worker():
    """A DetectorWorker with no live thread/queues — safe for exercising
    per-frame processing helpers in isolation."""
    return DetectorWorker.__new__(DetectorWorker)


def test_process_one_swallows_build_result_exceptions(monkeypatch):
    """If classification blows up on a frame (e.g. a shape mismatch from a
    custom --model), that one frame should be dropped and logged — not
    propagate out of the loop and silently kill the worker thread for the
    rest of the process lifetime."""
    worker = _bare_worker()
    monkeypatch.setattr(dw, "locate_hand", lambda hd, frame: ([[0, 0, 0]] * 21, 10, 10))

    def _boom(*a, **k):
        raise ValueError("shape mismatch")

    monkeypatch.setattr(worker, "_build_result", _boom)

    result, _ = worker._process_one(
        hd=None, model=None, frame_bgr=None, smoothed_pts=None
    )

    assert result is None
