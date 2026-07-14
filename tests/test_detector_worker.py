from src.detector_worker import smooth_landmarks


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
