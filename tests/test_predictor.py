from src.predictor import disambiguate, distance


def test_distance_pythagorean():
    assert distance([0, 0], [3, 4]) == 5.0


def test_distance_same_point():
    assert distance([5, 5], [5, 5]) == 0.0


def test_distance_symmetry():
    assert distance([1, 2], [4, 6]) == distance([4, 6], [1, 2])


# --- disambiguate: group 2 (C vs O) ---


def _pts_with_distance(pt4, pt12):
    """21 landmark points, only pt4 and pt12 matter for group 2."""
    pts = [[0, 0]] * 21
    pts[4] = list(pt4)
    pts[12] = list(pt12)
    return pts


def test_group2_c_when_distance_gt_42():
    pts = _pts_with_distance(pt4=[0, 0], pt12=[30, 30])  # dist ~42.4 > 42
    assert disambiguate(pts, ch1=2) == "C"


def test_group2_o_when_distance_lte_42():
    pts = _pts_with_distance(pt4=[0, 0], pt12=[20, 20])  # dist ~28 < 42
    assert disambiguate(pts, ch1=2) == "O"


# --- disambiguate: group 4 (L) ---


def test_group4_always_l():
    pts = [[0, 0]] * 21
    assert disambiguate(pts, ch1=4) == "L"


# --- disambiguate: group 6 (X) ---


def test_group6_always_x():
    pts = [[0, 0]] * 21
    assert disambiguate(pts, ch1=6) == "X"


# --- disambiguate: group 3 (G vs H) ---


def _pts_for_group3(pt8, pt12):
    pts = [[0, 0]] * 21
    pts[8] = list(pt8)
    pts[12] = list(pt12)
    return pts


def test_group3_g_when_distance_gt_72():
    pts = _pts_for_group3(pt8=[0, 0], pt12=[60, 40])  # dist ~72.1 > 72
    assert disambiguate(pts, ch1=3) == "G"


def test_group3_h_when_distance_lte_72():
    pts = _pts_for_group3(pt8=[0, 0], pt12=[30, 30])  # dist ~42 < 72
    assert disambiguate(pts, ch1=3) == "H"


# --- disambiguate: group 7 (Y vs J) ---


def _pts_for_group7(pt4, pt8):
    pts = [[0, 0]] * 21
    pts[4] = list(pt4)
    pts[8] = list(pt8)
    return pts


def test_group7_y_when_distance_gt_42():
    pts = _pts_for_group7(pt4=[0, 0], pt8=[30, 30])  # dist ~42.4 > 42
    assert disambiguate(pts, ch1=7) == "Y"


def test_group7_j_when_distance_lte_42():
    pts = _pts_for_group7(pt4=[0, 0], pt8=[20, 20])  # dist ~28 < 42
    assert disambiguate(pts, ch1=7) == "J"
