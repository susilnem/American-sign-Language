import math

import cv2
import numpy as np


def draw_skeleton(canvas, pts, os_x, os_y):
    """Draw hand skeleton onto a canvas given 21 landmark points and an offset."""
    finger_segments = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    for start, end in finger_segments:
        for t in range(start, end):
            cv2.line(
                canvas,
                (pts[t][0] + os_x, pts[t][1] + os_y),
                (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y),
                (0, 255, 0),
                3,
            )
    for a, b in [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]:
        cv2.line(
            canvas,
            (pts[a][0] + os_x, pts[a][1] + os_y),
            (pts[b][0] + os_x, pts[b][1] + os_y),
            (0, 255, 0),
            3,
        )
    for i in range(21):
        cv2.circle(canvas, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)


def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def predict_letter(pts, prob):
    """
    Takes 21 hand landmark points and CNN output probabilities (8 groups).
    Applies inter-group correction rules then within-group disambiguation.
    Returns the predicted letter (str), ' ' for space, 'next', or 'Backspace'.
    """
    prob = np.array(prob, dtype="float32")
    ch1 = int(np.argmax(prob))
    prob[ch1] = 0
    ch2 = int(np.argmax(prob))

    pl = [ch1, ch2]

    # all-fingers-curled → group 0 (aemnst)
    if pl in [
        [5, 2],
        [5, 3],
        [3, 5],
        [3, 6],
        [3, 0],
        [3, 2],
        [6, 4],
        [6, 1],
        [6, 2],
        [6, 6],
        [6, 7],
        [6, 0],
        [6, 5],
        [4, 1],
        [1, 0],
        [1, 1],
        [6, 3],
        [1, 6],
        [5, 6],
        [5, 1],
        [4, 5],
        [1, 4],
        [1, 5],
        [2, 0],
        [2, 6],
        [4, 6],
        [1, 0],
        [5, 7],
        [1, 6],
        [6, 1],
        [7, 6],
        [2, 5],
        [7, 1],
        [5, 4],
        [7, 0],
        [7, 5],
        [7, 2],
    ]:
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = 0

    if pl in [[2, 2], [2, 1]]:
        if pts[5][0] < pts[4][0]:
            ch1 = 0

    pl = [ch1, ch2]
    if pl in [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]:
        if (
            pts[0][0] > pts[8][0]
            and pts[0][0] > pts[4][0]
            and pts[0][0] > pts[12][0]
            and pts[0][0] > pts[16][0]
            and pts[0][0] > pts[20][0]
        ) and pts[5][0] > pts[4][0]:
            ch1 = 2

    pl = [ch1, ch2]
    if pl in [[6, 0], [6, 6], [6, 2]]:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    pl = [ch1, ch2]
    if pl in [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]:
        if (
            pts[6][1] > pts[8][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[0][0] < pts[8][0]
            and pts[0][0] < pts[12][0]
            and pts[0][0] < pts[16][0]
            and pts[0][0] < pts[20][0]
        ):
            ch1 = 3

    pl = [ch1, ch2]
    if pl in [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    pl = [ch1, ch2]
    if pl in [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    pl = [ch1, ch2]
    if pl in [[6, 4], [6, 1], [6, 2]]:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    pl = [ch1, ch2]
    if pl in [[1, 4], [1, 6], [1, 1]]:
        if (
            distance(pts[4], pts[11]) > 50
            and pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = 4

    pl = [ch1, ch2]
    if pl in [[3, 6], [3, 4]]:
        if pts[4][0] < pts[0][0]:
            ch1 = 4

    pl = [ch1, ch2]
    if pl in [[2, 2], [2, 5], [2, 4]]:
        if pts[1][0] < pts[12][0]:
            ch1 = 4

    pl = [ch1, ch2]
    if pl in [[3, 6], [3, 5], [3, 4]]:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] > pts[10][1]
        ):
            ch1 = 5

    pl = [ch1, ch2]
    if pl in [[3, 2], [3, 1], [3, 6]]:
        if (
            pts[4][1] + 17 > pts[8][1]
            and pts[4][1] + 17 > pts[12][1]
            and pts[4][1] + 17 > pts[16][1]
            and pts[4][1] + 17 > pts[20][1]
        ):
            ch1 = 5

    pl = [ch1, ch2]
    if pl in [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    pl = [ch1, ch2]
    if pl in [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]:
        if (
            pts[0][0] < pts[8][0]
            and pts[0][0] < pts[12][0]
            and pts[0][0] < pts[16][0]
            and pts[0][0] < pts[20][0]
        ):
            ch1 = 5

    pl = [ch1, ch2]
    if pl in [[5, 7], [5, 2], [5, 6]]:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    pl = [ch1, ch2]
    if pl in [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    pl = [ch1, ch2]
    if pl in [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    pl = [ch1, ch2]
    if pl in [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    pl = [ch1, ch2]
    if pl in [[7, 2]]:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    pl = [ch1, ch2]
    if pl in [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    pl = [ch1, ch2]
    if pl in [[4, 6], [4, 2], [4, 1], [4, 4]]:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    pl = [ch1, ch2]
    if pl in [[1, 4], [1, 6], [1, 0], [1, 2]]:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    pl = [ch1, ch2]
    if pl in [
        [5, 0],
        [5, 1],
        [5, 4],
        [5, 5],
        [5, 6],
        [6, 1],
        [7, 6],
        [0, 2],
        [7, 1],
        [7, 4],
        [6, 6],
        [7, 2],
        [6, 3],
        [6, 4],
        [7, 5],
    ]:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [
        [6, 1],
        [6, 0],
        [0, 3],
        [6, 4],
        [2, 2],
        [0, 6],
        [6, 2],
        [7, 6],
        [4, 6],
        [4, 1],
        [4, 2],
        [0, 2],
        [7, 1],
        [7, 4],
        [6, 6],
        [7, 2],
        [7, 5],
    ]:
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]:
        if (
            pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]:
        if (
            (
                pts[6][1] > pts[8][1]
                and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
            )
            and pts[2][0] < pts[0][0]
            and pts[4][1] > pts[14][1]
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [[4, 1], [4, 2], [4, 4]]:
        if (
            distance(pts[4], pts[11]) < 50
            and pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]:
        if (
            (
                pts[6][1] > pts[8][1]
                and pts[10][1] < pts[12][1]
                and pts[14][1] < pts[16][1]
                and pts[18][1] < pts[20][1]
            )
            and pts[2][0] < pts[0][0]
            and pts[14][1] < pts[4][1]
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [[6, 6], [6, 4], [6, 1], [6, 2]]:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [
        [5, 4],
        [5, 5],
        [5, 1],
        [0, 3],
        [0, 7],
        [5, 0],
        [0, 2],
        [6, 2],
        [7, 5],
        [7, 1],
        [7, 6],
        [7, 7],
    ]:
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]:
        if (
            pts[4][0] < pts[5][0] + 15
            and pts[6][1] < pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 7

    pl = [ch1, ch2]
    if pl in [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] > pts[14][1]
        ):
            ch1 = 1

    fg = 13
    pl = [ch1, ch2]
    if pl in [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]:
        if (
            not (
                pts[0][0] + fg < pts[8][0]
                and pts[0][0] + fg < pts[12][0]
                and pts[0][0] + fg < pts[16][0]
                and pts[0][0] + fg < pts[20][0]
            )
            and not (
                pts[0][0] > pts[8][0]
                and pts[0][0] > pts[12][0]
                and pts[0][0] > pts[16][0]
                and pts[0][0] > pts[20][0]
            )
            and distance(pts[4], pts[11]) < 50
        ):
            ch1 = 1

    pl = [ch1, ch2]
    if pl in [[5, 0], [5, 5], [0, 1]]:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
        ):
            ch1 = 1

    # within-group disambiguation
    ch1 = disambiguate(pts, ch1)

    # space gesture: index+middle up, ring+pinky down
    # (the bare int 1 this used to also match was the raw, unmapped group-1
    # code that could leak out of disambiguate() before its "D" default was
    # added — disambiguate() now always returns a letter, so that case can't
    # happen anymore and the literal was removed.)
    if ch1 in ("E", "S", "X", "Y", "B"):
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = " "

    # next gesture: open flat hand
    if ch1 in ("E", "Y", "B"):
        if (
            pts[4][0] < pts[5][0]
            and pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "next"

    # backspace gesture: thumb up, all fingers above wrist
    if (
        pts[0][0] > pts[8][0]
        and pts[0][0] > pts[12][0]
        and pts[0][0] > pts[16][0]
        and pts[0][0] > pts[20][0]
        and pts[4][1] < pts[8][1]
        and pts[4][1] < pts[12][1]
        and pts[4][1] < pts[16][1]
        and pts[4][1] < pts[20][1]
        and pts[4][1] < pts[6][1]
        and pts[4][1] < pts[10][1]
        and pts[4][1] < pts[14][1]
        and pts[4][1] < pts[18][1]
    ):
        ch1 = "Backspace"

    return ch1


def disambiguate(pts, ch1):
    """Within-group disambiguation using hand geometry."""
    if ch1 == 0:
        ch1 = "S"
        if (
            pts[4][0] < pts[6][0]
            and pts[4][0] < pts[10][0]
            and pts[4][0] < pts[14][0]
            and pts[4][0] < pts[18][0]
        ):
            ch1 = "A"
        if (
            pts[4][0] > pts[6][0]
            and pts[4][0] < pts[10][0]
            and pts[4][0] < pts[14][0]
            and pts[4][0] < pts[18][0]
            and pts[4][1] < pts[14][1]
            and pts[4][1] < pts[18][1]
        ):
            ch1 = "T"
        if (
            pts[4][1] > pts[8][1]
            and pts[4][1] > pts[12][1]
            and pts[4][1] > pts[16][1]
            and pts[4][1] > pts[20][1]
        ):
            ch1 = "E"
        if (
            pts[4][0] > pts[6][0]
            and pts[4][0] > pts[10][0]
            and pts[4][0] > pts[14][0]
            and pts[4][1] < pts[18][1]
        ):
            ch1 = "M"
        if (
            pts[4][0] > pts[6][0]
            and pts[4][0] > pts[10][0]
            and pts[4][1] < pts[18][1]
            and pts[4][1] < pts[14][1]
        ):
            ch1 = "N"

    elif ch1 == 2:
        ch1 = "C" if distance(pts[12], pts[4]) > 42 else "O"

    elif ch1 == 3:
        ch1 = "G" if distance(pts[8], pts[12]) > 72 else "H"

    elif ch1 == 4:
        ch1 = "L"

    elif ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            ch1 = "Z" if pts[8][1] < pts[5][1] else "Q"
        else:
            ch1 = "P"

    elif ch1 == 6:
        ch1 = "X"

    elif ch1 == 7:
        ch1 = "Y" if distance(pts[8], pts[4]) > 42 else "J"

    elif ch1 == 1:
        # Unlike every other group above, this one previously had no default
        # assignment before its chain of geometry checks — if none matched,
        # `disambiguate` returned the raw internal group number (int `1`)
        # instead of a letter. That wasn't purely a bug: the original
        # pre-refactor code relied on exactly this fallthrough — the int `1`
        # leaking into predict_letter's space-gesture check (which tests the
        # same geometry as one of the uncovered cases below) — as its way of
        # producing a space for group-1 hand poses none of the letter branches
        # below match. Defaulting to a letter (e.g. "D") like group 0 does
        # broke that: it silently ate the space case. Defaulting to " "
        # directly instead fixes the original bug (never leaks a bare int)
        # while preserving the original space-fallback behavior.
        ch1 = " "
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "B"
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "D"
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "F"
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "I"
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "W"
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] < pts[9][1]
        ):
            ch1 = "K"
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "U"
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] > pts[9][1]
        ):
            ch1 = "V"
        if (pts[8][0] > pts[12][0]) and (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "R"

    return ch1
