# ASL Sign Language to Text — How It Works

## What This App Does

Point your webcam at your hand, make an ASL letter sign, and the app converts it to text in real time. It also suggests word completions and can read the sentence aloud.

---

## The Pipeline

```
Your Hand
    │
    ▼
Webcam Frame
    │
    ▼
Hand Detection (MediaPipe via cvzone)
  → Finds your hand and its bounding box
    │
    ▼
21 Landmark Points
  → Finger joints, knuckles, wrist — mapped as (x, y) coordinates
    │
    ▼
Skeleton Drawing
  → Lines and dots drawn on a blank 400×400 white image
  → This is what the model actually sees — not your raw hand photo
    │
    ▼
CNN Model (cnn8grps_rad1_model.h5)
  → Classifies the skeleton into 1 of 8 visual groups
    │
    ▼
Geometry Disambiguation
  → Uses distances and positions of specific landmarks
  → Picks the exact letter within the group
    │
    ▼
Predicted Letter
    │
    ▼
Sentence Builder + Word Suggestions (pyenchant)
    │
    ▼
Text-to-Speech (Piper)
```

---

## Why 8 Groups Instead of 26 Letters?

Many ASL letters look visually similar — A, E, M, N, S, T all have fingers curled in. The CNN would struggle to tell them apart directly. So instead:

1. The CNN learns 8 broad groups (easier classification problem)
2. Hand geometry rules then disambiguate within each group

| Group | Letters |
|-------|---------|
| 0 | A, E, M, N, S, T |
| 1 | B, D, F, I, K, R, U, V, W |
| 2 | C, O |
| 3 | G, H |
| 4 | L |
| 5 | P, Q, Z |
| 6 | X |
| 7 | Y, J |

Example — Group 2 (C vs O): measure the distance between thumb tip and middle fingertip. If distance > 42 pixels → C. Otherwise → O.

---

## Special Gestures

| Gesture | Action |
|---------|--------|
| Index + pinky extended, others curled | Space |
| Flat open hand facing camera | Next (commits current letter) |
| Thumbs up pointing inward | Backspace |

---

## Files

| File | What it does |
|------|-------------|
| `src/app.py` | Main app — run this |
| `src/predict.py` | Same prediction, no GUI (terminal only) |
| `src/collect.py` | Collect training images from webcam |
| `src/predictor.py` | Shared functions: `draw_skeleton()`, `predict_letter()`, `disambiguate()` |
| `assets/cnn8grps_rad1_model.h5` | Trained model weights |
| `assets/signs.png` | ASL alphabet reference chart shown in the GUI |
| `data/` | Training skeleton images (A–Z, ~180 per letter) |

---

## Running the App

```bash
# Install dependencies
uv sync

# Run
uv run python src/app.py
```

Requires `enchant` system library:
```bash
# Arch Linux
sudo pacman -S enchant aspell aspell-en

# Ubuntu/Debian
sudo apt install enchant-2 aspell aspell-en
```

---

## Collecting New Training Data

```bash
uv run python src/collect.py
```

- Point webcam at your hand
- Press `a` to start/stop auto-saving frames
- Press `n` to move to the next letter
- Images save to `data/<Letter>/`
- ~180 images per letter is enough
