# Getting Started

## Prerequisites

- Python 3.12+
- A webcam
- [uv](https://docs.astral.sh/uv/) — install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### System libraries

These enable spell suggestions and text-to-speech. Both are optional — the app runs without them, those features just won't work.

**Arch Linux**
```bash
sudo pacman -S enchant aspell aspell-en espeak-ng
```

**Ubuntu / Debian**
```bash
sudo apt install enchant-2 aspell aspell-en espeak-ng
```

## Installation

```bash
uv sync
```

This creates a virtual environment and installs all Python dependencies automatically.

## Running

```bash
uv run python src/app.py
```

The GUI opens and immediately starts reading from your webcam. Hold up an ASL hand sign in front of the camera.

## Other scripts

**Headless mode** — prediction without a GUI, prints the letter to the terminal:
```bash
uv run python src/predict.py
```

**Collect training data** — point your webcam at your hand, press `a` to auto-save 180 frames, press `n` to move to the next letter:
```bash
uv run python src/collect.py
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not found | Try a different index: change `cv2.VideoCapture(0)` to `1` or `2` in `src/app.py` |
| Word suggestions not working | Install `enchant aspell aspell-en` (see above) |
| Speak button silent | Install `espeak-ng` (see above) |
| App crashes on startup | Make sure `assets/cnn8grps_rad1_model.h5` exists |
