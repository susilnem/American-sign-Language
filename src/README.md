# Getting Started

## Prerequisites

- Python 3.12+
- A webcam
- [uv](https://docs.astral.sh/uv/) — install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### System libraries

This enables spell suggestions. It's optional — the app runs without it, suggestions just won't work. (Text-to-speech needs no system library — Piper's voice model downloads once, automatically, on first use, then works fully offline.) `uv sync` can't install this itself (it's a Python dependency manager, not a system package manager, and doing so would need sudo) — use `scripts/setup.sh` below, or install manually:

**Arch Linux**
```bash
sudo pacman -S enchant aspell aspell-en
```

**Ubuntu / Debian**
```bash
sudo apt install enchant-2 aspell aspell-en
```

## Installation

```bash
./scripts/setup.sh
```

Installs the system libraries above (detects pacman/apt/dnf/brew), runs `uv sync`, and fetches the Piper text-to-speech voice model (~60MB, one-time, needs internet) so it's ready before you ever click Speak. If you'd rather skip any of that or do it yourself:

```bash
uv sync                        # just the Python dependencies
uv run python src/tts_worker.py  # just the voice download
```

Skipping the voice download isn't fatal — Speak downloads it on its first click instead, it'll just pause for a few seconds that first time.

## Windows

Not tested till now (developed and used on Linux) — but every dependency either
ships a Windows wheel or is pure-Python (verified on PyPI), so it should
install and run. If you hit something broken, please report it.

- `./scripts/setup.sh` is a bash script and won't run on Windows. Skip it —
  just run `uv sync`, then optionally `uv run python src/tts_worker.py` to
  fetch the Piper voice ahead of time (see above).
- Word suggestions (enchant) likely work with no extra install: `pyenchant`
  ships dedicated Windows wheels that bundle the enchant library, unlike on
  Linux where a separate system package is required.
- The camera exposure/framerate fix (`_disable_exposure_dynamic_framerate` in
  `src/app.py`) shells out to `v4l2-ctl`, a Linux-only tool — it silently
  no-ops on Windows. If your camera seems capped at a low framerate under dim
  lighting, look for an exposure/framerate override in your camera's Windows
  driver or manufacturer software instead.
- GPU training is still NVIDIA+CUDA-only either way (see "GPU support"
  below); the no-GPU-detected fallback relies on `nvidia-smi` being on PATH,
  which is normally true once NVIDIA drivers are installed.

## Running

```bash
uv run python src/app.py
```

The GUI opens and immediately starts reading from your webcam. Hold up an ASL hand sign in front of the camera.

Uses the shipped model by default. Pass `--model path/to/model.h5` to use a model you trained yourself (see "Training your own model" below).

## Other scripts

**Headless mode** — prediction without a GUI, prints the letter to the terminal:
```bash
uv run python src/predict.py
```

**Collect training data** — point your webcam at your hand, press `a` to auto-save 180 frames, press `n` to move to the next letter:
```bash
uv run python src/collect.py
```

## Training your own model

**Short answer: yes, but until now this repo only had half of it.** `collect.py`
(data collection) already existed, but there was no training script anywhere in
the repo — the shipped `assets/cnn8grps_rad1_model.h5` was trained entirely
outside this project, with no way to reproduce or retrain it here. `src/train.py`
closes that gap: a real, runnable script that trains a model with the exact
same architecture the app already expects.

### Workflow

```bash
# 1. Collect images for each letter (repeat/redo per letter as needed)
uv run python src/collect.py
#    press `a` to start/stop auto-saving frames (up to 180 per letter)
#    press `n` to move to the next letter (wraps Z -> A)

# 2. Train on whatever you've collected
uv run python src/train.py
```

By default `train.py` saves to `assets/cnn8grps_rad1_model_custom.h5` — it
**never overwrites the shipped model**. Run the app with your model:

```bash
uv run python src/app.py --model assets/cnn8grps_rad1_model_custom.h5
```

No `--model` flag means the app just uses the original shipped model, so
`uv run python src/app.py` keeps working exactly as before for anyone who
hasn't retrained. If you do want to replace the shipped model, pass
`--output assets/cnn8grps_rad1_model.h5` explicitly — any existing file at
that path is backed up to `.h5.bak` first, never silently clobbered.

Useful flags: `--data-dir`, `--output`, `--epochs` (default 25, with early
stopping on validation accuracy), `--batch-size` (default 32), `--val-split`
(default 0.2). Run `uv run python src/train.py --help` for the full list.

### The 8-group subtlety

The CNN does **not** output 26 letters — it outputs 8 groups of
visually-similar letters (see `workflow.md` for the full table and the
reasoning behind it). `predictor.py`'s `predict_letter`/`disambiguate`
functions hardcode that 8-group scheme and then use hand-geometry rules
(landmark distances/positions) to pick the exact letter within whichever
group the CNN picked. So `train.py` relabels each `data/<LETTER>/` folder to
its group index (0-7) before training — it does not train a 26-class model.

This also means you don't strictly need all 26 letters collected: training
only fails if an *entire group* has zero images (because then the model could
never predict that group at all), and it will tell you exactly which
letters/groups are missing rather than crashing.

### Expectations

- `collect.py` caps out at 180 images per letter, so a full run across all 26
  letters is roughly 4,700 images — a small dataset by CNN standards, which is
  why the defaults above (few epochs, early stopping) are deliberately modest
  rather than tuned for a huge corpus.
- Retraining on your own hand, camera, and lighting is exactly the fix for the
  "the model works great for the original author but poorly for me" problem
  mentioned elsewhere in this project — the shipped model only ever saw one
  person's hand under one set of conditions.

### GPU support

Yes — `train.py` uses plain TensorFlow/`tf_keras`, which automatically trains
on a GPU if one is visible, with no flags or code changes needed. At the start
of every run it prints which device it picked:

```
Using GPU: /physical_device:GPU:0
```
or
```
No GPU detected — training on CPU
```

What that actually requires, though:

- **NVIDIA + CUDA/cuDNN only.** Plain TensorFlow's GPU backend only supports
  NVIDIA hardware via CUDA. You need an NVIDIA GPU with a driver and
  CUDA/cuDNN versions matching this project's TensorFlow version — see
  [tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu). No
  extra Python-side setup is needed; `uv sync` already installs the same
  `tensorflow` package either way, and it uses the GPU automatically once the
  system-level CUDA/cuDNN install is in place.
- **AMD GPUs (including AMD iGPUs) are not supported** by plain TensorFlow —
  there's no ROCm build wired up here, so an AMD card is treated the same as
  having no GPU at all (this is the case on the machine this script was
  developed on: no NVIDIA/CUDA, only an AMD iGPU).
- **No usable GPU -> CPU, and that's fine.** Training runs on CPU without any
  changes; on a dataset this small (a few thousand 400x400 images), expect
  somewhere from a few minutes to a bit over an hour depending on your CPU and
  epoch count. It's slower, not broken.
- If a GPU is present but you'd rather force CPU (e.g. to keep it free for
  something else), set `CUDA_VISIBLE_DEVICES=-1` before running — the same
  guard `app.py` applies automatically when no NVIDIA driver is found.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not found | Try a different index: change `CAMERA_INDEX = 0` to `1` or `2` near the top of `src/app.py` |
| Word suggestions not working | Run `./scripts/setup.sh`, or install `enchant aspell aspell-en` manually (see above) |
| Speak button silent | Check the terminal for a Piper voice-download error (needs internet on first run only — downloads once to `assets/`, then works offline) |
| App crashes on startup | Make sure `assets/cnn8grps_rad1_model.h5` exists |
