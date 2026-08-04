"""Train the 8-group CNN used by app.py / predict.py.

Reads data/<LETTER>/*.jpg (from collect.py), relabels each letter to its
group (0-7, see GROUPS — predictor.py hardcodes this exact scheme), trains,
and saves a model. Does NOT overwrite the shipped model by default — pass
--output to opt into that. Run the trained model in the app with:
    uv run python src/app.py --model assets/cnn8grps_rad1_model_custom.h5

Usage: uv run python src/train.py [--data-dir data] [--epochs 30] ...
See src/README.md ("Training your own model") for the full workflow.
"""

import argparse
import os
import shutil
from pathlib import Path
from string import ascii_uppercase

import cv2
import numpy as np
from loguru import logger

# Must run before tensorflow/tf_keras import — avoids noisy CUDA probing
# on machines with no NVIDIA driver (same guard as app.py).
if not shutil.which("nvidia-smi") and not Path("/proc/driver/nvidia").exists():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
from tf_keras import Sequential
from tf_keras.callbacks import EarlyStopping
from tf_keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from logging_config import configure_logging

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "assets"
DEFAULT_DATA_DIR = ROOT / "data"
SHIPPED_MODEL = ASSETS / "cnn8grps_rad1_model.h5"
# Default output is a separate file — never clobbers the shipped model
# unless the user explicitly passes --output pointing at it.
DEFAULT_OUTPUT = ASSETS / "cnn8grps_rad1_model_custom.h5"

IMAGE_SIZE = 400  # matches draw_skeleton's 400x400 canvas

# Letter -> group mapping (must exactly match predictor.predict_letter's
# hardcoded group semantics — see class docstring above and workflow.md).
GROUPS = [
    "AEMNST",  # 0: fingers curled into a fist
    "BDFIKRUVW",  # 1: index finger (+others) extended
    "CO",  # 2: curved hand, thumb-to-middle-finger distance
    "GH",  # 3: index+middle fingers pointing sideways
    "L",  # 4: thumb + index at a right angle
    "PQZ",  # 5: thumb crossed over, fingers down
    "X",  # 6: index finger hooked
    "YJ",  # 7: thumb + pinky extended
]
NUM_GROUPS = len(GROUPS)

LETTER_TO_GROUP = {
    letter: group for group, letters in enumerate(GROUPS) for letter in letters
}
assert set(LETTER_TO_GROUP) == set(ascii_uppercase), "GROUPS must cover exactly A-Z"


def build_model() -> Sequential:
    """Same architecture as the shipped model (verified via get_config()).
    No rescaling layer — inference feeds raw 0-255 pixels, so we match that."""
    model = Sequential(
        [
            Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
            ),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(16, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(16, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(96, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dense(NUM_GROUPS, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def _discover_letter_dirs(data_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Return (letter -> dir) for letters with at least one image, plus a list
    of letters that are missing or empty."""
    present = {}
    missing = []
    for letter in ascii_uppercase:
        letter_dir = data_dir / letter
        images = sorted(letter_dir.glob("*.jpg")) if letter_dir.is_dir() else []
        if images:
            present[letter] = letter_dir
        else:
            missing.append(letter)
    return present, missing


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load images, map letters to groups. Errors clearly if a whole group
    has no data (would be unlearnable)."""
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `uv run python src/collect.py` first to collect training images."
        )

    present, missing = _discover_letter_dirs(data_dir)

    if missing:
        missing_str = ", ".join(missing)
        logger.warning(
            "No images found for {} letter(s): {}. Training will proceed using "
            "only the letters that do have data — groups are still learnable "
            "as long as at least one letter per group has images.",
            len(missing),
            missing_str,
        )

    if not present:
        raise SystemExit(
            f"No training images found under {data_dir}. Expected subfolders like "
            f"{data_dir / 'A'}/*.jpg — run `uv run python src/collect.py` first."
        )

    groups_covered = {LETTER_TO_GROUP[letter] for letter in present}
    uncovered = [g for g in range(NUM_GROUPS) if g not in groups_covered]
    if uncovered:
        lines = []
        for g in uncovered:
            letters_in_group = GROUPS[g]
            lines.append(
                f"  group {g} ({'/'.join(letters_in_group)}): no data for any of it"
            )
        raise SystemExit(
            "Cannot train: the following output group(s) have zero training images, "
            "so the model could never learn to predict them:\n"
            + "\n".join(lines)
            + "\nCollect at least one letter per group with "
            "`uv run python src/collect.py` and try again."
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    for letter, letter_dir in sorted(present.items()):
        group = LETTER_TO_GROUP[letter]
        count = 0
        for path in sorted(letter_dir.glob("*.jpg")):
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Could not read {}, skipping", path)
                continue
            if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(group)
            count += 1
        logger.info("  {} -> group {}: {} images", letter, group, count)

    x = np.asarray(images, dtype="float32")
    y = np.asarray(labels, dtype="int64")
    n_letters, n_groups = len(present), len(groups_covered)
    logger.info(
        "Loaded {} images across {} letters / {} groups.", len(x), n_letters, n_groups
    )
    return x, y


def stratified_split(
    x: np.ndarray, y: np.ndarray, val_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/val, keeping each class's proportion roughly stable,
    without depending on scikit-learn (not a project dependency)."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in np.unique(y):
        class_idx = np.where(y == label)[0]
        rng.shuffle(class_idx)
        n_val = (
            max(1, int(round(len(class_idx) * val_split))) if len(class_idx) > 1 else 0
        )
        val_idx.extend(class_idx[:n_val].tolist())
        train_idx.extend(class_idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def backup_existing_model(output: Path) -> None:
    """Back up output path if it already exists (e.g. user retargeted the
    shipped model on purpose) — never lose a model silently."""
    if not output.exists():
        return
    backup = output.with_suffix(output.suffix + ".bak")
    logger.info("Backing up existing {} -> {}", output, backup)
    shutil.move(str(output), str(backup))


def train(
    data_dir: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    val_split: float,
    seed: int,
) -> float:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("Using GPU: {}", gpus[0].name)
    else:
        logger.info("No GPU detected — training on CPU")

    logger.info("Loading images from {} ...", data_dir)
    x, y = load_dataset(data_dir)

    x_train, y_train, x_val, y_val = stratified_split(x, y, val_split, seed)
    logger.info("Train: {} images, Validation: {} images", len(x_train), len(x_val))

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_accuracy = float(history.history["val_accuracy"][-1])
    logger.info("Final validation accuracy: {:.4f}", val_accuracy)

    backup_existing_model(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    logger.info("Saved trained model to {}", output)

    return val_accuracy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing data/<LETTER>/*.jpg (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Save path (default: {DEFAULT_OUTPUT}, never the shipped model "
        "unless you point --output at it yourself)",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Max training epochs (default: 25)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of each class held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for the train/val split"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    train(
        data_dir=args.data_dir,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
