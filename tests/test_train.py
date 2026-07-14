from string import ascii_uppercase

import cv2
import numpy as np
import pytest

from src.train import (
    LETTER_TO_GROUP,
    NUM_GROUPS,
    SHIPPED_MODEL,
    build_model,
    load_dataset,
    parse_args,
    stratified_split,
)


def test_letter_to_group_covers_every_letter_exactly_once():
    assert set(LETTER_TO_GROUP) == set(ascii_uppercase)


def test_default_output_does_not_overwrite_shipped_model():
    """Training should be opt-in to overwrite the shipped model — default
    output goes to a separate file so `uv run python src/app.py` keeps
    working with the original model unless a user explicitly retrains it."""
    assert parse_args([]).output != SHIPPED_MODEL


def test_user_can_still_target_shipped_model_explicitly():
    assert parse_args(["--output", str(SHIPPED_MODEL)]).output == SHIPPED_MODEL


@pytest.mark.parametrize(
    "letter,expected_group",
    [
        ("A", 0),
        ("S", 0),
        ("T", 0),
        ("B", 1),
        ("W", 1),
        ("C", 2),
        ("O", 2),
        ("G", 3),
        ("H", 3),
        ("L", 4),
        ("P", 5),
        ("Z", 5),
        ("X", 6),
        ("Y", 7),
        ("J", 7),
    ],
)
def test_letter_maps_to_documented_group(letter, expected_group):
    """Pins down the letter->group table against workflow.md's documented
    grouping — predictor.predict_letter/disambiguate hardcode this scheme,
    so a silent drift here would train a model incompatible with the rest
    of the app."""
    assert LETTER_TO_GROUP[letter] == expected_group


def test_build_model_matches_shipped_model_architecture():
    """The model this script builds must be an exact architectural match
    for assets/cnn8grps_rad1_model.h5 (same input/output shape and param
    count) so predictor.py keeps working unchanged against a retrained
    model."""
    model = build_model()
    assert model.input_shape == (None, 400, 400, 3)
    assert model.output_shape == (None, NUM_GROUPS)
    assert model.count_params() == 1_119_720


def test_stratified_split_keeps_every_class_in_both_splits():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, size=(40, 4, 4, 3)).astype("float32")
    y = np.array([label for label in range(NUM_GROUPS) for _ in range(5)])

    x_train, y_train, x_val, y_val = stratified_split(x, y, val_split=0.2, seed=0)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert len(x_train) + len(x_val) == len(x)
    assert set(y_train.tolist()) | set(y_val.tolist()) == set(range(NUM_GROUPS))


def test_load_dataset_errors_clearly_when_data_dir_missing(tmp_path):
    with pytest.raises(SystemExit, match="Data directory not found"):
        load_dataset(tmp_path / "does-not-exist")


def test_load_dataset_errors_when_an_entire_group_has_no_images(tmp_path):
    """Only letter A (group 0) is collected, so groups 1-7 have zero images —
    that must fail with a clear message rather than silently training a
    model that can never predict those groups."""
    letter_dir = tmp_path / "A"
    letter_dir.mkdir()
    cv2.imwrite(str(letter_dir / "0.jpg"), np.zeros((400, 400, 3), dtype="uint8"))

    with pytest.raises(SystemExit, match="Cannot train"):
        load_dataset(tmp_path)


def test_load_dataset_succeeds_with_one_letter_per_group(tmp_path):
    letters_covering_all_groups = ["A", "B", "C", "G", "L", "P", "X", "Y"]
    for letter in letters_covering_all_groups:
        letter_dir = tmp_path / letter
        letter_dir.mkdir()
        blank = np.zeros((400, 400, 3), dtype="uint8")
        for i in range(2):
            cv2.imwrite(str(letter_dir / f"{i}.jpg"), blank)

    x, y = load_dataset(tmp_path)

    assert len(x) == len(letters_covering_all_groups) * 2
    assert set(y.tolist()) == set(range(NUM_GROUPS))
