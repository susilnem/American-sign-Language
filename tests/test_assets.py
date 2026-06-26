from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_model_exists():
    assert (ROOT / "assets" / "cnn8grps_rad1_model.h5").exists()


def test_signs_exists():
    assert (ROOT / "assets" / "signs.png").exists()


def test_white_exists():
    assert (ROOT / "assets" / "white.jpg").exists()
