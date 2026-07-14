from pathlib import Path

from src.app import ASSETS, resolve_model_path


def test_resolve_model_path_defaults_to_shipped_model():
    assert resolve_model_path(None) == ASSETS / "cnn8grps_rad1_model.h5"


def test_resolve_model_path_uses_custom_path_when_given():
    custom = "assets/my_custom_model.h5"
    assert resolve_model_path(custom) == Path(custom)
