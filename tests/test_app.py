from pathlib import Path

from src.app import ASSETS, STABLE_FRAMES, Application, resolve_model_path


def test_resolve_model_path_defaults_to_shipped_model():
    assert resolve_model_path(None) == ASSETS / "cnn8grps_rad1_model.h5"


def test_resolve_model_path_uses_custom_path_when_given():
    custom = "assets/my_custom_model.h5"
    assert resolve_model_path(custom) == Path(custom)


def _bare_application():
    app = Application.__new__(Application)
    app.prev_char = ""
    app.count = -1
    app.ten_prev_char = [" "] * 10
    app._pending_char = None
    app._pending_count = 0
    app.str = " "
    app.current_symbol = "C"
    app.word1 = app.word2 = app.word3 = app.word4 = " "
    return app


def test_apply_prediction_ignores_next_gesture_before_any_letter_committed():
    app = _bare_application()
    for _ in range(STABLE_FRAMES):
        app._apply_prediction("next")
    assert app.str == " ", "no letter was ever committed — str must stay empty"
