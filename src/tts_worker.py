"""Text-to-speech via Piper (neural TTS), replacing pyttsx3/espeak-ng.

espeak-ng is a format synthesizer — it's supposed to sound robotic, that's
the technique. Piper is a small offline neural model that sounds like an
actual voice. It has no streaming API, so `speak()` synthesizes the full
sentence up front and plays it via `sounddevice` (already an indirect
mediapipe dependency) rather than piper's own `AudioPlayer`, which shells out
to `ffplay` — an extra system binary this app doesn't otherwise need.

Runs on its own daemon thread per call so a long sentence doesn't block the
Tk main thread.

The voice model (~60MB) downloads once, ever, to .downloaded_cache/ — every
call after is fully offline. See `_ensure_voice` for why the download is
atomic (a non-atomic version could leave a permanently corrupt cached file
after an interrupted download).
"""

import os
import queue
import tempfile
import threading
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd
from loguru import logger
from piper.voice import PiperVoice

VOICE_NAME = "en_US-lessac-medium"
# .downloaded_cache/, not assets/: assets/ is for files this project actually
# ships (checked into git); this is a runtime download cache, gitignored
# wholesale via .gitignore rather than needing a per-file entry — same
# reasoning as detector_worker.py's hand_landmarker.task.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".downloaded_cache"
_MODEL_PATH = CACHE_DIR / f"{VOICE_NAME}.onnx"
_CONFIG_PATH = CACHE_DIR / f"{VOICE_NAME}.onnx.json"

_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
)


def _ensure_voice():
    """Download the voice model + config once, if not already present —
    same pattern as detector_worker.py's mediapipe model download.

    Downloads to a temp file in the same directory first, then renames it
    into place only once the transfer completes — never straight to the
    final path. A download that's interrupted partway (network drop, app
    closed mid-download) previously left a truncated/empty file sitting at
    the final path; `path.exists()` alone treated that as "already have it"
    forever after, permanently breaking Speak until someone noticed and
    deleted the file by hand. An existing empty file is also treated as
    not-downloaded, to self-heal machines that already hit that bug."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path, filename in (
        (_MODEL_PATH, f"{VOICE_NAME}.onnx"),
        (_CONFIG_PATH, f"{VOICE_NAME}.onnx.json"),
    ):
        if path.exists() and path.stat().st_size > 0:
            continue
        fd, tmp_name = tempfile.mkstemp(dir=str(CACHE_DIR))
        os.close(fd)
        try:
            urllib.request.urlretrieve(
                f"{_BASE_URL}/{filename}?download=true", tmp_name
            )
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)


class TTSWorker:
    """Owns one loaded PiperVoice; each `speak()` call synthesizes+plays on
    its own short-lived daemon thread so it never blocks the Tk main thread.

    `sd.stop()` interrupts whatever's currently playing so hitting Speak
    again says the new sentence immediately instead of queueing behind the
    old one — the load-lock must not be held across `sd.wait()`, or a second
    call would block on the lock instead of ever reaching `sd.stop()`.

    State changes go through `latest_state()`, a non-blocking queue poll —
    same pattern as `DetectorWorker.latest_result()`, not a callback that
    calls `root.after()` from this background thread. An earlier version did
    that, and it's a real bug: Tcl's event queue isn't guaranteed thread-safe
    to poke from a non-Tcl thread, so later callbacks could silently go
    missing (symptom: Speak button stuck on "Loading voice..." forever even
    though audio had already played). Polling from the main thread avoids it."""

    def __init__(self):
        self._voice = None
        self._load_lock = threading.Lock()
        self._state_q = queue.Queue(maxsize=1)

    def _get_voice(self):
        # No permanent error-latch on purpose: a failed load is often a
        # transient network hiccup (see _ensure_voice's docstring on
        # interrupted downloads) — retrying on the *next* Speak click should
        # work without restarting the whole app, not fail forever until then.
        with self._load_lock:
            if self._voice is None:
                try:
                    _ensure_voice()
                    self._voice = PiperVoice.load(
                        str(_MODEL_PATH), config_path=str(_CONFIG_PATH)
                    )
                except Exception:
                    logger.exception("TTS voice failed to load")
            return self._voice

    def _push_state(self, state):
        try:
            self._state_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._state_q.put_nowait(state)
        except queue.Full:
            pass

    def latest_state(self):
        """Non-blocking poll for the newest state change since the last
        call, or None if nothing new. Safe to call every tick from the Tk
        main loop."""
        try:
            return self._state_q.get_nowait()
        except queue.Empty:
            return None

    def speak(self, text):
        if not text or not text.strip():
            return
        threading.Thread(target=self._speak_now, args=(text,), daemon=True).start()

    def _speak_now(self, text):
        if self._voice is None:
            self._push_state("loading")
        voice = self._get_voice()
        if voice is None:
            self._push_state("error")
            return
        try:
            sd.stop()
            self._push_state("speaking")
            chunks = list(voice.synthesize(text))
            if not chunks:
                self._push_state("done")
                return
            audio = np.concatenate([c.audio_float_array for c in chunks])
            sd.play(audio, samplerate=chunks[0].sample_rate)
            sd.wait()
            self._push_state("done")
        except Exception:
            logger.exception("TTS synthesis/playback failed")
            self._push_state("error")


if __name__ == "__main__":
    from logging_config import configure_logging

    configure_logging()
    logger.info("Downloading Piper voice '{}' to {}...", VOICE_NAME, CACHE_DIR)
    _ensure_voice()
    logger.info("Done — Speak will work fully offline from now on.")
