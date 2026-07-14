#!/usr/bin/env bash
# One-command setup: installs the system library that enables spell
# suggestions (enchant + its aspell backend), then runs `uv sync`. `uv sync`
# itself can't do this step — it's a Python dependency manager with no
# concept of system package managers, and installing OS packages needs sudo,
# which uv intentionally never invokes. Text-to-speech (Piper) needs no
# system library at all — its voice model downloads once, automatically, on
# first use, handled entirely by `src/tts_worker.py`.
# App runs fine without this script too; suggestions just won't work.
set -euo pipefail

install_packages() {
    if command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --needed enchant aspell aspell-en
    elif command -v apt >/dev/null 2>&1; then
        sudo apt update && sudo apt install -y enchant-2 aspell aspell-en
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y enchant2 aspell aspell-en
    elif command -v brew >/dev/null 2>&1; then
        brew install enchant
    else
        echo "Unrecognized package manager — install enchant + aspell (+ aspell-en) manually." >&2
        echo "See src/README.md's System libraries section for per-OS package names." >&2
        return 1
    fi
}

if ! install_packages; then
    echo "Skipping system libraries; continuing with uv sync (suggestions just won't work)." >&2
fi

uv sync

# Best-effort: fetch the Piper voice model now instead of on the first Speak
# click. Not fatal if this fails (e.g. no internet right now) — Speak just
# retries the download on its next click.
uv run python src/tts_worker.py || echo "Skipping Piper voice download; Speak will fetch it on first use instead." >&2
