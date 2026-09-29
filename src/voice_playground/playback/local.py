"""Local playback: afplay (macOS) -> ffplay -> error (PLAN.md T5)."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from voice_playground import audio
from voice_playground.errors import ConfigError, ProviderError
from voice_playground.providers.base import AudioResult

PLAYER_INSTALL_HINT = (
    "no audio player found; install ffmpeg (which provides ffplay), "
    "e.g. `brew install ffmpeg` or `apt install ffmpeg`"
)
#: Seconds to wait for the player to exit after `terminate()` before `kill()`.
TERMINATE_TIMEOUT_S = 2.0

_SUFFIXES = {"audio/wav": ".wav", "audio/mpeg": ".mp3"}


def player_command() -> list[str]:
    """The player command (without the file argument).

    `afplay` on macOS if present, else `ffplay -nodisp -autoexit -loglevel quiet`.
    Neither available -> `ConfigError` with an install hint.
    """
    if sys.platform == "darwin":
        afplay = shutil.which("afplay")
        if afplay:
            return [afplay]
    ffplay = shutil.which("ffplay")
    if ffplay:
        return [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet"]
    raise ConfigError(PLAYER_INSTALL_HINT)


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the player, escalating to kill if it doesn't exit promptly."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=TERMINATE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def play_local(result: AudioResult) -> None:
    """Play `result` on the local speakers, blocking until playback finishes.

    Raw PCM (`audio/l16`) is wrapped as WAV first. The audio is written to a temp file and
    played with `afplay` on macOS, else `ffplay -nodisp -autoexit -loglevel quiet`. No
    player available -> `ConfigError` with an install hint; the player exiting non-zero ->
    `ProviderError`. The temp file is always removed.

    Ctrl-C: the player process is terminated (then killed if it lingers) and the
    `KeyboardInterrupt` is re-raised so the caller decides how to exit.
    """
    cmd = player_command()
    mime = result.mime_type.split(";", 1)[0].strip().lower()
    if mime not in _SUFFIXES:
        # PCM (and any other non wav/mp3 audio) is played as WAV.
        result = audio.convert(result, "wav")
        mime = audio.MIME_TYPES["wav"]

    fd, name = tempfile.mkstemp(prefix="vp-play-", suffix=_SUFFIXES[mime])
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(result.data)
        try:
            proc = subprocess.Popen(
                [*cmd, str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ProviderError(f"could not start audio player {cmd[0]!r}: {exc}") from exc
        try:
            returncode = proc.wait()
        except BaseException:  # KeyboardInterrupt (Ctrl-C) and anything else: stop the player
            _stop(proc)
            raise
        if returncode != 0:
            raise ProviderError(f"audio player {Path(cmd[0]).name} exited with code {returncode}")
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
