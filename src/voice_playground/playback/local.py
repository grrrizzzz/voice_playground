"""Local playback: afplay (macOS) -> ffplay -> error. Implemented in T5."""

from __future__ import annotations

from voice_playground.providers.base import AudioResult


def play_local(result: AudioResult) -> None:
    """Play `result` on the local speakers, blocking until playback finishes.

    Writes a temp file and plays it with `afplay` on macOS, else
    `ffplay -nodisp -autoexit -loglevel quiet`. No player available -> `ProviderError`.
    Ctrl-C stops playback cleanly (terminates the player process).
    """
    raise NotImplementedError("playback.local.play_local is implemented in T5")
