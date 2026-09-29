"""Audio helpers: PCM -> WAV, WAV info, ffmpeg conversion, writing output. Implemented in T5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_playground.providers.base import AudioResult

__all__ = [
    "MIME_TYPES",
    "AudioResult",
    "WavInfo",
    "convert",
    "format_from_path",
    "pcm_to_wav",
    "wav_info",
    "write_output",
]

#: Output format -> mime type used in `AudioResult.mime_type`.
MIME_TYPES: dict[str, str] = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/l16"}


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int  # bytes per sample
    frames: int

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def pcm_to_wav(data: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM in a WAV container."""
    raise NotImplementedError("audio.pcm_to_wav is implemented in T5")


def wav_info(data: bytes) -> WavInfo:
    """Parse WAV bytes with the `wave` module. Invalid data raises `ProviderError`."""
    raise NotImplementedError("audio.wav_info is implemented in T5")


def convert(result: AudioResult, fmt: str) -> AudioResult:
    """Convert `result` to `fmt` ("wav" | "mp3" | "pcm") using the ffmpeg CLI.

    Returns `result` unchanged if it is already in `fmt`. Raises `ProviderError` with a clear
    message if ffmpeg is not installed or fails.
    """
    raise NotImplementedError("audio.convert is implemented in T5")


def format_from_path(path: Path) -> str:
    """Infer "wav" | "mp3" | "pcm" from the file extension; unknown raises `ConfigError`."""
    raise NotImplementedError("audio.format_from_path is implemented in T5")


def write_output(result: AudioResult, path: Path, fmt: str | None = None) -> Path:
    """Write `result` to `path`, converting to `fmt` (default: inferred from the extension).

    Creates parent directories. Returns the path written.
    """
    raise NotImplementedError("audio.write_output is implemented in T5")
