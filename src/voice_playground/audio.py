"""Audio helpers: PCM -> WAV, WAV info, ffmpeg conversion, writing output (PLAN.md T5).

Conversions between WAV and raw PCM (`audio/l16`, 16-bit little-endian) are done in pure
Python. Anything involving MP3 (or an unrecognised source mime type) goes through the
`ffmpeg` CLI over stdin/stdout pipes. A missing ffmpeg raises `ConfigError` with an install
hint; an ffmpeg failure raises `ProviderError`.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from voice_playground.errors import ConfigError, ProviderError
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

#: Accepted spellings of each format's mime type (lower-case, parameters stripped).
_MIME_ALIASES: dict[str, str] = {
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/vnd.wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mpeg3": "mp3",
    "audio/x-mpeg-3": "mp3",
    "audio/l16": "pcm",
    "audio/pcm": "pcm",
}

_EXTENSIONS: dict[str, str] = {".wav": "wav", ".mp3": "mp3", ".pcm": "pcm"}

FFMPEG_INSTALL_HINT = (
    "install it (macOS: `brew install ffmpeg`; Debian/Ubuntu: `apt install ffmpeg`)"
)
FFMPEG_TIMEOUT_S = 300


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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return buf.getvalue()


def wav_info(data: bytes) -> WavInfo:
    """Parse WAV bytes with the `wave` module. Invalid data raises `ProviderError`."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return WavInfo(
                sample_rate=w.getframerate(),
                channels=w.getnchannels(),
                sample_width=w.getsampwidth(),
                frames=w.getnframes(),
            )
    except (wave.Error, EOFError) as exc:
        raise ProviderError(f"invalid WAV audio: {exc}") from exc


def _read_wav(data: bytes) -> tuple[WavInfo, bytes]:
    """Return the WAV parameters and all PCM frames.

    Reads frames until EOF instead of trusting the header's frame count, so streamed WAVs
    (e.g. ffmpeg writing to a pipe, which sets the sizes to 0xFFFFFFFF) are handled.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            chunks = []
            while chunk := w.readframes(65536):
                chunks.append(chunk)
            frames_data = b"".join(chunks)
            frame_size = w.getsampwidth() * w.getnchannels()
            info = WavInfo(
                sample_rate=w.getframerate(),
                channels=w.getnchannels(),
                sample_width=w.getsampwidth(),
                frames=len(frames_data) // frame_size if frame_size else 0,
            )
    except (wave.Error, EOFError) as exc:
        raise ProviderError(f"invalid WAV audio: {exc}") from exc
    return info, frames_data


def _source_format(result: AudioResult) -> str | None:
    """ "wav" | "mp3" | "pcm" for a known mime type, else `None`."""
    mime = result.mime_type.split(";", 1)[0].strip().lower()
    return _MIME_ALIASES.get(mime)


def _check_format(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt not in MIME_TYPES:
        raise ConfigError(
            f"unsupported audio format {fmt!r}; expected one of: {', '.join(MIME_TYPES)}"
        )
    return fmt


def _ffmpeg(input_args: list[str], data: bytes, output_args: list[str]) -> bytes:
    """Run ffmpeg reading `data` on stdin and return stdout."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise ConfigError(
            f"ffmpeg is required for this audio conversion but was not found; {FFMPEG_INSTALL_HINT}"
        )
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", *input_args, "-i", "pipe:0"]
    cmd += [*output_args, "pipe:1"]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s") from exc
    except OSError as exc:
        raise ProviderError(f"could not run ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = stderr[-1] if stderr else f"exit code {proc.returncode}"
        raise ProviderError(f"ffmpeg conversion failed: {detail}")
    if not proc.stdout:
        raise ProviderError("ffmpeg conversion produced no output")
    return bytes(proc.stdout)


def _ffmpeg_input_args(result: AudioResult, src: str | None) -> list[str]:
    if src != "pcm":
        return []  # let ffmpeg probe the container
    if not result.sample_rate:
        raise ProviderError("cannot convert raw PCM audio without a sample rate")
    return ["-f", "s16le", "-ar", str(result.sample_rate), "-ac", str(result.channels)]


def _to_wav(result: AudioResult, src: str | None) -> AudioResult:
    if src == "pcm":
        if not result.sample_rate:
            raise ProviderError("cannot convert raw PCM audio without a sample rate")
        data = pcm_to_wav(result.data, result.sample_rate, result.channels)
        return AudioResult(data, MIME_TYPES["wav"], result.sample_rate, result.channels)
    raw = _ffmpeg(_ffmpeg_input_args(result, src), result.data, ["-f", "wav", "-c:a", "pcm_s16le"])
    # Re-wrap so the header carries real sizes (ffmpeg can't seek back on a pipe).
    info, frames = _read_wav(raw)
    data = pcm_to_wav(frames, info.sample_rate, info.channels, info.sample_width)
    return AudioResult(data, MIME_TYPES["wav"], info.sample_rate, info.channels)


def _wav_to_pcm(result: AudioResult) -> AudioResult:
    info, frames = _read_wav(result.data)
    if info.sample_width != 2:
        # audio/l16 is 16-bit: let ffmpeg resample the sample width.
        frames = _ffmpeg([], result.data, ["-f", "s16le", "-c:a", "pcm_s16le"])
    return AudioResult(frames, MIME_TYPES["pcm"], info.sample_rate, info.channels)


def convert(result: AudioResult, fmt: str) -> AudioResult:
    """Convert `result` to `fmt` ("wav" | "mp3" | "pcm").

    Returns `result` unchanged if it is already in `fmt`. WAV <-> PCM is pure Python; MP3
    (and unknown source mime types) use the ffmpeg CLI. Raises `ConfigError` for an unknown
    `fmt` or if ffmpeg is needed but not installed (with an install hint), and
    `ProviderError` if ffmpeg fails or the input audio is invalid.
    """
    fmt = _check_format(fmt)
    src = _source_format(result)
    if src == fmt:
        return result
    if fmt == "wav":
        return _to_wav(result, src)
    if fmt == "pcm":
        wav = result if src == "wav" else _to_wav(result, src)
        return _wav_to_pcm(wav)
    # fmt == "mp3"
    data = _ffmpeg(_ffmpeg_input_args(result, src), result.data, ["-f", "mp3"])
    return AudioResult(data, MIME_TYPES["mp3"], result.sample_rate, result.channels)


def format_from_path(path: Path) -> str:
    """Infer "wav" | "mp3" | "pcm" from the file extension; unknown raises `ConfigError`."""
    fmt = _EXTENSIONS.get(Path(path).suffix.lower())
    if fmt is None:
        raise ConfigError(
            f"cannot infer audio format from {str(path)!r}; use a "
            f"{'/'.join(_EXTENSIONS)} extension or pass --format"
        )
    return fmt


def write_output(result: AudioResult, path: Path, fmt: str | None = None) -> Path:
    """Write `result` to `path`, converting to `fmt` (default: inferred from the extension).

    Creates parent directories. Returns the path written.
    """
    path = Path(path)
    target = _check_format(fmt) if fmt is not None else format_from_path(path)
    converted = convert(result, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(converted.data)
    return path
