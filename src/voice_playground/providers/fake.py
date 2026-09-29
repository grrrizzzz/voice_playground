"""Deterministic fake provider used by tests. Needs no API key and makes no network calls."""

from __future__ import annotations

import io
import math
import struct
import wave
from typing import TYPE_CHECKING, Any, ClassVar

from voice_playground.providers.base import (
    AudioResult,
    BaseProvider,
    Capability,
    CreatedVoice,
    LibraryVoice,
    STTRequest,
    Transcript,
    TTSRequest,
)

if TYPE_CHECKING:
    from voice_playground.voices import VoiceConfig

DEFAULT_SAMPLE_RATE = 24_000
DURATION_S = 0.5
FREQUENCY_HZ = 440.0
AMPLITUDE = 0.3

LIBRARY: tuple[LibraryVoice, ...] = (
    LibraryVoice(
        id="fake_alto",
        name="Alto",
        description="Warm, low fake voice",
        extra={"language": "en-US", "gender": "female"},
    ),
    LibraryVoice(
        id="fake_baritone",
        name="Baritone",
        description="Deep fake voice",
        extra={"language": "en-GB", "gender": "male"},
    ),
    LibraryVoice(
        id="fake_tenor",
        name="Tenor",
        description="Bright fake voice",
        extra={"language": "en-US", "gender": "male"},
    ),
)


def sine_pcm(sample_rate: int = DEFAULT_SAMPLE_RATE, duration_s: float = DURATION_S) -> bytes:
    """Mono 16-bit little-endian PCM of a 440 Hz sine tone."""
    frames = int(sample_rate * duration_s)
    peak = int(32767 * AMPLITUDE)
    samples = (
        int(peak * math.sin(2 * math.pi * FREQUENCY_HZ * i / sample_rate)) for i in range(frames)
    )
    return b"".join(struct.pack("<h", s) for s in samples)


def sine_wav(sample_rate: int = DEFAULT_SAMPLE_RATE, duration_s: float = DURATION_S) -> bytes:
    """WAV container around `sine_pcm`."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(sine_pcm(sample_rate, duration_s))
    return buf.getvalue()


class FakeProvider(BaseProvider):
    """Implements every capability deterministically.

    - `tts` returns a 0.5 s 440 Hz sine: raw PCM (`audio/l16`) for `pcm`, otherwise WAV
      (`audio/wav`, also for `mp3`; callers convert when the mime type doesn't match).
      Sample rate is `req.sample_rate`, else the voice's, else 24 kHz.
    - `stt` echoes the audio file name as the transcript.
    - `create_voice` returns `fake_voice_<name>` with no expiry.
    """

    name: ClassVar[str] = "fake"
    capabilities: ClassVar[frozenset[Capability]] = frozenset(Capability)
    default_tts_model: ClassVar[str | None] = "fake-tts"
    default_stt_model: ClassVar[str | None] = "fake-stt"
    api_key_env: ClassVar[str | None] = None
    default_voice: ClassVar[str | None] = "fake-voice"

    def tts(self, req: TTSRequest) -> AudioResult:
        rate = req.sample_rate or req.voice.sample_rate or DEFAULT_SAMPLE_RATE
        if req.output_format == "pcm":
            return AudioResult(data=sine_pcm(rate), mime_type="audio/l16", sample_rate=rate)
        return AudioResult(data=sine_wav(rate), mime_type="audio/wav", sample_rate=rate)

    def stt(self, req: STTRequest) -> Transcript:
        return Transcript(text=req.audio_path.name, language=req.language)

    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice:
        return CreatedVoice(remote_id=f"fake_voice_{cfg.name}", expires_at=None)

    def delete_voice(self, remote_id: str) -> None:
        return None

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        search = str(filters.get("search") or "").lower()
        language = filters.get("language")
        gender = filters.get("gender")
        result = []
        for v in LIBRARY:
            haystack = f"{v.name} {v.description or ''}".lower()
            if search and search not in haystack:
                continue
            if language and v.extra.get("language") != language:
                continue
            if gender and v.extra.get("gender") != gender:
                continue
            result.append(v)
        return result
