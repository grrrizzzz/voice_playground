"""Live smoke tests for the Google Gemini provider. Skipped unless GOOGLE_API_KEY is set."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from voice_playground.providers.base import ResolvedVoice, STTRequest, TTSRequest
from voice_playground.providers.google import GoogleProvider
from voice_playground.settings import get_settings

pytestmark = pytest.mark.live("google")

SENTENCE = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def provider() -> GoogleProvider:
    return GoogleProvider(get_settings())


@pytest.fixture(scope="module")
def spoken_wav(provider: GoogleProvider) -> bytes:
    result = provider.tts(TTSRequest(text=SENTENCE, model=None, voice=ResolvedVoice("Kore")))
    assert result.mime_type == "audio/wav"
    return result.data


def _duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def test_tts_returns_valid_wav(spoken_wav: bytes) -> None:
    assert _duration(spoken_wav) > 0.3


def test_stt_transcribes_tts_output(
    provider: GoogleProvider, spoken_wav: bytes, tmp_path: Path
) -> None:
    audio = tmp_path / "fox.wav"
    audio.write_bytes(spoken_wav)
    transcript = provider.stt(STTRequest(audio_path=audio, model=None, language="en"))
    assert "fox" in transcript.text.lower()
