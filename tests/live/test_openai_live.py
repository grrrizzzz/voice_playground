"""Live smoke tests for the OpenAI provider. Skipped unless OPENAI_API_KEY is set."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from voice_playground.providers.base import ResolvedVoice, STTRequest, TTSRequest
from voice_playground.providers.openai import OpenAIProvider
from voice_playground.settings import get_settings

pytestmark = pytest.mark.live("openai")

TEXT = "The purple elephant ate a banana in the garden."


def test_tts_stt_round_trip(tmp_path: Path) -> None:
    provider = OpenAIProvider(get_settings())
    audio = provider.tts(
        TTSRequest(
            text=TEXT,
            model=None,
            voice=ResolvedVoice(provider_voice="marin", style="clear, neutral delivery"),
            output_format="wav",
        )
    )
    assert audio.mime_type == "audio/wav"
    assert audio.sample_rate == 24_000
    with wave.open(io.BytesIO(audio.data)) as w:
        assert w.getframerate() == 24_000
        assert w.getnframes() / w.getframerate() > 0.3

    path = tmp_path / "round_trip.wav"
    path.write_bytes(audio.data)
    transcript = provider.stt(STTRequest(audio_path=path, model=None, language="en"))
    assert "banana" in transcript.text.lower()


def test_tts_legacy_model_mp3() -> None:
    provider = OpenAIProvider(get_settings())
    audio = provider.tts(
        TTSRequest(
            text="Hello.",
            model="tts-1",
            voice=ResolvedVoice(provider_voice="coral"),
            output_format="mp3",
        )
    )
    assert audio.mime_type == "audio/mpeg"
    assert len(audio.data) > 1000
