"""Live ElevenLabs smoke tests. Skipped unless ELEVENLABS_API_KEY is set (see conftest.py)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from voice_playground.providers.base import ResolvedVoice, STTRequest, TTSRequest
from voice_playground.providers.elevenlabs import ElevenLabsProvider
from voice_playground.settings import get_settings

#: A premade ElevenLabs voice ("George"), used in the ElevenLabs API reference examples.
PREMADE_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

pytestmark = pytest.mark.live("elevenlabs")


def test_tts_stt_round_trip(tmp_path: Path) -> None:
    provider = ElevenLabsProvider(get_settings())
    result = provider.tts(
        TTSRequest(
            text="The purple elephant is dancing in the garden.",
            model=None,
            voice=ResolvedVoice(provider_voice=PREMADE_VOICE_ID),
        )
    )
    assert result.mime_type == "audio/wav"
    with wave.open(io.BytesIO(result.data)) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() / w.getframerate() > 0.3

    audio = tmp_path / "round_trip.wav"
    audio.write_bytes(result.data)
    transcript = provider.stt(STTRequest(audio_path=audio, model=None, language="en"))
    assert "elephant" in transcript.text.lower()
