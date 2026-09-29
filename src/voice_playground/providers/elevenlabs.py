"""ElevenLabs provider. Stub written in T1; implemented in T4 (see PLAN.md)."""

from __future__ import annotations

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
from voice_playground.settings import require_secret

if TYPE_CHECKING:
    from voice_playground.settings import Settings
    from voice_playground.voices import VoiceConfig


class ElevenLabsProvider(BaseProvider):
    name: ClassVar[str] = "elevenlabs"
    capabilities: ClassVar[frozenset[Capability]] = frozenset(
        {
            Capability.TTS,
            Capability.STT,
            Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE,
            Capability.VOICE_LIBRARY,
        }
    )
    default_tts_model: ClassVar[str | None] = "eleven_v3"
    default_stt_model: ClassVar[str | None] = "scribe_v1"  # T4: verify against current docs

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key = require_secret(settings.elevenlabs_api_key, "ELEVENLABS_API_KEY")

    def tts(self, req: TTSRequest) -> AudioResult:
        raise NotImplementedError("elevenlabs tts is implemented in T4")

    def stt(self, req: STTRequest) -> Transcript:
        raise NotImplementedError("elevenlabs stt is implemented in T4")

    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice:
        raise NotImplementedError("elevenlabs create_voice is implemented in T4")

    def delete_voice(self, remote_id: str) -> None:
        raise NotImplementedError("elevenlabs delete_voice is implemented in T4")

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        raise NotImplementedError("elevenlabs list_library is implemented in T4")
