"""Google Gemini provider. Stub written in T1; implemented in T2 (see PLAN.md)."""

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


class GoogleProvider(BaseProvider):
    name: ClassVar[str] = "google"
    capabilities: ClassVar[frozenset[Capability]] = frozenset(
        {
            Capability.TTS,
            Capability.STT,
            Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE,
            Capability.VOICE_LIBRARY,
        }
    )
    default_tts_model: ClassVar[str | None] = "gemini-3.8-flash-tts"
    default_stt_model: ClassVar[str | None] = None  # T2: verify against current docs

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key = require_secret(settings.google_api_key, "GOOGLE_API_KEY")

    def tts(self, req: TTSRequest) -> AudioResult:
        raise NotImplementedError("google tts is implemented in T2")

    def stt(self, req: STTRequest) -> Transcript:
        raise NotImplementedError("google stt is implemented in T2")

    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice:
        raise NotImplementedError("google create_voice is implemented in T2")

    def delete_voice(self, remote_id: str) -> None:
        raise NotImplementedError("google delete_voice is implemented in T2")

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        raise NotImplementedError("google list_library is implemented in T2")
