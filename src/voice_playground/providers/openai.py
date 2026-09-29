"""OpenAI provider. Stub written in T1; implemented in T3 (see PLAN.md)."""

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


class OpenAIProvider(BaseProvider):
    name: ClassVar[str] = "openai"
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.TTS, Capability.STT})
    default_tts_model: ClassVar[str | None] = "gpt-4o-mini-tts"
    default_stt_model: ClassVar[str | None] = "gpt-4o-transcribe"  # T3: verify against current docs

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key = require_secret(settings.openai_api_key, "OPENAI_API_KEY")

    def tts(self, req: TTSRequest) -> AudioResult:
        raise NotImplementedError("openai tts is implemented in T3")

    def stt(self, req: STTRequest) -> Transcript:
        raise NotImplementedError("openai stt is implemented in T3")

    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice:
        raise NotImplementedError("openai create_voice is implemented in T3")

    def delete_voice(self, remote_id: str) -> None:
        raise NotImplementedError("openai delete_voice is implemented in T3")

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        raise NotImplementedError("openai list_library is implemented in T3")
