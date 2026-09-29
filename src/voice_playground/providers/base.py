"""Core provider contracts (PLAN.md §4). Frozen after T1: do not change without sign-off.

Every provider is constructed as `Cls(settings)`. A provider whose API key is missing raises
`ConfigError` naming the env var at construction time (never at import time). Methods a
provider does not support raise `UnsupportedCapability`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

from voice_playground.errors import ConfigError, UnsupportedCapability
from voice_playground.settings import require_secret

if TYPE_CHECKING:
    from voice_playground.settings import Settings
    from voice_playground.voices import VoiceConfig


class Capability(StrEnum):
    """What a provider can do."""

    TTS = "tts"
    STT = "stt"
    VOICE_DESIGN = "voice_design"
    VOICE_CLONE = "voice_clone"
    VOICE_LIBRARY = "voice_library"


@dataclass(frozen=True)
class ResolvedVoice:
    """A voice ready to hand to a provider.

    `provider_voice` is either a provider voice name/id (e.g. `Kore`, `coral`, an ElevenLabs
    voice_id) or the remote id of a created custom voice. `options` is the voice config's
    `provider_options`, passed through as-is. `config_name` is the voice config name, or
    `None` for a raw provider voice.
    """

    provider_voice: str
    style: str | None = None
    language: str | None = None
    sample_rate: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    config_name: str | None = None


@dataclass(frozen=True)
class CreatedVoice:
    """Result of creating a designed/cloned voice remotely."""

    remote_id: str
    expires_at: datetime | None = None


@dataclass(frozen=True)
class LibraryVoice:
    """One entry of a provider's remote voice catalog."""

    id: str
    name: str
    description: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSRequest:
    text: str
    model: str | None
    voice: ResolvedVoice  # provider voice name OR remote custom-voice id + style/lang/options
    output_format: Literal["wav", "mp3", "pcm"] = "wav"
    sample_rate: int | None = None


@dataclass(frozen=True)
class AudioResult:
    data: bytes
    mime_type: str  # "audio/wav", "audio/mpeg", "audio/l16"
    sample_rate: int | None
    channels: int = 1


@dataclass(frozen=True)
class STTRequest:
    audio_path: Path
    model: str | None
    language: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    segments: list[dict[str, Any]] | None = None


@runtime_checkable
class Provider(Protocol):
    name: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]
    default_tts_model: ClassVar[str | None]
    default_stt_model: ClassVar[str | None]

    def tts(self, req: TTSRequest) -> AudioResult: ...
    def stt(self, req: STTRequest) -> Transcript: ...
    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice: ...  # remote_id, expires_at
    def delete_voice(self, remote_id: str) -> None: ...
    def list_library(self, **filters: Any) -> list[LibraryVoice]: ...


class BaseProvider:
    """Optional convenience base class implementing `Provider`.

    Stores `settings` and makes every method raise `UnsupportedCapability` by default, so a
    subclass only overrides what it supports. Subclasses must set the four `Provider`
    ClassVars, and may set:

    - `api_key_env`: the env var holding the provider's API key (`None` = no key). It drives
      `vp providers` key status and secret masking in CLI errors, and `_require_api_key()`.
    - `default_voice`: the voice used when `vp tts` gets no `--voice` (`None` = required).

    These extras live here, not on the `Provider` Protocol; code reading them uses
    `getattr(cls, "api_key_env", None)` so a Protocol-only provider still works.
    """

    name: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]] = frozenset()
    default_tts_model: ClassVar[str | None] = None
    default_stt_model: ClassVar[str | None] = None
    api_key_env: ClassVar[str | None] = None
    default_voice: ClassVar[str | None] = None

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _require_api_key(self) -> str:
        """The plain API key from `api_key_env`, or `ConfigError` naming the env var."""
        if self.api_key_env is None:
            raise ConfigError(f"provider '{self.name}' does not declare api_key_env")
        return require_secret(self.settings.api_key(self.api_key_env), self.api_key_env)

    def validate_tts(
        self, *, model: str | None, custom_voice: bool, sample_rate: int | None
    ) -> None:
        """Cheap provider-side checks run BEFORE a designed/cloned voice is auto-created.

        `model` is the effective model (None = provider default), `custom_voice` is True when
        the request will use a created (designed/cloned) voice, `sample_rate` the requested
        rate. Raise `ConfigError` for a combination `tts` would reject. Default: no checks.
        """
        return None

    def _unsupported(self, capability: Capability) -> UnsupportedCapability:
        return UnsupportedCapability(f"provider '{self.name}' does not support {capability.value}")

    def tts(self, req: TTSRequest) -> AudioResult:
        raise self._unsupported(Capability.TTS)

    def stt(self, req: STTRequest) -> Transcript:
        raise self._unsupported(Capability.STT)

    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice:
        if cfg.type == "cloned":
            raise self._unsupported(Capability.VOICE_CLONE)
        raise self._unsupported(Capability.VOICE_DESIGN)

    def delete_voice(self, remote_id: str) -> None:
        raise self._unsupported(Capability.VOICE_DESIGN)

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        raise self._unsupported(Capability.VOICE_LIBRARY)
