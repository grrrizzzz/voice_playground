"""Voice config models, loading, and the remote-voice cache (PLAN.md §3). Implemented in T7.

The pydantic model shapes are defined in T1 because `providers/base.py` references
`VoiceConfig`. The functions and `VoiceCache` methods are typed stubs for T7 to fill in.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from voice_playground.providers.base import ResolvedVoice


class _VoiceConfigBase(BaseModel):
    """Fields shared by every voice config type. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str  # must match the YAML file name (without extension)
    provider: str | None = None  # google | openai | elevenlabs (| fake in tests)
    model: str | None = None
    style: str | None = None  # google speech_metadata.style / openai instructions
    language: str | None = None
    sample_rate: int | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)  # passed through as-is


class PrebuiltVoiceConfig(_VoiceConfigBase):
    """A provider's built-in voice (name/id in `voice`), optionally with a style."""

    type: Literal["prebuilt"] = "prebuilt"
    voice: str


class DesignedVoiceConfig(_VoiceConfigBase):
    """A voice created remotely from a natural-language `description`."""

    type: Literal["designed"] = "designed"
    description: str
    store: bool = True  # google: stateful voice_ (1y) vs stateless voicekey_ (7d)


class ClonedVoiceConfig(_VoiceConfigBase):
    """A voice cloned remotely from `reference_audio` (path relative to the repo root)."""

    type: Literal["cloned"] = "cloned"
    reference_audio: Path
    store: bool = True  # google: stateful voice_ (1y) vs stateless voicekey_ (7d)


VoiceConfig = Annotated[
    PrebuiltVoiceConfig | DesignedVoiceConfig | ClonedVoiceConfig,
    Field(discriminator="type"),
]


class CacheEntry(BaseModel):
    """One entry in `.vp_cache/voices.json`, keyed by voice config name."""

    provider: str
    remote_id: str
    config_hash: str
    created_at: datetime
    expires_at: datetime | None = None


def load_voice(name: str, voices_dir: Path) -> VoiceConfig:
    """Load and validate `voices_dir/<name>.yaml`.

    Must raise `ConfigError` (message names the file) when the file is missing, is invalid
    YAML, fails validation (wrong/missing fields for its `type`), or when the config's `name`
    does not match the file name.
    """
    raise NotImplementedError("voices.load_voice is implemented in T7")


def list_voices(voices_dir: Path, provider: str | None = None) -> list[VoiceConfig]:
    """Load every `*.yaml` voice config in `voices_dir`, optionally filtered by provider.

    Sorted by name. Invalid files raise `ConfigError` naming the file.
    """
    raise NotImplementedError("voices.list_voices is implemented in T7")


def config_hash(cfg: VoiceConfig) -> str:
    """Stable hash of the fields that define a remote voice (used to detect stale cache)."""
    raise NotImplementedError("voices.config_hash is implemented in T7")


class VoiceCache:
    """Remote ids of created voices, stored as JSON at `cache_dir/voices.json`.

    Writes are atomic (write temp file + rename). Entries are keyed by voice config name and
    store `{provider, remote_id, config_hash, created_at, expires_at}`.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.path = cache_dir / "voices.json"

    def get(self, name: str) -> CacheEntry | None:
        """Return the entry for `name`, or None."""
        raise NotImplementedError("VoiceCache.get is implemented in T7")

    def put(self, name: str, entry: CacheEntry) -> None:
        """Insert or replace the entry for `name` (atomic write)."""
        raise NotImplementedError("VoiceCache.put is implemented in T7")

    def delete(self, name: str) -> None:
        """Remove the entry for `name` if present (atomic write)."""
        raise NotImplementedError("VoiceCache.delete is implemented in T7")

    def is_valid(self, name: str, cfg: VoiceConfig, now: datetime | None = None) -> bool:
        """True if an entry exists, its hash matches `cfg`, and it has not expired.

        A stale hash should warn on stderr (the caller then recreates with `--force` behavior).
        """
        raise NotImplementedError("VoiceCache.is_valid is implemented in T7")


def resolve_voice(
    name_or_raw: str,
    provider_flag: str | None,
    *,
    voices_dir: Path,
    cache: VoiceCache,
) -> tuple[str | None, ResolvedVoice, bool]:
    """Resolve `--voice` into `(provider_name, ResolvedVoice, needs_create)`. PLAN.md §2 rules 2-4.

    - If `voices_dir/<name_or_raw>.yaml` exists, load it. Otherwise treat `name_or_raw` as a raw
      provider voice name/id: return `(provider_flag, ResolvedVoice(provider_voice=name_or_raw),
      False)`.
    - If the config sets `provider` and `provider_flag` is a different value, raise
      `ConfigError`. If `provider_flag` is None, use the config's provider.
    - prebuilt: `provider_voice` = config `voice`. designed/cloned: `provider_voice` = cached
      remote id when valid; otherwise `needs_create=True` (provider_voice left empty for the
      caller to fill after creation).
    - `style`, `language`, `sample_rate`, `provider_options` (-> `options`) and `config_name`
      come from the config.
    """
    raise NotImplementedError("voices.resolve_voice is implemented in T7")
