"""Orchestration layer between the CLI and providers/voices/audio/playback. Implemented in T8.

The CLI stays thin: it parses flags and calls one function here per command. Functions raise
`VPError` subclasses; the CLI maps them to exit codes (PLAN.md §2 rule 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_playground.errors import VPError
from voice_playground.providers.base import Capability, CreatedVoice, LibraryVoice, Transcript
from voice_playground.providers.registry import API_KEY_ENV, get_provider_class, provider_names
from voice_playground.settings import Settings
from voice_playground.voices import VoiceConfig


@dataclass(frozen=True)
class ProviderInfo:
    """What `vp providers` shows. `key_present` is a yes/no; key values are never exposed."""

    name: str
    capabilities: frozenset[Capability]
    default_tts_model: str | None
    default_stt_model: str | None
    key_env: str | None
    key_present: bool | None  # None = the provider needs no key
    error: str | None = None  # set if the provider module could not be imported


def providers_info(settings: Settings) -> list[ProviderInfo]:
    """Describe every registered provider without constructing it (no key needed)."""
    infos: list[ProviderInfo] = []
    for name in provider_names():
        key_env = API_KEY_ENV.get(name)
        key_present: bool | None = None
        if key_env is not None:
            secret = getattr(settings, key_env.lower(), None)
            key_present = secret is not None and bool(secret.get_secret_value().strip())
        try:
            cls = get_provider_class(name)
        except VPError as exc:
            infos.append(
                ProviderInfo(name, frozenset(), None, None, key_env, key_present, error=str(exc))
            )
            continue
        infos.append(
            ProviderInfo(
                name=name,
                capabilities=cls.capabilities,
                default_tts_model=cls.default_tts_model,
                default_stt_model=cls.default_stt_model,
                key_env=key_env,
                key_present=key_present,
            )
        )
    return infos


def run_tts(
    *,
    settings: Settings,
    text: str | None,
    text_file: Path | None,
    provider: str | None,
    model: str | None,
    voice: str | None,
    output: Path | None,
    style: str | None,
    fmt: str | None,
    play: str | None,
    speaker: str | None,
    auto_create: bool = True,
) -> Path | None:
    """Synthesize speech (PLAN.md §2 rules 1-7). Returns the written path, or None if only played.

    - Text: `text`, else `text_file`, else stdin when it isn't a TTY, else `ConfigError`.
    - Voice resolution via `voices.resolve_voice`; auto-create designed/cloned voices unless
      `auto_create` is False (then `ConfigError`).
    - Precedence: CLI flag > voice config field > provider default.
    - With `output`: write (format from extension unless `fmt`), play only if `play` is set.
      Without `output`: play (`local` default; `sonos` or `speaker` -> Sonos).
    """
    raise NotImplementedError("service.run_tts is implemented in T8")


def run_stt(
    *,
    settings: Settings,
    input_path: Path,
    provider: str | None,
    model: str | None,
    output: Path | None,
    language: str | None,
    prompt: str | None,
) -> Transcript:
    """Transcribe `input_path`; write the text to `output` if given. Returns the transcript."""
    raise NotImplementedError("service.run_stt is implemented in T8")


def voices_list(settings: Settings, provider: str | None = None) -> list[VoiceConfig]:
    """Voice configs in `settings.voices_dir`, optionally filtered by provider."""
    raise NotImplementedError("service.voices_list is implemented in T8")


def voices_show(settings: Settings, name: str) -> dict[str, Any]:
    """Resolved config plus cached remote id/expiry for voice `name`."""
    raise NotImplementedError("service.voices_show is implemented in T8")


def voices_create(settings: Settings, name: str, force: bool = False) -> CreatedVoice:
    """Create the designed/cloned voice `name` remotely and cache its id.

    If a valid cached id exists and `force` is False, return it without calling the provider.
    """
    raise NotImplementedError("service.voices_create is implemented in T8")


def voices_library(settings: Settings, provider: str, **filters: Any) -> list[LibraryVoice]:
    """List the provider's remote voice catalog (filters: search, language, gender, ...)."""
    raise NotImplementedError("service.voices_library is implemented in T8")


def voices_delete(settings: Settings, name: str) -> None:
    """Delete voice `name`'s remote voice and its cache entry."""
    raise NotImplementedError("service.voices_delete is implemented in T8")


def play_file(
    settings: Settings, path: Path, play: str | None = None, speaker: str | None = None
) -> None:
    """Play an existing audio file locally (default) or on Sonos (`play="sonos"` or `speaker`)."""
    raise NotImplementedError("service.play_file is implemented in T8")
