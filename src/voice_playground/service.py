"""Orchestration layer between the CLI and providers/voices/audio/playback (PLAN.md §2).

The CLI stays thin: it parses flags and calls one function here per command. Functions raise
`VPError` subclasses; the CLI maps them to exit codes (PLAN.md §2 rule 7).

TTS pipeline (`run_tts`):
1. Text: `--text`, else `--text-file`, else stdin when it isn't a TTY, else `ConfigError`.
2. Voice: `voices.resolve_voice` (config file or raw provider voice name). With no `--voice`,
   the provider class's `default_voice` is used. A provider is required (flag or config);
   no default provider is guessed.
3. Designed/cloned voices without a valid cached id are created remotely and cached, unless
   `auto_create` is False (`ConfigError`). Cheap provider checks (`validate_tts`) run first,
   and the replaced remote voice (stale / expired / other provider / `--force`) is deleted.
4. Precedence: CLI flag > voice config field > provider default (model, style).
5. The provider result is converted once with `audio.convert` (for `-o` and local playback),
   so the requested format is guaranteed even when a provider returns WAV for an MP3 request.
   Sonos gets the unconverted result (it converts to MP3 itself).
6. Output: with `-o`, write (format from extension unless `--format`), play only if `--play`
   or `--speaker` is given. Without `-o`, play locally (default) or on Sonos.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from voice_playground import audio
from voice_playground.errors import ConfigError, VPError
from voice_playground.playback import local as local_playback
from voice_playground.playback import sonos as sonos_playback
from voice_playground.providers.base import (
    AudioResult,
    Capability,
    CreatedVoice,
    LibraryVoice,
    Provider,
    ResolvedVoice,
    STTRequest,
    Transcript,
    TTSRequest,
)
from voice_playground.providers.registry import (
    get_provider,
    get_provider_class,
    provider_names,
)
from voice_playground.settings import Settings
from voice_playground.voices import (
    PrebuiltVoiceConfig,
    VoiceCache,
    VoiceConfig,
    list_voices,
    load_voice,
    resolve_voice,
)

AudioFormat = Literal["wav", "mp3", "pcm"]

#: Audio format used when playing without `-o` and no `--format`.
DEFAULT_PLAY_FORMAT: AudioFormat = "wav"


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


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
        try:
            cls = get_provider_class(name)
        except VPError as exc:
            infos.append(ProviderInfo(name, frozenset(), None, None, None, None, error=str(exc)))
            continue
        key_env: str | None = getattr(cls, "api_key_env", None)
        key_present = None if key_env is None else settings.api_key(key_env) is not None
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


# --- helpers -------------------------------------------------------------------------------


def _read_text(text: str | None, text_file: Path | None, stdin: TextIO | None) -> str:
    """PLAN.md §2 rule 5."""
    if text is not None and text_file is not None:
        raise ConfigError("pass either --text or --text-file, not both")
    if text is not None:
        content = text
    elif text_file is not None:
        try:
            content = text_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"cannot read --text-file {text_file}: {exc}") from exc
    else:
        stream = stdin if stdin is not None else sys.stdin
        if stream is None or stream.isatty():
            raise ConfigError("no text: pass --text, --text-file, or pipe text on stdin")
        content = stream.read()
    if not content.strip():
        raise ConfigError("no text to speak: the input is empty")
    return content.strip()


def _output_format(fmt: str | None, output: Path | None) -> AudioFormat:
    """`--format`, else the `-o` extension, else WAV (PLAN.md §2 rule 6)."""
    if fmt is None:
        return cast(AudioFormat, audio.format_from_path(output)) if output else DEFAULT_PLAY_FORMAT
    if fmt not in audio.MIME_TYPES:
        raise ConfigError(f"unsupported --format '{fmt}'; use one of: wav, mp3, pcm")
    return cast(AudioFormat, fmt)


def _play_target(play: str | None, speaker: str | None, has_output: bool) -> str | None:
    """Where to play: "local", "sonos", or None (only write). PLAN.md §2 rule 6."""
    if play == "local" and speaker:
        raise ConfigError("--speaker plays on Sonos; it cannot be combined with --play local")
    if play == "sonos" or speaker:
        return "sonos"
    if play == "local":
        return "local"
    return None if has_output else "local"


def _play(settings: Settings, result: AudioResult, target: str, speaker: str | None) -> None:
    if target == "sonos":
        sonos_playback.play_sonos(result, settings, entity_id=speaker)
    else:
        local_playback.play_local(result)


def _create_voice(
    settings: Settings, cache: VoiceCache, prov: Provider, provider: str, cfg: VoiceConfig
) -> CreatedVoice:
    """Create `cfg` on `prov`, delete the remote voice it replaces, and cache the new id.

    The previous cache entry (stale, expired, created on another provider, or replaced with
    `--force`) would otherwise be orphaned remotely. A failed delete never blocks the new
    voice: it prints a one-line warning with the orphaned id (silently tolerated when the
    old entry had expired, since the provider may already have removed it).
    """
    old = cache.get(cfg.name)
    created = prov.create_voice(cfg)
    if old is not None and not (old.provider == provider and old.remote_id == created.remote_id):
        _delete_old_voice(settings, cfg.name, old.provider, old.remote_id, old.expires_at)
    cache.record(cfg, provider, created)
    return created


def _delete_old_voice(
    settings: Settings,
    name: str,
    provider: str,
    remote_id: str,
    expires_at: datetime | None,
) -> None:
    expired = expires_at is not None and (
        expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=UTC)
    ) <= datetime.now(UTC)
    try:
        get_provider(provider, settings).delete_voice(remote_id)
    except VPError as exc:
        if not expired:
            _warn(
                f"could not delete the old {provider} voice {remote_id} replaced by '{name}' "
                f"({' '.join(str(exc).split())}); it is orphaned, delete it manually"
            )


def _resolve_tts_voice(
    settings: Settings,
    cache: VoiceCache,
    voice: str | None,
    provider: str | None,
    auto_create: bool,
    model: str | None = None,
) -> tuple[str, Provider, ResolvedVoice, VoiceConfig | None]:
    """Resolve provider + voice (rules 2-4); create designed/cloned voices on first use.

    `model` is the `--model` flag; it is only used to validate the request with the provider
    (`validate_tts`) before a voice is created.
    """
    if voice is None:
        if provider is None:
            raise ConfigError("pass --provider (and optionally --voice)")
        default: str | None = getattr(get_provider_class(provider), "default_voice", None)
        if default is None:
            raise ConfigError(f"provider '{provider}' has no default voice; pass --voice")
        return provider, get_provider(provider, settings), ResolvedVoice(default), None

    name, resolved, needs_create = resolve_voice(
        voice, provider, voices_dir=settings.voices_dir, cache=cache
    )
    if name is None:
        raise ConfigError(
            f"'{voice}' is not a voice config in {settings.voices_dir}; to use it as a raw "
            "provider voice name, pass --provider"
        )
    prov = get_provider(name, settings)
    cfg = load_voice(resolved.config_name, settings.voices_dir) if resolved.config_name else None
    if needs_create:
        assert cfg is not None  # only voice configs need creating
        if not auto_create:
            raise ConfigError(
                f"voice '{cfg.name}' has not been created on {name} yet; run "
                f"`vp voices create {cfg.name}` or drop --no-auto-create"
            )
        validate = getattr(prov, "validate_tts", None)
        if validate is not None:
            validate(
                model=model or cfg.model or type(prov).default_tts_model,
                custom_voice=True,
                sample_rate=resolved.sample_rate,
            )
        print(f"creating {cfg.type} voice '{cfg.name}' on {name}...", file=sys.stderr)
        created = _create_voice(settings, cache, prov, name, cfg)
        resolved = dataclasses.replace(resolved, provider_voice=created.remote_id)
    return name, prov, resolved, cfg


# --- commands ------------------------------------------------------------------------------


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
    stdin: TextIO | None = None,
) -> Path | None:
    """Synthesize speech (PLAN.md §2 rules 1-7). Returns the written path, or None if only played.

    `stdin` defaults to `sys.stdin` (read only when no text/text_file is given and it isn't
    a TTY).
    """
    content = _read_text(text, text_file, stdin)
    target_fmt = _output_format(fmt, output)
    target = _play_target(play, speaker, output is not None)

    cache = VoiceCache(settings.cache_dir)
    provider_name, prov, resolved, cfg = _resolve_tts_voice(
        settings, cache, voice, provider, auto_create, model
    )
    if style is not None:
        resolved = dataclasses.replace(resolved, style=style)
    effective_model = model or (cfg.model if cfg else None) or type(prov).default_tts_model

    request = TTSRequest(
        text=content,
        model=effective_model,
        voice=resolved,
        output_format=target_fmt,
        sample_rate=resolved.sample_rate,
    )
    result = prov.tts(request)
    if (
        resolved.sample_rate is not None
        and result.sample_rate is not None
        and result.sample_rate != resolved.sample_rate
    ):
        _warn(
            f"{provider_name} returned {result.sample_rate} Hz audio "
            f"(requested {resolved.sample_rate} Hz); not resampling"
        )

    # Convert at most once: the same converted audio is written and played locally.
    final = audio.convert(result, target_fmt) if output is not None or target == "local" else None
    written: Path | None = None
    if output is not None:
        assert final is not None
        written = audio.write_output(final, output, target_fmt)
    if target == "sonos":
        _play(settings, result, target, speaker)  # sonos converts to MP3 itself
    elif target == "local":
        assert final is not None
        _play(settings, final, target, speaker)
    return written


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
    if provider is None:
        raise ConfigError("pass --provider for stt")
    if not input_path.is_file():
        raise ConfigError(f"input audio file not found: {input_path}")
    prov = get_provider(provider, settings)
    request = STTRequest(
        audio_path=input_path,
        model=model or type(prov).default_stt_model,
        language=language,
        prompt=prompt,
    )
    transcript = prov.stt(request)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(transcript.text + "\n", encoding="utf-8")
    return transcript


def voices_list(settings: Settings, provider: str | None = None) -> list[VoiceConfig]:
    """Voice configs in `settings.voices_dir`, optionally filtered by provider."""
    if provider is not None:
        get_provider_class(provider)  # unknown provider -> ConfigError
    return list_voices(settings.voices_dir, provider)


def voices_show(settings: Settings, name: str) -> dict[str, Any]:
    """Resolved config plus cache status / cached remote id and expiry for voice `name`."""
    cfg = load_voice(name, settings.voices_dir)
    info: dict[str, Any] = {
        k: v for k, v in cfg.model_dump(mode="json").items() if v not in (None, {}, "")
    }
    if cfg.model is None and cfg.provider is not None:
        default = get_provider_class(cfg.provider).default_tts_model
        info["model"] = f"{default} (provider default)"
    if isinstance(cfg, PrebuiltVoiceConfig):
        info["cache"] = "n/a (prebuilt voice)"
        return info
    cache = VoiceCache(settings.cache_dir)
    info["cache"] = cache.status(name, cfg, provider=cfg.provider)
    entry = cache.get(name)
    if entry is not None:
        info["remote_id"] = entry.remote_id
        info["remote_provider"] = entry.provider
        info["created_at"] = entry.created_at.isoformat()
        info["expires_at"] = entry.expires_at.isoformat() if entry.expires_at else "never"
    return info


def voices_create(settings: Settings, name: str, force: bool = False) -> CreatedVoice:
    """Create the designed/cloned voice `name` remotely and cache its id.

    If a valid cached id exists and `force` is False, return it without calling the provider.
    """
    cfg = load_voice(name, settings.voices_dir)
    if isinstance(cfg, PrebuiltVoiceConfig):
        raise ConfigError(f"voice '{name}' is a prebuilt voice; there is nothing to create")
    cache = VoiceCache(settings.cache_dir)
    provider, _, needs_create = resolve_voice(
        name, None, voices_dir=settings.voices_dir, cache=cache
    )
    assert provider is not None  # resolve_voice requires a provider for designed/cloned
    if not needs_create and not force:
        entry = cache.get(name)
        assert entry is not None
        return CreatedVoice(remote_id=entry.remote_id, expires_at=entry.expires_at)
    return _create_voice(settings, cache, get_provider(provider, settings), provider, cfg)


def voices_library(settings: Settings, provider: str, **filters: Any) -> list[LibraryVoice]:
    """List the provider's remote voice catalog (filters: search, language, gender, ...)."""
    return get_provider(provider, settings).list_library(**filters)


def voices_delete(settings: Settings, name: str) -> None:
    """Delete voice `name`'s remote voice and its cache entry."""
    cfg = load_voice(name, settings.voices_dir)
    if isinstance(cfg, PrebuiltVoiceConfig):
        raise ConfigError(f"voice '{name}' is a prebuilt voice; there is no remote voice to delete")
    cache = VoiceCache(settings.cache_dir)
    entry = cache.get(name)
    if entry is None:
        raise ConfigError(f"voice '{name}' has no cached remote voice; nothing to delete")
    get_provider(entry.provider, settings).delete_voice(entry.remote_id)
    cache.delete(name)


def play_file(
    settings: Settings, path: Path, play: str | None = None, speaker: str | None = None
) -> None:
    """Play an existing audio file locally (default) or on Sonos (`play="sonos"` or `speaker`).

    Supported files: `.wav`, `.mp3`, and `.pcm` (assumed 24 kHz mono s16le).
    """
    if not path.is_file():
        raise ConfigError(f"audio file not found: {path}")
    fmt = audio.format_from_path(path)
    data = path.read_bytes()
    sample_rate: int | None = None
    channels = 1
    if fmt == "wav":
        info = audio.wav_info(data)
        sample_rate, channels = info.sample_rate, info.channels
    elif fmt == "pcm":
        sample_rate = 24_000
    result = AudioResult(data, audio.MIME_TYPES[fmt], sample_rate, channels)
    target = _play_target(play, speaker, has_output=False)
    assert target is not None
    _play(settings, result, target, speaker)
