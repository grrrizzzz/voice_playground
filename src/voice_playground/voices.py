"""Voice config models, loading, and the remote-voice cache (PLAN.md §3).

- A voice config is `voices_dir/<name>.yaml` (or `.yml`), validated by a pydantic
  discriminated union on `type` (prebuilt | designed | cloned).
- `VoiceCache` stores the remote ids of created designed/cloned voices in
  `cache_dir/voices.json` (atomic writes, config-hash staleness, expiry).
- `resolve_voice` implements PLAN.md §2 resolution rules 2-4.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from voice_playground.errors import ConfigError
from voice_playground.providers.base import CreatedVoice, ResolvedVoice
from voice_playground.providers.registry import provider_names

#: File extensions loaded as voice configs. Anything else (e.g. `*.yaml.example`) is ignored.
VOICE_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")

#: Names that may be looked up as files. Anything else (paths, `..`) is never a file lookup.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

CacheStatus = Literal["valid", "missing", "provider_mismatch", "stale", "expired"]


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


class _VoiceConfigBase(BaseModel):
    """Fields shared by every voice config type. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str  # must match the YAML file name (without extension)
    provider: str | None = None  # google | openai | elevenlabs (| fake in tests)
    model: str | None = None
    style: str | None = None  # google speech_metadata.style / openai instructions
    language: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    provider_options: dict[str, Any] = Field(default_factory=dict)  # passed through as-is

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str | None) -> str | None:
        if value is not None and value not in provider_names():
            raise ValueError(
                f"unknown provider '{value}'; valid providers: {', '.join(provider_names())}"
            )
        return value


class PrebuiltVoiceConfig(_VoiceConfigBase):
    """A provider's built-in voice (name/id in `voice`), optionally with a style."""

    type: Literal["prebuilt"] = "prebuilt"
    voice: str = Field(min_length=1)


class DesignedVoiceConfig(_VoiceConfigBase):
    """A voice created remotely from a natural-language `description`."""

    type: Literal["designed"] = "designed"
    description: str = Field(min_length=1)
    store: bool = True  # google: stateful voice_ (1y) vs stateless voicekey_ (7d)


class ClonedVoiceConfig(_VoiceConfigBase):
    """A voice cloned remotely from `reference_audio` (path relative to the repo root = cwd).

    The file's existence is checked when the voice is resolved, not when the config loads,
    so `vp voices list` works without the (gitignored) personal recording.
    """

    type: Literal["cloned"] = "cloned"
    reference_audio: Path
    store: bool = True  # google: stateful voice_ (1y) vs stateless voicekey_ (7d)


VoiceConfig = Annotated[
    PrebuiltVoiceConfig | DesignedVoiceConfig | ClonedVoiceConfig,
    Field(discriminator="type"),
]

_VOICE_ADAPTER: TypeAdapter[VoiceConfig] = TypeAdapter(VoiceConfig)


class CacheEntry(BaseModel):
    """One entry in `.vp_cache/voices.json`, keyed by voice config name."""

    provider: str
    remote_id: str
    config_hash: str
    created_at: datetime
    expires_at: datetime | None = None


# --- loading -------------------------------------------------------------------------------


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _parse_file(path: Path) -> VoiceConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read voice config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        detail = " ".join(str(exc).split())
        raise ConfigError(f"invalid YAML in voice config {path}: {detail}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"invalid voice config {path}: expected a mapping of fields")
    if "type" not in raw:
        raise ConfigError(
            f"invalid voice config {path}: missing 'type' (prebuilt | designed | cloned)"
        )
    try:
        cfg = _VOICE_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid voice config {path}: {_format_validation_error(exc)}") from exc
    if cfg.name != path.stem:
        raise ConfigError(
            f"invalid voice config {path}: name '{cfg.name}' must match the file name '{path.stem}'"
        )
    return cfg


def voice_path(name: str, voices_dir: Path) -> Path | None:
    """Path of the config file for `name` (`.yaml` preferred over `.yml`), or None.

    Names that are not plain file names (contain `/`, start with `.`, ...) never match a file.
    """
    if not _SAFE_NAME.match(name):
        return None
    for suffix in VOICE_SUFFIXES:
        candidate = voices_dir / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_voice(name: str, voices_dir: Path) -> VoiceConfig:
    """Load and validate `voices_dir/<name>.yaml` (or `.yml`).

    Raises `ConfigError` (message names the file) when the file is missing, is invalid
    YAML, fails validation (wrong/missing fields for its `type`), or when the config's `name`
    does not match the file name.
    """
    path = voice_path(name, voices_dir)
    if path is None:
        raise ConfigError(f"voice config not found: {voices_dir / (name + '.yaml')}")
    return _parse_file(path)


def list_voices(voices_dir: Path, provider: str | None = None) -> list[VoiceConfig]:
    """Load every `*.yaml` / `*.yml` voice config in `voices_dir`, optionally by provider.

    Sorted by name. Invalid files raise `ConfigError` naming the file. A missing directory
    yields an empty list.
    """
    if not voices_dir.is_dir():
        return []
    files = [p for p in voices_dir.iterdir() if p.is_file() and p.suffix in VOICE_SUFFIXES]
    configs = [_parse_file(p) for p in files]
    if provider is not None:
        configs = [c for c in configs if c.provider == provider]
    return sorted(configs, key=lambda c: c.name)


# --- hashing -------------------------------------------------------------------------------


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def config_hash(cfg: VoiceConfig, root: Path | None = None) -> str:
    """Stable sha256 of the fields that define a remote voice (detects a stale cache).

    Covers `provider`, `type`, `store`, and `description` (designed) or the `reference_audio`
    path plus a hash of its contents (cloned; `null` if the file is unreadable). `style`,
    `model`, `language`, etc. are applied per request, so changing them does not invalidate
    the remote voice. `root` resolves relative `reference_audio` paths (default: cwd).
    """
    payload: dict[str, Any] = {"provider": cfg.provider, "type": cfg.type}
    if isinstance(cfg, DesignedVoiceConfig):
        payload |= {"description": cfg.description, "store": cfg.store}
    elif isinstance(cfg, ClonedVoiceConfig):
        audio = _audio_path(cfg, root)
        payload |= {
            "reference_audio": cfg.reference_audio.as_posix(),
            "reference_audio_sha256": _file_sha256(audio),
            "store": cfg.store,
        }
    else:
        payload |= {"voice": cfg.voice}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _audio_path(cfg: ClonedVoiceConfig, root: Path | None) -> Path:
    if cfg.reference_audio.is_absolute():
        return cfg.reference_audio
    return (root or Path.cwd()) / cfg.reference_audio


# --- cache ---------------------------------------------------------------------------------


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class VoiceCache:
    """Remote ids of created voices, stored as JSON at `cache_dir/voices.json`.

    Writes are atomic (temp file in the same dir + `os.replace`). Entries are keyed by voice
    config name and store `{provider, remote_id, config_hash, created_at, expires_at}`.
    A missing file is an empty cache; a corrupt file warns on stderr and is treated as empty
    (the next write replaces it).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.path = cache_dir / "voices.json"

    def _read(self) -> dict[str, CacheEntry]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            _warn(f"cannot read voice cache {self.path} ({exc}); treating it as empty")
            return {}
        try:
            raw = json.loads(text)
            if not isinstance(raw, dict):
                raise ValueError("expected a JSON object")
            return {str(k): CacheEntry.model_validate(v) for k, v in raw.items()}
        except (ValueError, ValidationError) as exc:
            detail = " ".join(str(exc).split())[:200]
            _warn(f"voice cache {self.path} is corrupt ({detail}); treating it as empty")
            return {}

    def _write(self, entries: dict[str, CacheEntry]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {name: e.model_dump(mode="json") for name, e in sorted(entries.items())}
        fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, prefix=".voices.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def entries(self) -> dict[str, CacheEntry]:
        """All cached entries, keyed by voice config name."""
        return self._read()

    def get(self, name: str) -> CacheEntry | None:
        """Return the entry for `name`, or None."""
        return self._read().get(name)

    def put(self, name: str, entry: CacheEntry) -> None:
        """Insert or replace the entry for `name` (atomic write)."""
        entries = self._read()
        entries[name] = entry
        self._write(entries)

    def record(
        self,
        cfg: VoiceConfig,
        provider: str,
        created: CreatedVoice,
        *,
        root: Path | None = None,
        now: datetime | None = None,
    ) -> CacheEntry:
        """Build a `CacheEntry` for a freshly created voice, store it under `cfg.name`."""
        entry = CacheEntry(
            provider=provider,
            remote_id=created.remote_id,
            config_hash=config_hash(cfg, root),
            created_at=now or datetime.now(UTC),
            expires_at=created.expires_at,
        )
        self.put(cfg.name, entry)
        return entry

    def delete(self, name: str) -> None:
        """Remove the entry for `name` if present (atomic write)."""
        entries = self._read()
        if entries.pop(name, None) is not None:
            self._write(entries)

    def status(
        self,
        name: str,
        cfg: VoiceConfig,
        now: datetime | None = None,
        *,
        provider: str | None = None,
        root: Path | None = None,
    ) -> CacheStatus:
        """Classify the entry for `name` against `cfg` (no warnings).

        `provider` is the effective provider (defaults to `cfg.provider`); an entry created on
        another provider is `provider_mismatch`. Expired means `expires_at <= now`.
        """
        entry = self.get(name)
        if entry is None:
            return "missing"
        want_provider = provider or cfg.provider
        if want_provider is not None and entry.provider != want_provider:
            return "provider_mismatch"
        if entry.config_hash != config_hash(cfg, root):
            return "stale"
        current = _utc(now or datetime.now(UTC))
        if entry.expires_at is not None and _utc(entry.expires_at) <= current:
            return "expired"
        return "valid"

    def is_valid(
        self,
        name: str,
        cfg: VoiceConfig,
        now: datetime | None = None,
        *,
        provider: str | None = None,
        root: Path | None = None,
    ) -> bool:
        """True if an entry exists for the provider, its hash matches `cfg`, and it's unexpired.

        Warns on stderr when the entry is stale (config changed) or expired; the caller then
        recreates the voice (with `--force` behavior).
        """
        state = self.status(name, cfg, now, provider=provider, root=root)
        if state == "stale":
            _warn(f"voice '{name}' changed since it was created; it will be recreated")
        elif state == "expired":
            _warn(f"cached remote voice for '{name}' has expired; it will be recreated")
        elif state == "provider_mismatch":
            _warn(f"cached remote voice for '{name}' belongs to another provider; recreating")
        return state == "valid"


# --- resolution ----------------------------------------------------------------------------


def resolve_voice(
    name_or_raw: str,
    provider_flag: str | None,
    *,
    voices_dir: Path,
    cache: VoiceCache,
    root: Path | None = None,
    now: datetime | None = None,
) -> tuple[str | None, ResolvedVoice, bool]:
    """Resolve `--voice` into `(provider_name, ResolvedVoice, needs_create)`. PLAN.md §2 rules 2-4.

    - If `voices_dir/<name_or_raw>.yaml` exists, load it. Otherwise treat `name_or_raw` as a raw
      provider voice name/id: return `(provider_flag, ResolvedVoice(provider_voice=name_or_raw),
      False)` (provider is None when no flag was given; the caller picks the default).
    - If the config sets `provider` and `provider_flag` is a different value, raise
      `ConfigError` (exit 2). If `provider_flag` is None, use the config's provider.
    - prebuilt: `provider_voice` = config `voice`, `needs_create=False`.
    - designed/cloned: a provider is required (config or flag), else `ConfigError`. Cloned
      `reference_audio` must exist (relative to `root`, default cwd), else `ConfigError`
      naming the voice file. If the cache holds a valid entry (same provider, same config
      hash, not expired), `provider_voice` = its remote id and `needs_create=False`. Otherwise
      `needs_create=True` and `provider_voice` is the empty-string placeholder `""`: the caller
      creates the voice, calls `cache.record(...)`, and builds the final voice with
      `dataclasses.replace(resolved, provider_voice=remote_id)`. A stale/expired entry warns
      on stderr.
    - `style`, `language`, `sample_rate`, `provider_options` (-> `options`) and `config_name`
      come from the config (CLI flags override them in the caller).
    """
    if not name_or_raw.strip():
        raise ConfigError("--voice must not be empty")
    path = voice_path(name_or_raw, voices_dir)
    if path is None:
        return provider_flag, ResolvedVoice(provider_voice=name_or_raw), False

    cfg = _parse_file(path)
    if cfg.provider is not None and provider_flag is not None and cfg.provider != provider_flag:
        raise ConfigError(
            f"voice '{cfg.name}' ({path}) is for provider '{cfg.provider}', "
            f"but --provider {provider_flag} was given"
        )
    provider = provider_flag or cfg.provider

    def build(provider_voice: str) -> ResolvedVoice:
        return ResolvedVoice(
            provider_voice=provider_voice,
            style=cfg.style,
            language=cfg.language,
            sample_rate=cfg.sample_rate,
            options=dict(cfg.provider_options),
            config_name=cfg.name,
        )

    if isinstance(cfg, PrebuiltVoiceConfig):
        return provider, build(cfg.voice), False

    if provider is None:
        raise ConfigError(
            f"voice '{cfg.name}' ({path}) is a {cfg.type} voice and needs a provider: "
            "set 'provider' in the file or pass --provider"
        )
    if isinstance(cfg, ClonedVoiceConfig):
        audio = _audio_path(cfg, root)
        if not audio.is_file():
            raise ConfigError(
                f"voice '{cfg.name}' ({path}): reference_audio not found: {cfg.reference_audio}"
            )
    entry = cache.get(cfg.name)
    if entry is not None and cache.is_valid(cfg.name, cfg, now, provider=provider, root=root):
        return provider, build(entry.remote_id), False
    return provider, build(""), True
