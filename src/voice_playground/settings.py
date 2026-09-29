"""Application settings loaded from environment variables and `.env` (PLAN.md §3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import AliasChoices, Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_playground.errors import ConfigError

#: Environment variable that overrides which dotenv file `get_settings()` reads.
ENV_FILE_VAR = "VP_ENV_FILE"


def _alias(env_name: str, field_name: str) -> AliasChoices:
    return AliasChoices(env_name, field_name)


class Settings(BaseSettings):
    """All runtime configuration. API keys and the HA token are `SecretStr` and never printed."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    # Provider API keys. Field names are the lower-cased env var names.
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None

    # Home Assistant / Sonos
    ha_url: str | None = None
    ha_token: SecretStr | None = None
    ha_sonos_entity: str | None = None

    # Temporary LAN HTTP server used for Sonos playback
    serve_host: str | None = Field(
        default=None, validation_alias=_alias("VP_SERVE_HOST", "serve_host")
    )
    serve_port: int = Field(default=0, validation_alias=_alias("VP_SERVE_PORT", "serve_port"))

    # Local directories
    voices_dir: Path = Field(
        default=Path("voices"), validation_alias=_alias("VP_VOICES_DIR", "voices_dir")
    )
    cache_dir: Path = Field(
        default=Path(".vp_cache"), validation_alias=_alias("VP_CACHE_DIR", "cache_dir")
    )

    #: Dotenv file `api_key()` falls back to for undeclared keys (set by `get_settings()`).
    _dotenv_path: Path | None = PrivateAttr(default=None)

    def api_key(self, env_name: str) -> SecretStr | None:
        """Secret for env var `env_name` (e.g. `ACME_API_KEY`), or None if unset/empty.

        Uses the declared field (`acme_api_key`) when there is one. Otherwise reads the
        environment, then the dotenv file `get_settings()` loaded, so a new provider needs no
        `Settings` field.
        """
        field = env_name.lower()
        if field in type(self).model_fields:
            value = getattr(self, field)
            if value is None or isinstance(value, SecretStr):
                return value if value is None or value.get_secret_value().strip() else None
            raise ConfigError(f"{env_name} is not a secret setting")
        raw = os.environ.get(env_name)
        if (raw is None or not raw.strip()) and self._dotenv_path is not None:
            try:
                raw = dotenv_values(self._dotenv_path).get(env_name)
            except OSError:
                raw = None
        if raw is None or not raw.strip():
            return None
        return SecretStr(raw)


def get_settings() -> Settings:
    """Build `Settings` from the environment plus the dotenv file.

    The dotenv path defaults to `.env` in the current directory and can be overridden with
    the `VP_ENV_FILE` environment variable (tests point it at a non-existent file).
    """
    # Passed via a dict: `_env_file` is a BaseSettings init arg mypy doesn't see on subclasses.
    env_file = os.environ.get(ENV_FILE_VAR, ".env")
    init_args: dict[str, Any] = {"_env_file": env_file}
    settings = Settings(**init_args)
    settings._dotenv_path = Path(env_file)
    return settings


def require_secret(value: SecretStr | None, env_var: str) -> str:
    """Return the secret's plain value, or raise `ConfigError` naming the missing env var."""
    if value is None or not value.get_secret_value().strip():
        raise ConfigError(f"{env_var} is not set; add it to .env (see .env.example)")
    return value.get_secret_value()
