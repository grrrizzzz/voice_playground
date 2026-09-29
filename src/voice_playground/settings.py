"""Application settings loaded from environment variables and `.env` (PLAN.md §3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr
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


def get_settings() -> Settings:
    """Build `Settings` from the environment plus the dotenv file.

    The dotenv path defaults to `.env` in the current directory and can be overridden with
    the `VP_ENV_FILE` environment variable (tests point it at a non-existent file).
    """
    # Passed via a dict: `_env_file` is a BaseSettings init arg mypy doesn't see on subclasses.
    init_args: dict[str, Any] = {"_env_file": os.environ.get(ENV_FILE_VAR, ".env")}
    return Settings(**init_args)


def require_secret(value: SecretStr | None, env_var: str) -> str:
    """Return the secret's plain value, or raise `ConfigError` naming the missing env var."""
    if value is None or not value.get_secret_value().strip():
        raise ConfigError(f"{env_var} is not set; add it to .env (see .env.example)")
    return value.get_secret_value()
