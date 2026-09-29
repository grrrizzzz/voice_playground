"""Provider registry: name -> lazy import path.

Imports are lazy so a missing SDK or API key only breaks that one provider.
To add a provider: create `providers/<name>.py` and add one line to `PROVIDERS`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

from voice_playground.errors import ConfigError, ProviderError

if TYPE_CHECKING:
    from voice_playground.providers.base import Provider
    from voice_playground.settings import Settings

PROVIDERS: dict[str, str] = {
    "google": "voice_playground.providers.google:GoogleProvider",
    "openai": "voice_playground.providers.openai:OpenAIProvider",
    "elevenlabs": "voice_playground.providers.elevenlabs:ElevenLabsProvider",
    "fake": "voice_playground.providers.fake:FakeProvider",
}

#: Env var holding each provider's API key (None = no key needed).
API_KEY_ENV: dict[str, str | None] = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "fake": None,
}


def provider_names() -> list[str]:
    """All registered provider names, in registration order."""
    return list(PROVIDERS)


def get_provider_class(name: str) -> type[Provider]:
    """Import and return the provider class for `name` without constructing it.

    Raises `ConfigError` for unknown names (listing the valid ones) and `ProviderError` if the
    provider module cannot be imported (e.g. its SDK is missing).
    """
    target = PROVIDERS.get(name)
    if target is None:
        valid = ", ".join(PROVIDERS)
        raise ConfigError(f"unknown provider '{name}'; valid providers: {valid}")
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProviderError(f"provider '{name}' is unavailable: {exc}") from exc
    return cast("type[Provider]", getattr(module, attr))


def get_provider(name: str, settings: Settings) -> Provider:
    """Construct provider `name` with `settings`. Raises `ConfigError` if its key is missing."""
    cls = get_provider_class(name)
    factory = cast("type", cls)
    return cast("Provider", factory(settings))
