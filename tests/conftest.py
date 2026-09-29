"""Shared fixtures.

- Unit tests never read the real `.env`: `VP_ENV_FILE` points at a non-existent file and all
  secrets are removed from the environment. `VP_VOICES_DIR`/`VP_CACHE_DIR` point at tmp dirs.
- Live tests (`@pytest.mark.live`) are skipped unless the relevant `*_API_KEY` is set (in the
  environment or `.env`). The provider is taken from the marker arg
  (`@pytest.mark.live("google")`) or inferred from the test file name (`test_google_live.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from voice_playground.providers.registry import API_KEY_ENV
from voice_playground.settings import ENV_FILE_VAR, Settings, get_settings

SECRET_ENV_VARS = ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "HA_TOKEN")


def _live_provider(item: pytest.Item) -> str | None:
    marker = item.get_closest_marker("live")
    if marker is not None and marker.args:
        return str(marker.args[0])
    stem = item.path.stem
    return next((name for name in API_KEY_ENV if name in stem), None)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live_items = [item for item in items if item.get_closest_marker("live")]
    if not live_items:
        return
    real = get_settings()  # reads the real .env; only presence is checked, never printed
    for item in live_items:
        provider = _live_provider(item)
        env_var = API_KEY_ENV.get(provider) if provider else None
        if env_var is None:
            continue
        secret = getattr(real, env_var.lower(), None)
        if secret is None or not secret.get_secret_value().strip():
            item.add_marker(pytest.mark.skip(reason=f"{env_var} not set"))


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    path = tmp_path / "voices"
    path.mkdir()
    return path


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / ".vp_cache"


@pytest.fixture(autouse=True)
def _isolated_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    voices_dir: Path,
    cache_dir: Path,
) -> Iterator[None]:
    """Isolate unit tests from the developer's `.env` and real directories."""
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setenv(ENV_FILE_VAR, str(tmp_path / "no-such.env"))
        for var in SECRET_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VP_VOICES_DIR", str(voices_dir))
    monkeypatch.setenv("VP_CACHE_DIR", str(cache_dir))
    yield


@pytest.fixture
def settings(voices_dir: Path, cache_dir: Path) -> Settings:
    """Settings with no keys, no HA config, and tmp voices/cache dirs (ignores `.env`)."""
    return Settings(_env_file=None, voices_dir=voices_dir, cache_dir=cache_dir)
