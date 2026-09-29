"""Sonos playback through Home Assistant (PLAN.md §4 "Sonos / Home Assistant flow"). T6."""

from __future__ import annotations

from typing import TYPE_CHECKING

from voice_playground.providers.base import AudioResult

if TYPE_CHECKING:
    from voice_playground.settings import Settings


def play_sonos(result: AudioResult, settings: Settings, entity_id: str | None = None) -> None:
    """Play `result` on a Sonos speaker via Home Assistant `media_player.play_media`.

    1. Convert to MP3 (ffmpeg).
    2. Serve exactly one file at `/<secrets.token_urlsafe(16)>.mp3` from a `ThreadingHTTPServer`
       on `0.0.0.0:settings.serve_port` in a background thread; everything else is 404.
    3. LAN IP = `settings.serve_host`, else the UDP-connect trick.
    4. POST `{HA_URL}/api/services/media_player/play_media` with Bearer `HA_TOKEN` and JSON
       `{"entity_id", "media_content_id": url, "media_content_type": "music"}`.
       `entity_id` defaults to `settings.ha_sonos_entity`.
    5. Serve until fully fetched + (duration + 5 s), or a 120 s timeout; then shut down.
    6. `ConfigError` for missing HA_URL/HA_TOKEN/entity; `ProviderError` for HA 401/404 and for
       the file never being fetched within 15 s (hint: macOS firewall / Sonos can't reach host).
    """
    raise NotImplementedError("playback.sonos.play_sonos is implemented in T6")
