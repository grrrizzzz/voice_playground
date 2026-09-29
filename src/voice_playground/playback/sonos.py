"""Sonos playback through Home Assistant (PLAN.md §4 "Sonos / Home Assistant flow"). T6.

Flow: convert to MP3 -> serve that one file from a temporary LAN HTTP server at an unguessable
path -> ask Home Assistant (`media_player.play_media`) to play the URL -> keep serving until the
Sonos has fetched the whole file plus (duration + grace), or until a hard timeout.
"""

from __future__ import annotations

import io
import re
import secrets
import socket
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from voice_playground import audio
from voice_playground.errors import ConfigError, ProviderError, VPError
from voice_playground.providers.base import AudioResult
from voice_playground.settings import require_secret

if TYPE_CHECKING:
    from voice_playground.settings import Settings

#: Interface the temporary server binds to. `None` = the LAN IP handed to the Sonos (falling
#: back to all interfaces if that address can't be bound, e.g. a NAT/external VP_SERVE_HOST).
#: Tests override this with 127.0.0.1.
BIND_HOST: str | None = None
#: Fallback bind address when the LAN IP itself can't be bound.
FALLBACK_BIND_HOST = "0.0.0.0"
#: Seconds to wait for the Sonos to request the file after Home Assistant accepted the call.
FETCH_TIMEOUT = 15.0
#: Extra seconds to keep serving after the audio's duration has elapsed.
GRACE_PERIOD = 5.0
#: Hard cap on how long the server stays up, in seconds.
MAX_SERVE_TIME = 120.0
#: Assumed duration when it can't be computed from the input audio.
DEFAULT_DURATION = 30.0
#: Timeout for the Home Assistant REST call (HA waits for the speaker to accept the media).
HA_TIMEOUT = 30.0

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def play_sonos(
    result: AudioResult,
    settings: Settings,
    entity_id: str | None = None,
    *,
    fetch_timeout: float = FETCH_TIMEOUT,
    grace: float = GRACE_PERIOD,
    max_serve: float = MAX_SERVE_TIME,
) -> None:
    """Play `result` on a Sonos speaker via Home Assistant `media_player.play_media`.

    `entity_id` defaults to `settings.ha_sonos_entity`. Raises `ConfigError` for missing
    HA_URL/HA_TOKEN/entity, `ProviderError` for HA failures (401/404/other/unreachable) and for
    the file never being fetched within `fetch_timeout` seconds.
    """
    ha_url, token, entity = _ha_config(settings, entity_id)
    duration = _duration(result)
    mp3 = audio.convert(result, "mp3")
    host = _lan_ip(settings)

    with _serve(mp3.data, BIND_HOST or host, settings.serve_port) as server:
        deadline = time.monotonic() + max_serve
        media_url = f"http://{host}:{server.port}{server.path}"
        _call_home_assistant(ha_url, token, entity, media_url)

        if not server.requested.wait(min(fetch_timeout, _remaining(deadline))):
            raise ProviderError(
                f"{entity} never fetched the audio from {host}:{server.port} within "
                f"{fetch_timeout:g} s; check that the macOS firewall allows incoming "
                "connections for Python and that the Sonos can reach this computer on the LAN "
                "(set VP_SERVE_HOST if the detected IP is wrong)"
            )
        if server.fetched.wait(_remaining(deadline)):
            play_time = (duration if duration is not None else DEFAULT_DURATION) + grace
            threading.Event().wait(min(play_time, _remaining(deadline)))


@contextmanager
def _serve(data: bytes, bind_host: str, port: int) -> Iterator[_OneFileServer]:
    """Run the one-file server on `bind_host` (the LAN IP, not all interfaces).

    If that address can't be bound (VP_SERVE_HOST may be an address this machine doesn't
    own, e.g. behind NAT), fall back to all interfaces.
    """
    server = _OneFileServer(data, host=bind_host, port=port)
    try:
        server.__enter__()
    except ProviderError as exc:
        if bind_host == FALLBACK_BIND_HOST or not isinstance(exc.__cause__, OSError):
            raise
        server = _OneFileServer(data, host=FALLBACK_BIND_HOST, port=port)
        server.__enter__()
    try:
        yield server
    finally:
        server.close()


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _ha_config(settings: Settings, entity_id: str | None) -> tuple[str, str, str]:
    """Return (HA_URL, HA_TOKEN, entity), raising `ConfigError` naming what's missing."""
    ha_url = (settings.ha_url or "").strip()
    if not ha_url:
        raise ConfigError("HA_URL is not set; add it to .env (see .env.example)")
    token = require_secret(settings.ha_token, "HA_TOKEN")
    entity = (entity_id or settings.ha_sonos_entity or "").strip()
    if not entity:
        raise ConfigError(
            "no Sonos entity: pass --speaker <entity_id> or set HA_SONOS_ENTITY in .env"
        )
    return ha_url.rstrip("/"), token, entity


def _duration(result: AudioResult) -> float | None:
    """Audio duration in seconds, or `None` if it can't be determined.

    WAV / L16 are measured directly; MP3 and unknown types are decoded to WAV with ffmpeg
    (`audio.convert`), so the server doesn't linger for `DEFAULT_DURATION` after short clips.
    """
    mime = result.mime_type.lower()
    if mime in ("audio/wav", "audio/x-wav", "audio/wave"):
        try:
            with wave.open(io.BytesIO(result.data), "rb") as wav:
                rate = wav.getframerate()
                return wav.getnframes() / rate if rate else None
        except (wave.Error, EOFError):
            return None
    if mime.startswith("audio/l16"):
        if not result.sample_rate or result.channels < 1:
            return None
        return len(result.data) / (result.sample_rate * 2 * result.channels)
    try:
        return audio.wav_info(audio.convert(result, "wav").data).duration
    except VPError:
        return None


def _lan_ip(settings: Settings) -> str:
    """The IP the Sonos should use to reach us: VP_SERVE_HOST, else the UDP-connect trick."""
    if settings.serve_host and settings.serve_host.strip():
        return settings.serve_host.strip()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))  # UDP connect sends no packets; it only picks a route
            ip = str(sock.getsockname()[0])
    except OSError as exc:
        raise ProviderError(
            f"could not detect this computer's LAN IP ({exc.__class__.__name__}); "
            "set VP_SERVE_HOST in .env"
        ) from exc
    if ip.startswith("127.") or ip == "0.0.0.0":
        raise ProviderError(
            f"detected LAN IP {ip} is not reachable from the Sonos; set VP_SERVE_HOST in .env"
        )
    return ip


def _call_home_assistant(
    ha_url: str, token: str, entity_id: str, media_url: str, *, timeout: float = HA_TIMEOUT
) -> None:
    """POST `media_player.play_media` to Home Assistant. Errors never include the token."""
    endpoint = f"{ha_url.rstrip('/')}/api/services/media_player/play_media"
    payload = {
        "entity_id": entity_id,
        "media_content_id": media_url,
        "media_content_type": "music",
    }
    try:
        response = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise ProviderError(
            f"could not reach Home Assistant at {ha_url} ({exc.__class__.__name__}); check HA_URL"
        ) from exc

    status = response.status_code
    if status == 401:
        raise ProviderError("Home Assistant returned 401 Unauthorized; check HA_TOKEN")
    if status == 404:
        raise ProviderError(
            f"Home Assistant returned 404 Not Found for {endpoint}; check HA_URL and that "
            f"the media_player integration and entity {entity_id} exist"
        )
    if status >= 400:
        detail = " ".join(response.text.split())[:200].replace(token, "***")
        raise ProviderError(
            f"Home Assistant returned {status} for media_player.play_media on {entity_id}"
            + (f": {detail}" if detail else "")
        )


class _OneFileServer:
    """Context manager: a threaded HTTP server that serves one in-memory file at one path.

    The path is `/<secrets.token_urlsafe(16)>.mp3`; every other path is 404. GET and HEAD are
    supported, as is a single `Range: bytes=...` (206). `requested` is set on the first GET or
    HEAD of the token path; `fetched` once any response has delivered the file's last byte.
    """

    def __init__(
        self,
        data: bytes,
        *,
        host: str = FALLBACK_BIND_HOST,
        port: int = 0,
        content_type: str = "audio/mpeg",
        suffix: str = ".mp3",
    ) -> None:
        self.data = data
        self.content_type = content_type
        self.path = f"/{secrets.token_urlsafe(16)}{suffix}"
        self.requested = threading.Event()
        self.fetched = threading.Event()
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("server is not running")
        return int(self._httpd.server_address[1])

    def __enter__(self) -> _OneFileServer:
        try:
            httpd = ThreadingHTTPServer((self._host, self._port), _make_handler(self))
        except OSError as exc:
            raise ProviderError(
                f"could not start the local audio server on {self._host}:{self._port} "
                f"({exc.strerror or exc.__class__.__name__}); try VP_SERVE_PORT=0"
            ) from exc
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None | bool:
    """Parse a single-range `Range` header.

    Returns (start, end) inclusive for a satisfiable range, `None` to serve the whole file
    (no header, or a form we don't support such as multiple ranges), or `False` when the
    range is unsatisfiable (416).
    """
    if not header:
        return None
    match = _RANGE_RE.match(header.strip())
    if match is None:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    if not first:  # suffix range: the last N bytes
        length = int(last)
        if length == 0 or size == 0:
            return False
        return max(0, size - length), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        return False
    return start, min(end, size - 1)


def _make_handler(server: _OneFileServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "vp-sonos"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the default per-request stderr logging."""

        def do_GET(self) -> None:
            self._serve(send_body=True)

        def do_HEAD(self) -> None:
            self._serve(send_body=False)

        def _serve(self, *, send_body: bool) -> None:
            if urlsplit(self.path).path != server.path:
                self.send_error(404)
                return
            server.requested.set()
            data = server.data
            size = len(data)
            byte_range = _parse_range(self.headers.get("Range"), size)
            if byte_range is False:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if isinstance(byte_range, tuple):
                start, end = byte_range
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                start, end = 0, size - 1
                self.send_response(200)
            self.send_header("Content-Type", server.content_type)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not send_body:
                return
            try:
                self.wfile.write(data[start : end + 1])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if end >= size - 1:
                server.fetched.set()

    return Handler
