"""Tests for Sonos playback via Home Assistant (T6). All network is local or mocked."""

from __future__ import annotations

import http.client
import io
import json
import shutil
import socket
import time
import urllib.request
import wave
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx

from voice_playground import audio
from voice_playground.errors import ConfigError, ProviderError
from voice_playground.playback import sonos
from voice_playground.providers.base import AudioResult
from voice_playground.settings import Settings

HA_URL = "http://ha.test:8123"
HA_ENDPOINT = f"{HA_URL}/api/services/media_player/play_media"
TOKEN = "dummy-ha-token-value"
ENTITY = "media_player.living_room"
MP3_BYTES = b"ID3" + bytes(range(256)) * 8


def _wav(seconds: float, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "ha_url": HA_URL,
        "ha_token": TOKEN,
        "ha_sonos_entity": ENTITY,
        "serve_host": "127.0.0.1",
        "serve_port": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _get(
    port: int, path: str, *, method: str = "GET", headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    """Raw request with http.client so the path is sent exactly as given (no normalization)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _local_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the server to loopback in tests (avoids firewall prompts on 0.0.0.0)."""
    monkeypatch.setattr(sonos, "BIND_HOST", "127.0.0.1")


@pytest.fixture
def fake_convert(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AudioResult, str]]:
    """Replace `audio.convert` so these tests don't depend on ffmpeg or T5's implementation."""
    calls: list[tuple[AudioResult, str]] = []

    def convert(result: AudioResult, fmt: str) -> AudioResult:
        calls.append((result, fmt))
        return AudioResult(data=MP3_BYTES, mime_type="audio/mpeg", sample_rate=None)

    monkeypatch.setattr(audio, "convert", convert)
    return calls


# --------------------------------------------------------------------------- server


def test_server_serves_file_at_token_path_only() -> None:
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        assert server.path.startswith("/") and server.path.endswith(".mp3")
        assert len(server.path) > 20  # token_urlsafe(16) -> 22 chars

        status, headers, body = _get(server.port, server.path)
        assert status == 200
        assert body == MP3_BYTES
        assert headers["Content-Type"] == "audio/mpeg"
        assert headers["Content-Length"] == str(len(MP3_BYTES))
        assert headers["Accept-Ranges"] == "bytes"
        assert server.fetched.wait(2)  # set by the handler thread just after sending

        # Query strings don't change which file is served.
        assert _get(server.port, server.path + "?x=1")[0] == 200


@pytest.mark.parametrize(
    "path_template",
    [
        "/",
        "/other.mp3",
        "/index.html",
        "{token}x",
        "{token}/",
        "/..{token}",
        "/../{token_name}",
        "/x/..{token}",
        "{token}/..",
        "/%2e%2e{token}",
        "/{token_name}.mp3",
        "/etc/passwd",
        "/../../etc/passwd",
    ],
)
def test_server_returns_404_for_other_paths(path_template: str) -> None:
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        path = path_template.format(token=server.path, token_name=server.path.lstrip("/"))
        status, _, body = _get(server.port, path)
        assert status == 404
        assert MP3_BYTES not in body
        assert not server.requested.is_set()
        assert not server.fetched.is_set()


def test_server_head_returns_headers_without_body_and_is_not_a_full_fetch() -> None:
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        status, headers, body = _get(server.port, server.path, method="HEAD")
        assert status == 200
        assert body == b""
        assert headers["Content-Length"] == str(len(MP3_BYTES))
        assert server.requested.is_set()
        assert not server.fetched.is_set()

        assert _get(server.port, "/nope.mp3", method="HEAD")[0] == 404


def test_server_range_requests_and_full_fetch_detection() -> None:
    size = len(MP3_BYTES)
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        status, headers, body = _get(server.port, server.path, headers={"Range": "bytes=0-99"})
        assert status == 206
        assert body == MP3_BYTES[:100]
        assert headers["Content-Range"] == f"bytes 0-99/{size}"
        assert headers["Content-Length"] == "100"
        assert not server.fetched.is_set()  # last byte not delivered yet

        status, headers, body = _get(server.port, server.path, headers={"Range": "bytes=100-"})
        assert status == 206
        assert body == MP3_BYTES[100:]
        assert headers["Content-Range"] == f"bytes 100-{size - 1}/{size}"
        assert server.fetched.wait(2)  # set by the handler thread just after sending


def test_server_suffix_range_delivers_last_byte() -> None:
    size = len(MP3_BYTES)
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        status, headers, body = _get(server.port, server.path, headers={"Range": "bytes=-10"})
        assert status == 206
        assert body == MP3_BYTES[-10:]
        assert headers["Content-Range"] == f"bytes {size - 10}-{size - 1}/{size}"
        assert server.fetched.wait(2)  # set by the handler thread just after sending


def test_server_unsatisfiable_range_is_416() -> None:
    size = len(MP3_BYTES)
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        status, headers, _ = _get(
            server.port, server.path, headers={"Range": f"bytes={size}-{size + 5}"}
        )
        assert status == 416
        assert headers["Content-Range"] == f"bytes */{size}"
        assert not server.fetched.is_set()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("bytes=0-9", (0, 9)),
        ("bytes=5-", (5, 99)),
        ("bytes=-10", (90, 99)),
        ("bytes=-500", (0, 99)),
        ("bytes=90-500", (90, 99)),
        ("bytes=0-1,5-6", None),  # multiple ranges: serve the whole file
        ("items=0-1", None),
        ("bytes=-", None),
        ("bytes=100-", False),
        ("bytes=9-3", False),
        ("bytes=-0", False),
    ],
)
def test_parse_range(header: str | None, expected: object) -> None:
    assert sonos._parse_range(header, 100) == expected


def test_server_does_not_log_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        _get(server.port, server.path)
        _get(server.port, "/missing")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_server_shuts_down_on_exit() -> None:
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as server:
        port = server.port
        assert _get(port, server.path)[0] == 200
    with pytest.raises(OSError):
        _get(port, server.path)


def test_server_bind_failure_is_provider_error() -> None:
    with sonos._OneFileServer(MP3_BYTES, host="127.0.0.1") as first:
        busy = first.port
        with (
            pytest.raises(ProviderError, match="VP_SERVE_PORT"),
            sonos._OneFileServer(MP3_BYTES, host="127.0.0.1", port=busy),
        ):
            pass


# --------------------------------------------------------------------------- Home Assistant


@respx.mock
def test_call_home_assistant_request_shape() -> None:
    route = respx.post(HA_ENDPOINT).mock(return_value=httpx.Response(200, json=[]))
    sonos._call_home_assistant(HA_URL + "/", TOKEN, ENTITY, "http://10.0.0.5:1234/abc.mp3")

    assert route.call_count == 1
    request = route.calls.last.request
    assert str(request.url) == HA_ENDPOINT
    assert request.method == "POST"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request.content) == {
        "entity_id": ENTITY,
        "media_content_id": "http://10.0.0.5:1234/abc.mp3",
        "media_content_type": "music",
    }


@respx.mock
def test_call_home_assistant_401_hints_token() -> None:
    respx.post(HA_ENDPOINT).mock(return_value=httpx.Response(401, text="401: Unauthorized"))
    with pytest.raises(ProviderError) as excinfo:
        sonos._call_home_assistant(HA_URL, TOKEN, ENTITY, "http://x/y.mp3")
    message = str(excinfo.value)
    assert "401" in message
    assert "HA_TOKEN" in message
    assert TOKEN not in message


@respx.mock
def test_call_home_assistant_404_hints_url_and_entity() -> None:
    respx.post(HA_ENDPOINT).mock(return_value=httpx.Response(404, text="Not Found"))
    with pytest.raises(ProviderError) as excinfo:
        sonos._call_home_assistant(HA_URL, TOKEN, ENTITY, "http://x/y.mp3")
    message = str(excinfo.value)
    assert "404" in message
    assert "HA_URL" in message
    assert ENTITY in message
    assert TOKEN not in message


@respx.mock
def test_call_home_assistant_other_error_redacts_token() -> None:
    respx.post(HA_ENDPOINT).mock(return_value=httpx.Response(400, text=f"bad request echo {TOKEN}"))
    with pytest.raises(ProviderError) as excinfo:
        sonos._call_home_assistant(HA_URL, TOKEN, ENTITY, "http://x/y.mp3")
    assert "400" in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)


@respx.mock
def test_call_home_assistant_unreachable() -> None:
    respx.post(HA_ENDPOINT).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(ProviderError, match="could not reach Home Assistant") as excinfo:
        sonos._call_home_assistant(HA_URL, TOKEN, ENTITY, "http://x/y.mp3")
    assert TOKEN not in str(excinfo.value)


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"ha_url": None}, "HA_URL"),
        ({"ha_token": None}, "HA_TOKEN"),
        ({"ha_token": "   "}, "HA_TOKEN"),
        ({"ha_sonos_entity": None}, "HA_SONOS_ENTITY"),
    ],
)
def test_missing_ha_config_raises_config_error(
    overrides: dict[str, Any], missing: str, fake_convert: list[Any]
) -> None:
    result = AudioResult(data=_wav(0.1), mime_type="audio/wav", sample_rate=8000)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(HA_ENDPOINT)
        with pytest.raises(ConfigError, match=missing) as excinfo:
            sonos.play_sonos(result, _settings(**overrides))
        assert not route.called
    assert excinfo.value.exit_code == 2
    assert fake_convert == []  # config is validated before any work


def test_explicit_entity_overrides_settings() -> None:
    settings = _settings(ha_sonos_entity=None)
    assert sonos._ha_config(settings, "media_player.kitchen") == (
        HA_URL,
        TOKEN,
        "media_player.kitchen",
    )
    assert sonos._ha_config(_settings(), None)[2] == ENTITY
    assert sonos._ha_config(_settings(), "media_player.den")[2] == "media_player.den"


# --------------------------------------------------------------------------- LAN IP / duration


def test_lan_ip_uses_serve_host_override() -> None:
    assert sonos._lan_ip(_settings(serve_host=" 192.168.1.50 ")) == "192.168.1.50"


class _FakeSocket:
    def __init__(self, ip: str | None) -> None:
        self.ip = ip
        self.connected_to: tuple[str, int] | None = None

    def __enter__(self) -> _FakeSocket:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def connect(self, addr: tuple[str, int]) -> None:
        if self.ip is None:
            raise OSError("Network is unreachable")
        self.connected_to = addr

    def getsockname(self) -> tuple[str, int]:
        assert self.ip is not None
        return (self.ip, 54321)


def _patch_socket(monkeypatch: pytest.MonkeyPatch, ip: str | None) -> list[_FakeSocket]:
    made: list[_FakeSocket] = []

    def factory(family: int, kind: int) -> _FakeSocket:
        assert family == socket.AF_INET
        assert kind == socket.SOCK_DGRAM
        made.append(_FakeSocket(ip))
        return made[-1]

    monkeypatch.setattr(sonos.socket, "socket", factory)
    return made


def test_lan_ip_udp_connect_trick(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _patch_socket(monkeypatch, "192.168.1.23")
    assert sonos._lan_ip(_settings(serve_host=None)) == "192.168.1.23"
    assert made[0].connected_to == ("8.8.8.8", 80)


@pytest.mark.parametrize("ip", [None, "127.0.0.1", "0.0.0.0"])
def test_lan_ip_failure_hints_serve_host(monkeypatch: pytest.MonkeyPatch, ip: str | None) -> None:
    _patch_socket(monkeypatch, ip)
    with pytest.raises(ProviderError, match="VP_SERVE_HOST"):
        sonos._lan_ip(_settings(serve_host=None))


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (AudioResult(data=_wav(0.5), mime_type="audio/wav", sample_rate=8000), 0.5),
        (AudioResult(data=b"\x00" * 48000, mime_type="audio/l16", sample_rate=24000), 1.0),
        (
            AudioResult(data=b"\x00" * 48000, mime_type="audio/l16", sample_rate=24000, channels=2),
            0.5,
        ),
        (AudioResult(data=b"\x00" * 10, mime_type="audio/l16", sample_rate=None), None),
        (AudioResult(data=b"not a wav", mime_type="audio/wav", sample_rate=None), None),
        (AudioResult(data=MP3_BYTES, mime_type="audio/mpeg", sample_rate=None), None),
    ],
)
def test_duration(result: AudioResult, expected: float | None) -> None:
    assert sonos._duration(result) == expected


# --------------------------------------------------------------------------- play_sonos


def _fetching_ha(
    fetch: Callable[[str], None] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """A fake HA endpoint that (like a real Sonos) fetches the media URL it was given."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = json.loads(request.content)["media_content_id"]
        if fetch is not None:
            fetch(url)
        return httpx.Response(200, json=[])

    return handler


def test_play_sonos_end_to_end(fake_convert: list[tuple[AudioResult, str]]) -> None:
    seen: dict[str, Any] = {}

    def fetch(url: str) -> None:
        seen["url"] = url
        with urllib.request.urlopen(url, timeout=5) as resp:
            seen["body"] = resp.read()

    duration, grace = 0.3, 0.2
    result = AudioResult(data=_wav(duration), mime_type="audio/wav", sample_rate=8000)
    with respx.mock:
        route = respx.post(HA_ENDPOINT).mock(side_effect=_fetching_ha(fetch))
        start = time.monotonic()
        sonos.play_sonos(result, _settings(), fetch_timeout=2, grace=grace, max_serve=10)
        elapsed = time.monotonic() - start

    assert fake_convert == [(result, "mp3")]
    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    body = json.loads(request.content)
    assert body["entity_id"] == ENTITY
    assert body["media_content_type"] == "music"
    assert body["media_content_id"] == seen["url"]
    assert seen["url"].startswith("http://127.0.0.1:")
    assert seen["url"].endswith(".mp3")
    assert seen["body"] == MP3_BYTES
    # Kept serving for duration + grace after the fetch, then shut down.
    assert duration + grace <= elapsed < 5
    with pytest.raises(OSError):
        urllib.request.urlopen(seen["url"], timeout=1)


def test_play_sonos_uses_explicit_entity(fake_convert: list[Any]) -> None:
    def fetch(url: str) -> None:
        with urllib.request.urlopen(url, timeout=5) as resp:
            resp.read()

    result = AudioResult(data=_wav(0.05), mime_type="audio/wav", sample_rate=8000)
    with respx.mock:
        route = respx.post(HA_ENDPOINT).mock(side_effect=_fetching_ha(fetch))
        sonos.play_sonos(
            result, _settings(), "media_player.kitchen", fetch_timeout=2, grace=0, max_serve=5
        )
    assert json.loads(route.calls.last.request.content)["entity_id"] == "media_player.kitchen"


def test_play_sonos_never_fetched_hints_firewall(fake_convert: list[Any]) -> None:
    result = AudioResult(data=_wav(0.1), mime_type="audio/wav", sample_rate=8000)
    with respx.mock:
        respx.post(HA_ENDPOINT).mock(side_effect=_fetching_ha(None))
        start = time.monotonic()
        with pytest.raises(ProviderError) as excinfo:
            sonos.play_sonos(result, _settings(), fetch_timeout=0.3, grace=0, max_serve=10)
        elapsed = time.monotonic() - start
    message = str(excinfo.value)
    assert "never fetched" in message
    assert "firewall" in message
    assert "reach" in message
    assert TOKEN not in message
    assert elapsed < 3


def test_play_sonos_stops_at_max_serve_time(fake_convert: list[Any]) -> None:
    def fetch(url: str) -> None:
        with urllib.request.urlopen(url, timeout=5) as resp:
            resp.read()

    # 60 s of audio, but the hard cap is 0.5 s.
    result = AudioResult(data=b"\x00" * 24000 * 2 * 60, mime_type="audio/l16", sample_rate=24000)
    with respx.mock:
        respx.post(HA_ENDPOINT).mock(side_effect=_fetching_ha(fetch))
        start = time.monotonic()
        sonos.play_sonos(result, _settings(), fetch_timeout=2, grace=5, max_serve=0.5)
        elapsed = time.monotonic() - start
    assert 0.4 <= elapsed < 3


def test_play_sonos_ha_401_shuts_server_down(
    fake_convert: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    servers: list[sonos._OneFileServer] = []
    real_enter = sonos._OneFileServer.__enter__

    def spy_enter(self: sonos._OneFileServer) -> sonos._OneFileServer:
        servers.append(self)
        return real_enter(self)

    monkeypatch.setattr(sonos._OneFileServer, "__enter__", spy_enter)
    result = AudioResult(data=_wav(0.1), mime_type="audio/wav", sample_rate=8000)
    with respx.mock:
        respx.post(HA_ENDPOINT).mock(return_value=httpx.Response(401))
        with pytest.raises(ProviderError, match="HA_TOKEN"):
            sonos.play_sonos(result, _settings(), fetch_timeout=1, grace=0, max_serve=5)
    assert len(servers) == 1
    with pytest.raises(RuntimeError):
        _ = servers[0].port  # closed


def test_play_sonos_propagates_conversion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(result: AudioResult, fmt: str) -> AudioResult:
        raise ProviderError("ffmpeg is not installed")

    monkeypatch.setattr(audio, "convert", broken)
    result = AudioResult(data=_wav(0.1), mime_type="audio/wav", sample_rate=8000)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(HA_ENDPOINT)
        with pytest.raises(ProviderError, match="ffmpeg"):
            sonos.play_sonos(result, _settings())
        assert not route.called


# --------------------------------------------------------------- L1: MP3 input duration


def test_duration_of_mp3_is_decoded_via_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def convert(result: AudioResult, fmt: str) -> AudioResult:
        calls.append(fmt)
        return AudioResult(data=_wav(1.5), mime_type="audio/wav", sample_rate=8000)

    monkeypatch.setattr(audio, "convert", convert)
    mp3 = AudioResult(data=MP3_BYTES, mime_type="audio/mpeg", sample_rate=None)
    assert sonos._duration(mp3) == 1.5
    assert calls == ["wav"]


def test_duration_falls_back_to_none_when_decoding_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(result: AudioResult, fmt: str) -> AudioResult:
        raise ConfigError("ffmpeg is required")

    monkeypatch.setattr(audio, "convert", broken)
    mp3 = AudioResult(data=MP3_BYTES, mime_type="audio/mpeg", sample_rate=None)
    assert sonos._duration(mp3) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_duration_of_real_mp3_with_ffmpeg() -> None:
    wav = AudioResult(data=_wav(0.5, rate=24000), mime_type="audio/wav", sample_rate=24000)
    mp3 = audio.convert(wav, "mp3")
    duration = sonos._duration(mp3)
    assert duration is not None
    assert 0.4 <= duration <= 0.7  # MP3 encoder padding adds a few ms


def test_play_sonos_mp3_input_does_not_wait_default_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def convert(result: AudioResult, fmt: str) -> AudioResult:
        if fmt == "wav":
            return AudioResult(data=_wav(0.2), mime_type="audio/wav", sample_rate=8000)
        return result  # already MP3

    monkeypatch.setattr(audio, "convert", convert)

    def fetch(url: str) -> None:
        with urllib.request.urlopen(url, timeout=5) as resp:
            resp.read()

    result = AudioResult(data=MP3_BYTES, mime_type="audio/mpeg", sample_rate=None)
    with respx.mock:
        respx.post(HA_ENDPOINT).mock(side_effect=_fetching_ha(fetch))
        start = time.monotonic()
        sonos.play_sonos(result, _settings(), fetch_timeout=2, grace=0.1, max_serve=10)
        elapsed = time.monotonic() - start
    assert 0.3 <= elapsed < 3  # 0.2 s audio + 0.1 s grace, not DEFAULT_DURATION


# ------------------------------------------------ S2: bind the LAN IP, not all interfaces


def test_play_sonos_binds_the_lan_ip(
    monkeypatch: pytest.MonkeyPatch, fake_convert: list[Any]
) -> None:
    monkeypatch.setattr(sonos, "BIND_HOST", None)  # production default
    hosts: list[str] = []
    real_init = sonos._OneFileServer.__init__

    def spy_init(self: sonos._OneFileServer, data: bytes, **kwargs: Any) -> None:
        hosts.append(kwargs["host"])
        real_init(self, data, **kwargs)

    monkeypatch.setattr(sonos._OneFileServer, "__init__", spy_init)

    def fetch(url: str) -> None:
        with urllib.request.urlopen(url, timeout=5) as resp:
            resp.read()

    result = AudioResult(data=_wav(0.05), mime_type="audio/wav", sample_rate=8000)
    with respx.mock:
        respx.post(HA_ENDPOINT).mock(side_effect=_fetching_ha(fetch))
        sonos.play_sonos(
            result, _settings(serve_host="127.0.0.1"), fetch_timeout=2, grace=0, max_serve=5
        )
    assert hosts == ["127.0.0.1"]


def test_serve_binds_requested_host() -> None:
    with sonos._serve(MP3_BYTES, "127.0.0.1", 0) as server:
        assert server._host == "127.0.0.1"
        status, _, body = _get(server.port, server.path)
        assert (status, body) == (200, MP3_BYTES)
    with pytest.raises(RuntimeError):
        _ = server.port  # closed on exit


def test_serve_falls_back_when_lan_ip_cannot_be_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # 203.0.113.5 (TEST-NET-3) is not an address of this machine, e.g. a NAT VP_SERVE_HOST.
    monkeypatch.setattr(sonos, "FALLBACK_BIND_HOST", "127.0.0.1")  # tests stay on loopback
    with sonos._serve(MP3_BYTES, "203.0.113.5", 0) as server:
        assert server._host == "127.0.0.1"
        status, _, body = _get(server.port, server.path)
        assert (status, body) == (200, MP3_BYTES)
