"""T8: service-level resolution, precedence, format handling, and voice management."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from voice_playground import service
from voice_playground.errors import ConfigError
from voice_playground.providers.base import AudioResult, CreatedVoice, TTSRequest
from voice_playground.providers.fake import FakeProvider, sine_wav
from voice_playground.providers.registry import PROVIDERS
from voice_playground.settings import Settings
from voice_playground.voices import VoiceCache, load_voice


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def playback(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    local = MagicMock(name="play_local")
    sonos = MagicMock(name="play_sonos")
    monkeypatch.setattr("voice_playground.playback.local.play_local", local)
    monkeypatch.setattr("voice_playground.playback.sonos.play_sonos", sonos)
    return {"local": local, "sonos": sonos}


@pytest.fixture
def tts_calls(monkeypatch: pytest.MonkeyPatch) -> list[TTSRequest]:
    calls: list[TTSRequest] = []
    original = FakeProvider.tts

    def spy(self: FakeProvider, req: TTSRequest) -> AudioResult:
        calls.append(req)
        return original(self, req)

    monkeypatch.setattr(FakeProvider, "tts", spy)
    return calls


def tts(settings: Settings, **kw: Any) -> Path | None:
    args: dict[str, Any] = {
        "text": "hi",
        "text_file": None,
        "provider": "fake",
        "model": None,
        "voice": None,
        "output": None,
        "style": None,
        "fmt": None,
        "play": None,
        "speaker": None,
    }
    args.update(kw)
    return service.run_tts(settings=settings, **args)


def write_voice(voices_dir: Path, name: str, body: str) -> None:
    (voices_dir / f"{name}.yaml").write_text(f"name: {name}\n{body}")


def test_default_voices_cover_every_provider() -> None:
    assert set(service.DEFAULT_VOICES) == set(PROVIDERS)


def test_no_voice_uses_provider_default_voice_and_model(
    settings: Settings, tts_calls: list[TTSRequest]
) -> None:
    tts(settings)
    assert tts_calls[0].voice.provider_voice == "fake-voice"
    assert tts_calls[0].model == "fake-tts"
    assert tts_calls[0].output_format == "wav"


def test_precedence_cli_over_config_over_default(
    settings: Settings, voices_dir: Path, tts_calls: list[TTSRequest]
) -> None:
    write_voice(
        voices_dir,
        "v",
        "provider: fake\nmodel: cfg-model\ntype: prebuilt\nvoice: fake_alto\nstyle: calm\n"
        "language: en-GB\nprovider_options: {speed: 1.2}\n",
    )
    tts(settings, voice="v")
    tts(settings, voice="v", model="cli-model", style="whispered")
    write_voice(voices_dir, "w", "provider: fake\ntype: prebuilt\nvoice: fake_alto\n")
    tts(settings, voice="w")
    cfg_req, cli_req, default_req = tts_calls
    assert (cfg_req.model, cfg_req.voice.style) == ("cfg-model", "calm")
    assert (cli_req.model, cli_req.voice.style) == ("cli-model", "whispered")
    assert cli_req.voice.language == "en-GB"
    assert cli_req.voice.options == {"speed": 1.2}
    assert default_req.model == "fake-tts"


def test_raw_voice_with_provider_flag_passes_through(
    settings: Settings, tts_calls: list[TTSRequest]
) -> None:
    tts(settings, voice="SomeRawVoice", style="urgent")
    assert tts_calls[0].voice.provider_voice == "SomeRawVoice"
    assert tts_calls[0].voice.style == "urgent"
    assert tts_calls[0].voice.config_name is None


def test_text_sources(settings: Settings, tmp_path: Path, tts_calls: list[TTSRequest]) -> None:
    tts(settings, text=None, stdin=io.StringIO("  piped  \n"))
    assert tts_calls[-1].text == "piped"
    with pytest.raises(ConfigError, match="no text"):
        tts(settings, text=None, stdin=_TTY("ignored"))
    with pytest.raises(ConfigError, match="not both"):
        tts(settings, text="a", text_file=tmp_path / "f.txt")
    with pytest.raises(ConfigError, match="empty"):
        tts(settings, text="   ")


def test_mp3_request_is_converted_even_if_provider_returns_wav(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = MagicMock(return_value=b"ID3-mp3")
    monkeypatch.setattr("voice_playground.audio._ffmpeg", ffmpeg)
    out = tts(settings, output=tmp_path / "o.mp3")
    assert out == tmp_path / "o.mp3"
    assert out.read_bytes() == b"ID3-mp3"


def test_playback_gets_requested_format(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, playback: dict[str, MagicMock]
) -> None:
    monkeypatch.setattr("voice_playground.audio._ffmpeg", MagicMock(return_value=b"ID3"))
    tts(settings, fmt="mp3")
    assert playback["local"].call_args.args[0].mime_type == "audio/mpeg"


def test_sample_rate_mismatch_warns_without_resampling(
    settings: Settings,
    voices_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_voice(voices_dir, "v", "provider: fake\ntype: prebuilt\nvoice: x\nsample_rate: 16000\n")

    def fixed_rate(self: FakeProvider, req: TTSRequest) -> AudioResult:
        assert req.sample_rate == 16_000
        return AudioResult(sine_wav(24_000), "audio/wav", 24_000)

    monkeypatch.setattr(FakeProvider, "tts", fixed_rate)
    out = tts(settings, voice="v", output=tmp_path / "a.wav")
    assert out is not None
    assert out.read_bytes() == sine_wav(24_000)
    assert "returned 24000 Hz audio (requested 16000 Hz)" in capsys.readouterr().err


def test_cloned_voice_auto_create_records_cache(
    settings: Settings,
    voices_dir: Path,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tts_calls: list[TTSRequest],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "me.wav").write_bytes(b"ref")
    write_voice(voices_dir, "me", "provider: fake\ntype: cloned\nreference_audio: me.wav\n")
    tts(settings, provider=None, voice="me", output=tmp_path / "a.wav")
    entry = VoiceCache(cache_dir).get("me")
    assert entry is not None
    assert entry.remote_id == "fake_voice_me"
    assert tts_calls[0].voice.provider_voice == "fake_voice_me"


def test_voices_create_returns_cached_without_provider_call(
    settings: Settings, voices_dir: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_voice(voices_dir, "d", "provider: fake\ntype: designed\ndescription: warm\n")
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "fake", CreatedVoice("cached_id"))
    create = MagicMock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr(FakeProvider, "create_voice", create)
    assert service.voices_create(settings, "d").remote_id == "cached_id"


def test_voices_create_without_provider_is_config_error(
    settings: Settings, voices_dir: Path
) -> None:
    write_voice(voices_dir, "d", "type: designed\ndescription: warm\n")
    with pytest.raises(ConfigError, match="needs a provider"):
        service.voices_create(settings, "d")


def test_voices_delete_uses_cached_provider_and_id(
    settings: Settings, voices_dir: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_voice(voices_dir, "d", "provider: fake\ntype: designed\ndescription: warm\n")
    cache = VoiceCache(cache_dir)
    cache.record(load_voice("d", voices_dir), "fake", CreatedVoice("remote_123"))
    delete = MagicMock()
    monkeypatch.setattr(FakeProvider, "delete_voice", delete)
    service.voices_delete(settings, "d")
    delete.assert_called_once_with("remote_123")
    assert cache.get("d") is None


def test_voices_show_includes_cache_details(
    settings: Settings, voices_dir: Path, cache_dir: Path
) -> None:
    write_voice(voices_dir, "d", "provider: fake\ntype: designed\ndescription: warm\n")
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "fake", CreatedVoice("rid"))
    info = service.voices_show(settings, "d")
    assert info["cache"] == "valid"
    assert info["remote_id"] == "rid"
    assert info["remote_provider"] == "fake"
    assert info["expires_at"] == "never"
    assert info["type"] == "designed"


def test_voices_list_unknown_provider_is_config_error(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="valid providers"):
        service.voices_list(settings, provider="nope")


def test_play_file_pcm_assumes_24k(
    settings: Settings, tmp_path: Path, playback: dict[str, MagicMock]
) -> None:
    raw = tmp_path / "a.pcm"
    raw.write_bytes(b"\x00\x00" * 10)
    service.play_file(settings, raw)
    played = playback["local"].call_args.args[0]
    assert (played.mime_type, played.sample_rate) == ("audio/l16", 24_000)


def test_play_file_unknown_extension_is_config_error(settings: Settings, tmp_path: Path) -> None:
    clip = tmp_path / "a.ogg"
    clip.write_bytes(b"x")
    with pytest.raises(ConfigError):
        service.play_file(settings, clip)
