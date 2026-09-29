"""T8: service-level resolution, precedence, format handling, and voice management."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from voice_playground import service
from voice_playground.errors import ConfigError, ProviderError, UnsupportedCapability
from voice_playground.providers.base import AudioResult, CreatedVoice, TTSRequest
from voice_playground.providers.fake import FakeProvider, sine_wav
from voice_playground.providers.registry import PROVIDERS, get_provider_class
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
    for name in PROVIDERS:
        assert getattr(get_provider_class(name), "default_voice", None), name


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


# --- M2: recreating a voice deletes the remote voice it replaces --------------------------

DESIGNED = "provider: fake\ntype: designed\ndescription: warm\n"


@pytest.fixture
def delete_calls(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    delete = MagicMock(name="delete_voice")
    monkeypatch.setattr(FakeProvider, "delete_voice", delete)
    return delete


@pytest.fixture
def new_ids(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make each created fake voice get a fresh id (the real fake is deterministic)."""
    ids: list[str] = []

    def create(self: FakeProvider, cfg: Any) -> CreatedVoice:
        ids.append(f"new_{len(ids)}")
        return CreatedVoice(ids[-1])

    monkeypatch.setattr(FakeProvider, "create_voice", create)
    return ids


def test_stale_voice_recreate_deletes_old_remote_voice(
    settings: Settings,
    voices_dir: Path,
    cache_dir: Path,
    delete_calls: MagicMock,
    new_ids: list[str],
    tts_calls: list[TTSRequest],
) -> None:
    write_voice(voices_dir, "d", DESIGNED)
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "fake", CreatedVoice("old_id"))
    write_voice(voices_dir, "d", DESIGNED.replace("warm", "cold"))  # now stale
    tts(settings, provider=None, voice="d", output=settings.cache_dir / "a.wav")
    delete_calls.assert_called_once_with("old_id")
    entry = VoiceCache(cache_dir).get("d")
    assert entry is not None
    assert entry.remote_id == "new_0"
    assert tts_calls[0].voice.provider_voice == "new_0"


def test_provider_mismatch_recreate_deletes_on_old_provider(
    settings: Settings,
    voices_dir: Path,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    delete_calls: MagicMock,
    new_ids: list[str],
) -> None:
    from pydantic import SecretStr

    from voice_playground.providers.google import GoogleProvider

    google_delete = MagicMock(name="google.delete_voice")
    monkeypatch.setattr(GoogleProvider, "delete_voice", google_delete)
    keyed = settings.model_copy(update={"google_api_key": SecretStr("dummy-google")})
    write_voice(voices_dir, "d", DESIGNED)
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "google", CreatedVoice("voice_g"))
    tts(keyed, provider=None, voice="d", output=cache_dir / "a.wav")
    google_delete.assert_called_once_with("voice_g")
    delete_calls.assert_not_called()
    entry = VoiceCache(cache_dir).get("d")
    assert entry is not None
    assert (entry.provider, entry.remote_id) == ("fake", "new_0")


def test_voices_create_force_deletes_old_remote_voice(
    settings: Settings,
    voices_dir: Path,
    cache_dir: Path,
    delete_calls: MagicMock,
    new_ids: list[str],
) -> None:
    write_voice(voices_dir, "d", DESIGNED)
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "fake", CreatedVoice("old_id"))
    created = service.voices_create(settings, "d", force=True)
    assert created.remote_id == "new_0"
    delete_calls.assert_called_once_with("old_id")
    entry = VoiceCache(cache_dir).get("d")
    assert entry is not None
    assert entry.remote_id == "new_0"


def test_recreate_same_remote_id_is_not_deleted(
    settings: Settings, voices_dir: Path, cache_dir: Path, delete_calls: MagicMock
) -> None:
    # The fake provider returns the same id again: deleting "the old one" would delete it.
    write_voice(voices_dir, "d", DESIGNED)
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "fake", CreatedVoice("fake_voice_d"))
    service.voices_create(settings, "d", force=True)
    delete_calls.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("google voice delete failed (404): not found"),
        UnsupportedCapability("provider 'fake' does not support voice_design"),
    ],
)
def test_delete_failure_warns_with_orphaned_id_and_continues(
    settings: Settings,
    voices_dir: Path,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    new_ids: list[str],
    error: Exception,
) -> None:
    monkeypatch.setattr(FakeProvider, "delete_voice", MagicMock(side_effect=error))
    write_voice(voices_dir, "d", DESIGNED)
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "fake", CreatedVoice("old_id"))
    created = service.voices_create(settings, "d", force=True)
    assert created.remote_id == "new_0"
    warnings = [line for line in capsys.readouterr().err.splitlines() if "old_id" in line]
    assert len(warnings) == 1
    assert warnings[0].startswith("warning:")
    assert "orphaned" in warnings[0]
    entry = VoiceCache(cache_dir).get("d")
    assert entry is not None
    assert entry.remote_id == "new_0"


def test_delete_on_old_provider_without_key_warns(
    settings: Settings,
    voices_dir: Path,
    cache_dir: Path,
    capsys: pytest.CaptureFixture[str],
    new_ids: list[str],
) -> None:
    write_voice(voices_dir, "d", DESIGNED)
    VoiceCache(cache_dir).record(load_voice("d", voices_dir), "google", CreatedVoice("voice_g"))
    tts(settings, provider=None, voice="d", output=cache_dir / "a.wav")  # no GOOGLE_API_KEY
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "voice_g" in err
    assert "GOOGLE_API_KEY" in err
    entry = VoiceCache(cache_dir).get("d")
    assert entry is not None
    assert entry.provider == "fake"


def test_expired_voice_delete_is_attempted_and_failure_tolerated(
    settings: Settings,
    voices_dir: Path,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    new_ids: list[str],
) -> None:
    delete = MagicMock(side_effect=ProviderError("gone"))
    monkeypatch.setattr(FakeProvider, "delete_voice", delete)
    write_voice(voices_dir, "d", DESIGNED)
    past = datetime.now(UTC) - timedelta(days=1)
    VoiceCache(cache_dir).record(
        load_voice("d", voices_dir), "fake", CreatedVoice("old_id", expires_at=past)
    )
    tts(settings, provider=None, voice="d", output=cache_dir / "a.wav")
    delete.assert_called_once_with("old_id")
    assert "orphaned" not in capsys.readouterr().err
    entry = VoiceCache(cache_dir).get("d")
    assert entry is not None
    assert entry.remote_id == "new_0"


# --- L2: provider validation runs before a voice is auto-created ---------------------------


def test_invalid_model_for_custom_voice_fails_before_creating(
    settings: Settings, voices_dir: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import SecretStr

    from voice_playground.providers.google import GoogleProvider

    create = MagicMock(side_effect=AssertionError("create_voice must not be called"))
    monkeypatch.setattr(GoogleProvider, "create_voice", create)
    keyed = settings.model_copy(update={"google_api_key": SecretStr("dummy-google")})
    write_voice(voices_dir, "s", "provider: google\ntype: designed\ndescription: warm\n")
    with pytest.raises(ConfigError, match="does not support custom voices"):
        tts(
            keyed,
            provider=None,
            voice="s",
            model="gemini-3.1-flash-tts-preview",
            output=cache_dir / "a.wav",
        )
    create.assert_not_called()
    assert VoiceCache(cache_dir).get("s") is None


def test_validate_tts_hook_is_called_before_create(
    settings: Settings, voices_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validate = MagicMock(side_effect=ConfigError("bad combo"))
    create = MagicMock()
    monkeypatch.setattr(FakeProvider, "validate_tts", validate)
    monkeypatch.setattr(FakeProvider, "create_voice", create)
    write_voice(voices_dir, "d", DESIGNED + "sample_rate: 16000\n")
    with pytest.raises(ConfigError, match="bad combo"):
        tts(settings, provider=None, voice="d", model="m1")
    validate.assert_called_once_with(model="m1", custom_voice=True, sample_rate=16000)
    create.assert_not_called()


# --- L3 / L1: convert once; Sonos gets the unconverted provider result ---------------------


def test_output_mp3_plus_local_play_converts_once(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    playback: dict[str, MagicMock],
) -> None:
    ffmpeg = MagicMock(return_value=b"ID3-mp3")
    monkeypatch.setattr("voice_playground.audio._ffmpeg", ffmpeg)
    out = tts(settings, output=tmp_path / "o.mp3", play="local")
    assert out is not None
    assert out.read_bytes() == b"ID3-mp3"
    assert ffmpeg.call_count == 1
    played = playback["local"].call_args.args[0]
    assert (played.mime_type, played.data) == ("audio/mpeg", b"ID3-mp3")


def test_sonos_gets_unconverted_provider_result(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    playback: dict[str, MagicMock],
) -> None:
    ffmpeg = MagicMock(return_value=b"ID3-mp3")
    monkeypatch.setattr("voice_playground.audio._ffmpeg", ffmpeg)
    tts(settings, output=tmp_path / "o.mp3", play="sonos")
    played = playback["sonos"].call_args.args[0]
    assert played.mime_type == "audio/wav"  # sonos measures the WAV, then converts to MP3
    assert ffmpeg.call_count == 1  # only for the -o file
