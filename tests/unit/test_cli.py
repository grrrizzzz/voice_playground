"""T8: `vp` end to end through Typer's CliRunner with the fake provider and mocked playback."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from voice_playground.cli import app
from voice_playground.errors import ProviderError
from voice_playground.providers.base import AudioResult, CreatedVoice, TTSRequest
from voice_playground.providers.fake import FakeProvider
from voice_playground.providers.registry import PROVIDERS
from voice_playground.settings import Settings
from voice_playground.voices import VoiceConfig

runner = CliRunner()


@pytest.fixture(autouse=True)
def playback(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Never play real audio: both playback targets are mocks."""
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


@pytest.fixture
def create_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    original = FakeProvider.create_voice

    def spy(self: FakeProvider, cfg: VoiceConfig) -> CreatedVoice:
        calls.append(cfg.name)
        return original(self, cfg)

    monkeypatch.setattr(FakeProvider, "create_voice", spy)
    return calls


def write_voice(voices_dir: Path, name: str, body: str) -> None:
    (voices_dir / f"{name}.yaml").write_text(f"name: {name}\n{body}")


def one_line_error(result: object) -> str:
    stderr = getattr(result, "stderr", "")
    lines = [line for line in stderr.splitlines() if line.startswith("error:")]
    assert len(lines) == 1, stderr
    return lines[0]


# --- tts ---------------------------------------------------------------------------------


def test_tts_writes_valid_wav_without_playback(
    tmp_path: Path, playback: dict[str, MagicMock]
) -> None:
    out = tmp_path / "out.wav"
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi", "-o", str(out)])
    assert result.exit_code == 0, result.output
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 24_000
        assert w.getnchannels() == 1
        assert w.getnframes() > 0
    playback["local"].assert_not_called()
    playback["sonos"].assert_not_called()


def test_tts_without_output_plays_locally_once(playback: dict[str, MagicMock]) -> None:
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi"])
    assert result.exit_code == 0, result.output
    playback["local"].assert_called_once()
    assert playback["local"].call_args.args[0].mime_type == "audio/wav"
    playback["sonos"].assert_not_called()


def test_tts_play_sonos_uses_settings_entity(
    monkeypatch: pytest.MonkeyPatch, playback: dict[str, MagicMock]
) -> None:
    monkeypatch.setenv("HA_SONOS_ENTITY", "media_player.kitchen")
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi", "--play", "sonos"])
    assert result.exit_code == 0, result.output
    playback["sonos"].assert_called_once()
    _, settings = playback["sonos"].call_args.args
    assert playback["sonos"].call_args.kwargs["entity_id"] is None  # sonos falls back to settings
    assert settings.ha_sonos_entity == "media_player.kitchen"
    playback["local"].assert_not_called()


def test_tts_speaker_implies_sonos(playback: dict[str, MagicMock]) -> None:
    result = runner.invoke(
        app, ["tts", "--provider", "fake", "--text", "hi", "--speaker", "media_player.office"]
    )
    assert result.exit_code == 0, result.output
    playback["sonos"].assert_called_once()
    assert playback["sonos"].call_args.kwargs["entity_id"] == "media_player.office"
    playback["local"].assert_not_called()


def test_tts_output_plus_play_writes_and_plays(
    tmp_path: Path, playback: dict[str, MagicMock]
) -> None:
    out = tmp_path / "out.wav"
    result = runner.invoke(
        app, ["tts", "--provider", "fake", "--text", "hi", "-o", str(out), "--play", "local"]
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    playback["local"].assert_called_once()


def test_tts_play_local_with_speaker_is_usage_error() -> None:
    result = runner.invoke(
        app,
        ["tts", "--provider", "fake", "--text", "hi", "--play", "local", "--speaker", "x.y"],
    )
    assert result.exit_code == 2
    assert "--speaker" in one_line_error(result)


def test_tts_mp3_output_is_converted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake returns WAV for mp3 requests; the service must convert (ffmpeg mocked here).
    ffmpeg = MagicMock(return_value=b"ID3-fake-mp3")
    monkeypatch.setattr("voice_playground.audio._ffmpeg", ffmpeg)
    out = tmp_path / "out.mp3"
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == b"ID3-fake-mp3"
    assert ffmpeg.call_args.args[2] == ["-f", "mp3"]


def test_tts_format_flag_overrides_extension(tmp_path: Path) -> None:
    out = tmp_path / "out.wav"
    result = runner.invoke(
        app, ["tts", "--provider", "fake", "--text", "hi", "-o", str(out), "--format", "pcm"]
    )
    assert result.exit_code == 0, result.output
    assert not out.read_bytes().startswith(b"RIFF")  # raw PCM despite the .wav name
    assert len(out.read_bytes()) == 24_000  # 0.5 s * 24 kHz * 2 bytes


def test_tts_unknown_extension_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["tts", "--provider", "fake", "--text", "hi", "-o", str(tmp_path / "x.ogg")]
    )
    assert result.exit_code == 2
    assert "--format" in one_line_error(result)


def test_tts_reads_stdin(tmp_path: Path, tts_calls: list[TTSRequest]) -> None:
    out = tmp_path / "x.wav"
    result = runner.invoke(
        app, ["tts", "--provider", "fake", "-o", str(out)], input="hello from stdin\n"
    )
    assert result.exit_code == 0, result.output
    assert tts_calls[0].text == "hello from stdin"
    assert out.is_file()


def test_tts_reads_text_file(tmp_path: Path, tts_calls: list[TTSRequest]) -> None:
    script = tmp_path / "script.txt"
    script.write_text("from a file\n")
    result = runner.invoke(
        app,
        ["tts", "--provider", "fake", "--text-file", str(script), "-o", str(tmp_path / "a.wav")],
    )
    assert result.exit_code == 0, result.output
    assert tts_calls[0].text == "from a file"


@pytest.mark.parametrize("stdin", ["", "   \n"])
def test_tts_missing_text_exits_2(stdin: str, playback: dict[str, MagicMock]) -> None:
    result = runner.invoke(app, ["tts", "--provider", "fake"], input=stdin)
    assert result.exit_code == 2
    assert "text" in one_line_error(result)
    playback["local"].assert_not_called()


def test_tts_missing_text_file_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["tts", "--provider", "fake", "--text-file", str(tmp_path / "nope.txt")]
    )
    assert result.exit_code == 2


def test_tts_designed_voice_auto_creates_once_then_uses_cache(
    voices_dir: Path,
    cache_dir: Path,
    tmp_path: Path,
    tts_calls: list[TTSRequest],
    create_calls: list[str],
) -> None:
    write_voice(voices_dir, "teller", "provider: fake\ntype: designed\ndescription: warm\n")
    args = ["tts", "--voice", "teller", "--text", "hi", "-o", str(tmp_path / "a.wav")]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert create_calls == ["teller"]
    assert [r.voice.provider_voice for r in tts_calls] == ["fake_voice_teller"] * 2
    cached = json.loads((cache_dir / "voices.json").read_text())
    assert cached["teller"]["remote_id"] == "fake_voice_teller"
    assert cached["teller"]["provider"] == "fake"


def test_tts_no_auto_create_exits_2(
    voices_dir: Path, tmp_path: Path, create_calls: list[str]
) -> None:
    write_voice(voices_dir, "teller", "provider: fake\ntype: designed\ndescription: warm\n")
    out = tmp_path / "a.wav"
    result = runner.invoke(
        app,
        ["tts", "--voice", "teller", "--text", "hi", "--no-auto-create", "-o", str(out)],
    )
    assert result.exit_code == 2
    assert "vp voices create teller" in one_line_error(result)
    assert create_calls == []


def test_tts_voice_config_provider_conflict_exits_2(voices_dir: Path) -> None:
    write_voice(voices_dir, "n", "provider: google\ntype: prebuilt\nvoice: Kore\n")
    result = runner.invoke(app, ["tts", "--provider", "fake", "--voice", "n", "--text", "hi"])
    assert result.exit_code == 2
    assert "google" in one_line_error(result)


def test_tts_voice_config_provider_used_when_flag_omitted(
    voices_dir: Path, tmp_path: Path, tts_calls: list[TTSRequest]
) -> None:
    write_voice(voices_dir, "n", "provider: fake\ntype: prebuilt\nvoice: fake_alto\nstyle: calm\n")
    out = tmp_path / "a.wav"
    result = runner.invoke(app, ["tts", "--voice", "n", "--text", "hi", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert tts_calls[0].voice.provider_voice == "fake_alto"
    assert tts_calls[0].voice.style == "calm"


def test_tts_raw_voice_without_provider_exits_2() -> None:
    result = runner.invoke(app, ["tts", "--voice", "Kore", "--text", "hi"])
    assert result.exit_code == 2
    assert "--provider" in one_line_error(result)


def test_tts_no_voice_no_provider_exits_2() -> None:
    result = runner.invoke(app, ["tts", "--text", "hi"])
    assert result.exit_code == 2
    assert "--provider" in one_line_error(result)


def test_tts_unknown_provider_exits_2() -> None:
    result = runner.invoke(app, ["tts", "--provider", "nope", "--text", "hi"])
    assert result.exit_code == 2
    assert "valid providers" in one_line_error(result)


def test_tts_missing_key_exits_2_naming_env_var() -> None:
    result = runner.invoke(app, ["tts", "--provider", "openai", "--text", "hi"])
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in one_line_error(result)


def test_tts_unsupported_capability_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_playground.errors import UnsupportedCapability

    def unsupported(self: FakeProvider, req: TTSRequest) -> None:
        raise UnsupportedCapability("provider 'fake' does not support tts")

    monkeypatch.setattr(FakeProvider, "tts", unsupported)
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi"])
    assert result.exit_code == 2


def test_provider_error_exits_1_one_line_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = "sk-dummy-SECRET-value-123"
    monkeypatch.setenv("OPENAI_API_KEY", dummy)

    def fail(self: FakeProvider, req: TTSRequest) -> None:
        raise ProviderError(f"boom\nwith key {dummy}")

    monkeypatch.setattr(FakeProvider, "tts", fail)
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi"])
    assert result.exit_code == 1
    line = one_line_error(result)
    assert dummy not in result.output
    assert line == "error: boom with key ***"


def test_keyboard_interrupt_exits_130_quietly(
    monkeypatch: pytest.MonkeyPatch, playback: dict[str, MagicMock]
) -> None:
    playback["local"].side_effect = KeyboardInterrupt
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi"])
    assert result.exit_code == 130
    assert "error" not in result.output
    assert "Traceback" not in result.output


def test_debug_flag_shows_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: FakeProvider, req: TTSRequest) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(FakeProvider, "tts", boom)
    quiet = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi"])
    assert quiet.exit_code == 1
    assert "unexpected RuntimeError: kaboom" in one_line_error(quiet)
    loud = runner.invoke(app, ["--debug", "tts", "--provider", "fake", "--text", "hi"])
    assert loud.exit_code == 1
    assert isinstance(loud.exception, RuntimeError)  # re-raised for a traceback


def test_invalid_settings_exit_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VP_SERVE_PORT", "not-a-port")
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 2
    assert "not-a-port" not in result.output


def test_bad_flag_value_exits_2() -> None:
    result = runner.invoke(app, ["tts", "--provider", "fake", "--format", "ogg", "--text", "x"])
    assert result.exit_code == 2


# --- stt ---------------------------------------------------------------------------------


def _wav(path: Path) -> Path:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\x00\x00" * 160)
    path.write_bytes(buf.getvalue())
    return path


def test_stt_prints_transcript(tmp_path: Path) -> None:
    clip = _wav(tmp_path / "x.wav")
    result = runner.invoke(app, ["stt", "--provider", "fake", "--input", str(clip)])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "x.wav"  # the fake echoes the file name


def test_stt_writes_output_file(tmp_path: Path) -> None:
    clip = _wav(tmp_path / "x.wav")
    out = tmp_path / "sub" / "transcript.txt"
    result = runner.invoke(app, ["stt", "--provider", "fake", "--input", str(clip), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text() == "x.wav\n"
    assert result.stdout == ""


def test_stt_missing_input_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["stt", "--provider", "fake", "--input", str(tmp_path / "missing.wav")]
    )
    assert result.exit_code == 2
    assert "not found" in one_line_error(result)


def test_stt_without_provider_exits_2(tmp_path: Path) -> None:
    clip = _wav(tmp_path / "x.wav")
    result = runner.invoke(app, ["stt", "--input", str(clip)])
    assert result.exit_code == 2


# --- providers / play --------------------------------------------------------------------


def test_providers_lists_capabilities_and_key_status_never_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy = "dummy-ELEVEN-key-987654"
    monkeypatch.setenv("ELEVENLABS_API_KEY", dummy)
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0, result.output
    assert dummy not in result.output
    for name in PROVIDERS:
        assert f"{name}:" in result.output
    assert "ELEVENLABS_API_KEY yes" in result.output
    assert "OPENAI_API_KEY no" in result.output
    fake_line = next(line for line in result.stdout.splitlines() if line.startswith("fake:"))
    for cap in ("tts", "stt", "voice_design", "voice_clone", "voice_library"):
        assert cap in fake_line


def test_play_command_local_and_sonos(tmp_path: Path, playback: dict[str, MagicMock]) -> None:
    clip = _wav(tmp_path / "clip.wav")
    assert runner.invoke(app, ["play", str(clip)]).exit_code == 0
    playback["local"].assert_called_once()
    played = playback["local"].call_args.args[0]
    assert (played.mime_type, played.sample_rate) == ("audio/wav", 16_000)
    result = runner.invoke(app, ["play", str(clip), "--speaker", "media_player.den"])
    assert result.exit_code == 0, result.output
    assert playback["sonos"].call_args.kwargs["entity_id"] == "media_player.den"


def test_play_missing_file_exits_2(tmp_path: Path) -> None:
    assert runner.invoke(app, ["play", str(tmp_path / "nope.wav")]).exit_code == 2


# --- voices ------------------------------------------------------------------------------


def test_voices_list_and_filter(voices_dir: Path) -> None:
    write_voice(voices_dir, "a", "provider: fake\ntype: prebuilt\nvoice: fake_alto\n")
    write_voice(voices_dir, "b", "provider: google\ntype: prebuilt\nvoice: Kore\n")
    result = runner.invoke(app, ["voices", "list"])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["a\tfake\tprebuilt", "b\tgoogle\tprebuilt"]
    filtered = runner.invoke(app, ["voices", "list", "--provider", "fake"])
    assert filtered.stdout.splitlines() == ["a\tfake\tprebuilt"]


def test_voices_list_invalid_config_exits_2(voices_dir: Path) -> None:
    (voices_dir / "bad.yaml").write_text("name: bad\ntype: prebuilt\n")
    result = runner.invoke(app, ["voices", "list"])
    assert result.exit_code == 2
    assert "bad.yaml" in one_line_error(result)


def test_voices_show_create_show_delete_cycle(voices_dir: Path, create_calls: list[str]) -> None:
    write_voice(voices_dir, "teller", "provider: fake\ntype: designed\ndescription: warm\n")
    before = runner.invoke(app, ["voices", "show", "teller"])
    assert before.exit_code == 0, before.output
    assert "cache: missing" in before.stdout
    assert "description: warm" in before.stdout
    assert "model: fake-tts (provider default)" in before.stdout

    created = runner.invoke(app, ["voices", "create", "teller"])
    assert created.exit_code == 0, created.output
    assert "fake_voice_teller" in created.stdout
    again = runner.invoke(app, ["voices", "create", "teller"])
    assert again.exit_code == 0
    assert create_calls == ["teller"]  # cached, not recreated
    forced = runner.invoke(app, ["voices", "create", "teller", "--force"])
    assert forced.exit_code == 0
    assert create_calls == ["teller", "teller"]

    after = runner.invoke(app, ["voices", "show", "teller"])
    assert "cache: valid" in after.stdout
    assert "remote_id: fake_voice_teller" in after.stdout

    deleted = runner.invoke(app, ["voices", "delete", "teller"])
    assert deleted.exit_code == 0, deleted.output
    assert "cache: missing" in runner.invoke(app, ["voices", "show", "teller"]).stdout
    assert runner.invoke(app, ["voices", "delete", "teller"]).exit_code == 2


def test_voices_prebuilt_create_and_delete_exit_2(voices_dir: Path) -> None:
    write_voice(voices_dir, "n", "provider: fake\ntype: prebuilt\nvoice: fake_alto\n")
    shown = runner.invoke(app, ["voices", "show", "n"])
    assert "cache: n/a (prebuilt voice)" in shown.stdout
    assert runner.invoke(app, ["voices", "create", "n"]).exit_code == 2
    result = runner.invoke(app, ["voices", "delete", "n"])
    assert result.exit_code == 2
    assert "prebuilt" in one_line_error(result)


def test_voices_show_unknown_exits_2() -> None:
    result = runner.invoke(app, ["voices", "show", "nope"])
    assert result.exit_code == 2
    assert "not found" in one_line_error(result)


def test_voices_library_with_filters() -> None:
    result = runner.invoke(
        app, ["voices", "library", "--provider", "fake", "--language", "en-US", "--gender", "male"]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["fake_tenor\tTenor\tBright fake voice"]
    searched = runner.invoke(app, ["voices", "library", "--provider", "fake", "--search", "warm"])
    assert searched.stdout.splitlines() == ["fake_alto\tAlto\tWarm, low fake voice"]


# --- M3: a new provider needs only its module + one registry line --------------------------

ACME_KEY = "acme-SECRET-key-424242"


@pytest.fixture
def acme_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a provider the way a new module would: a class + one `PROVIDERS` line.

    No `Settings` field, no `API_KEY_ENV`/`DEFAULT_VOICES` entries, no `cli._scrub` edit.
    """
    import sys
    import types
    from typing import ClassVar

    from voice_playground.providers import registry
    from voice_playground.providers.base import BaseProvider, Capability

    class AcmeProvider(BaseProvider):
        name: ClassVar[str] = "acme"
        capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.TTS})
        default_tts_model: ClassVar[str | None] = "acme-tts-1"
        api_key_env: ClassVar[str | None] = "ACME_API_KEY"
        default_voice: ClassVar[str | None] = "acme-voice"

        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self._key = self._require_api_key()

        def tts(self, req: TTSRequest) -> AudioResult:
            # A careless provider that echoes its key in an error: the CLI must mask it.
            voice = req.voice.provider_voice
            raise ProviderError(f"acme rejected key {self._key} for voice {voice}")

    module = types.ModuleType("vp_test_acme")
    module.__dict__["AcmeProvider"] = AcmeProvider
    monkeypatch.setitem(sys.modules, "vp_test_acme", module)
    monkeypatch.setitem(registry.PROVIDERS, "acme", "vp_test_acme:AcmeProvider")


@pytest.mark.usefixtures("acme_provider")
def test_new_provider_key_is_masked_in_cli_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACME_API_KEY", ACME_KEY)
    result = runner.invoke(app, ["tts", "--provider", "acme", "--text", "hi"])
    assert result.exit_code == 1
    line = one_line_error(result)
    assert ACME_KEY not in result.output
    assert "acme rejected key *** for voice acme-voice" in line  # default_voice was used


@pytest.mark.usefixtures("acme_provider")
def test_new_provider_key_read_from_dotenv_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / "acme.env"
    env_file.write_text(f"ACME_API_KEY={ACME_KEY}\n")
    monkeypatch.setenv("VP_ENV_FILE", str(env_file))
    result = runner.invoke(app, ["tts", "--provider", "acme", "--text", "hi"])
    assert result.exit_code == 1
    assert ACME_KEY not in result.output
    assert "acme rejected key ***" in one_line_error(result)


@pytest.mark.usefixtures("acme_provider")
def test_new_provider_missing_key_and_providers_status(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["tts", "--provider", "acme", "--text", "hi"])
    assert result.exit_code == 2
    assert "ACME_API_KEY is not set" in one_line_error(result)
    assert "acme: capabilities=[tts]" in runner.invoke(app, ["providers"]).output
    assert "ACME_API_KEY no" in runner.invoke(app, ["providers"]).output
    monkeypatch.setenv("ACME_API_KEY", ACME_KEY)
    out = runner.invoke(app, ["providers"]).output
    assert "ACME_API_KEY yes" in out
    assert ACME_KEY not in out


def test_scrub_skips_providers_that_fail_to_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_playground.providers import registry

    monkeypatch.setitem(registry.PROVIDERS, "broken", "vp_no_such_module:Nope")
    dummy = "dummy-OPENAI-key-555"
    monkeypatch.setenv("OPENAI_API_KEY", dummy)

    def boom(**_: object) -> None:
        raise ProviderError(f"failed with {dummy}")

    monkeypatch.setattr("voice_playground.service.run_tts", boom)
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi"])
    assert result.exit_code == 1
    assert dummy not in result.output
    assert "failed with ***" in one_line_error(result)
