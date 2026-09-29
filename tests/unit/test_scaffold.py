"""T1 scaffold tests: CLI surface, registry, fake provider, settings, gitignore."""

from __future__ import annotations

import io
import re
import subprocess
import wave
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import SecretStr, TypeAdapter, ValidationError
from typer.testing import CliRunner

from voice_playground.cli import app
from voice_playground.errors import ConfigError, ProviderError, UnsupportedCapability, VPError
from voice_playground.providers.base import (
    BaseProvider,
    Capability,
    Provider,
    ResolvedVoice,
    STTRequest,
    TTSRequest,
)
from voice_playground.providers.fake import FakeProvider
from voice_playground.providers.registry import (
    PROVIDERS,
    get_provider,
    get_provider_class,
    provider_names,
)
from voice_playground.settings import Settings, get_settings
from voice_playground.voices import (
    ClonedVoiceConfig,
    DesignedVoiceConfig,
    PrebuiltVoiceConfig,
    VoiceConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _help(*args: str) -> str:
    result = runner.invoke(app, [*args, "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"})
    assert result.exit_code == 0, result.output
    return ANSI.sub("", result.output)


# --- CLI surface -------------------------------------------------------------------------


def test_root_help_lists_commands() -> None:
    out = _help()
    for cmd in ("tts", "stt", "voices", "providers", "play"):
        assert cmd in out


@pytest.mark.parametrize(
    ("args", "flags"),
    [
        (
            ("tts",),
            [
                "--provider",
                "--model",
                "--voice",
                "--text",
                "--text-file",
                "--output",
                "-o",
                "--style",
                "--format",
                "--play",
                "--speaker",
                "--no-auto-create",
            ],
        ),
        (
            ("stt",),
            ["--provider", "--model", "--input", "--output", "-o", "--language", "--prompt"],
        ),
        (("voices",), ["list", "show", "create", "library", "delete"]),
        (("voices", "list"), ["--provider"]),
        (("voices", "show"), ["name"]),
        (("voices", "create"), ["name", "--force"]),
        (("voices", "library"), ["--provider", "--search", "--language", "--gender"]),
        (("voices", "delete"), ["name"]),
        (("play",), ["path", "--play", "--speaker"]),
        (("providers",), []),
    ],
)
def test_help_lists_every_flag(args: tuple[str, ...], flags: list[str]) -> None:
    out = _help(*args)
    for flag in flags:
        assert flag in out, f"{flag} missing from `vp {' '.join(args)} --help`"


def test_help_lists_enum_choices() -> None:
    tts_help = _help("tts")
    for choice in ("wav", "mp3", "pcm", "local", "sonos"):
        assert choice in tts_help


def test_unexpected_error_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    # T8 implemented the service; an unexpected exception still exits 1 with one line.
    def boom(**_: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("voice_playground.service.run_tts", boom)
    result = runner.invoke(app, ["tts", "--provider", "fake", "--text", "hi"])
    assert result.exit_code == 1
    assert result.output.strip().startswith("error: unexpected RuntimeError: kaboom")
    assert len(result.output.strip().splitlines()) == 1


def test_providers_command_shows_key_status_never_key(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = "sk-dummy-SECRET-value-123"
    monkeypatch.setenv("OPENAI_API_KEY", dummy)
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0, result.output
    assert dummy not in result.output
    for name in PROVIDERS:
        assert f"{name}:" in result.output
    assert "OPENAI_API_KEY yes" in result.output
    assert "GOOGLE_API_KEY no" in result.output


# --- registry ----------------------------------------------------------------------------


def test_registry_has_all_providers() -> None:
    assert set(provider_names()) == {"google", "openai", "elevenlabs", "fake"}


def test_get_provider_fake(settings: Settings) -> None:
    provider = get_provider("fake", settings)
    assert isinstance(provider, FakeProvider)
    assert isinstance(provider, Provider)
    assert provider.name == "fake"


def test_get_provider_unknown_lists_valid_names(settings: Settings) -> None:
    with pytest.raises(ConfigError) as excinfo:
        get_provider("nope", settings)
    message = str(excinfo.value)
    assert "nope" in message
    for name in PROVIDERS:
        assert name in message
    assert excinfo.value.exit_code == 2


@pytest.mark.parametrize(
    ("name", "env_var"),
    [
        ("google", "GOOGLE_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("elevenlabs", "ELEVENLABS_API_KEY"),
    ],
)
def test_real_provider_missing_key_raises_config_error(
    settings: Settings, name: str, env_var: str
) -> None:
    with pytest.raises(ConfigError, match=env_var):
        get_provider(name, settings)


@pytest.mark.parametrize("name", ["google", "openai", "elevenlabs"])
def test_real_provider_classes_declare_contract(name: str) -> None:
    cls = get_provider_class(name)
    assert cls.name == name
    assert {Capability.TTS, Capability.STT} <= cls.capabilities
    assert cls.default_tts_model


def test_real_provider_constructs_with_key(settings: Settings) -> None:
    keyed = settings.model_copy(update={"google_api_key": SecretStr("dummy")})
    assert get_provider("google", keyed).name == "google"


# --- errors ------------------------------------------------------------------------------


def test_error_exit_codes() -> None:
    assert VPError().exit_code == 1
    assert ConfigError().exit_code == 2
    assert ProviderError().exit_code == 1
    assert UnsupportedCapability().exit_code == 2


def test_base_provider_defaults_raise_unsupported(settings: Settings) -> None:
    class Bare(BaseProvider):
        name = "bare"

    with pytest.raises(UnsupportedCapability, match="bare"):
        Bare(settings).tts(TTSRequest(text="x", model=None, voice=ResolvedVoice("v")))


# --- fake provider -----------------------------------------------------------------------


def test_fake_tts_returns_half_second_24khz_wav(settings: Settings) -> None:
    result = FakeProvider(settings).tts(
        TTSRequest(text="hi", model=None, voice=ResolvedVoice(provider_voice="x"))
    )
    assert result.mime_type == "audio/wav"
    assert result.sample_rate == 24_000
    with wave.open(io.BytesIO(result.data)) as w:
        assert w.getframerate() == 24_000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 12_000


def test_fake_tts_is_deterministic_and_supports_pcm(settings: Settings) -> None:
    fake = FakeProvider(settings)
    req = TTSRequest(text="hi", model=None, voice=ResolvedVoice("x"), output_format="pcm")
    first, second = fake.tts(req), fake.tts(req)
    assert first == second
    assert first.mime_type == "audio/l16"
    assert len(first.data) == 24_000  # 12000 frames * 2 bytes


def test_fake_stt_echoes_file_name(settings: Settings, tmp_path: Path) -> None:
    transcript = FakeProvider(settings).stt(
        STTRequest(audio_path=tmp_path / "clip.wav", model=None, language="en")
    )
    assert transcript.text == "clip.wav"
    assert transcript.language == "en"


def test_fake_create_voice_and_library(settings: Settings) -> None:
    fake = FakeProvider(settings)
    cfg = DesignedVoiceConfig(name="storyteller", provider="fake", description="warm")
    created = fake.create_voice(cfg)
    assert created.remote_id == "fake_voice_storyteller"
    assert created.expires_at is None
    fake.delete_voice(created.remote_id)
    assert len(fake.list_library()) == 3
    assert [v.name for v in fake.list_library(search="deep")] == ["Baritone"]
    assert [v.id for v in fake.list_library(language="en-US", gender="male")] == ["fake_tenor"]


def test_fake_has_every_capability() -> None:
    assert frozenset(Capability) == FakeProvider.capabilities


# --- voice config shapes -----------------------------------------------------------------


def test_voice_config_discriminated_union() -> None:
    adapter: TypeAdapter[VoiceConfig] = TypeAdapter(VoiceConfig)
    assert isinstance(
        adapter.validate_python({"name": "n", "type": "prebuilt", "voice": "Kore"}),
        PrebuiltVoiceConfig,
    )
    assert isinstance(
        adapter.validate_python({"name": "n", "type": "cloned", "reference_audio": "a.wav"}),
        ClonedVoiceConfig,
    )
    with pytest.raises(ValidationError):
        adapter.validate_python({"name": "n", "type": "designed"})  # missing description


# --- settings ----------------------------------------------------------------------------


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setenv("HA_URL", "http://ha:8123")
    monkeypatch.setenv("VP_SERVE_PORT", "8765")
    s = get_settings()
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "g-key"
    assert "g-key" not in repr(s)
    assert s.ha_url == "http://ha:8123"
    assert s.serve_port == 8765
    assert s.voices_dir.name == "voices"
    assert s.cache_dir.name == ".vp_cache"
    assert s.openai_api_key is None


def test_settings_read_dotenv_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text("ELEVENLABS_API_KEY=el-key\nVP_SERVE_HOST=\nHA_TOKEN=\n")
    monkeypatch.setenv("VP_ENV_FILE", str(env_file))
    s = get_settings()
    assert s.elevenlabs_api_key is not None
    assert s.elevenlabs_api_key.get_secret_value() == "el-key"
    assert s.serve_host is None
    assert s.ha_token is None


def test_env_example_lists_every_setting() -> None:
    text = (REPO_ROOT / ".env.example").read_text()
    for var in (
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "HA_URL",
        "HA_TOKEN",
        "HA_SONOS_ENTITY",
        "VP_SERVE_HOST",
        "VP_SERVE_PORT",
        "VP_VOICES_DIR",
        "VP_CACHE_DIR",
    ):
        assert re.search(rf"^{var}=", text, re.MULTILINE), var


def test_env_example_has_no_inline_comment_values() -> None:
    # python-dotenv parses `KEY=   # note` as the value "# note", so a copied .env.example
    # would silently set e.g. HA_TOKEN to a comment. Comments must be on their own lines.
    values = dotenv_values(REPO_ROOT / ".env.example")
    assert all(not (v or "").startswith("#") for v in values.values()), values


# --- gitignore ---------------------------------------------------------------------------


def _ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path], cwd=REPO_ROOT, check=False
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".vp_cache/voices.json",
        "voices/audio/me.wav",
        "out/a.wav",
        "x.wav",
        "x.mp3",
        ".venv/bin/python",
    ],
)
def test_gitignore_ignores(path: str) -> None:
    assert _ignored(path)


@pytest.mark.parametrize(
    "path", [".env.example", "voices/narrator.yaml", "src/voice_playground/cli.py"]
)
def test_gitignore_keeps(path: str) -> None:
    assert not _ignored(path)
