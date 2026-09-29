"""Unit tests for the OpenAI provider. The SDK client is always mocked; no network."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import httpx2
import openai as openai_sdk
import pytest
from pydantic import SecretStr

from voice_playground.errors import ConfigError, ProviderError, UnsupportedCapability
from voice_playground.providers import openai as mod
from voice_playground.providers.base import Capability, ResolvedVoice, STTRequest, TTSRequest
from voice_playground.providers.openai import OpenAIProvider
from voice_playground.settings import Settings
from voice_playground.voices import ClonedVoiceConfig, DesignedVoiceConfig

FAKE_KEY = "sk-test-DO-NOT-LEAK-1234567890"
PCM = b"\x01\x00\x02\x00" * 240  # 480 frames of s16le


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(name="OpenAI()")
    fake.audio.speech.create.return_value = MagicMock(content=PCM)
    fake.audio.transcriptions.create.return_value = MagicMock(
        text=" hello world ", languages=None, segments=None, spec=["text", "languages", "segments"]
    )
    factory = MagicMock(return_value=fake)
    monkeypatch.setattr(mod.openai, "OpenAI", factory)
    fake.factory = factory
    return fake


@pytest.fixture
def provider(settings: Settings, client: MagicMock) -> OpenAIProvider:
    return OpenAIProvider(settings.model_copy(update={"openai_api_key": SecretStr(FAKE_KEY)}))


def tts_req(
    *,
    model: str | None = None,
    fmt: Literal["wav", "mp3", "pcm"] = "wav",
    style: str | None = None,
    voice: str = "coral",
    options: dict[str, Any] | None = None,
    sample_rate: int | None = None,
    text: str = "Hello there",
) -> TTSRequest:
    return TTSRequest(
        text=text,
        model=model,
        voice=ResolvedVoice(provider_voice=voice, style=style, options=options or {}),
        output_format=fmt,
        sample_rate=sample_rate,
    )


def speech_kwargs(client: MagicMock) -> dict[str, Any]:
    client.audio.speech.create.assert_called_once()
    return dict(client.audio.speech.create.call_args.kwargs)


# --------------------------------------------------------------------- module / construction


def test_import_openai_resolves_to_sdk_not_this_module() -> None:
    assert mod.openai is openai_sdk
    assert mod.openai.__name__ == "openai"
    assert hasattr(mod.openai, "OpenAI")
    assert mod.__name__ == "voice_playground.providers.openai"


def test_missing_key_raises_config_error(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OpenAIProvider(settings)


def test_client_built_with_key(provider: OpenAIProvider, client: MagicMock) -> None:
    client.factory.assert_called_once_with(api_key=FAKE_KEY)


def test_class_metadata() -> None:
    assert OpenAIProvider.name == "openai"
    assert OpenAIProvider.default_tts_model == "gpt-4o-mini-tts"
    assert OpenAIProvider.default_stt_model == "gpt-transcribe"
    assert Capability.TTS in OpenAIProvider.capabilities
    assert Capability.STT in OpenAIProvider.capabilities
    assert Capability.VOICE_LIBRARY in OpenAIProvider.capabilities
    assert Capability.VOICE_DESIGN not in OpenAIProvider.capabilities
    assert Capability.VOICE_CLONE not in OpenAIProvider.capabilities


def test_registry_loads_openai(settings: Settings, client: MagicMock) -> None:
    from voice_playground.providers.registry import get_provider

    p = get_provider("openai", settings.model_copy(update={"openai_api_key": SecretStr(FAKE_KEY)}))
    assert isinstance(p, OpenAIProvider)


# ----------------------------------------------------------------------------------- TTS


def test_tts_wav_requests_pcm_and_wraps_header(provider: OpenAIProvider, client: MagicMock) -> None:
    result = provider.tts(tts_req(fmt="wav"))
    assert speech_kwargs(client) == {
        "model": "gpt-4o-mini-tts",
        "voice": "coral",
        "input": "Hello there",
        "response_format": "pcm",
    }
    assert result.mime_type == "audio/wav"
    assert result.sample_rate == 24_000
    with wave.open(io.BytesIO(result.data)) as w:
        assert w.getframerate() == 24_000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == len(PCM) // 2
        assert w.readframes(w.getnframes()) == PCM


def test_tts_pcm(provider: OpenAIProvider, client: MagicMock) -> None:
    result = provider.tts(tts_req(fmt="pcm"))
    assert speech_kwargs(client)["response_format"] == "pcm"
    assert result.data == PCM
    assert result.mime_type == "audio/l16"
    assert result.sample_rate == 24_000


def test_tts_mp3(provider: OpenAIProvider, client: MagicMock) -> None:
    client.audio.speech.create.return_value = MagicMock(content=b"ID3mp3data")
    result = provider.tts(tts_req(fmt="mp3", model="tts-1-hd"))
    assert speech_kwargs(client) == {
        "model": "tts-1-hd",
        "voice": "coral",
        "input": "Hello there",
        "response_format": "mp3",
    }
    assert result.data == b"ID3mp3data"
    assert result.mime_type == "audio/mpeg"
    assert result.sample_rate == 24_000


@pytest.mark.parametrize("model", ["gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15"])
def test_tts_style_becomes_instructions_on_gpt_models(
    provider: OpenAIProvider, client: MagicMock, model: str, capsys: pytest.CaptureFixture[str]
) -> None:
    provider.tts(tts_req(model=model, style="whispered, urgent"))
    kwargs = speech_kwargs(client)
    assert kwargs["instructions"] == "whispered, urgent"
    assert kwargs["model"] == model
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("model", ["tts-1", "tts-1-hd"])
def test_tts_style_ignored_with_single_warning_on_tts1(
    provider: OpenAIProvider, client: MagicMock, model: str, capsys: pytest.CaptureFixture[str]
) -> None:
    provider.tts(tts_req(model=model, style="calm"))
    provider.tts(tts_req(model=model, style="calm"))
    for call in client.audio.speech.create.call_args_list:
        assert "instructions" not in call.kwargs
    err = capsys.readouterr().err
    assert err.count("warning:") == 1
    assert model in err and "style" in err


def test_tts_no_style_no_instructions(provider: OpenAIProvider, client: MagicMock) -> None:
    provider.tts(tts_req())
    assert "instructions" not in speech_kwargs(client)


def test_tts_mismatched_sample_rate_does_not_fail(
    provider: OpenAIProvider, client: MagicMock
) -> None:
    result = provider.tts(tts_req(sample_rate=16_000))
    assert result.sample_rate == 24_000
    assert "sample_rate" not in speech_kwargs(client)


def test_tts_speed_option(provider: OpenAIProvider, client: MagicMock) -> None:
    provider.tts(tts_req(options={"speed": 1.25}))
    assert speech_kwargs(client)["speed"] == 1.25


def test_tts_unknown_option_is_config_error(provider: OpenAIProvider, client: MagicMock) -> None:
    with pytest.raises(ConfigError, match="stability"):
        provider.tts(tts_req(options={"stability": 0.5}))
    client.audio.speech.create.assert_not_called()


def test_tts_custom_voice_id(provider: OpenAIProvider, client: MagicMock) -> None:
    provider.tts(tts_req(voice="voice_1234"))
    assert speech_kwargs(client)["voice"] == {"id": "voice_1234"}


def test_tts_text_too_long(provider: OpenAIProvider, client: MagicMock) -> None:
    with pytest.raises(ConfigError, match="4096"):
        provider.tts(tts_req(text="a" * 4097))
    client.audio.speech.create.assert_not_called()


# ----------------------------------------------------------------------------------- STT


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFFfake")
    return path


def test_stt_default_model_uses_languages_and_prompt(
    provider: OpenAIProvider, client: MagicMock, audio_file: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> Any:
        fh = kwargs.pop("file")
        seen["mode"] = fh.mode
        seen["content"] = fh.read()
        seen["kwargs"] = kwargs
        return MagicMock(text=" hello world ", languages=None, segments=None)

    client.audio.transcriptions.create.side_effect = fake_create
    result = provider.stt(
        STTRequest(audio_path=audio_file, model=None, language="en-US", prompt="domain words")
    )
    assert seen["mode"] == "rb"
    assert seen["content"] == b"RIFFfake"
    assert seen["kwargs"] == {
        "model": "gpt-transcribe",
        "languages": ["en"],
        "prompt": "domain words",
    }
    assert result.text == "hello world"
    assert result.language == "en-US"


@pytest.mark.parametrize("model", ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"])
def test_stt_single_language_models(
    provider: OpenAIProvider, client: MagicMock, audio_file: Path, model: str
) -> None:
    provider.stt(STTRequest(audio_path=audio_file, model=model, language="de", prompt="p"))
    kwargs = dict(client.audio.transcriptions.create.call_args.kwargs)
    kwargs.pop("file")
    assert kwargs == {"model": model, "language": "de", "prompt": "p"}


def test_stt_minimal_kwargs(provider: OpenAIProvider, client: MagicMock, audio_file: Path) -> None:
    provider.stt(STTRequest(audio_path=audio_file, model="whisper-1"))
    kwargs = dict(client.audio.transcriptions.create.call_args.kwargs)
    assert kwargs.pop("file").mode == "rb"
    assert kwargs == {"model": "whisper-1"}


def test_stt_diarize_drops_prompt(
    provider: OpenAIProvider,
    client: MagicMock,
    audio_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider.stt(
        STTRequest(audio_path=audio_file, model="gpt-4o-transcribe-diarize", prompt="names")
    )
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert "prompt" not in kwargs
    assert kwargs["chunking_strategy"] == "auto"
    assert "prompt" in capsys.readouterr().err


def test_stt_detected_language(
    provider: OpenAIProvider, client: MagicMock, audio_file: Path
) -> None:
    client.audio.transcriptions.create.return_value = MagicMock(
        text="bonjour", languages=[MagicMock(code="fr")], segments=None
    )
    result = provider.stt(STTRequest(audio_path=audio_file, model=None))
    assert result.language == "fr"


def test_stt_string_response(provider: OpenAIProvider, client: MagicMock, audio_file: Path) -> None:
    client.audio.transcriptions.create.return_value = "plain text\n"
    assert provider.stt(STTRequest(audio_path=audio_file, model=None)).text == "plain text"


def test_stt_missing_file_is_config_error(
    provider: OpenAIProvider, client: MagicMock, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="missing.wav"):
        provider.stt(STTRequest(audio_path=tmp_path / "missing.wav", model=None))
    client.audio.transcriptions.create.assert_not_called()


# ---------------------------------------------------------------------------- errors


def _status_error(cls: type[openai_sdk.APIStatusError], status: int) -> Exception:
    request = httpx2.Request("POST", "https://api.openai.com/v1/audio/speech")
    response = httpx2.Response(status, request=request)
    message = f"Incorrect API key provided: {FAKE_KEY}. You can find your API key at ..."
    body = {"message": message, "type": "invalid_request_error", "code": "invalid_api_key"}
    return cls(message, response=response, body=body)


def test_tts_auth_error_maps_to_provider_error_without_key(
    provider: OpenAIProvider, client: MagicMock
) -> None:
    client.audio.speech.create.side_effect = _status_error(openai_sdk.AuthenticationError, 401)
    with pytest.raises(ProviderError) as info:
        provider.tts(tts_req())
    msg = str(info.value)
    assert "401" in msg
    assert "openai tts failed" in msg
    assert FAKE_KEY not in msg
    assert "DO-NOT-LEAK" not in msg
    assert "\n" not in msg
    assert info.value.exit_code == 1


def test_stt_rate_limit_error(
    provider: OpenAIProvider, client: MagicMock, audio_file: Path
) -> None:
    client.audio.transcriptions.create.side_effect = _status_error(openai_sdk.RateLimitError, 429)
    with pytest.raises(ProviderError, match="openai stt failed: HTTP 429") as info:
        provider.stt(STTRequest(audio_path=audio_file, model=None))
    assert FAKE_KEY not in str(info.value)


def test_connection_error(provider: OpenAIProvider, client: MagicMock) -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/audio/speech")
    client.audio.speech.create.side_effect = openai_sdk.APIConnectionError(request=request)
    with pytest.raises(ProviderError, match="could not connect"):
        provider.tts(tts_req())


# ------------------------------------------------------------------- voices / library


def test_create_voice_unsupported(provider: OpenAIProvider) -> None:
    with pytest.raises(UnsupportedCapability, match="voice_design"):
        provider.create_voice(DesignedVoiceConfig(name="x", description="warm"))
    with pytest.raises(UnsupportedCapability, match="voice_clone"):
        provider.create_voice(ClonedVoiceConfig(name="x", reference_audio=Path("a.wav")))
    with pytest.raises(UnsupportedCapability):
        provider.delete_voice("voice_1")


def test_list_library_static(provider: OpenAIProvider, client: MagicMock) -> None:
    names = [v.name for v in provider.list_library()]
    assert names == [
        "alloy", "ash", "ballad", "coral", "echo", "fable", "nova",
        "onyx", "sage", "shimmer", "verse", "marin", "cedar",
    ]  # fmt: skip
    assert client.audio.method_calls == []  # no network


def test_list_library_filters(provider: OpenAIProvider) -> None:
    assert [v.id for v in provider.list_library(search="MAR")] == ["marin"]
    tts1 = {v.id for v in provider.list_library(model="tts-1")}
    assert "marin" not in tts1 and "coral" in tts1 and len(tts1) == 9
    assert len(provider.list_library(model="gpt-4o-mini-tts-2025-12-15")) == 13
    assert len(provider.list_library(language="en-US", gender=None)) == 13


def test_list_library_unsupported_filter(provider: OpenAIProvider) -> None:
    with pytest.raises(ConfigError, match="gender"):
        provider.list_library(gender="female")
