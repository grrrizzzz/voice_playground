"""Unit tests for the Google Gemini provider. `genai.Client` is fully mocked."""

from __future__ import annotations

import base64
import io
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic import SecretStr

from voice_playground.errors import ConfigError, ProviderError
from voice_playground.providers import google as google_mod
from voice_playground.providers.base import ResolvedVoice, STTRequest, TTSRequest
from voice_playground.providers.google import GoogleProvider, pcm_to_wav
from voice_playground.providers.registry import get_provider_class
from voice_playground.settings import Settings
from voice_playground.voices import ClonedVoiceConfig, DesignedVoiceConfig, PrebuiltVoiceConfig

FAKE_KEY = "test-google-key-123"
PCM = b"\x01\x00" * 2400  # 0.1 s at 24 kHz


def _wav(rate: int = 24_000, pcm: bytes = PCM) -> bytes:
    return pcm_to_wav(pcm, rate)


def _interaction(data: bytes, mime: str = "audio/wav") -> SimpleNamespace:
    audio = SimpleNamespace(data=base64.b64encode(data).decode(), mime_type=mime)
    return SimpleNamespace(output_audio=audio, output_text=None, steps=[])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(name="genai.Client()")
    factory = MagicMock(return_value=mock)
    monkeypatch.setattr(google_mod.genai, "Client", factory)
    mock.factory = factory
    mock.interactions.create.return_value = _interaction(_wav())
    return mock


@pytest.fixture
def provider(settings: Settings, client: MagicMock) -> GoogleProvider:
    return GoogleProvider(settings.model_copy(update={"google_api_key": SecretStr(FAKE_KEY)}))


def _tts(provider: GoogleProvider, voice: ResolvedVoice, **kwargs: Any) -> Any:
    return provider.tts(TTSRequest(text=kwargs.pop("text", "Hello there"), voice=voice, **kwargs))


# ------------------------------------------------------------------ construction
def test_registry_and_metadata() -> None:
    assert get_provider_class("google") is GoogleProvider
    assert GoogleProvider.default_tts_model == "gemini-3.8-flash-tts"
    assert GoogleProvider.default_stt_model == "gemini-3.8-flash"


def test_missing_key_is_config_error(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        GoogleProvider(settings)


def test_client_built_lazily_with_key(provider: GoogleProvider, client: MagicMock) -> None:
    client.factory.assert_not_called()
    _tts(provider, ResolvedVoice("Kore"), model=None)
    client.factory.assert_called_once_with(api_key=FAKE_KEY)


# ---------------------------------------------------------------------------- tts
def test_tts_prebuilt_with_style_payload(provider: GoogleProvider, client: MagicMock) -> None:
    voice = ResolvedVoice("Kore", style="whispered, urgent", language="en-US")
    result = _tts(provider, voice, model=None, text="Wait <sigh> what?")

    client.interactions.create.assert_called_once_with(
        model="gemini-3.8-flash-tts",
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": "Wait <sigh> what?",
                        "annotations": [{"type": "speech_metadata", "style": "whispered, urgent"}],
                    }
                ],
            }
        ],
        response_format={"type": "audio", "mime_type": "audio/wav", "sample_rate": 24000},
        generation_config={"speech_config": [{"voice": "Kore", "language": "en-US"}]},
    )
    assert result.mime_type == "audio/wav"
    assert result.sample_rate == 24000
    assert result.data == _wav()


def test_tts_without_style_has_no_annotations(provider: GoogleProvider, client: MagicMock) -> None:
    _tts(provider, ResolvedVoice("Puck"), model="gemini-3.8-flash-lite-tts")
    kwargs = client.interactions.create.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-flash-lite-tts"
    assert kwargs["input"][0]["content"][0] == {"type": "text", "text": "Hello there"}
    assert kwargs["generation_config"] == {"speech_config": [{"voice": "Puck"}]}


@pytest.mark.parametrize("voice_id", ["voice_abc123", "voicekey_xyz789"])
def test_tts_custom_voice_ids(provider: GoogleProvider, client: MagicMock, voice_id: str) -> None:
    voice = ResolvedVoice(voice_id, style="calm", config_name="narrator")
    _tts(provider, voice, model=None)
    kwargs = client.interactions.create.call_args.kwargs
    assert kwargs["generation_config"] == {"speech_config": [{"voice": voice_id}]}
    assert kwargs["model"] == "gemini-3.8-flash-tts"
    assert kwargs["input"][0]["content"][0]["annotations"] == [
        {"type": "speech_metadata", "style": "calm"}
    ]


def test_tts_custom_sample_rate(provider: GoogleProvider, client: MagicMock) -> None:
    client.interactions.create.return_value = _interaction(_wav(16_000))
    result = _tts(provider, ResolvedVoice("Kore"), model=None, sample_rate=16_000)
    kwargs = client.interactions.create.call_args.kwargs
    assert kwargs["response_format"] == {
        "type": "audio",
        "mime_type": "audio/wav",
        "sample_rate": 16000,
    }
    assert result.sample_rate == 16_000


def test_tts_sample_rate_from_voice(provider: GoogleProvider, client: MagicMock) -> None:
    client.interactions.create.return_value = _interaction(_wav(8_000))
    result = _tts(provider, ResolvedVoice("Kore", sample_rate=8_000), model=None)
    assert client.interactions.create.call_args.kwargs["response_format"]["sample_rate"] == 8000
    assert result.sample_rate == 8_000


def test_tts_rejects_unsupported_sample_rate(provider: GoogleProvider, client: MagicMock) -> None:
    with pytest.raises(ConfigError, match="sample_rate"):
        _tts(provider, ResolvedVoice("Kore"), model=None, sample_rate=44_100)
    client.interactions.create.assert_not_called()


def test_tts_pcm_requests_l16(provider: GoogleProvider, client: MagicMock) -> None:
    client.interactions.create.return_value = _interaction(PCM, "audio/l16")
    result = _tts(provider, ResolvedVoice("Kore"), model=None, output_format="pcm")
    assert (
        client.interactions.create.call_args.kwargs["response_format"]["mime_type"] == "audio/l16"
    )
    assert result.mime_type == "audio/l16"
    assert result.data == PCM
    assert result.sample_rate == 24_000


def test_tts_pcm_strips_wav_header_if_returned(provider: GoogleProvider) -> None:
    result = _tts(provider, ResolvedVoice("Kore"), model=None, output_format="pcm")
    assert result.mime_type == "audio/l16"
    assert result.data == PCM


def test_tts_mp3_returns_wav_for_caller_to_convert(
    provider: GoogleProvider, client: MagicMock
) -> None:
    result = _tts(provider, ResolvedVoice("Kore"), model=None, output_format="mp3")
    assert (
        client.interactions.create.call_args.kwargs["response_format"]["mime_type"] == "audio/wav"
    )
    assert result.mime_type == "audio/wav"


def test_tts_preview_model_path(provider: GoogleProvider, client: MagicMock) -> None:
    client.interactions.create.return_value = _interaction(PCM, "audio/l16")
    voice = ResolvedVoice("Kore", style="cheerful")
    result = _tts(provider, voice, model="gemini-3.1-flash-tts-preview", text="Hi!")

    client.interactions.create.assert_called_once_with(
        model="gemini-3.1-flash-tts-preview",
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi!",
                        "annotations": [{"type": "speech_metadata", "style": "cheerful"}],
                    }
                ],
            }
        ],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": "Kore"}]},
    )
    # Headerless 24 kHz s16le PCM is wrapped into a WAV container locally.
    assert result.mime_type == "audio/wav"
    with wave.open(io.BytesIO(result.data)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24_000, 1, 2)
        assert w.readframes(w.getnframes()) == PCM


def test_tts_preview_rejects_custom_voice(provider: GoogleProvider, client: MagicMock) -> None:
    with pytest.raises(ConfigError, match="custom voices"):
        _tts(provider, ResolvedVoice("voice_abc"), model="gemini-3.1-flash-tts-preview")
    client.interactions.create.assert_not_called()


def test_tts_preview_rejects_other_sample_rate(provider: GoogleProvider) -> None:
    with pytest.raises(ConfigError, match="24000"):
        _tts(
            provider, ResolvedVoice("Kore"), model="gemini-3.1-flash-tts-preview", sample_rate=8000
        )


def test_tts_falls_back_to_steps(provider: GoogleProvider, client: MagicMock) -> None:
    block = SimpleNamespace(type="audio", data=base64.b64encode(_wav()).decode())
    client.interactions.create.return_value = SimpleNamespace(
        output_audio=None, steps=[SimpleNamespace(content=[block])]
    )
    assert _tts(provider, ResolvedVoice("Kore"), model=None).data == _wav()


def test_tts_no_audio_is_provider_error(provider: GoogleProvider, client: MagicMock) -> None:
    client.interactions.create.return_value = SimpleNamespace(output_audio=None, steps=[])
    with pytest.raises(ProviderError, match="no audio"):
        _tts(provider, ResolvedVoice("Kore"), model=None)


# ------------------------------------------------------------------------- errors
def _api_error(code: int = 403) -> genai_errors.APIError:
    body = {"error": {"code": code, "message": f"API key {FAKE_KEY} not valid", "status": "DENIED"}}
    return genai_errors.APIError(code, body)


@pytest.mark.parametrize(
    "exc",
    [
        _api_error(),
        httpx.ConnectError(f"connect failed for key={FAKE_KEY}"),
    ],
)
def test_sdk_error_becomes_provider_error_without_key(
    provider: GoogleProvider, client: MagicMock, exc: Exception
) -> None:
    client.interactions.create.side_effect = exc
    with pytest.raises(ProviderError) as info:
        _tts(provider, ResolvedVoice("Kore"), model=None)
    message = str(info.value)
    assert FAKE_KEY not in message
    assert message.startswith("google tts failed")
    assert "\n" not in message


def test_api_error_message_has_status_and_reason(
    provider: GoogleProvider, client: MagicMock
) -> None:
    client.voices.delete.side_effect = _api_error(404)
    with pytest.raises(ProviderError) as info:
        provider.delete_voice("voice_gone")
    assert "404" in str(info.value)
    assert "DENIED" in str(info.value)
    assert FAKE_KEY not in str(info.value)


def test_nextgen_sdk_error_is_mapped(provider: GoogleProvider, client: MagicMock) -> None:
    from google.genai._gaos.lib.compat_errors import APIError as NextGenAPIError

    request = httpx.Request("POST", "https://example.invalid/v1beta/voices")
    client.voices.create.side_effect = NextGenAPIError(
        f"bad request {FAKE_KEY}", request, body={"error": {"code": 400, "status": "INVALID"}}
    )
    with pytest.raises(ProviderError) as info:
        provider.create_voice(DesignedVoiceConfig(name="n", description="warm"))
    assert "400 INVALID" in str(info.value)
    assert FAKE_KEY not in str(info.value)


# ---------------------------------------------------------------------------- stt
def test_stt_inline_payload(provider: GoogleProvider, client: MagicMock, tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(_wav())
    client.interactions.create.return_value = SimpleNamespace(output_text="  Hello there.\n")

    transcript = provider.stt(
        STTRequest(audio_path=audio, model=None, language="en", prompt="Gemini, Sonos")
    )

    kwargs = client.interactions.create.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-flash"
    text_block, audio_block = kwargs["input"]
    assert text_block["type"] == "text"
    assert "transcript" in text_block["text"]
    assert "'en'" in text_block["text"]
    assert "Gemini, Sonos" in text_block["text"]
    assert audio_block == {
        "type": "audio",
        "mime_type": "audio/wav",
        "data": base64.b64encode(_wav()).decode(),
    }
    assert transcript.text == "Hello there."
    assert transcript.language == "en"


def test_stt_large_file_is_uploaded(
    provider: GoogleProvider, client: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(google_mod, "MAX_INLINE_AUDIO_BYTES", 10)
    audio = tmp_path / "big.mp3"
    audio.write_bytes(b"x" * 100)
    client.files.upload.return_value = SimpleNamespace(uri="files/abc", mime_type="audio/mpeg")
    client.interactions.create.return_value = SimpleNamespace(output_text="hi")

    provider.stt(STTRequest(audio_path=audio, model="gemini-3.8-flash"))

    client.files.upload.assert_called_once_with(file=str(audio))
    audio_block = client.interactions.create.call_args.kwargs["input"][1]
    assert audio_block == {"type": "audio", "mime_type": "audio/mpeg", "uri": "files/abc"}


def test_stt_missing_file(provider: GoogleProvider, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        provider.stt(STTRequest(audio_path=tmp_path / "nope.wav", model=None))


def test_stt_unknown_extension(provider: GoogleProvider, tmp_path: Path) -> None:
    path = tmp_path / "clip.xyz"
    path.write_bytes(b"x")
    with pytest.raises(ConfigError, match="unsupported audio type"):
        provider.stt(STTRequest(audio_path=path, model=None))


# ------------------------------------------------------------------ create_voice
def _created(**fields: Any) -> SimpleNamespace:
    base: dict[str, Any] = {"id": None, "key": None, "expire_time": None}
    base.update(fields)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(("store", "ttl_days"), [(True, 365), (False, 7)])
def test_create_designed_voice_payload(
    provider: GoogleProvider, client: MagicMock, store: bool, ttl_days: int
) -> None:
    client.voices.create.return_value = _created(id="voice_abc")
    cfg = DesignedVoiceConfig(
        name="astronomer",
        description="A warm, thoughtful astronomer",
        language="en-GB",
        store=store,
        provider_options={"gender": "male", "unrelated": 1},
    )
    before = datetime.now(UTC)
    created = provider.create_voice(cfg)
    after = datetime.now(UTC)

    client.voices.create.assert_called_once_with(
        voice={
            "type": "prompted",
            "model": "gemini-3.8-flash-tts",
            "display_name": "astronomer",
            "language_code": "en-GB",
            "gender": "male",
            "prompted": {"input": "A warm, thoughtful astronomer"},
        },
        store=store,
    )
    assert created.remote_id == "voice_abc"
    assert created.expires_at is not None
    ttl = timedelta(days=ttl_days)
    assert before + ttl <= created.expires_at <= after + ttl


@pytest.mark.parametrize(
    ("store", "response", "remote_id", "ttl_days"),
    [
        (True, {"id": "voice_rep"}, "voice_rep", 365),
        (False, {"key": "voicekey_rep"}, "voicekey_rep", 7),
    ],
)
def test_create_cloned_voice_payload(
    provider: GoogleProvider,
    client: MagicMock,
    tmp_path: Path,
    store: bool,
    response: dict[str, str],
    remote_id: str,
    ttl_days: int,
) -> None:
    source = tmp_path / "me.wav"
    source.write_bytes(b"source-audio")
    consent = tmp_path / "consent.wav"
    consent.write_bytes(b"consent-audio")
    client.voices.create.return_value = _created(**response)
    cfg = ClonedVoiceConfig(
        name="me",
        reference_audio=source,
        store=store,
        model="gemini-3.8-flash-lite-tts",
        provider_options={"consent_audio": str(consent)},
    )
    before = datetime.now(UTC)
    created = provider.create_voice(cfg)

    client.voices.create.assert_called_once_with(
        voice={
            "type": "replicated",
            "model": "gemini-3.8-flash-lite-tts",
            "display_name": "me",
            "replicated": {
                "source_audio": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(b"source-audio").decode(),
                },
                "consent_audio": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(b"consent-audio").decode(),
                },
            },
        },
        store=store,
    )
    assert created.remote_id == remote_id
    assert created.expires_at is not None
    ttl = timedelta(days=ttl_days)
    assert before + ttl <= created.expires_at <= datetime.now(UTC) + ttl


def test_create_voice_uses_server_expire_time(provider: GoogleProvider, client: MagicMock) -> None:
    expire = datetime(2027, 9, 28, tzinfo=UTC)
    client.voices.create.return_value = _created(id="voice_x", expire_time=expire)
    created = provider.create_voice(DesignedVoiceConfig(name="n", description="warm"))
    assert created.expires_at == expire


def test_create_cloned_voice_requires_consent(
    provider: GoogleProvider, client: MagicMock, tmp_path: Path
) -> None:
    source = tmp_path / "me.wav"
    source.write_bytes(b"x")
    with pytest.raises(ConfigError, match="consent_audio"):
        provider.create_voice(ClonedVoiceConfig(name="me", reference_audio=source))
    client.voices.create.assert_not_called()


def test_create_cloned_voice_missing_reference(provider: GoogleProvider, tmp_path: Path) -> None:
    consent = tmp_path / "c.wav"
    consent.write_bytes(b"x")
    cfg = ClonedVoiceConfig(
        name="me",
        reference_audio=tmp_path / "missing.wav",
        provider_options={"consent_audio": str(consent)},
    )
    with pytest.raises(ConfigError, match="not found"):
        provider.create_voice(cfg)


def test_create_prebuilt_is_config_error(provider: GoogleProvider) -> None:
    with pytest.raises(ConfigError, match="prebuilt"):
        provider.create_voice(PrebuiltVoiceConfig(name="k", voice="Kore"))


def test_create_voice_without_id_is_provider_error(
    provider: GoogleProvider, client: MagicMock
) -> None:
    client.voices.create.return_value = _created()
    with pytest.raises(ProviderError, match="no voice id"):
        provider.create_voice(DesignedVoiceConfig(name="n", description="warm"))


# ------------------------------------------------------------ delete / library
def test_delete_voice(provider: GoogleProvider, client: MagicMock) -> None:
    provider.delete_voice("voice_abc")
    client.voices.delete.assert_called_once_with(id="voice_abc")


def test_delete_stateless_key_is_local_only(provider: GoogleProvider, client: MagicMock) -> None:
    provider.delete_voice("voicekey_abc")
    client.voices.delete.assert_not_called()


def _page(voices: list[dict[str, Any]], token: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(voices=[SimpleNamespace(**v) for v in voices], next_page_token=token)


def test_list_library_filters_and_pagination(provider: GoogleProvider, client: MagicMock) -> None:
    client.voices.list.side_effect = [
        _page(
            [
                {
                    "id": "Kore",
                    "display_name": "Kore",
                    "description": "Firm",
                    "type": "prebuilt",
                    "gender": "female",
                    "language_code": "en-US",
                }
            ],
            token="t2",
        ),
        _page([{"id": "Aoede", "display_name": "Aoede", "type": "prebuilt"}]),
    ]

    voices = provider.list_library(
        language="en-US",
        gender="female",
        pitch=["medium", "low"],
        contexts="Audiobook",
        search="warm",
        type_="prebuilt",
        page_size=1,
        accent=None,
    )

    expected = {
        "language_code": ["en-US"],
        "gender": ["female"],
        "pitch": ["medium", "low"],
        "contexts": ["Audiobook"],
        "search": "warm",
        "type_": ["prebuilt"],
        "page_size": 1,
    }
    first, second = client.voices.list.call_args_list
    assert first.kwargs == expected
    assert second.kwargs == {**expected, "page_token": "t2"}
    assert [v.id for v in voices] == ["Kore", "Aoede"]
    assert voices[0].name == "Kore"
    assert voices[0].description == "Firm"
    assert voices[0].extra == {"type": "prebuilt", "gender": "female", "language_code": "en-US"}


def test_list_library_single_page_with_token(provider: GoogleProvider, client: MagicMock) -> None:
    client.voices.list.return_value = _page([{"id": "Puck", "display_name": "Puck"}], token="t3")
    voices = provider.list_library(page_token="t2")
    client.voices.list.assert_called_once_with(page_token="t2")
    assert [v.id for v in voices] == ["Puck"]


def test_list_library_limit(provider: GoogleProvider, client: MagicMock) -> None:
    client.voices.list.return_value = _page(
        [{"id": n, "display_name": n} for n in ("A", "B", "C")], token="more"
    )
    voices = provider.list_library(limit=2)
    client.voices.list.assert_called_once_with()
    assert [v.id for v in voices] == ["A", "B"]


def test_list_library_unknown_filter(provider: GoogleProvider) -> None:
    with pytest.raises(ConfigError, match="bogus"):
        provider.list_library(bogus="x")
