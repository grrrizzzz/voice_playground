"""Unit tests for the ElevenLabs provider. The SDK client is always a mock; no network."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from elevenlabs import VoiceSettings
from elevenlabs.core.api_error import ApiError

from voice_playground.errors import ConfigError, ProviderError
from voice_playground.providers.base import (
    Capability,
    ResolvedVoice,
    STTRequest,
    TTSRequest,
)
from voice_playground.providers.elevenlabs import (
    DEFAULT_DESIGN_MODEL,
    ElevenLabsProvider,
    output_format_for,
)
from voice_playground.providers.registry import get_provider
from voice_playground.settings import Settings
from voice_playground.voices import (
    ClonedVoiceConfig,
    DesignedVoiceConfig,
    PrebuiltVoiceConfig,
)

FAKE_KEY = "sk_unit_test_dummy_key_0123456789"
PCM = b"\x01\x00\x02\x00" * 600  # 1200 frames of s16le mono


@pytest.fixture
def client() -> MagicMock:
    c = MagicMock()
    c.text_to_speech.convert.return_value = iter([PCM[:1000], PCM[1000:]])
    return c


@pytest.fixture
def provider(settings: Settings, client: MagicMock) -> ElevenLabsProvider:
    keyed = settings.model_copy(update={"elevenlabs_api_key": _secret(FAKE_KEY)})
    return ElevenLabsProvider(keyed, client=client)


def _secret(value: str) -> Any:
    from pydantic import SecretStr

    return SecretStr(value)


def _req(**overrides: Any) -> TTSRequest:
    voice = overrides.pop("voice", ResolvedVoice(provider_voice="voice123"))
    return TTSRequest(
        text="Hello there", model=overrides.pop("model", None), voice=voice, **overrides
    )


# -- construction / metadata -------------------------------------------------------------------


def test_missing_key_raises_config_error(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="ELEVENLABS_API_KEY"):
        ElevenLabsProvider(settings)


def test_registry_constructs_provider(settings: Settings) -> None:
    keyed = settings.model_copy(update={"elevenlabs_api_key": _secret(FAKE_KEY)})
    p = get_provider("elevenlabs", keyed)
    assert isinstance(p, ElevenLabsProvider)
    assert p.default_tts_model == "eleven_v3"
    assert p.default_stt_model == "scribe_v2"
    assert p.capabilities == frozenset(Capability)


def test_real_client_built_lazily_with_key(settings: Settings) -> None:
    keyed = settings.model_copy(update={"elevenlabs_api_key": _secret(FAKE_KEY)})
    p = ElevenLabsProvider(keyed)
    from elevenlabs import ElevenLabs

    assert isinstance(p._client, ElevenLabs)


# -- output format mapping ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "rate", "expected"),
    [
        ("mp3", None, ("mp3_44100_128", 44100)),
        ("mp3", 16000, ("mp3_44100_128", 44100)),
        ("wav", None, ("pcm_24000", 24000)),
        ("pcm", None, ("pcm_24000", 24000)),
        ("wav", 16000, ("pcm_16000", 16000)),
        ("pcm", 8000, ("pcm_8000", 8000)),
        ("wav", 22050, ("pcm_22050", 22050)),
        ("wav", 44100, ("pcm_44100", 44100)),
        ("pcm", 48000, ("pcm_48000", 48000)),
    ],
)
def test_output_format_mapping(fmt: str, rate: int | None, expected: tuple[str, int]) -> None:
    assert output_format_for(fmt, rate) == expected


def test_output_format_rejects_bad_rate_and_format() -> None:
    with pytest.raises(ConfigError, match="sample rate 11025"):
        output_format_for("wav", 11025)
    with pytest.raises(ConfigError, match="format 'ogg'"):
        output_format_for("ogg", None)


# -- TTS ---------------------------------------------------------------------------------------


def test_tts_wav_default_payload_and_header(
    provider: ElevenLabsProvider, client: MagicMock
) -> None:
    result = provider.tts(_req())
    client.text_to_speech.convert.assert_called_once_with(
        "voice123", text="Hello there", model_id="eleven_v3", output_format="pcm_24000"
    )
    assert result.mime_type == "audio/wav"
    assert result.sample_rate == 24000
    with wave.open(io.BytesIO(result.data)) as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == len(PCM) // 2
        assert w.readframes(w.getnframes()) == PCM


def test_tts_pcm_returns_raw(provider: ElevenLabsProvider, client: MagicMock) -> None:
    result = provider.tts(_req(output_format="pcm", sample_rate=16000))
    assert client.text_to_speech.convert.call_args.kwargs["output_format"] == "pcm_16000"
    assert result.data == PCM
    assert result.mime_type == "audio/l16"
    assert result.sample_rate == 16000


def test_tts_mp3(provider: ElevenLabsProvider, client: MagicMock) -> None:
    client.text_to_speech.convert.return_value = iter([b"ID3", b"mp3data"])
    result = provider.tts(_req(output_format="mp3", model="eleven_flash_v2_5"))
    kwargs = client.text_to_speech.convert.call_args.kwargs
    assert kwargs["output_format"] == "mp3_44100_128"
    assert kwargs["model_id"] == "eleven_flash_v2_5"
    assert result.data == b"ID3mp3data"
    assert result.mime_type == "audio/mpeg"


def test_tts_voice_sample_rate_used_when_request_has_none(
    provider: ElevenLabsProvider, client: MagicMock
) -> None:
    voice = ResolvedVoice(provider_voice="v", sample_rate=22050)
    result = provider.tts(_req(voice=voice))
    assert client.text_to_speech.convert.call_args.kwargs["output_format"] == "pcm_22050"
    assert result.sample_rate == 22050


def test_tts_provider_options_mapping(provider: ElevenLabsProvider, client: MagicMock) -> None:
    voice = ResolvedVoice(
        provider_voice="custom_id",
        options={
            "stability": 0.3,
            "similarity_boost": 0.8,
            "style": 0.2,
            "speed": 1.1,
            "use_speaker_boost": True,
            "seed": 42,
            "language_code": "en",
            "apply_text_normalization": "on",
        },
    )
    provider.tts(_req(voice=voice, model="eleven_multilingual_v2"))
    args, kwargs = client.text_to_speech.convert.call_args
    assert args == ("custom_id",)
    assert kwargs["voice_settings"] == VoiceSettings(
        stability=0.3, similarity_boost=0.8, style=0.2, speed=1.1, use_speaker_boost=True
    )
    assert kwargs["seed"] == 42
    assert kwargs["language_code"] == "en"
    assert kwargs["apply_text_normalization"] == "on"
    assert kwargs["model_id"] == "eleven_multilingual_v2"


def test_tts_unknown_option_is_config_error(provider: ElevenLabsProvider) -> None:
    voice = ResolvedVoice(provider_voice="v", options={"stabilty": 0.3})
    with pytest.raises(ConfigError, match="stabilty"):
        provider.tts(_req(voice=voice))


def test_tts_text_style_warns_and_is_not_sent(
    provider: ElevenLabsProvider, client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    provider.tts(_req(voice=ResolvedVoice(provider_voice="v", style="whispered")))
    assert "style" in capsys.readouterr().err
    assert "voice_settings" not in client.text_to_speech.convert.call_args.kwargs


def test_tts_empty_audio_is_provider_error(provider: ElevenLabsProvider, client: MagicMock) -> None:
    client.text_to_speech.convert.return_value = iter([])
    with pytest.raises(ProviderError, match="empty"):
        provider.tts(_req())


# -- STT ---------------------------------------------------------------------------------------


def _stt_response() -> SimpleNamespace:
    word = SimpleNamespace(text="Hello", start=0.0, end=0.4, type="word", speaker_id=None)
    return SimpleNamespace(text=" Hello there ", language_code="eng", words=[word])


def test_stt_payload_and_result(
    provider: ElevenLabsProvider, client: MagicMock, tmp_path: Path
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFdata")
    seen: dict[str, Any] = {}

    def convert(**kwargs: Any) -> SimpleNamespace:
        fh = kwargs.pop("file")
        seen["mode"] = fh.mode
        seen["bytes"] = fh.read()
        seen.update(kwargs)
        return _stt_response()

    client.speech_to_text.convert.side_effect = convert
    t = provider.stt(STTRequest(audio_path=audio, model=None, language="en-US"))
    assert seen == {
        "mode": "rb",
        "bytes": b"RIFFdata",
        "model_id": "scribe_v2",
        "language_code": "en",
    }
    assert t.text == "Hello there"
    assert t.language == "eng"
    assert t.segments == [
        {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": None}
    ]


def test_stt_no_language_and_explicit_model(
    provider: ElevenLabsProvider,
    client: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"x")
    client.speech_to_text.convert.return_value = _stt_response()
    provider.stt(STTRequest(audio_path=audio, model="scribe_v1", prompt="jargon"))
    kwargs = client.speech_to_text.convert.call_args.kwargs
    assert kwargs["model_id"] == "scribe_v1"
    assert "language_code" not in kwargs
    assert "prompt" in capsys.readouterr().err


def test_stt_missing_file(provider: ElevenLabsProvider, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        provider.stt(STTRequest(audio_path=tmp_path / "nope.wav", model=None))


# -- create / delete voices --------------------------------------------------------------------


def test_create_designed_voice(provider: ElevenLabsProvider, client: MagicMock) -> None:
    client.text_to_voice.design.return_value = SimpleNamespace(
        previews=[
            SimpleNamespace(generated_voice_id="gen_1"),
            SimpleNamespace(generated_voice_id="gen_2"),
        ],
        text="sample",
    )
    client.text_to_voice.create.return_value = SimpleNamespace(voice_id="new_voice")
    cfg = DesignedVoiceConfig(name="storyteller", description="A warm, raspy old sailor voice")
    created = provider.create_voice(cfg)
    client.text_to_voice.design.assert_called_once_with(
        voice_description="A warm, raspy old sailor voice",
        model_id=DEFAULT_DESIGN_MODEL,
        auto_generate_text=True,
    )
    client.text_to_voice.create.assert_called_once_with(
        voice_name="storyteller",
        voice_description="A warm, raspy old sailor voice",
        generated_voice_id="gen_1",
    )
    assert created.remote_id == "new_voice"
    assert created.expires_at is None


def test_create_designed_voice_model_override(
    provider: ElevenLabsProvider, client: MagicMock
) -> None:
    client.text_to_voice.design.return_value = SimpleNamespace(
        previews=[SimpleNamespace(generated_voice_id="g")]
    )
    client.text_to_voice.create.return_value = SimpleNamespace(voice_id="v")
    cfg = DesignedVoiceConfig(
        name="n",
        description="desc",
        provider_options={"design_model_id": "eleven_multilingual_ttv_v2"},
    )
    provider.create_voice(cfg)
    assert client.text_to_voice.design.call_args.kwargs["model_id"] == "eleven_multilingual_ttv_v2"


def test_create_designed_voice_no_previews(provider: ElevenLabsProvider, client: MagicMock) -> None:
    client.text_to_voice.design.return_value = SimpleNamespace(previews=[])
    with pytest.raises(ProviderError, match="no previews"):
        provider.create_voice(DesignedVoiceConfig(name="n", description="d"))
    client.text_to_voice.create.assert_not_called()


def test_create_cloned_voice(
    provider: ElevenLabsProvider, client: MagicMock, tmp_path: Path
) -> None:
    ref = tmp_path / "me.wav"
    ref.write_bytes(b"RIFFme")
    seen: dict[str, Any] = {}

    def create(**kwargs: Any) -> SimpleNamespace:
        files = kwargs.pop("files")
        seen["modes"] = [f.mode for f in files]
        seen["bytes"] = [f.read() for f in files]
        seen.update(kwargs)
        return SimpleNamespace(voice_id="clone_1", requires_verification=False)

    client.voices.ivc.create.side_effect = create
    cfg = ClonedVoiceConfig(
        name="me-clone", reference_audio=ref, provider_options={"remove_background_noise": True}
    )
    created = provider.create_voice(cfg)
    assert seen == {
        "modes": ["rb"],
        "bytes": [b"RIFFme"],
        "name": "me-clone",
        "remove_background_noise": True,
    }
    assert created.remote_id == "clone_1"
    assert created.expires_at is None


def test_create_cloned_voice_missing_file(provider: ElevenLabsProvider, tmp_path: Path) -> None:
    cfg = ClonedVoiceConfig(name="me", reference_audio=tmp_path / "missing.wav")
    with pytest.raises(ConfigError, match="reference_audio"):
        provider.create_voice(cfg)


def test_create_cloned_voice_rejects_non_audio_file(
    provider: ElevenLabsProvider, client: MagicMock, tmp_path: Path
) -> None:
    secret = tmp_path / "credentials.json"
    secret.write_text("{}")
    cfg = ClonedVoiceConfig(name="me", reference_audio=secret)
    with pytest.raises(ConfigError, match="reference_audio must be an audio file"):
        provider.create_voice(cfg)
    client.voices.ivc.create.assert_not_called()


def test_create_prebuilt_voice_is_config_error(provider: ElevenLabsProvider) -> None:
    with pytest.raises(ConfigError, match="prebuilt"):
        provider.create_voice(PrebuiltVoiceConfig(name="rachel", voice="abc"))


def test_delete_voice(provider: ElevenLabsProvider, client: MagicMock) -> None:
    provider.delete_voice("voice_9")
    client.voices.delete.assert_called_once_with("voice_9")


# -- library -----------------------------------------------------------------------------------


def _voice(i: int) -> SimpleNamespace:
    return SimpleNamespace(
        voice_id=f"id{i}",
        name=f"Voice {i}",
        description=f"desc {i}",
        category="premade",
        labels={"gender": "female"},
    )


def test_list_library_filters_and_pagination(
    provider: ElevenLabsProvider, client: MagicMock
) -> None:
    client.voices.search.side_effect = [
        SimpleNamespace(voices=[_voice(1), _voice(2)], has_more=True, next_page_token="p2"),
        SimpleNamespace(voices=[_voice(3)], has_more=False, next_page_token=None),
    ]
    voices = provider.list_library(search="warm", language="en-US", gender="female")
    first, second = client.voices.search.call_args_list
    assert first.kwargs == {
        "search": "warm",
        "gender": "female",
        "language": "en",
        "page_size": 100,
    }
    assert second.kwargs == {
        "search": "warm",
        "gender": "female",
        "language": "en",
        "page_size": 98,
        "next_page_token": "p2",
    }
    assert [v.id for v in voices] == ["id1", "id2", "id3"]
    assert voices[0].name == "Voice 1"
    assert voices[0].description == "desc 1"
    assert voices[0].extra == {"category": "premade", "labels": {"gender": "female"}}


def test_list_library_respects_limit(provider: ElevenLabsProvider, client: MagicMock) -> None:
    client.voices.search.return_value = SimpleNamespace(
        voices=[_voice(1), _voice(2)], has_more=True, next_page_token="p2"
    )
    voices = provider.list_library(limit=2)
    client.voices.search.assert_called_once_with(page_size=2)
    assert len(voices) == 2


# -- error mapping -----------------------------------------------------------------------------


def test_api_error_becomes_provider_error_without_key(
    provider: ElevenLabsProvider, client: MagicMock
) -> None:
    client.text_to_speech.convert.side_effect = ApiError(
        status_code=401,
        headers={"xi-api-key": FAKE_KEY},
        body={"detail": {"status": "invalid_api_key", "message": f"Invalid key {FAKE_KEY}"}},
    )
    with pytest.raises(ProviderError) as exc_info:
        provider.tts(_req())
    msg = str(exc_info.value)
    assert "401" in msg
    assert "Invalid key" in msg
    assert FAKE_KEY not in msg
    assert exc_info.value.exit_code == 1


def test_generic_error_becomes_provider_error_without_key(
    provider: ElevenLabsProvider, client: MagicMock
) -> None:
    client.voices.delete.side_effect = RuntimeError(f"boom api_key={FAKE_KEY}")
    with pytest.raises(ProviderError) as exc_info:
        provider.delete_voice("v")
    msg = str(exc_info.value)
    assert "RuntimeError" in msg
    assert FAKE_KEY not in msg


def test_api_error_on_stt_and_library(
    provider: ElevenLabsProvider, client: MagicMock, tmp_path: Path
) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    client.speech_to_text.convert.side_effect = ApiError(status_code=422, body="bad model")
    with pytest.raises(ProviderError, match="stt failed: HTTP 422: bad model"):
        provider.stt(STTRequest(audio_path=audio, model="nope"))
    client.voices.search.side_effect = ApiError(status_code=500, body=None)
    with pytest.raises(ProviderError, match=r"voice library failed: HTTP 500$"):
        provider.list_library()
