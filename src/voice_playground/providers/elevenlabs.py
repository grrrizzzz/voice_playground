"""ElevenLabs provider: TTS, Scribe STT, Voice Design, Instant Voice Clone, voice library.

SDK: `elevenlabs` (verified against 2.70.0 and https://elevenlabs.io/docs/api-reference).

- TTS: `client.text_to_speech.convert(voice_id, text=..., model_id=..., output_format=...,
  voice_settings=VoiceSettings(...))` returns an iterator of byte chunks.
  `wav` / `pcm` request `pcm_<rate>` (s16le mono); `wav` is wrapped in a WAV header here.
  `mp3` requests `mp3_44100_128`.
- STT: `client.speech_to_text.convert(model_id=..., file=<binary file>, language_code=...)`.
- Voice Design: `client.text_to_voice.design(voice_description=...)` gives previews, then
  `client.text_to_voice.create(voice_name=..., voice_description=..., generated_voice_id=...)`.
- Instant Voice Clone: `client.voices.ivc.create(name=..., files=[<binary file>])`.
- `client.voices.delete(voice_id)` and `client.voices.search(...)` (paginated).

`provider_options` keys (voice config): `stability`, `similarity_boost`, `style` (float),
`speed`, `use_speaker_boost` -> `voice_settings`; `seed`, `language_code`,
`apply_text_normalization` -> TTS request; `design_model_id` -> Voice Design model;
`remove_background_noise` -> Instant Voice Clone. Unknown keys raise `ConfigError`.
"""

from __future__ import annotations

import io
import sys
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from voice_playground.errors import ConfigError, ProviderError, VPError
from voice_playground.providers.base import (
    AudioResult,
    BaseProvider,
    Capability,
    CreatedVoice,
    LibraryVoice,
    STTRequest,
    Transcript,
    TTSRequest,
)
from voice_playground.settings import require_secret

if TYPE_CHECKING:
    from voice_playground.settings import Settings
    from voice_playground.voices import VoiceConfig

T = TypeVar("T")

#: TTS models (default first). Verified at https://elevenlabs.io/docs/overview/models.
TTS_MODELS: tuple[str, ...] = ("eleven_v3", "eleven_multilingual_v2", "eleven_flash_v2_5")
#: Scribe models. `scribe_v1` is deprecated in favour of `scribe_v2`.
STT_MODELS: tuple[str, ...] = ("scribe_v2",)
#: Voice Design model (alternative: `eleven_multilingual_ttv_v2`).
DEFAULT_DESIGN_MODEL = "eleven_ttv_v3"

DEFAULT_SAMPLE_RATE = 24_000
#: Sample rates ElevenLabs offers as raw `pcm_<rate>` output.
PCM_RATES: frozenset[int] = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})
MP3_FORMAT = "mp3_44100_128"
MP3_SAMPLE_RATE = 44_100

VOICE_SETTING_KEYS = frozenset(
    {"stability", "similarity_boost", "style", "speed", "use_speaker_boost"}
)
TTS_OPTION_KEYS = frozenset({"seed", "language_code", "apply_text_normalization"})
OTHER_OPTION_KEYS = frozenset({"design_model_id", "remove_background_noise"})
KNOWN_OPTION_KEYS = VOICE_SETTING_KEYS | TTS_OPTION_KEYS | OTHER_OPTION_KEYS

MAX_ERROR_DETAIL = 200
DEFAULT_LIBRARY_LIMIT = 100


def output_format_for(fmt: str, sample_rate: int | None) -> tuple[str, int]:
    """Map our output format (+ optional sample rate) to `(elevenlabs output_format, rate)`."""
    if fmt == "mp3":
        return MP3_FORMAT, MP3_SAMPLE_RATE
    if fmt not in ("wav", "pcm"):
        raise ConfigError(f"elevenlabs: unsupported output format '{fmt}' (use wav, mp3 or pcm)")
    rate = sample_rate or DEFAULT_SAMPLE_RATE
    if rate not in PCM_RATES:
        valid = ", ".join(str(r) for r in sorted(PCM_RATES))
        raise ConfigError(f"elevenlabs: unsupported sample rate {rate}; supported: {valid}")
    return f"pcm_{rate}", rate


def _pcm_to_wav(data: bytes, sample_rate: int) -> bytes:
    """Wrap mono s16le PCM in a WAV header (private; audio.py is not a dependency here)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return buf.getvalue()


def _base_language(code: str | None) -> str | None:
    """`en-US` -> `en`. ElevenLabs uses ISO-639 language codes without a region."""
    if not code:
        return None
    return code.split("-")[0].split("_")[0].lower() or None


def _error_detail(body: Any) -> str:
    """Best-effort short reason from an ElevenLabs error body."""
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("status") or ""
    elif isinstance(detail, list) and detail:
        first = detail[0]
        detail = first.get("msg", "") if isinstance(first, dict) else first
    return str(detail or "").strip()


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


class ElevenLabsProvider(BaseProvider):
    name: ClassVar[str] = "elevenlabs"
    capabilities: ClassVar[frozenset[Capability]] = frozenset(
        {
            Capability.TTS,
            Capability.STT,
            Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE,
            Capability.VOICE_LIBRARY,
        }
    )
    default_tts_model: ClassVar[str | None] = TTS_MODELS[0]
    default_stt_model: ClassVar[str | None] = STT_MODELS[0]

    def __init__(self, settings: Settings, client: Any = None) -> None:
        super().__init__(settings)
        self._api_key = require_secret(settings.elevenlabs_api_key, "ELEVENLABS_API_KEY")
        self._client_obj = client

    # -- plumbing ---------------------------------------------------------------------------

    @property
    def _client(self) -> Any:
        if self._client_obj is None:
            from elevenlabs import ElevenLabs

            self._client_obj = ElevenLabs(api_key=self._api_key)
        return self._client_obj

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _call(self, action: str, fn: Callable[[], T]) -> T:
        """Run an SDK call, mapping any SDK/network error to a short `ProviderError`."""
        try:
            return fn()
        except VPError:
            raise
        except Exception as exc:
            raise ProviderError(self._redact(self._describe(action, exc))) from exc

    @staticmethod
    def _describe(action: str, exc: Exception) -> str:
        from elevenlabs.core.api_error import ApiError

        if isinstance(exc, ApiError):
            reason = _error_detail(exc.body)[:MAX_ERROR_DETAIL]
            status = exc.status_code if exc.status_code is not None else "?"
            return f"elevenlabs {action} failed: HTTP {status}" + (f": {reason}" if reason else "")
        # Don't echo arbitrary exception text (could contain request details); type is enough.
        return f"elevenlabs {action} failed: {type(exc).__name__}"

    @staticmethod
    def _check_options(options: dict[str, Any]) -> None:
        unknown = sorted(set(options) - KNOWN_OPTION_KEYS)
        if unknown:
            valid = ", ".join(sorted(KNOWN_OPTION_KEYS))
            raise ConfigError(
                f"elevenlabs: unknown provider_options {', '.join(unknown)}; valid: {valid}"
            )

    # -- TTS --------------------------------------------------------------------------------

    def tts(self, req: TTSRequest) -> AudioResult:
        options = dict(req.voice.options)
        self._check_options(options)
        output_format, rate = output_format_for(
            req.output_format, req.sample_rate or req.voice.sample_rate
        )
        if req.voice.style:
            _warn(
                "elevenlabs ignores the text 'style' field; use provider_options.style (0-1) "
                "or inline audio tags like [whispers] with eleven_v3"
            )

        kwargs: dict[str, Any] = {
            "text": req.text,
            "model_id": req.model or self.default_tts_model,
            "output_format": output_format,
        }
        settings = {k: options[k] for k in VOICE_SETTING_KEYS if k in options}
        if settings:
            from elevenlabs import VoiceSettings

            kwargs["voice_settings"] = VoiceSettings(**settings)
        kwargs.update({k: options[k] for k in TTS_OPTION_KEYS if k in options})

        def run() -> bytes:
            chunks: Iterable[bytes] = self._client.text_to_speech.convert(
                req.voice.provider_voice, **kwargs
            )
            return b"".join(chunks)

        data = self._call("tts", run)
        if not data:
            raise ProviderError("elevenlabs tts failed: empty audio response")
        if req.output_format == "mp3":
            return AudioResult(data=data, mime_type="audio/mpeg", sample_rate=rate)
        if req.output_format == "pcm":
            return AudioResult(data=data, mime_type="audio/l16", sample_rate=rate)
        return AudioResult(data=_pcm_to_wav(data, rate), mime_type="audio/wav", sample_rate=rate)

    # -- STT --------------------------------------------------------------------------------

    def stt(self, req: STTRequest) -> Transcript:
        if not req.audio_path.is_file():
            raise ConfigError(f"audio file not found: {req.audio_path}")
        if req.prompt:
            _warn("elevenlabs Scribe does not take a free-text prompt; --prompt is ignored")
        kwargs: dict[str, Any] = {"model_id": req.model or self.default_stt_model}
        language = _base_language(req.language)
        if language:
            kwargs["language_code"] = language

        def run() -> Any:
            with req.audio_path.open("rb") as fh:
                return self._client.speech_to_text.convert(file=fh, **kwargs)

        resp = self._call("stt", run)
        text = getattr(resp, "text", None)
        if text is None:
            raise ProviderError("elevenlabs stt failed: response contained no transcript")
        segments = [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "type": w.type,
                "speaker_id": w.speaker_id,
            }
            for w in (getattr(resp, "words", None) or [])
        ]
        return Transcript(
            text=str(text).strip(),
            language=getattr(resp, "language_code", None) or language,
            segments=segments or None,
        )

    # -- custom voices ----------------------------------------------------------------------

    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice:
        options = dict(cfg.provider_options)
        self._check_options(options)
        if cfg.type == "designed":
            return self._design_voice(cfg.name, cfg.description, options)
        if cfg.type == "cloned":
            return self._clone_voice(cfg.name, cfg.reference_audio, options)
        raise ConfigError(f"voice '{cfg.name}' is prebuilt; nothing to create")

    def _design_voice(self, name: str, description: str, options: dict[str, Any]) -> CreatedVoice:
        model_id = options.get("design_model_id") or DEFAULT_DESIGN_MODEL
        design = self._call(
            "voice design",
            lambda: self._client.text_to_voice.design(
                voice_description=description, model_id=model_id, auto_generate_text=True
            ),
        )
        previews = getattr(design, "previews", None) or []
        if not previews:
            raise ProviderError("elevenlabs voice design failed: no previews returned")
        generated_id = previews[0].generated_voice_id
        voice = self._call(
            "voice create",
            lambda: self._client.text_to_voice.create(
                voice_name=name, voice_description=description, generated_voice_id=generated_id
            ),
        )
        return CreatedVoice(remote_id=voice.voice_id, expires_at=None)

    def _clone_voice(self, name: str, path: Path, options: dict[str, Any]) -> CreatedVoice:
        if not path.is_file():
            raise ConfigError(f"voice '{name}': reference_audio not found: {path}")
        kwargs: dict[str, Any] = {"name": name}
        if "remove_background_noise" in options:
            kwargs["remove_background_noise"] = bool(options["remove_background_noise"])

        def run() -> Any:
            with path.open("rb") as fh:
                return self._client.voices.ivc.create(files=[fh], **kwargs)

        resp = self._call("voice clone", run)
        if getattr(resp, "requires_verification", False):
            _warn(f"elevenlabs: cloned voice '{name}' requires verification before use")
        return CreatedVoice(remote_id=resp.voice_id, expires_at=None)

    def delete_voice(self, remote_id: str) -> None:
        self._call("voice delete", lambda: self._client.voices.delete(remote_id))

    # -- library ----------------------------------------------------------------------------

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        """Search the account's voices (`voices.search`), following pagination.

        Filters: `search`, `language` (region stripped: en-US -> en), `gender`, `age`,
        `accent`, `category`, `voice_type`, `limit` (default 100 results).
        """
        limit = int(filters.get("limit") or DEFAULT_LIBRARY_LIMIT)
        query: dict[str, Any] = {
            k: filters[k]
            for k in ("search", "gender", "age", "accent", "category", "voice_type")
            if filters.get(k)
        }
        language = _base_language(filters.get("language"))
        if language:
            query["language"] = language

        result: list[LibraryVoice] = []
        token: str | None = None
        while len(result) < limit:
            page_size = min(100, limit - len(result))
            page_token = token

            def fetch(size: int = page_size, tok: str | None = page_token) -> Any:
                kw = dict(query, page_size=size)
                if tok:
                    kw["next_page_token"] = tok
                return self._client.voices.search(**kw)

            page = self._call("voice library", fetch)
            for v in page.voices:
                result.append(
                    LibraryVoice(
                        id=v.voice_id,
                        name=v.name or v.voice_id,
                        description=v.description,
                        extra={"category": v.category, "labels": dict(v.labels or {})},
                    )
                )
            token = page.next_page_token
            if not page.has_more or not token or not page.voices:
                break
        return result[:limit]
