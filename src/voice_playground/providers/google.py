"""Google Gemini provider (PLAN.md T2).

API surface (verified against https://aistudio.google.com/docs/speech-generation,
/docs/voice-design, /docs/voice-replication, https://ai.google.dev/gemini-api/docs/audio and the
installed `google-genai` SDK):

- TTS: `client.interactions.create(model, input=[user_input step], response_format={"type":
  "audio", ...}, generation_config={"speech_config": [{"voice": ...}]})`. The audio comes back
  base64-encoded in `interaction.output_audio.data`. The 3.8 models return a WAV (RIFF) file by
  default. `gemini-3.1-flash-tts-preview` also uses the interactions API, but returns headerless
  24 kHz s16le PCM (`audio/l16`) and supports neither custom voices nor other sample rates, so it
  gets its own request shape and the PCM is wrapped into WAV locally.
- Custom voices: `client.voices.create(voice={...}, store=...)`. Stored voices have ids
  `voice_...` (1 year TTL); stateless replicated voices return a `key` `voicekey_...` (7 days).
  Both ids go into `speech_config` exactly like a prebuilt voice name.
- STT: audio understanding with `gemini-3.8-flash` through `client.interactions.create` with a
  text prompt plus an inline (base64) audio block, or an uploaded file for large inputs.
"""

from __future__ import annotations

import base64
import io
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai._gaos.lib.compat_errors import GeminiNextGenAPIClientError

from voice_playground.errors import ConfigError, ProviderError
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

#: TTS models served by this provider (default first).
TTS_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
)
#: Preview model: returns raw 24 kHz PCM and does not support custom voices.
PREVIEW_TTS_MODEL = "gemini-3.1-flash-tts-preview"
#: Audio-understanding model used for transcription (ai.google.dev/gemini-api/docs/audio).
STT_MODEL = "gemini-3.8-flash"
TRANSCRIBE_PROMPT = "Generate a transcript of the speech. Return only the transcript text."

DEFAULT_SAMPLE_RATE = 24_000
SUPPORTED_SAMPLE_RATES = frozenset({24_000, 16_000, 8_000})
CUSTOM_VOICE_PREFIXES = ("voice_", "voicekey_")
STORED_VOICE_TTL = timedelta(days=365)
STATELESS_VOICE_TTL = timedelta(days=7)
#: Above this size, STT uploads the file instead of inlining it (request limit is 20 MB).
MAX_INLINE_AUDIO_BYTES = 14 * 1024 * 1024
MAX_ERROR_DETAIL = 200

#: `provider_options` keys copied into the `voices.create` voice payload.
VOICE_INPUT_OPTION_KEYS = (
    "display_name",
    "gender",
    "accent",
    "persona",
    "pitch",
    "context",
    "region_code",
    "description",
)
#: `list_library` filters that the SDK takes as lists of strings.
LIST_FILTER_KEYS = (
    "language_code",
    "gender",
    "pitch",
    "contexts",
    "type_",
    "accent",
    "persona",
    "region_code",
)
#: Friendly aliases accepted by `list_library`.
FILTER_ALIASES = {"language": "language_code", "type": "type_", "context": "contexts"}

AUDIO_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".m4a": "audio/m4a",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".webm": "audio/webm",
    ".pcm": "audio/l16",
}

#: Exceptions the SDK raises for API / transport failures.
SDK_ERRORS: tuple[type[BaseException], ...] = (
    genai_errors.APIError,
    GeminiNextGenAPIClientError,
    httpx.HTTPError,
)


def is_custom_voice(voice_id: str) -> bool:
    """True for remote custom-voice ids (`voice_...` stored, `voicekey_...` stateless)."""
    return voice_id.startswith(CUSTOM_VOICE_PREFIXES)


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw s16le PCM in a WAV (RIFF) container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm(data: bytes) -> tuple[bytes, int, int]:
    """Return `(pcm_frames, sample_rate, channels)` from a WAV file."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels()


def _is_wav(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _get(obj: Any, key: str) -> Any:
    """Read `key` from an SDK model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class GoogleProvider(BaseProvider):
    name: ClassVar[str] = "google"
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
    default_stt_model: ClassVar[str | None] = STT_MODEL

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key = require_secret(settings.google_api_key, "GOOGLE_API_KEY")
        self._client_obj: genai.Client | None = None

    # ----------------------------------------------------------------- plumbing
    @property
    def _client(self) -> genai.Client:
        if self._client_obj is None:
            self._client_obj = genai.Client(api_key=self._api_key)
        return self._client_obj

    def _error(self, action: str, exc: BaseException) -> ProviderError:
        """Short `ProviderError` (status + reason) with the API key scrubbed."""
        status: Any = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        reason: Any = getattr(exc, "status", None)
        detail: Any = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            status = status or error.get("code")
            reason = error.get("status") or reason
            detail = error.get("message") or detail
        if reason and not isinstance(reason, int) and str(reason) != str(status):
            status = f"{status} {reason}" if status else reason
        detail = " ".join(str(detail).split())
        if len(detail) > MAX_ERROR_DETAIL:
            detail = detail[: MAX_ERROR_DETAIL - 3] + "..."
        prefix = f"google {action} failed"
        if status is not None:
            prefix += f" ({status})"
        message = f"{prefix}: {detail}"
        if self._api_key:
            message = message.replace(self._api_key, "***")
        return ProviderError(message)

    def _call(self, action: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except SDK_ERRORS as exc:
            raise self._error(action, exc) from exc

    # ----------------------------------------------------------------------- tts
    def tts(self, req: TTSRequest) -> AudioResult:
        model = req.model or self.default_tts_model or TTS_MODELS[0]
        voice = req.voice
        sample_rate = req.sample_rate or voice.sample_rate or DEFAULT_SAMPLE_RATE
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            rates = ", ".join(str(r) for r in sorted(SUPPORTED_SAMPLE_RATES, reverse=True))
            raise ConfigError(f"google tts sample_rate must be one of {rates}, got {sample_rate}")
        if not voice.provider_voice:
            raise ConfigError("google tts needs a voice (e.g. Kore) or a custom voice id")

        preview = model == PREVIEW_TTS_MODEL
        if preview:
            if is_custom_voice(voice.provider_voice):
                raise ConfigError(
                    f"{PREVIEW_TTS_MODEL} does not support custom voices; "
                    f"use {TTS_MODELS[0]} or {TTS_MODELS[1]}"
                )
            if sample_rate != DEFAULT_SAMPLE_RATE:
                raise ConfigError(f"{PREVIEW_TTS_MODEL} only outputs {DEFAULT_SAMPLE_RATE} Hz")

        payload = self._tts_payload(req, model, sample_rate, preview=preview)
        interaction = self._call("tts", self._client.interactions.create, **payload)
        data = self._extract_audio(interaction)
        return self._shape_audio(data, req.output_format, sample_rate)

    def _tts_payload(
        self, req: TTSRequest, model: str, sample_rate: int, *, preview: bool
    ) -> dict[str, Any]:
        voice = req.voice
        text_block: dict[str, Any] = {"type": "text", "text": req.text}
        if voice.style:
            text_block["annotations"] = [{"type": "speech_metadata", "style": voice.style}]
        speech: dict[str, Any] = {"voice": voice.provider_voice}
        if voice.language:
            speech["language"] = voice.language

        response_format: dict[str, Any] = {"type": "audio"}
        if not preview:
            # mp3 is produced as WAV and converted by the caller (mime type says audio/wav).
            response_format["mime_type"] = (
                "audio/l16" if req.output_format == "pcm" else "audio/wav"
            )
            response_format["sample_rate"] = sample_rate

        return {
            "model": model,
            "input": [{"type": "user_input", "content": [text_block]}],
            "response_format": response_format,
            "generation_config": {"speech_config": [speech]},
        }

    def _extract_audio(self, interaction: Any) -> bytes:
        audio = _get(interaction, "output_audio")
        if audio is None:
            # Fall back to the raw step list: steps[].content[] with type "audio".
            for step in _get(interaction, "steps") or []:
                for block in _get(step, "content") or []:
                    if _get(block, "type") == "audio" and _get(block, "data"):
                        audio = block
        raw = _get(audio, "data") if audio is not None else None
        if not raw:
            raise ProviderError("google tts returned no audio")
        if isinstance(raw, bytes):
            return raw
        try:
            return base64.b64decode(raw)
        except ValueError as exc:
            raise ProviderError("google tts returned audio that is not valid base64") from exc

    @staticmethod
    def _shape_audio(data: bytes, output_format: str, sample_rate: int) -> AudioResult:
        """Normalise returned bytes (WAV or headerless PCM) to the requested format."""
        if output_format == "pcm":
            if _is_wav(data):
                pcm, rate, channels = wav_to_pcm(data)
                return AudioResult(pcm, "audio/l16", rate, channels)
            return AudioResult(data, "audio/l16", sample_rate)
        if _is_wav(data):
            _, rate, channels = wav_to_pcm(data)
            return AudioResult(data, "audio/wav", rate, channels)
        return AudioResult(pcm_to_wav(data, sample_rate), "audio/wav", sample_rate)

    # ----------------------------------------------------------------------- stt
    def stt(self, req: STTRequest) -> Transcript:
        path = Path(req.audio_path)
        if not path.is_file():
            raise ConfigError(f"audio file not found: {path}")
        mime = AUDIO_MIME_BY_SUFFIX.get(path.suffix.lower())
        if mime is None:
            known = ", ".join(sorted(AUDIO_MIME_BY_SUFFIX))
            raise ConfigError(f"unsupported audio type '{path.suffix}' (supported: {known})")
        model = req.model or self.default_stt_model or STT_MODEL

        prompt = TRANSCRIBE_PROMPT
        if req.language:
            prompt += f" The speech is in language '{req.language}'."
        if req.prompt:
            prompt += f" Context and vocabulary hints: {req.prompt}"

        audio_block: dict[str, Any] = {"type": "audio", "mime_type": mime}
        if path.stat().st_size <= MAX_INLINE_AUDIO_BYTES:
            audio_block["data"] = base64.b64encode(path.read_bytes()).decode("ascii")
        else:
            uploaded = self._call("file upload", self._client.files.upload, file=str(path))
            audio_block["uri"] = uploaded.uri
            audio_block["mime_type"] = uploaded.mime_type or mime

        interaction = self._call(
            "stt",
            self._client.interactions.create,
            model=model,
            input=[{"type": "text", "text": prompt}, audio_block],
        )
        text = _get(interaction, "output_text")
        if text is None:
            raise ProviderError("google stt returned no text")
        return Transcript(text=str(text).strip(), language=req.language)

    # -------------------------------------------------------------------- voices
    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice:
        if cfg.type == "designed":
            voice = self._voice_input(cfg, "prompted")
            voice["prompted"] = {"input": cfg.description}
        elif cfg.type == "cloned":
            voice = self._voice_input(cfg, "replicated")
            consent = cfg.provider_options.get("consent_audio")
            if not consent:
                raise ConfigError(
                    f"voice '{cfg.name}': google voice replication needs a consent recording; "
                    "set provider_options.consent_audio to its path"
                )
            voice["replicated"] = {
                "source_audio": self._audio_data(Path(cfg.reference_audio)),
                "consent_audio": self._audio_data(Path(str(consent))),
            }
        else:
            raise ConfigError(f"voice '{cfg.name}' is prebuilt; nothing to create")

        created = self._call(
            "voice create", self._client.voices.create, voice=voice, store=cfg.store
        )
        remote_id = _get(created, "id") or _get(created, "key")
        if not remote_id:
            raise ProviderError("google voice create returned no voice id or key")
        expires_at = _get(created, "expire_time")
        if not isinstance(expires_at, datetime):
            ttl = STORED_VOICE_TTL if cfg.store else STATELESS_VOICE_TTL
            expires_at = datetime.now(UTC) + ttl
        return CreatedVoice(remote_id=str(remote_id), expires_at=expires_at)

    def _voice_input(self, cfg: VoiceConfig, voice_type: str) -> dict[str, Any]:
        model = cfg.model if cfg.model and cfg.model != PREVIEW_TTS_MODEL else TTS_MODELS[0]
        voice: dict[str, Any] = {"type": voice_type, "model": model, "display_name": cfg.name}
        if cfg.language:
            voice["language_code"] = cfg.language
        for key in VOICE_INPUT_OPTION_KEYS:
            if key in cfg.provider_options:
                voice[key] = cfg.provider_options[key]
        return voice

    @staticmethod
    def _audio_data(path: Path) -> dict[str, str]:
        if not path.is_file():
            raise ConfigError(f"audio file not found: {path}")
        mime = AUDIO_MIME_BY_SUFFIX.get(path.suffix.lower(), "audio/wav")
        return {"mime_type": mime, "data": base64.b64encode(path.read_bytes()).decode("ascii")}

    def delete_voice(self, remote_id: str) -> None:
        if remote_id.startswith("voicekey_"):
            # Stateless keys live only on the client; there is nothing to delete remotely.
            return
        self._call("voice delete", self._client.voices.delete, id=remote_id)

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        """List catalog/custom voices.

        Filters: language_code (alias language), gender, pitch, contexts, type_ (alias type),
        accent, persona, region_code (str or list), search, page_size, page_token, limit.
        With `page_token`, only that page is fetched; otherwise pages are followed until
        exhausted or `limit` voices were collected.
        """
        kwargs: dict[str, Any] = {}
        limit: int | None = None
        single_page = False
        for raw_key, value in filters.items():
            if value is None or value == [] or value == "":
                continue
            key = FILTER_ALIASES.get(raw_key, raw_key)
            if key in LIST_FILTER_KEYS:
                kwargs[key] = _as_list(value)
            elif key == "search":
                kwargs["search"] = str(value)
            elif key == "page_size":
                kwargs["page_size"] = int(value)
            elif key == "page_token":
                kwargs["page_token"] = str(value)
                single_page = True
            elif key == "limit":
                limit = int(value)
            else:
                raise ConfigError(f"unknown google voice library filter '{raw_key}'")

        voices: list[LibraryVoice] = []
        while True:
            page = self._call("voice list", self._client.voices.list, **kwargs)
            voices.extend(self._library_voice(v) for v in _get(page, "voices") or [])
            token = _get(page, "next_page_token")
            if single_page or not token or (limit is not None and len(voices) >= limit):
                break
            kwargs["page_token"] = token
        return voices[:limit] if limit is not None else voices

    @staticmethod
    def _library_voice(v: Any) -> LibraryVoice:
        voice_id = _get(v, "id") or _get(v, "display_name") or ""
        extra: dict[str, Any] = {}
        for key in (
            "type",
            "language_code",
            "gender",
            "pitch",
            "accent",
            "persona",
            "context",
            "region_code",
        ):
            value = _get(v, key)
            if value is not None:
                extra[key] = value
        expire = _get(v, "expire_time")
        if expire is not None:
            extra["expire_time"] = expire.isoformat() if isinstance(expire, datetime) else expire
        return LibraryVoice(
            id=str(voice_id),
            name=str(_get(v, "display_name") or voice_id),
            description=_get(v, "description"),
            extra=extra,
        )
