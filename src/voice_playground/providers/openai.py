"""OpenAI provider: TTS via `audio.speech.create`, STT via `audio.transcriptions.create`.

Verified against (2026-09-28) https://developers.openai.com/api/docs/guides/text-to-speech,
https://developers.openai.com/api/docs/guides/transcription and the installed `openai` SDK.

TTS
- Models: `gpt-4o-mini-tts` (default, newest; also the `gpt-4o-mini-tts-2025-12-15`
  snapshot), `tts-1`, `tts-1-hd`.
- `style` maps to `instructions`. The `tts-1` family does not accept `instructions`; for those
  models the style is ignored with a one-time warning on stderr.
- Output is always 24 kHz mono 16-bit. OpenAI has no sample-rate option, so a requested
  `sample_rate` is not an error: the actual rate (24000) is returned in `AudioResult` and any
  resampling is the caller's job.
- `wav` is produced by requesting `pcm` and adding a WAV header locally, because OpenAI's own
  WAV output is written for streaming and carries placeholder chunk sizes. `mp3` and `pcm`
  are requested directly.
- `provider_options` may contain `speed` (0.25-4.0). Other keys are a `ConfigError`.
- A voice id starting with `voice_` is sent as a custom voice reference `{"id": ...}`.

STT
- Models: `gpt-transcribe` (default, recommended), `gpt-4o-transcribe`,
  `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1`.
- `language` is normalised to ISO-639-1 (`en-US` -> `en`). `gpt-transcribe` takes it as
  `languages=[code]`; older models take `language=code`. `prompt` is passed through (the
  diarize model does not support it, so it is dropped with a warning there).

Voice library
- Design, cloning and deletion raise `UnsupportedCapability`. `list_library` returns the
  static list of built-in voices (OpenAI has no voice-catalog endpoint), filterable by
  `search` and `model`. `language` is accepted and ignored (every voice is multilingual);
  any other non-empty filter is a `ConfigError`.
"""

from __future__ import annotations

import io
import re
import sys
import wave
from typing import TYPE_CHECKING, Any, ClassVar

import openai

from voice_playground.errors import ConfigError, ProviderError
from voice_playground.providers.base import (
    AudioResult,
    BaseProvider,
    Capability,
    LibraryVoice,
    STTRequest,
    Transcript,
    TTSRequest,
)

if TYPE_CHECKING:
    from voice_playground.settings import Settings

#: OpenAI TTS output is always 24 kHz, mono, signed 16-bit little-endian.
OPENAI_SAMPLE_RATE = 24_000
MAX_INPUT_CHARS = 4096

#: TTS model prefixes that do not accept `instructions`.
_NO_INSTRUCTIONS_PREFIXES = ("tts-1",)
#: STT models that take `languages=[...]` instead of `language=...`.
_MULTI_LANGUAGE_STT_PREFIXES = ("gpt-transcribe", "gpt-live-transcribe")
_DIARIZE_STT_PREFIX = "gpt-4o-transcribe-diarize"

_LEGACY_TTS_MODELS = ("tts-1", "tts-1-hd")
_ALL_TTS_MODELS = ("gpt-4o-mini-tts", *_LEGACY_TTS_MODELS)

#: Built-in voices: (name, models that support it). Recommended: marin, cedar.
BUILTIN_VOICES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("alloy", _ALL_TTS_MODELS),
    ("ash", _ALL_TTS_MODELS),
    ("ballad", ("gpt-4o-mini-tts",)),
    ("coral", _ALL_TTS_MODELS),
    ("echo", _ALL_TTS_MODELS),
    ("fable", _ALL_TTS_MODELS),
    ("nova", _ALL_TTS_MODELS),
    ("onyx", _ALL_TTS_MODELS),
    ("sage", _ALL_TTS_MODELS),
    ("shimmer", _ALL_TTS_MODELS),
    ("verse", ("gpt-4o-mini-tts",)),
    ("marin", ("gpt-4o-mini-tts",)),
    ("cedar", ("gpt-4o-mini-tts",)),
)
_RECOMMENDED_VOICES = frozenset({"marin", "cedar"})

_ALLOWED_OPTIONS = frozenset({"speed"})
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*.]+")


def supports_instructions(model: str) -> bool:
    """True if the TTS `model` accepts `instructions` (everything except the tts-1 family)."""
    return not model.startswith(_NO_INSTRUCTIONS_PREFIXES)


def _iso639_1(language: str) -> str:
    """`en-US` / `en_us` / `EN` -> `en`."""
    return re.split(r"[-_]", language.strip(), maxsplit=1)[0].lower()


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class OpenAIProvider(BaseProvider):
    name: ClassVar[str] = "openai"
    capabilities: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.TTS, Capability.STT, Capability.VOICE_LIBRARY}
    )
    default_tts_model: ClassVar[str | None] = "gpt-4o-mini-tts"
    default_stt_model: ClassVar[str | None] = "gpt-transcribe"
    api_key_env: ClassVar[str | None] = "OPENAI_API_KEY"
    default_voice: ClassVar[str | None] = "coral"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key = self._require_api_key()
        self._client = openai.OpenAI(api_key=self._api_key)
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ helpers

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"warning: {message}", file=sys.stderr)

    def _scrub(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return _SECRET_PATTERN.sub("sk-***", text)

    def _provider_error(self, action: str, exc: Exception) -> ProviderError:
        """Short, key-free one-line message for an SDK exception."""
        if isinstance(exc, openai.APIStatusError):
            reason = ""
            body = exc.body
            if isinstance(body, dict):
                err = body.get("error", body)
                if isinstance(err, dict):
                    reason = str(err.get("message") or err.get("code") or "")
            reason = reason or exc.message or type(exc).__name__
            detail = f"HTTP {exc.status_code}: {reason}"
        elif isinstance(exc, openai.APITimeoutError):
            detail = "request timed out"
        elif isinstance(exc, openai.APIConnectionError):
            detail = "could not connect to the OpenAI API"
        else:
            detail = f"{type(exc).__name__}: {exc}"
        detail = self._scrub(" ".join(detail.split()))
        return ProviderError(f"openai {action} failed: {detail}")

    # ---------------------------------------------------------------------- TTS

    def _tts_kwargs(self, req: TTSRequest, model: str) -> dict[str, Any]:
        if not req.text.strip():
            raise ConfigError("openai tts: text is empty")
        if len(req.text) > MAX_INPUT_CHARS:
            raise ConfigError(
                f"openai tts: text is {len(req.text)} characters; the limit is {MAX_INPUT_CHARS}"
            )
        unknown = sorted(set(req.voice.options) - _ALLOWED_OPTIONS)
        if unknown:
            raise ConfigError(
                f"openai tts: unsupported provider_options {unknown}; "
                f"allowed: {sorted(_ALLOWED_OPTIONS)}"
            )

        voice_id = req.voice.provider_voice
        voice: str | dict[str, str] = (
            {"id": voice_id} if voice_id.startswith("voice_") else voice_id
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "voice": voice,
            "input": req.text,
            # wav is built locally from pcm (see module docstring).
            "response_format": "mp3" if req.output_format == "mp3" else "pcm",
        }
        if "speed" in req.voice.options:
            kwargs["speed"] = float(req.voice.options["speed"])
        if req.voice.style:
            if supports_instructions(model):
                kwargs["instructions"] = req.voice.style
            else:
                self._warn_once(
                    f"instructions:{model}",
                    f"openai model '{model}' does not support style instructions; ignoring style",
                )
        return kwargs

    def tts(self, req: TTSRequest) -> AudioResult:
        model = req.model or self.default_tts_model or "gpt-4o-mini-tts"
        kwargs = self._tts_kwargs(req, model)
        try:
            response = self._client.audio.speech.create(**kwargs)
            data = response.content
        except openai.OpenAIError as exc:
            raise self._provider_error("tts", exc) from exc

        if req.output_format == "mp3":
            return AudioResult(data=data, mime_type="audio/mpeg", sample_rate=OPENAI_SAMPLE_RATE)
        if req.output_format == "pcm":
            return AudioResult(data=data, mime_type="audio/l16", sample_rate=OPENAI_SAMPLE_RATE)
        return AudioResult(
            data=_pcm_to_wav(data, OPENAI_SAMPLE_RATE),
            mime_type="audio/wav",
            sample_rate=OPENAI_SAMPLE_RATE,
        )

    # ---------------------------------------------------------------------- STT

    def _stt_kwargs(self, req: STTRequest, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model}
        if req.language:
            code = _iso639_1(req.language)
            if model.startswith(_MULTI_LANGUAGE_STT_PREFIXES):
                kwargs["languages"] = [code]
            else:
                kwargs["language"] = code
        if model.startswith(_DIARIZE_STT_PREFIX):
            kwargs["chunking_strategy"] = "auto"
            if req.prompt:
                self._warn_once(
                    f"prompt:{model}", f"openai model '{model}' does not support prompt; ignoring"
                )
        elif req.prompt:
            kwargs["prompt"] = req.prompt
        return kwargs

    def stt(self, req: STTRequest) -> Transcript:
        model = req.model or self.default_stt_model or "gpt-transcribe"
        kwargs = self._stt_kwargs(req, model)
        try:
            fh = req.audio_path.open("rb")
        except OSError as exc:
            raise ConfigError(f"cannot read audio file {req.audio_path}: {exc.strerror}") from exc
        try:
            with fh:
                result = self._client.audio.transcriptions.create(file=fh, **kwargs)
        except openai.OpenAIError as exc:
            raise self._provider_error("stt", exc) from exc

        if isinstance(result, str):
            return Transcript(text=result.strip(), language=req.language)
        text = str(getattr(result, "text", "") or "").strip()
        language = req.language
        detected = getattr(result, "languages", None) or []
        if detected and getattr(detected[0], "code", None):
            language = str(detected[0].code)
        elif isinstance(getattr(result, "language", None), str):
            language = result.language
        segments = getattr(result, "segments", None)
        seg_dicts = [s.model_dump() for s in segments] if segments else None
        return Transcript(text=text, language=language, segments=seg_dicts)

    # ------------------------------------------------------------ voice library

    def list_library(self, **filters: Any) -> list[LibraryVoice]:
        search = str(filters.pop("search", None) or "").lower()
        model = filters.pop("model", None)
        if model and str(model).startswith("gpt-4o-mini-tts"):
            model = "gpt-4o-mini-tts"  # dated snapshots share the alias's voices
        filters.pop("language", None)  # every OpenAI voice is multilingual
        unsupported = sorted(k for k, v in filters.items() if v not in (None, "", [], ()))
        if unsupported:
            raise ConfigError(
                f"openai voice library does not support filters: {', '.join(unsupported)} "
                "(supported: search, model, language)"
            )
        voices = []
        for name, models in BUILTIN_VOICES:
            if search and search not in name:
                continue
            if model and model not in models:
                continue
            description = "Built-in voice" + (
                " (recommended)" if name in _RECOMMENDED_VOICES else ""
            )
            voices.append(
                LibraryVoice(
                    id=name,
                    name=name,
                    description=description,
                    extra={"models": list(models), "recommended": name in _RECOMMENDED_VOICES},
                )
            )
        return voices
