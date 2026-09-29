# voice_playground

`vp` is a command-line playground for trying text-to-speech (TTS) and speech-to-text (STT)
models from **Google Gemini**, **OpenAI**, and **ElevenLabs**. It can:

- synthesize speech with built-in voices, voices designed from a text description, or voices
  cloned from your own recordings,
- transcribe audio files,
- play the result on your computer's speakers, or on a **Sonos** speaker through **Home
  Assistant**.

Providers are plug-ins: adding a new one means writing one module (see
[Adding a provider](#adding-a-provider)).

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 for you)
- [ffmpeg](https://ffmpeg.org/) for MP3 conversion, Sonos playback, and `ffplay`
  (macOS: `brew install ffmpeg`; Debian/Ubuntu: `apt install ffmpeg`)
- On macOS, local playback uses the built-in `afplay`. Elsewhere it uses `ffplay`.
- An API key for each provider you want to use.

## Setup

```bash
uv sync
cp .env.example .env      # then fill in the keys you have
uv run vp --help
uv run vp providers       # which providers are usable, and whether their key is set
```

`.env` is gitignored. Never commit real keys. `vp` never prints key values: `vp providers`
only says `yes`/`no`.

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key (https://aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | OpenAI API key |
| `ELEVENLABS_API_KEY` | ElevenLabs API key |
| `HA_URL` | Home Assistant URL, e.g. `http://homeassistant.local:8123` |
| `HA_TOKEN` | Home Assistant long-lived access token |
| `HA_SONOS_ENTITY` | Default Sonos entity, e.g. `media_player.living_room` |
| `VP_SERVE_HOST` | Optional: the LAN IP the Sonos should use to reach this computer |
| `VP_SERVE_PORT` | Port for the temporary audio server (`0` = random free port) |
| `VP_VOICES_DIR` | Voice config directory (default `voices`) |
| `VP_CACHE_DIR` | Cache directory for created voice ids (default `.vp_cache`) |

Keep comments in `.env` on their own lines: `KEY=  # note` sets `KEY` to `# note`.

## Usage overview

```bash
# Text-to-speech. With no -o it plays on the local speakers.
vp tts --provider google --text "Hello there"
vp tts --voice narrator --text "Once upon a time..."           # voice config (voices/narrator.yaml)
vp tts --provider openai --voice coral --text "Hi" -o out.mp3   # write a file (format from extension)
vp tts --provider openai --text-file script.txt -o out.wav --play local   # write AND play
echo "Piped text" | vp tts --provider elevenlabs -o piped.wav   # text from stdin
vp tts --voice narrator --text "Hi" --play sonos                # play on Sonos via Home Assistant
vp tts --voice narrator --text "Hi" --speaker media_player.kitchen   # a specific speaker

# Speech-to-text. Prints the transcript, or writes it with -o.
vp stt --provider openai --input clip.m4a
vp stt --provider elevenlabs --input clip.wav --language en -o transcript.txt
vp stt --provider google --input clip.mp3 --prompt "Kubernetes, kubectl, etcd"

# Voices
vp voices list [--provider google]      # voice configs in ./voices
vp voices show narrator                 # resolved config + cache status / remote id
vp voices create storyteller [--force]  # create a designed/cloned voice now and cache its id
vp voices delete storyteller            # delete the remote voice and its cache entry
vp voices library --provider elevenlabs --search warm --language en --gender female

# Utilities
vp providers                            # capabilities, default models, key present?
vp play out.wav [--play sonos] [--speaker media_player.kitchen]
vp --debug tts ...                      # full traceback for unexpected errors (or VP_DEBUG=1)
```

### Rules worth knowing

- **Precedence:** CLI flag > voice config field > provider default (for model, style, ...).
- `--voice X` loads `voices/X.yaml` if it exists. Otherwise `X` is passed to the provider as a
  raw voice name/id (`Kore`, `coral`, an ElevenLabs voice_id), and then `--provider` is required.
- With no `--voice`, each provider uses a default voice: google `Kore`, openai `coral`,
  elevenlabs `JBFqnCBsd6RMkjVDRZzb` (George).
- If a voice config sets `provider` and you pass a different `--provider`, `vp` exits with an
  error. If you omit `--provider`, the config's provider is used.
- Text comes from `--text`, else `--text-file`, else stdin (when piped).
- With `-o`, the file is written and not played, unless you also pass `--play`/`--speaker`.
  The format comes from the extension (`.wav`, `.mp3`, `.pcm`) unless `--format` is set.
  Whatever format a provider returns, the output is converted to the format you asked for.
- A voice config's `sample_rate` is passed to the provider. Providers that can't honor it
  (OpenAI is always 24 kHz) produce a warning; audio is not resampled.
- **Exit codes:** `0` ok, `1` provider/runtime error, `2` usage/config error, `130` Ctrl-C.
  Errors are a single `error: ...` line on stderr and never contain API keys.

## Providers

### Google Gemini

| | |
|---|---|
| TTS models | `gemini-3.8-flash-tts` (default), `gemini-3.8-flash-lite-tts`, `gemini-3.1-flash-tts-preview` |
| STT model | `gemini-3.8-flash` (audio understanding with a transcription prompt) |
| Voices | prebuilt (e.g. `Kore`, `Puck`, `Charon`), designed, cloned, library |
| Sample rates | 24000 (default), 16000, 8000. The 3.1 preview only does 24000 and no custom voices. |

```bash
vp tts --provider google --voice Kore --text "Hello from Gemini" -o gemini.wav
vp tts --provider google --model gemini-3.8-flash-lite-tts --voice Puck --text "Quick one"
vp tts --provider google --voice Kore --style "whispered, urgent" --text "They're here."
vp stt --provider google --input gemini.wav
vp voices library --provider google --language en-US --gender female
vp tts --voice storyteller --text "A designed voice, created on first use"
```

**Style and inline tags.** Gemini takes a free-text delivery style, from `--style` or the voice
config's `style` field (sent as `speech_metadata.style`). You can also put expressive tags
inline in the text; they are passed through untouched:

```bash
vp tts --voice narrator --text "Well <sigh> I suppose so. <short pause> Fine, let's go."
```

**Custom voices.** A `designed` voice is created with `voices.create(type="prompted")` from its
`description`. A `cloned` voice uses `voices.create(type="replicated")` and **needs two
recordings**: `reference_audio` (the voice to clone) and `consent_audio` (the speaker saying
they consent to the clone). `store: true` creates a stored `voice_...` id that lasts 1 year;
`store: false` creates a stateless `voicekey_...` that lasts 7 days.

```bash
# record voices/audio/me.wav and voices/audio/me-consent.wav, then
cp voices/me-clone.yaml.example voices/me-clone.yaml
vp tts --voice me-clone --text "Hello from my clone"
```

### OpenAI

| | |
|---|---|
| TTS models | `gpt-4o-mini-tts` (default; supports `instructions`), `tts-1`, `tts-1-hd` |
| STT models | `gpt-transcribe` (default), `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, `whisper-1` |
| Voices | built-in only: `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`, `verse`, `marin`, `cedar` |
| Sample rate | always 24 kHz |

```bash
vp tts --provider openai --voice marin --text "Hello from OpenAI" -o openai.mp3
vp tts --voice openai-coral --text "Styled via instructions"
vp tts --provider openai --model tts-1-hd --voice onyx --text "Classic model"   # --style ignored with a warning
vp stt --provider openai --input openai.mp3 --language en --prompt "voice_playground"
vp stt --provider openai --model whisper-1 --input clip.m4a
vp voices library --provider openai --search ce     # static list of built-in voices
```

`style` maps to `instructions` on `gpt-4o-mini-tts`. OpenAI has no voice design or cloning:
`designed`/`cloned` voice configs for openai fail with an "unsupported" error (exit 2).
`provider_options` may contain `speed` (0.25-4.0).

### ElevenLabs

| | |
|---|---|
| TTS models | `eleven_v3` (default), `eleven_multilingual_v2`, `eleven_flash_v2_5` |
| STT model | `scribe_v2` |
| Voices | premade/library voice_ids, Voice Design, Instant Voice Clone, library search |

```bash
vp tts --provider elevenlabs --voice JBFqnCBsd6RMkjVDRZzb --text "Hello from ElevenLabs"
vp tts --voice eleven-rachel --text "Voice settings from the config" -o rachel.wav
vp stt --provider elevenlabs --input rachel.wav --language en
vp voices library --provider elevenlabs --search narrator --gender male
```

Custom voices: a `designed` voice uses Voice Design (a preview is generated from the
`description`, then saved as a voice). A `cloned` voice uses Instant Voice Clone from
`reference_audio`. `provider_options` keys: `stability`, `similarity_boost`, `style`, `speed`,
`use_speaker_boost` (voice settings), `seed`, `language_code`, `apply_text_normalization`
(request), `design_model_id` (Voice Design), `remove_background_noise` (cloning).

## Voice configs

A voice config is `voices/<name>.yaml`. Use it with `--voice <name>`.

```yaml
name: narrator                 # required; must match the file name
provider: google               # google | openai | elevenlabs (optional for prebuilt + --provider)
model: gemini-3.8-flash-tts    # optional; --model overrides it
type: prebuilt                 # prebuilt | designed | cloned
style: "calm, measured audiobook delivery"   # google speech_metadata.style / openai instructions
language: en-US                # optional language hint
sample_rate: 24000             # optional; passed to the provider (warning if not honored)
provider_options: {}           # passed through to the provider as-is
```

Fields by `type` (unknown fields are rejected, and errors name the file):

| Type | Required | Optional type-specific fields |
|---|---|---|
| `prebuilt` | `voice`: the provider voice name/id (`Kore`, `coral`, an ElevenLabs voice_id) | |
| `designed` | `description`: natural-language description of the voice | `store` (google, default `true`) |
| `cloned` | `reference_audio`: path to a recording (relative to the repo root) | `consent_audio` (consent recording; **required by google**), `store` (google, default `true`) |

Examples in `voices/`: `narrator.yaml` (google prebuilt + style), `storyteller.yaml` (google
designed), `openai-coral.yaml`, `eleven-rachel.yaml`, and `me-clone.yaml.example` (google
cloned; copy it to `me-clone.yaml` once you have recordings). Only `*.yaml`/`*.yml` files are
loaded.

`voices/audio/` is gitignored: recordings of real voices are personal data.

### The voice cache (`.vp_cache/voices.json`)

Designed and cloned voices are created remotely on first use (or with `vp voices create`),
and their ids are cached in `.vp_cache/voices.json` (gitignored). Each entry stores
`provider`, `remote_id`, `config_hash`, `created_at`, and `expires_at`.

- `--no-auto-create` makes a missing voice an error instead of creating it.
- The hash covers the fields that define the remote voice: `provider`, `type`, `store`,
  `description`, and the audio file paths plus a hash of their contents. If you edit one of
  them (or re-record the audio), the cached voice is **stale**: `vp` warns and recreates it.
  Changing `style`, `model`, or `language` doesn't recreate anything.
- Expired entries (Google: 1 year stored / 7 days stateless) are recreated automatically.
- `vp voices show NAME` shows the cache status: `valid`, `missing`, `stale`, `expired`, or
  `provider_mismatch`.
- `vp voices create NAME --force` recreates the voice even if the cache is valid. The old
  remote voice is not deleted; use `vp voices delete NAME` first if you want that.

## Sonos via Home Assistant

`vp` sends audio to Sonos by starting a temporary HTTP server on your computer and asking Home
Assistant to play its URL (`media_player.play_media`). The server serves one MP3 at an
unguessable URL, then shuts down once the file has been fetched and played (at most 120 s).

1. In Home Assistant, open your **profile page** (click your user name at the bottom left),
   go to **Security**, and under **Long-lived access tokens** click **Create token**. Put it
   in `.env` as `HA_TOKEN`.
2. Set `HA_URL` (e.g. `http://homeassistant.local:8123`) and `HA_SONOS_ENTITY` (the Sonos
   entity id from *Settings > Devices & services > Entities*, e.g. `media_player.living_room`).
3. **macOS firewall:** if the firewall is on, allow incoming connections for Python (System
   Settings > Network > Firewall > Options, or accept the prompt the first time). Otherwise
   the Sonos can't fetch the file and `vp` reports that the file was never fetched.
4. `vp` detects this computer's LAN IP automatically. If it picks the wrong interface (VPN,
   several networks), set `VP_SERVE_HOST` to the IP the Sonos can reach. Set `VP_SERVE_PORT`
   if you need a fixed port for a firewall rule.

```bash
vp tts --voice narrator --text "Dinner is ready" --play sonos
vp tts --voice narrator --text "Dinner is ready" --speaker media_player.kitchen
vp play out.wav --play sonos
```

## Adding a provider

1. Create `src/voice_playground/providers/<name>.py` with a subclass of `BaseProvider`:

   ```python
   from typing import ClassVar

   from voice_playground.providers.base import (
       AudioResult, BaseProvider, Capability, STTRequest, Transcript, TTSRequest,
   )
   from voice_playground.settings import require_secret


   class AcmeProvider(BaseProvider):
       name: ClassVar[str] = "acme"
       capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.TTS, Capability.STT})
       default_tts_model: ClassVar[str | None] = "acme-tts-1"
       default_stt_model: ClassVar[str | None] = "acme-stt-1"

       def __init__(self, settings):
           super().__init__(settings)
           self._key = require_secret(settings.acme_api_key, "ACME_API_KEY")  # ConfigError if missing

       def tts(self, req: TTSRequest) -> AudioResult: ...
       def stt(self, req: STTRequest) -> Transcript: ...
   ```

   Anything you don't override (`create_voice`, `delete_voice`, `list_library`, ...) raises
   `UnsupportedCapability`. Return audio with the right `mime_type` (`audio/wav`,
   `audio/mpeg`, `audio/l16`); the service converts it to the requested format. Map SDK
   exceptions to `ProviderError` with a short message that never includes the key.
2. Register it with one line in `providers/registry.py`:
   `"acme": "voice_playground.providers.acme:AcmeProvider",` in `PROVIDERS`, and add
   `"acme": "ACME_API_KEY"` to `API_KEY_ENV` (use `None` if no key is needed).
3. Add the key field to `Settings` (`acme_api_key: SecretStr | None = None`) and to
   `.env.example`, and a default voice in `service.DEFAULT_VOICES`.
4. Add tests: `tests/unit/test_acme.py` with the SDK mocked (request payloads, error mapping,
   no key in messages), and `tests/live/test_acme_live.py` marked `@pytest.mark.live` (it is
   skipped automatically when `ACME_API_KEY` isn't set).

The registry imports providers lazily, so a missing SDK or key only breaks that one provider.

## Development and testing

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -m "not live" -q     # unit tests: no network, no keys, no audio playback
uv run pytest -m live -v           # live API smoke tests; each runs only if its key is in .env
uv run pytest -m live -v tests/live/test_openai_live.py   # one provider
```

Live tests cost a little money (a few short TTS/STT calls per provider). Unit tests never read
your `.env`. The `fake` provider (`--provider fake`) needs no key or network and is handy for
trying the CLI:

```bash
uv run vp tts --provider fake --text hi -o /tmp/beep.wav
uv run vp stt --provider fake --input /tmp/beep.wav
```

See `PLAN.md` for the full spec and architecture.
