# voice_playground — Build Plan

A CLI for trying out text-to-speech (TTS) and speech-to-text (STT) models from different providers.
The first providers are **Google Gemini**, **OpenAI**, and **ElevenLabs**. Adding a new provider means adding one module.

> Anthropic has no public TTS/STT API. It was replaced by ElevenLabs for the first set of providers.
> The provider interface is based on capabilities, so an LLM or "voice agent" stage can be added later without redesign.

---

## 1. Decisions (confirmed with the user)

| Topic | Decision |
|---|---|
| Stack | Python 3.13 (pinned with `uv python pin 3.13`), `uv`, **Typer** CLI, **pydantic** / pydantic-settings, `python-dotenv`, pytest, ruff, mypy |
| Providers (first set) | `google`, `openai`, `elevenlabs` |
| Gemini default model | `gemini-3.8-flash-tts`. `gemini-3.8-flash-lite-tts` and `gemini-3.1-flash-tts-preview` can be picked with `--model` |
| Custom voices | Voice config YAML files. Three types: **prebuilt + style**, **designed** (from a text description), **cloned** (from reference audio) |
| Secrets | `.env` (gitignored). `.env.example` is committed |
| Playback | With no `-o`: play locally by default. `--play sonos` (or `--speaker <entity_id>`) sends the audio to Sonos through Home Assistant |
| Sonos transport | A temporary local HTTP server serves the file on the LAN. Home Assistant `media_player.play_media` is pointed at that URL |
| Verification | Mocked unit tests are required. Live smoke tests (`@pytest.mark.live`) run only when the matching key is in `.env` |

---

## 2. CLI specification

Entry point: `vp` (defined in `[project.scripts]`, e.g. `vp = "voice_playground.cli:app"`). The mode (TTS or STT) is the subcommand.

```bash
# TTS
vp tts --provider google --model gemini-3.8-flash-tts --voice narrator \
       --text "Hello there" [--text-file script.txt] [-o out.wav] \
       [--style "whispered, urgent"] [--format wav|mp3|pcm] \
       [--play local|sonos] [--speaker media_player.living_room]

# STT
vp stt --provider openai --model gpt-4o-transcribe --input clip.m4a \
       [-o transcript.txt] [--language en] [--prompt "domain words"]

# Voices
vp voices list [--provider google]          # voice configs in ./voices
vp voices show narrator                     # resolved config + cached remote id
vp voices create narrator [--force]         # create designed/cloned voice remotely, cache its id
vp voices library --provider google [--search warm --language en-US --gender female]  # remote catalog
vp voices delete narrator                   # delete the remote voice + cache entry

# Utilities
vp providers                                # providers, capabilities, default models, key present? (yes/no, never the key)
vp play out.wav [--play sonos] [--speaker ...]
```

### Resolution rules (must be implemented exactly)
1. **Precedence:** CLI flag > voice config field > provider default.
2. `--voice X`: if `voices/X.yaml` exists, load it. If not, pass `X` to the provider as a raw provider voice name (e.g. `Kore`, `coral`, or an ElevenLabs voice_id).
3. If the voice config sets `provider` and `--provider` is a different value, exit with code 2 and a clear message. If `--provider` is omitted, use the config's provider.
4. For `designed` or `cloned` voices with no cached remote id, create the voice automatically on first use and cache the id. `--no-auto-create` makes this an error instead.
5. If neither `--text` nor `--text-file` is given, read stdin when it isn't a TTY. Otherwise exit with code 2.
6. When `-o` is given, write the file and don't play it unless `--play` is also passed. The output format comes from the file extension unless `--format` is set.
7. Exit codes: `0` ok, `1` provider/runtime error, `2` usage/config error. Errors are one line on stderr and never contain API keys.

---

## 3. Configuration

### `.env` (gitignored) / `.env.example` (committed)
```dotenv
GOOGLE_API_KEY=
OPENAI_API_KEY=
ELEVENLABS_API_KEY=

HA_URL=http://homeassistant.local:8123
HA_TOKEN=                       # long-lived access token
HA_SONOS_ENTITY=media_player.living_room
VP_SERVE_HOST=                  # optional: override the detected LAN IP
VP_SERVE_PORT=0                 # 0 = random free port
VP_VOICES_DIR=voices
```

### Voice config — `voices/<name>.yaml`
```yaml
name: narrator                 # must match the filename
provider: google               # google | openai | elevenlabs
model: gemini-3.8-flash-tts    # optional
type: prebuilt                 # prebuilt | designed | cloned
voice: Kore                    # prebuilt: provider voice name/id
description: >-                # designed: natural-language voice description
  Warm, engaging narrator with a slight British accent
reference_audio: voices/audio/me.wav   # cloned: path relative to repo root
store: true                    # google: stateful voice_ (1y) vs stateless voicekey_ (7d)
style: "calm, measured audiobook delivery"   # google speech_metadata.style / openai instructions
language: en-US
sample_rate: 24000
provider_options: {}           # passed through as-is (e.g. elevenlabs stability, similarity_boost, speed)
```
- Validated with a pydantic model and a discriminated union on `type`. Each type has its own required fields: `voice` for prebuilt, `description` for designed, `reference_audio` for cloned.
- Remote ids of created voices go in `.vp_cache/voices.json` (gitignored). Each entry is keyed by `name` and stores `{provider, remote_id, config_hash, created_at, expires_at}`. If the config hash changes, the voice is stale: warn and recreate on next use (with `--force` behavior).
- `voices/audio/` is gitignored because reference recordings are personal data. Commit example configs only.

### Capability matrix (the first provider tasks must fill this in; agents verify every model ID against current docs)
| Provider | TTS models (default first) | STT models | prebuilt | designed | cloned |
|---|---|---|---|---|---|
| google | `gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`, `gemini-3.1-flash-tts-preview` | current Gemini Flash multimodal model (audio understanding, verify ID) | ✅ | ✅ `voices.create(type="prompted")` | ✅ `voices.create(type="replicated")` |
| openai | `gpt-4o-mini-tts` (supports `instructions`), `tts-1`, `tts-1-hd` | `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1` (+ newer if present) | ✅ | ❌ → `UnsupportedCapability` | ❌ → `UnsupportedCapability` |
| elevenlabs | `eleven_v3`, `eleven_multilingual_v2`, `eleven_flash_v2_5` | `scribe_v1` / latest Scribe | ✅ | ✅ Voice Design | ✅ Instant Voice Clone |

---

## 4. Architecture

```
pyproject.toml  .python-version  .env.example  .gitignore  README.md  CLAUDE.md  PLAN.md
voices/                         # example voice configs (committed); voices/audio/ gitignored
src/voice_playground/
  __init__.py
  cli.py                        # Typer app; thin; delegates to services
  settings.py                   # pydantic-settings, loads .env
  errors.py                     # VPError, ConfigError(exit 2), ProviderError(exit 1), UnsupportedCapability
  audio.py                      # AudioResult, pcm->wav, format conversion via ffmpeg, write_output
  voices.py                     # VoiceConfig models, load/list, VoiceCache
  service.py                    # orchestration: resolve voice/provider/model, run tts/stt, handle output/playback
  providers/
    base.py                     # Provider protocol + request/response dataclasses
    registry.py                 # name -> lazy import path; get_provider(name, settings)
    fake.py                     # deterministic fake provider used by tests
    google.py  openai.py  elevenlabs.py
  playback/
    local.py                    # afplay (macOS) → ffplay → error
    sonos.py                    # temp HTTP server + Home Assistant REST call
tests/
  conftest.py                   # fixtures: tmp voices dir, fake settings, live-marker skip logic
  unit/...  live/...
```

### Core contracts (`providers/base.py`), written in T1 and frozen after
```python
@dataclass(frozen=True)
class TTSRequest:
    text: str
    model: str | None
    voice: ResolvedVoice          # provider voice name OR remote custom-voice id + style/lang/options
    output_format: Literal["wav", "mp3", "pcm"] = "wav"
    sample_rate: int | None = None

@dataclass(frozen=True)
class AudioResult:
    data: bytes
    mime_type: str               # "audio/wav", "audio/mpeg", "audio/l16"
    sample_rate: int | None
    channels: int = 1

@dataclass(frozen=True)
class STTRequest:
    audio_path: Path
    model: str | None
    language: str | None = None
    prompt: str | None = None

@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    segments: list[dict] | None = None

class Provider(Protocol):
    name: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]   # TTS, STT, VOICE_DESIGN, VOICE_CLONE, VOICE_LIBRARY
    default_tts_model: ClassVar[str | None]
    default_stt_model: ClassVar[str | None]
    def tts(self, req: TTSRequest) -> AudioResult: ...
    def stt(self, req: STTRequest) -> Transcript: ...
    def create_voice(self, cfg: VoiceConfig) -> CreatedVoice: ...      # remote_id, expires_at
    def delete_voice(self, remote_id: str) -> None: ...
    def list_library(self, **filters) -> list[LibraryVoice]: ...
```
Methods a provider doesn't support raise `UnsupportedCapability`. The registry maps names to import strings, e.g. `{"google": "voice_playground.providers.google:GoogleProvider", ...}`, so a missing SDK or key only breaks that one provider.

**To add a provider:** create `providers/<name>.py`, add one line to `registry.py`, add tests. That's all. The README must document this.

### Sonos / Home Assistant flow (`playback/sonos.py`)
1. Convert the audio to MP3 (ffmpeg) for the best Sonos compatibility.
2. Start a `ThreadingHTTPServer` on `0.0.0.0:VP_SERVE_PORT` in a background thread. It serves exactly one file at an unguessable path (`/<secrets.token_urlsafe(16)>.mp3`) and returns 404 for anything else.
3. LAN IP = `VP_SERVE_HOST`, or else the UDP-connect trick (`socket.connect(("8.8.8.8", 80))`, no packets sent).
4. `POST {HA_URL}/api/services/media_player/play_media` with Bearer `HA_TOKEN` and JSON body `{"entity_id": ..., "media_content_id": url, "media_content_type": "music"}`.
5. Keep serving until the file has been fully fetched and then (audio duration + 5 s), or until a timeout of 120 s. Then shut down.
6. Clear errors for: missing HA config, HA 401/404, the file never being fetched within 15 s (hint: macOS firewall, or the Sonos can't reach this Mac).

---

## 5. Task breakdown for sub-agents

### Waves
```
Wave 0:  T1 (scaffold + contracts)            — one agent, blocks everything
Wave 1:  T2 google | T3 openai | T4 elevenlabs | T5 audio+local playback | T6 sonos/HA | T7 voices+cache
Wave 2:  T8 service + CLI wiring + README + examples
Wave 3:  T9 review → fix loop (code-reviewer + security-auditor agents), T10 live smoke run (with user's keys)
```
Every Wave 1 task **owns a disjoint set of files**. T1 creates all module stubs, all dependencies, and the registry entries, so Wave 1 agents never edit shared files. If a Wave 1 agent believes a contract in `base.py` must change, it stops and reports `BLOCKED: contract change needed: <details>` instead of editing it.

### Global Definition of Done (applies to every task)
All of these must pass from the repo root:
```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -m "not live" -q
```
Also:
- Every acceptance criterion in the task is ticked, with evidence (test name or command output).
- No edits outside the task's owned files, except new test files under the task's test paths.
- No secrets in code, tests, fixtures, or logs. `git status` shows no `.env`, `voices/audio/*`, or `.vp_cache`.
- No skipped, xfail'd, or deleted tests, and no `# type: ignore` or `noqa` without a one-line justification.

### Retry protocol (mandatory for every agent)
1. Implement, then run the full Global DoD command set.
2. If anything fails: read the full failure, find the root cause, fix it, and run the **whole** set again, not just the failing check.
3. Repeat until everything passes. Weakening a test, loosening lint config, or catching and ignoring exceptions to make things green is **forbidden**.
4. After 6 full cycles without success, stop and report `BLOCKED` with the failing command, the full error, what you tried, and your hypothesis. Never report success with anything failing.
5. The final report must include: files changed, the acceptance checklist with evidence, and the last output of the DoD commands.

The **orchestrator** (main session) independently re-runs the DoD and spot-checks the acceptance criteria after each task. If anything fails, it sends the task back to the same agent (SendMessage) with the failure output. A task is done only when the orchestrator's own verification passes.

---

### T1 — Scaffold, contracts, fake provider, CLI skeleton
**Owns:** `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `src/voice_playground/{__init__,settings,errors}.py`, `providers/{base,registry,fake}.py`, stub `providers/{google,openai,elevenlabs}.py`, stub `playback/{local,sonos}.py`, stub `audio.py`, `voices.py`, `service.py`, `cli.py`, `tests/conftest.py`, `tests/unit/test_scaffold.py`.

**Requirements**
- uv project with a `src/` layout and Python pinned to 3.13. Deps: `typer`, `pydantic`, `pydantic-settings`, `python-dotenv`, `pyyaml`, `httpx`, `google-genai` (a version with `client.interactions` and `client.voices`), `openai`, `elevenlabs`. Dev deps: `pytest`, `pytest-mock`, `respx`, `ruff`, `mypy`, `types-PyYAML`.
- ruff config (line length 100, rule sets E,F,I,UP,B,SIM). mypy with `ignore_missing_imports = true` and `disallow_untyped_defs = true` for `voice_playground.*`.
- The pytest `live` marker is registered. `conftest.py` skips live tests unless the relevant `*_API_KEY` is present, and sets `VP_VOICES_DIR` and cache dir to tmp paths for unit tests.
- `.gitignore` covers `.env`, `.env.*` (but not `.env.example`), `.vp_cache/`, `voices/audio/`, `out/`, `*.wav`/`*.mp3` at the repo root, `.venv`, and caches.
- `base.py` implements the §4 contracts exactly. `registry.py` has all three real providers plus `fake`. Real provider stubs are classes with the correct `name` and `capabilities` whose methods raise `NotImplementedError`.
- `fake.py` implements every capability deterministically: it returns a 0.5 s 24 kHz sine WAV, echoes the file name as the transcript, and gives created voices ids like `fake_voice_<name>`.
- `cli.py` has all commands from §2 with every flag declared, and help text. Bodies may call stubbed service functions.

**Acceptance**
- [ ] `uv run vp --help`, `vp tts --help`, `vp stt --help`, and `vp voices --help` list every flag from §2.
- [ ] `get_provider("fake")` works. `get_provider("nope")` raises `ConfigError` listing the valid names.
- [ ] Tests confirm `.env` is ignored by git (`git check-ignore .env`) and `.env.example` is not.
- [ ] Global DoD passes.

### T2 — Google Gemini provider
**Owns:** `providers/google.py`, `tests/unit/test_google.py`, `tests/live/test_google_live.py`.
**Reference:** https://aistudio.google.com/docs/speech-generation. Fetch it first, since the API is new (`client.interactions.create`, `client.voices.*`). Don't rely on memory.

**Requirements**
- TTS through `client.interactions.create(..., response_format={"type":"audio", ...}, generation_config={"speech_config":[{"voice": id}]})`. `style` goes in `annotations=[{"type":"speech_metadata","style":...}]`. Inline tags like `<sigh>` pass through untouched.
- Default model `gemini-3.8-flash-tts`. It must also work with `gemini-3.8-flash-lite-tts` and `gemini-3.1-flash-tts-preview`. If the 3.1 preview needs the older `generate_content` + `speech_config` API, implement that path and choose it by model.
- Output: unary WAV (24 kHz mono s16le). Honor `sample_rate` (24000/16000/8000). `pcm` format asks for `audio/l16`.
- `create_voice`: `designed` → `type="prompted"` with description. `cloned` → `type="replicated"` with reference audio. `store` is passed through. Return `expires_at` (1 year if stored, 7 days if stateless).
- `delete_voice`, and `list_library` with filters language_code/gender/pitch/contexts/search/type_ and pagination.
- STT: send the audio file to the current Gemini Flash multimodal model with a transcription prompt (verify the model ID from the docs) and return `Transcript`.
- Map SDK errors to `ProviderError` with a short message (status + reason). Never include the key.

**Acceptance**
- [ ] Unit tests with a mocked `genai.Client` check the exact request payloads for: prebuilt+style, designed-voice id, cloned-voice id, custom sample rate, and the 3.1-preview path.
- [ ] Unit tests cover the create_voice payloads for prompted and replicated, `store` true/false, and the expires_at math.
- [ ] Unit test checks that an SDK exception becomes a `ProviderError` with no key in the message.
- [ ] Live tests (skipped without key): a 1-sentence TTS gives valid WAV bytes (`wave` module parses them, duration > 0.3 s), and a short STT of that WAV contains the expected word.
- [ ] Global DoD passes.

### T3 — OpenAI provider
**Owns:** `providers/openai.py`, `tests/unit/test_openai.py`, `tests/live/test_openai_live.py`.
**Reference:** https://developers.openai.com/api/docs/guides/text-to-speech and `/guides/transcription`. Verify the current model IDs and voices there.

**Requirements**
- TTS through `audio.speech.create`. `style` maps to `instructions`, and only for models that support it. For other models, warn once on stderr and ignore it. `response_format` is wav/mp3/pcm (OpenAI pcm is 24 kHz s16le).
- STT through `audio.transcriptions.create` with language and prompt. The file is opened in binary mode.
- Voice design, cloning, and library raise `UnsupportedCapability`. `list_library` may instead return the static list of built-in voices. Pick one and document it.

**Acceptance**
- [ ] Unit tests (mocked client) check the request kwargs for each format, the instructions handling on both model families, and STT kwargs.
- [ ] Error mapping test, same rules as T2.
- [ ] Live tests: the TTS→STT round trip contains the expected word.
- [ ] Global DoD passes.

### T4 — ElevenLabs provider
**Owns:** `providers/elevenlabs.py`, `tests/unit/test_elevenlabs.py`, `tests/live/test_elevenlabs_live.py`.
**Reference:** https://elevenlabs.io/docs. Verify the current SDK method names for TTS, Speech-to-Text (Scribe), Voice Design, Instant Voice Clone, and the voices list.

**Requirements**
- TTS with voice_id, model, and `provider_options` mapped to voice settings (stability, similarity_boost, style, speed). Output format maps to the ElevenLabs `output_format` strings. wav/pcm come from `pcm_24000` wrapped in a WAV header when needed.
- STT through Scribe with an optional language code.
- `designed` → Voice Design (generate a preview from the description, then save it as a voice). `cloned` → Instant Voice Clone from `reference_audio`. `delete_voice` and `list_library` (search/filter) are implemented.

**Acceptance**
- [ ] Unit tests (mocked SDK) cover every method's payload and the output-format mapping table.
- [ ] Error mapping test.
- [ ] Live tests: TTS→STT round trip.
- [ ] Global DoD passes.

### T5 — Audio utilities + local playback
**Owns:** `audio.py`, `playback/local.py`, `tests/unit/test_audio.py`, `tests/unit/test_local_playback.py`.

**Requirements**
- `pcm_to_wav(data, sample_rate, channels=1, sample_width=2)`, `wav_info(bytes)`, `convert(result, fmt)` using the ffmpeg CLI (clear error if ffmpeg is missing), `write_output(result, path, fmt|None)` which infers the format from the extension and creates parent dirs.
- `play_local(result)` writes a temp file and plays it with `afplay` on macOS, falling back to `ffplay -nodisp -autoexit -loglevel quiet`. It blocks until playback finishes, and Ctrl-C stops playback cleanly.

**Acceptance**
- [ ] Round trip: the `pcm_to_wav` output parses with the `wave` module and has the correct rate, channels, and frame count.
- [ ] Format inference covers `.wav/.mp3/.pcm`, and unknown extensions raise `ConfigError`.
- [ ] Player selection is tested with `shutil.which` and `subprocess` mocked.
- [ ] A conversion test runs ffmpeg for real if it's installed, and is skipped otherwise.
- [ ] Global DoD passes.

### T6 — Sonos via Home Assistant
**Owns:** `playback/sonos.py`, `tests/unit/test_sonos.py`.

**Requirements:** implement the §4 Sonos flow exactly. Use `httpx` for Home Assistant. The server must be killable and must never serve paths other than the token path.

**Acceptance**
- [ ] Test: the server serves the file at the token path, returns 404 for `/`, other paths, and `..` tricks, and shuts down after the fetch plus the grace period (use a short grace in the test).
- [ ] Test (respx): the HA request has the correct URL, the Bearer header, and the JSON body. A 401 becomes a `ProviderError` with a hint to check `HA_TOKEN`.
- [ ] Test: missing `HA_URL`, `HA_TOKEN`, or entity raises `ConfigError` naming the missing variable.
- [ ] Test: the "never fetched" timeout produces the firewall/reachability hint.
- [ ] Global DoD passes.

### T7 — Voice configs + cache
**Owns:** `voices.py`, `voices/examples` (committed samples: `narrator.yaml` google prebuilt+style, `storyteller.yaml` google designed, `me-clone.yaml.example` google cloned, `openai-coral.yaml`, `eleven-rachel.yaml`), `tests/unit/test_voices.py`.

**Requirements:** the §3 pydantic models (discriminated union), `load_voice(name)`, `list_voices()`, `VoiceCache` (atomic JSON writes, config hash, expiry check), and `resolve_voice(name_or_raw, provider_flag) -> (provider_name, ResolvedVoice, needs_create)` implementing rules 2–4 from §2.

**Acceptance**
- [ ] Tests for each type's required fields. Wrong field sets give clear errors that name the file.
- [ ] Tests for the name/filename mismatch, a provider conflict with `--provider` (ConfigError), the raw voice fallback, and stale-hash and expired cache entries.
- [ ] All committed example configs load without errors (a test globs them).
- [ ] Global DoD passes.

### T8 — Service wiring, CLI end-to-end, docs
**Owns:** `service.py`, `cli.py`, `README.md`, `tests/unit/test_cli.py`, `tests/unit/test_service.py`. Depends on T1–T7.

**Requirements:** implement every command and all 7 resolution rules in §2 on top of the finished modules. The README covers setup (`uv sync`, copying `.env.example`), per-provider examples, the voice config reference, Home Assistant/Sonos setup (long-lived token, firewall note), and a "Adding a provider" walkthrough.

**Acceptance** (all through Typer `CliRunner` with the `fake` provider plus mocked playback)
- [ ] `vp tts --provider fake --text hi -o out.wav` writes a valid WAV and doesn't call playback.
- [ ] Without `-o`, local playback is called once. With `--play sonos`, the sonos playback is called with the entity from settings or `--speaker`.
- [ ] Stdin text input works. Missing text exits 2.
- [ ] A designed voice auto-creates once, and a second run reuses the cached id. `--no-auto-create` exits 2.
- [ ] `vp stt --provider fake --input x.wav` prints the transcript. `-o` writes it to a file.
- [ ] `vp providers` shows each provider's capabilities and key status, and the output contains no key values (test with a dummy key string).
- [ ] Every exit-code rule from §2 is tested.
- [ ] Global DoD passes.

### T9 — Review and hardening
Run the `agent-skills:code-reviewer` and `agent-skills:security-auditor` agents over the whole repo. Every finding of severity medium or higher gets fixed by a fix agent (retry protocol applies), or it is recorded in `docs/known-issues.md` with a reason. **Done** when reviewers return no unresolved medium+ findings and the Global DoD passes.

### T10 — Live smoke run (after the user adds keys)
`uv run pytest -m live -v` passes for every provider whose key is present. Manual check: `vp tts --provider google --voice narrator --text "Testing one two three"` plays locally, and with `--play sonos` it plays on the Sonos.

---

## 6. Orchestration notes (for the main session)
- Launch Wave 1 as six parallel `general-purpose` agents in one message. Each prompt = "You are implementing task Tn of PLAN.md. Read PLAN.md §1–§5 and CLAUDE.md first. Follow the Global DoD and the retry protocol exactly." plus the task section pasted in full.
- Parallel agents share one working tree, so ruff and pytest see everyone's in-progress files. If a DoD failure is in a file another task owns, the agent waits and re-runs rather than editing that file. Alternative: run each agent with `isolation: "worktree"` and merge after.
- Commit after each task passes orchestrator verification: `feat(<area>): Tn <summary>`.
