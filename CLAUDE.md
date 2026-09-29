# CLAUDE.md

A CLI (`vp`) for trying out TTS/STT models from Google Gemini, OpenAI, and ElevenLabs. See `PLAN.md` for the spec and task breakdown.

## Rules for agents
- Read `PLAN.md` §1–§5 before starting work. Your task section is your contract. Don't edit files your task doesn't own.
- Provider SDKs change quickly. **Check the current provider docs before writing SDK calls.** Don't rely on memory for model IDs or method names.
- Don't change `src/voice_playground/providers/base.py` contracts. Report `BLOCKED: contract change needed` instead.
- Secrets live only in `.env` (gitignored). Never print, log, commit, or put API keys or `HA_TOKEN` in test fixtures. Error messages must not include keys.
- Unit tests mock all network and SDK calls. Tests that hit real APIs must have `@pytest.mark.live`.

## Commands
```bash
uv sync
uv run vp --help
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -m "not live" -q     # required to pass
uv run pytest -m live -v           # optional; needs keys in .env
```

## Done means
All the commands above except the live tests pass, every acceptance box in your task is ticked with evidence, and you followed the retry protocol in `PLAN.md` §5. Never report success while any of them fail.
