"""`vp` command-line interface (PLAN.md §2). Thin: parses flags and delegates to `service`."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from voice_playground import service
from voice_playground.errors import VPError
from voice_playground.settings import Settings, get_settings

app = typer.Typer(
    name="vp",
    help="Try out text-to-speech and speech-to-text models from different providers.",
    no_args_is_help=True,
    add_completion=False,
)
voices_app = typer.Typer(
    help="Manage voice configs (./voices) and remote custom voices.",
    no_args_is_help=True,
)
app.add_typer(voices_app, name="voices")


class PlayTarget(StrEnum):
    local = "local"
    sonos = "sonos"


class AudioFormat(StrEnum):
    wav = "wav"
    mp3 = "mp3"
    pcm = "pcm"


def _err(message: str) -> None:
    """Print a one-line error on stderr."""
    typer.echo(f"error: {' '.join(message.split())}", err=True)


@contextmanager
def _errors() -> Iterator[None]:
    """Map expected errors to one-line stderr messages and exit codes (§2 rule 7)."""
    try:
        yield
    except VPError as exc:
        _err(str(exc))
        raise typer.Exit(exc.exit_code) from None
    except NotImplementedError as exc:
        _err(f"not implemented yet: {exc}")
        raise typer.Exit(1) from None


def _run[T](fn: Callable[[Settings], T]) -> T:
    with _errors():
        return fn(get_settings())


# --- shared option types -----------------------------------------------------------------

ProviderOpt = Annotated[
    str | None,
    typer.Option("--provider", "-p", help="Provider name (google, openai, elevenlabs, fake)."),
]
ModelOpt = Annotated[
    str | None, typer.Option("--model", "-m", help="Model id (default: provider default).")
]
PlayOpt = Annotated[
    PlayTarget | None,
    typer.Option("--play", help="Where to play the audio: local speakers or Sonos via HA."),
]
SpeakerOpt = Annotated[
    str | None,
    typer.Option(
        "--speaker",
        help="Home Assistant media_player entity id (implies --play sonos).",
    ),
]


# --- commands ----------------------------------------------------------------------------


@app.command()
def tts(
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    voice: Annotated[
        str | None,
        typer.Option(
            "--voice",
            "-v",
            help="Voice config name (voices/<name>.yaml) or a raw provider voice name/id.",
        ),
    ] = None,
    text: Annotated[
        str | None, typer.Option("--text", "-t", help="Text to speak (else --text-file/stdin).")
    ] = None,
    text_file: Annotated[
        Path | None,
        typer.Option("--text-file", help="Read the text to speak from this file.", dir_okay=False),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write audio to this file instead of playing it."),
    ] = None,
    style: Annotated[
        str | None,
        typer.Option("--style", help='Delivery style, e.g. "whispered, urgent".'),
    ] = None,
    fmt: Annotated[
        AudioFormat | None,
        typer.Option("--format", "-f", help="Output format (default: from -o extension, or wav)."),
    ] = None,
    play: PlayOpt = None,
    speaker: SpeakerOpt = None,
    auto_create: Annotated[
        bool,
        typer.Option(
            "--auto-create/--no-auto-create",
            help="Create designed/cloned voices remotely on first use.",
        ),
    ] = True,
) -> None:
    """Text-to-speech: synthesize TEXT and play it or write it to a file."""
    path = _run(
        lambda s: service.run_tts(
            settings=s,
            text=text,
            text_file=text_file,
            provider=provider,
            model=model,
            voice=voice,
            output=output,
            style=style,
            fmt=fmt.value if fmt else None,
            play=play.value if play else None,
            speaker=speaker,
            auto_create=auto_create,
        )
    )
    if path is not None:
        typer.echo(f"wrote {path}", err=True)


@app.command()
def stt(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Audio file to transcribe.", dir_okay=False),
    ],
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the transcript to this file."),
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", "-l", help="Spoken language hint, e.g. en.")
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="Context/vocabulary hint for the transcriber."),
    ] = None,
) -> None:
    """Speech-to-text: transcribe an audio file."""
    transcript = _run(
        lambda s: service.run_stt(
            settings=s,
            input_path=input_path,
            provider=provider,
            model=model,
            output=output,
            language=language,
            prompt=prompt,
        )
    )
    if output is None:
        typer.echo(transcript.text)
    else:
        typer.echo(f"wrote {output}", err=True)


@app.command()
def providers() -> None:
    """List providers, their capabilities, default models, and whether a key is set."""
    infos = _run(service.providers_info)
    for info in infos:
        if info.error:
            typer.echo(f"{info.name}: unavailable ({info.error})")
            continue
        caps = ", ".join(sorted(c.value for c in info.capabilities)) or "-"
        if info.key_present is None:
            key = "not needed"
        else:
            key = f"{info.key_env} {'yes' if info.key_present else 'no'}"
        typer.echo(
            f"{info.name}: capabilities=[{caps}] tts={info.default_tts_model or '-'} "
            f"stt={info.default_stt_model or '-'} key={key}"
        )


@app.command()
def play(
    path: Annotated[Path, typer.Argument(help="Audio file to play.", dir_okay=False)],
    play: PlayOpt = None,
    speaker: SpeakerOpt = None,
) -> None:
    """Play an existing audio file locally or on Sonos."""
    target = play.value if play else None
    _run(lambda s: service.play_file(s, path, play=target, speaker=speaker))


# --- voices subcommands ------------------------------------------------------------------


@voices_app.command("list")
def voices_list(
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Only show this provider's voices.")
    ] = None,
) -> None:
    """List voice configs in the voices directory."""
    configs = _run(lambda s: service.voices_list(s, provider=provider))
    for cfg in configs:
        typer.echo(f"{cfg.name}\t{cfg.provider or '-'}\t{cfg.type}")


@voices_app.command("show")
def voices_show(name: Annotated[str, typer.Argument(help="Voice config name.")]) -> None:
    """Show a voice config resolved, plus its cached remote id."""
    info = _run(lambda s: service.voices_show(s, name))
    for key, value in info.items():
        typer.echo(f"{key}: {value}")


@voices_app.command("create")
def voices_create(
    name: Annotated[str, typer.Argument(help="Voice config name.")],
    force: Annotated[
        bool, typer.Option("--force", help="Recreate even if a valid cached id exists.")
    ] = False,
) -> None:
    """Create a designed/cloned voice remotely and cache its id."""
    created = _run(lambda s: service.voices_create(s, name, force=force))
    expires = created.expires_at.isoformat() if created.expires_at else "never"
    typer.echo(f"{name}: {created.remote_id} (expires {expires})")


@voices_app.command("library")
def voices_library(
    provider: Annotated[str, typer.Option("--provider", "-p", help="Provider to query.")],
    search: Annotated[str | None, typer.Option("--search", help="Free-text search.")] = None,
    language: Annotated[
        str | None, typer.Option("--language", help="Language code, e.g. en-US.")
    ] = None,
    gender: Annotated[str | None, typer.Option("--gender", help="Voice gender filter.")] = None,
) -> None:
    """Browse a provider's remote voice catalog."""
    filters: dict[str, Any] = {
        k: v for k, v in {"search": search, "language": language, "gender": gender}.items() if v
    }
    voices = _run(lambda s: service.voices_library(s, provider, **filters))
    for v in voices:
        typer.echo(f"{v.id}\t{v.name}\t{v.description or ''}")


@voices_app.command("delete")
def voices_delete(name: Annotated[str, typer.Argument(help="Voice config name.")]) -> None:
    """Delete a voice's remote custom voice and its cache entry."""
    _run(lambda s: service.voices_delete(s, name))
    typer.echo(f"deleted {name}", err=True)


if __name__ == "__main__":  # pragma: no cover
    app()
