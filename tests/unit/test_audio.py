"""Tests for voice_playground.audio (T5)."""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from voice_playground import audio
from voice_playground.audio import (
    MIME_TYPES,
    AudioResult,
    convert,
    format_from_path,
    pcm_to_wav,
    wav_info,
    write_output,
)
from voice_playground.errors import ConfigError, ProviderError
from voice_playground.providers.fake import sine_pcm, sine_wav

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

RATE = 24_000


def pcm_result(rate: int = RATE, channels: int = 1) -> AudioResult:
    data = _interleave(sine_pcm(rate), channels)
    return AudioResult(data=data, mime_type="audio/l16", sample_rate=rate, channels=channels)


def _interleave(mono: bytes, channels: int) -> bytes:
    samples = [mono[i : i + 2] for i in range(0, len(mono), 2)]
    return b"".join(s * channels for s in samples)


def wav_result(rate: int = RATE) -> AudioResult:
    return AudioResult(data=sine_wav(rate), mime_type="audio/wav", sample_rate=rate)


# --- pcm_to_wav / wav_info ------------------------------------------------------------


@pytest.mark.parametrize(("rate", "channels"), [(24_000, 1), (16_000, 2), (44_100, 1)])
def test_pcm_to_wav_round_trip(rate: int, channels: int) -> None:
    pcm = pcm_result(rate, channels).data
    wav_bytes = pcm_to_wav(pcm, rate, channels)
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == rate
        assert w.getnchannels() == channels
        assert w.getsampwidth() == 2
        assert w.getnframes() == len(pcm) // (2 * channels)
        assert w.readframes(w.getnframes()) == pcm


def test_wav_info_reports_params_and_duration() -> None:
    info = wav_info(sine_wav(RATE, 0.5))
    assert (info.sample_rate, info.channels, info.sample_width) == (RATE, 1, 2)
    assert info.frames == RATE // 2
    assert info.duration == pytest.approx(0.5)


def test_wav_info_invalid_raises_provider_error() -> None:
    with pytest.raises(ProviderError, match="invalid WAV"):
        wav_info(b"not a wav file at all")


# --- format_from_path -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("a.wav", "wav"), ("a.mp3", "mp3"), ("a.pcm", "pcm"), ("DIR/A.WAV", "wav"), ("x.Mp3", "mp3")],
)
def test_format_from_path(name: str, fmt: str) -> None:
    assert format_from_path(Path(name)) == fmt


@pytest.mark.parametrize("name", ["a.ogg", "a.txt", "noext", "a.wav.bak"])
def test_format_from_path_unknown_raises_config_error(name: str) -> None:
    with pytest.raises(ConfigError, match="cannot infer audio format"):
        format_from_path(Path(name))


# --- convert: pure-Python paths ---------------------------------------------------------


def test_convert_same_format_is_noop() -> None:
    for result in (wav_result(), pcm_result()):
        fmt = "wav" if result.mime_type == "audio/wav" else "pcm"
        assert convert(result, fmt) is result


def test_convert_unknown_format_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="unsupported audio format"):
        convert(wav_result(), "flac")


def test_convert_pcm_to_wav_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)
    src = pcm_result(16_000, 2)
    out = convert(src, "wav")
    assert out.mime_type == "audio/wav"
    assert (out.sample_rate, out.channels) == (16_000, 2)
    info = wav_info(out.data)
    assert (info.sample_rate, info.channels) == (16_000, 2)
    assert info.frames == len(src.data) // 4


def test_convert_wav_to_pcm_strips_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)
    out = convert(wav_result(), "pcm")
    assert out.mime_type == "audio/l16"
    assert out.sample_rate == RATE
    assert out.data == sine_pcm(RATE)


def test_convert_pcm_mime_with_params_is_recognised() -> None:
    src = AudioResult(sine_pcm(), "audio/L16;codec=pcm;rate=24000", RATE)
    assert convert(src, "pcm") is src
    assert convert(src, "wav").mime_type == "audio/wav"


def test_convert_pcm_without_sample_rate_raises() -> None:
    with pytest.raises(ProviderError, match="sample rate"):
        convert(AudioResult(sine_pcm(), "audio/l16", None), "wav")


# --- convert: ffmpeg errors (mocked) ----------------------------------------------------


def test_convert_mp3_without_ffmpeg_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)
    with pytest.raises(ConfigError, match="ffmpeg.*brew install ffmpeg"):
        convert(wav_result(), "mp3")


def test_convert_ffmpeg_failure_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, 1, b"", b"line one\npipe:0: Invalid data\n")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(ProviderError, match="ffmpeg conversion failed: pipe:0: Invalid data"):
        convert(wav_result(), "mp3")


def test_convert_ffmpeg_command_for_pcm_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, b"MP3DATA", b"")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    src = pcm_result()
    out = convert(src, "mp3")
    assert out == AudioResult(b"MP3DATA", "audio/mpeg", RATE, 1)
    cmd, kwargs = calls[0]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert cmd[cmd.index("-loglevel") + 1] == "error"
    assert cmd[cmd.index("-f") : cmd.index("-f") + 6] == ["-f", "s16le", "-ar", "24000", "-ac", "1"]
    assert cmd[-3:] == ["-f", "mp3", "pipe:1"]
    assert "pipe:0" in cmd
    assert kwargs["input"] == src.data


def test_convert_ffmpeg_timeout_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(ProviderError, match="timed out"):
        convert(wav_result(), "mp3")


# --- convert: real ffmpeg ---------------------------------------------------------------


@needs_ffmpeg
def test_real_ffmpeg_wav_mp3_round_trip() -> None:
    mp3 = convert(wav_result(), "mp3")
    assert mp3.mime_type == "audio/mpeg"
    assert mp3.sample_rate == RATE
    assert mp3.data[:3] == b"ID3" or mp3.data[0] == 0xFF

    back = convert(mp3, "wav")
    assert back.mime_type == "audio/wav"
    assert back.sample_rate == RATE
    info = wav_info(back.data)
    assert (info.sample_rate, info.channels, info.sample_width) == (RATE, 1, 2)
    # Header sizes are real (not ffmpeg's streaming 0xFFFFFFFF); mp3 adds encoder padding.
    assert 0.45 < info.duration < 0.7
    assert len(back.data) == 44 + info.frames * 2


@needs_ffmpeg
def test_real_ffmpeg_pcm_to_mp3_and_mp3_to_pcm() -> None:
    mp3 = convert(pcm_result(), "mp3")
    assert mp3.mime_type == "audio/mpeg"
    pcm = convert(mp3, "pcm")
    assert pcm.mime_type == "audio/l16"
    assert pcm.sample_rate == RATE
    assert len(pcm.data) % 2 == 0
    assert len(pcm.data) >= len(sine_pcm(RATE)) * 0.9


@needs_ffmpeg
def test_real_ffmpeg_invalid_input_raises_provider_error() -> None:
    with pytest.raises(ProviderError, match="ffmpeg"):
        convert(AudioResult(b"garbage" * 10, "audio/mpeg", None), "wav")


# --- write_output -----------------------------------------------------------------------


def test_write_output_infers_format_and_creates_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.wav"
    written = write_output(pcm_result(), target)
    assert written == target
    info = wav_info(target.read_bytes())
    assert (info.sample_rate, info.frames) == (RATE, len(sine_pcm(RATE)) // 2)


def test_write_output_pcm_extension(tmp_path: Path) -> None:
    target = write_output(wav_result(), tmp_path / "out.pcm")
    assert target.read_bytes() == sine_pcm(RATE)


def test_write_output_explicit_format_overrides_extension(tmp_path: Path) -> None:
    target = write_output(wav_result(), tmp_path / "out.bin", "pcm")
    assert target.read_bytes() == sine_pcm(RATE)


def test_write_output_unknown_extension_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        write_output(wav_result(), tmp_path / "out.ogg")
    assert not (tmp_path / "out.ogg").exists()


@needs_ffmpeg
def test_write_output_mp3_real(tmp_path: Path) -> None:
    target = write_output(wav_result(), tmp_path / "sub" / "out.mp3")
    data = target.read_bytes()
    assert data[:3] == b"ID3" or data[0] == 0xFF


def test_mime_types_constant() -> None:
    assert MIME_TYPES == {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/l16"}
