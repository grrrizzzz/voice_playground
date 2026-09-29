"""Tests for voice_playground.playback.local (T5). `shutil.which` and `subprocess` are mocked."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from voice_playground.audio import AudioResult, wav_info
from voice_playground.errors import ConfigError, ProviderError
from voice_playground.playback import local
from voice_playground.providers.fake import sine_pcm, sine_wav

AFPLAY = "/usr/bin/afplay"
FFPLAY = "/opt/homebrew/bin/ffplay"


def fake_which(available: dict[str, str]) -> Any:
    return lambda name: available.get(name)


class FakeProc:
    """Stands in for `subprocess.Popen`; records the command and the played file's bytes."""

    instances: list[FakeProc] = []

    def __init__(
        self,
        cmd: list[str],
        *,
        returncode: int = 0,
        interrupt: bool = False,
        hang_on_terminate: bool = False,
        **kwargs: Any,
    ) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.path = Path(cmd[-1])
        self.played = self.path.read_bytes()
        self._rc = returncode
        self._interrupt = interrupt
        self._hang = hang_on_terminate
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []
        FakeProc.instances.append(self)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self._interrupt and len(self.wait_calls) == 1:
            raise KeyboardInterrupt
        if self.terminated and self._hang and not self.killed:
            raise subprocess.TimeoutExpired(self.cmd, timeout or 0)
        if self.returncode is None:
            self.returncode = -15 if self.terminated else self._rc
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.fixture(autouse=True)
def _reset_procs() -> None:
    FakeProc.instances = []


def install_popen(monkeypatch: pytest.MonkeyPatch, **proc_kwargs: Any) -> None:
    def popen(cmd: list[str], **kwargs: Any) -> FakeProc:
        return FakeProc(cmd, **proc_kwargs, **kwargs)

    monkeypatch.setattr(local.subprocess, "Popen", popen)


def wav() -> AudioResult:
    return AudioResult(sine_wav(), "audio/wav", 24_000)


# --- player selection -------------------------------------------------------------------


def test_prefers_afplay_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY, "ffplay": FFPLAY}))
    assert local.player_command() == [AFPLAY]


def test_falls_back_to_ffplay_when_no_afplay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"ffplay": FFPLAY}))
    assert local.player_command() == [FFPLAY, "-nodisp", "-autoexit", "-loglevel", "quiet"]


def test_uses_ffplay_off_macos_even_if_afplay_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "linux")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY, "ffplay": FFPLAY}))
    assert local.player_command()[0] == FFPLAY


def test_no_player_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.shutil, "which", fake_which({}))
    popen_called = False

    def popen(*args: Any, **kwargs: Any) -> None:
        nonlocal popen_called
        popen_called = True

    monkeypatch.setattr(local.subprocess, "Popen", popen)
    with pytest.raises(ConfigError, match="no audio player found.*ffmpeg"):
        local.play_local(wav())
    assert not popen_called


# --- play_local -------------------------------------------------------------------------


def test_play_local_plays_temp_file_and_removes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY}))
    install_popen(monkeypatch)
    result = wav()
    local.play_local(result)
    (proc,) = FakeProc.instances
    assert proc.cmd == [AFPLAY, str(proc.path)]
    assert proc.path.suffix == ".wav"
    assert proc.played == result.data
    assert proc.wait_calls == [None]  # blocked until the player exited
    assert not proc.path.exists()


def test_play_local_mp3_uses_mp3_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.shutil, "which", fake_which({"ffplay": FFPLAY}))
    monkeypatch.setattr(local.sys, "platform", "linux")
    install_popen(monkeypatch)
    local.play_local(AudioResult(b"\xff\xfbfake-mp3", "audio/mpeg", 24_000))
    (proc,) = FakeProc.instances
    assert proc.cmd[:-1] == [FFPLAY, "-nodisp", "-autoexit", "-loglevel", "quiet"]
    assert proc.path.suffix == ".mp3"
    assert proc.played == b"\xff\xfbfake-mp3"


def test_play_local_converts_pcm_to_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY}))
    install_popen(monkeypatch)
    local.play_local(AudioResult(sine_pcm(16_000), "audio/l16", 16_000))
    (proc,) = FakeProc.instances
    assert proc.path.suffix == ".wav"
    info = wav_info(proc.played)
    assert info.sample_rate == 16_000
    assert info.frames == len(sine_pcm(16_000)) // 2


def test_play_local_nonzero_exit_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY}))
    install_popen(monkeypatch, returncode=1)
    with pytest.raises(ProviderError, match="afplay exited with code 1"):
        local.play_local(wav())
    assert not FakeProc.instances[0].path.exists()


def test_play_local_popen_oserror_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY}))
    seen: list[Path] = []

    def popen(cmd: list[str], **kwargs: Any) -> None:
        seen.append(Path(cmd[-1]))
        raise PermissionError("denied")

    monkeypatch.setattr(local.subprocess, "Popen", popen)
    with pytest.raises(ProviderError, match="could not start audio player"):
        local.play_local(wav())
    assert not seen[0].exists()


def test_ctrl_c_terminates_player_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY}))
    install_popen(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        local.play_local(wav())
    (proc,) = FakeProc.instances
    assert proc.terminated
    assert not proc.killed
    assert proc.wait_calls[1] == local.TERMINATE_TIMEOUT_S
    assert not proc.path.exists()


def test_ctrl_c_kills_player_that_ignores_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local.sys, "platform", "darwin")
    monkeypatch.setattr(local.shutil, "which", fake_which({"afplay": AFPLAY}))
    install_popen(monkeypatch, interrupt=True, hang_on_terminate=True)
    with pytest.raises(KeyboardInterrupt):
        local.play_local(wav())
    (proc,) = FakeProc.instances
    assert proc.terminated and proc.killed
    assert proc.returncode == -9
    assert not proc.path.exists()
