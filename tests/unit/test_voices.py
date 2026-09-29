"""T7: voice config models, loading, cache, and resolution (PLAN.md §2 rules 2-4, §3)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from voice_playground.errors import ConfigError
from voice_playground.providers.base import CreatedVoice
from voice_playground.voices import (
    CacheEntry,
    ClonedVoiceConfig,
    DesignedVoiceConfig,
    PrebuiltVoiceConfig,
    VoiceCache,
    config_hash,
    list_voices,
    load_voice,
    resolve_voice,
)

REPO_VOICES = Path(__file__).resolve().parents[2] / "voices"
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def write(voices_dir: Path, name: str, body: str, suffix: str = ".yaml") -> Path:
    path = voices_dir / f"{name}{suffix}"
    path.write_text(body)
    return path


@pytest.fixture
def cache(cache_dir: Path) -> VoiceCache:
    return VoiceCache(cache_dir)


# --- models / loading --------------------------------------------------------------------


def test_load_prebuilt(voices_dir: Path) -> None:
    write(
        voices_dir,
        "narrator",
        "name: narrator\nprovider: google\ntype: prebuilt\nvoice: Kore\nstyle: calm\n"
        "sample_rate: 24000\nprovider_options: {speed: 1.1}\n",
    )
    cfg = load_voice("narrator", voices_dir)
    assert isinstance(cfg, PrebuiltVoiceConfig)
    assert cfg.voice == "Kore"
    assert cfg.provider_options == {"speed": 1.1}


def test_load_designed_and_cloned(voices_dir: Path) -> None:
    write(voices_dir, "d", "name: d\ntype: designed\ndescription: warm\nstore: false\n")
    write(voices_dir, "c", "name: c\ntype: cloned\nreference_audio: voices/audio/me.wav\n")
    d = load_voice("d", voices_dir)
    c = load_voice("c", voices_dir)
    assert isinstance(d, DesignedVoiceConfig)
    assert d.store is False
    assert isinstance(c, ClonedVoiceConfig)
    assert c.reference_audio == Path("voices/audio/me.wav")  # existence not checked at load


def test_load_yml_extension(voices_dir: Path) -> None:
    write(voices_dir, "alt", "name: alt\ntype: prebuilt\nvoice: coral\n", suffix=".yml")
    assert load_voice("alt", voices_dir).name == "alt"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("name: x\ntype: prebuilt\n", "voice"),  # prebuilt missing voice
        ("name: x\ntype: designed\n", "description"),  # designed missing description
        ("name: x\ntype: cloned\n", "reference_audio"),  # cloned missing reference_audio
        ("name: x\ntype: prebuilt\nvoice: Kore\ndescription: nope\n", "description"),
        ("name: x\ntype: designed\ndescription: d\nvoice: Kore\n", "voice"),
        ("name: x\ntype: cloned\nreference_audio: a.wav\ndescription: d\n", "description"),
        ("name: x\ntype: robot\nvoice: Kore\n", "type"),
        ("name: x\nvoice: Kore\n", "type"),
        ("name: x\ntype: prebuilt\nvoice: Kore\nprovider: nope\n", "unknown provider"),
        ("name: x\ntype: prebuilt\nvoice: Kore\nsample_rate: -1\n", "sample_rate"),
        ("- just\n- a list\n", "mapping"),
        ("name: [unclosed\n", "invalid YAML"),
    ],
)
def test_invalid_config_names_file_and_field(voices_dir: Path, body: str, expected: str) -> None:
    path = write(voices_dir, "x", body)
    with pytest.raises(ConfigError) as excinfo:
        load_voice("x", voices_dir)
    message = str(excinfo.value)
    assert str(path) in message
    assert expected in message
    assert "\n" not in message
    assert excinfo.value.exit_code == 2


def test_name_filename_mismatch(voices_dir: Path) -> None:
    path = write(voices_dir, "narrator", "name: someone-else\ntype: prebuilt\nvoice: Kore\n")
    with pytest.raises(ConfigError, match="must match the file name") as excinfo:
        load_voice("narrator", voices_dir)
    assert str(path) in str(excinfo.value)


def test_load_missing_file(voices_dir: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_voice("ghost", voices_dir)


def test_list_voices_sorted_filtered_and_ignores_other_files(voices_dir: Path) -> None:
    write(voices_dir, "b", "name: b\nprovider: openai\ntype: prebuilt\nvoice: coral\n")
    write(voices_dir, "a", "name: a\nprovider: google\ntype: prebuilt\nvoice: Kore\n")
    write(voices_dir, "c", "name: c\nprovider: google\ntype: designed\ndescription: d\n", ".yml")
    write(voices_dir, "skip", "name: skip\ntype: nonsense\n", ".yaml.example")
    (voices_dir / "notes.txt").write_text("not a voice")
    (voices_dir / "audio").mkdir()
    assert [v.name for v in list_voices(voices_dir)] == ["a", "b", "c"]
    assert [v.name for v in list_voices(voices_dir, provider="google")] == ["a", "c"]


def test_list_voices_invalid_file_raises(voices_dir: Path) -> None:
    path = write(voices_dir, "bad", "name: bad\ntype: prebuilt\n")
    with pytest.raises(ConfigError) as excinfo:
        list_voices(voices_dir)
    assert str(path) in str(excinfo.value)


def test_list_voices_missing_dir(tmp_path: Path) -> None:
    assert list_voices(tmp_path / "nope") == []


def test_committed_example_configs_load() -> None:
    files = sorted(REPO_VOICES.glob("*.yaml")) + sorted(REPO_VOICES.glob("*.yml"))
    names = {p.stem for p in files}
    assert {"narrator", "storyteller", "openai-coral", "eleven-rachel"} <= names
    for path in files:
        assert load_voice(path.stem, REPO_VOICES).name == path.stem
    assert {v.name for v in list_voices(REPO_VOICES)} == names


def test_committed_clone_example_is_valid_but_not_loaded(voices_dir: Path) -> None:
    example = REPO_VOICES / "me-clone.yaml.example"
    assert "me-clone" not in {v.name for v in list_voices(REPO_VOICES)}
    shutil.copy(example, voices_dir / "me-clone.yaml")
    cfg = load_voice("me-clone", voices_dir)
    assert isinstance(cfg, ClonedVoiceConfig)
    assert cfg.provider == "google"


# --- config hash -------------------------------------------------------------------------


def test_config_hash_stable_and_ignores_per_request_fields() -> None:
    base = DesignedVoiceConfig(name="s", provider="google", description="warm")
    same = DesignedVoiceConfig(
        name="s", provider="google", description="warm", style="loud", model="m", language="fr"
    )
    assert config_hash(base) == config_hash(same)
    assert len(config_hash(base)) == 64
    for changed in (
        DesignedVoiceConfig(name="s", provider="google", description="cold"),
        DesignedVoiceConfig(name="s", provider="elevenlabs", description="warm"),
        DesignedVoiceConfig(name="s", provider="google", description="warm", store=False),
    ):
        assert config_hash(changed) != config_hash(base)


def test_config_hash_tracks_reference_audio_content(tmp_path: Path) -> None:
    audio = tmp_path / "me.wav"
    audio.write_bytes(b"one")
    cfg = ClonedVoiceConfig(name="c", provider="google", reference_audio=Path("me.wav"))
    first = config_hash(cfg, root=tmp_path)
    assert config_hash(cfg, root=tmp_path) == first
    audio.write_bytes(b"two")
    assert config_hash(cfg, root=tmp_path) != first


# --- cache -------------------------------------------------------------------------------


def _entry(cfg: DesignedVoiceConfig, **kw: object) -> CacheEntry:
    data: dict[str, object] = {
        "provider": "google",
        "remote_id": "voice_123",
        "config_hash": config_hash(cfg),
        "created_at": NOW - timedelta(days=1),
        "expires_at": NOW + timedelta(days=6),
    }
    data.update(kw)
    return CacheEntry.model_validate(data)


def test_cache_put_get_delete_entries(cache: VoiceCache) -> None:
    cfg = DesignedVoiceConfig(name="s", provider="google", description="warm")
    assert cache.get("s") is None
    assert cache.entries() == {}
    cache.put("s", _entry(cfg))
    assert cache.path.is_file()
    assert VoiceCache(cache.cache_dir).get("s") == _entry(cfg)  # persisted
    assert list(cache.entries()) == ["s"]
    cache.delete("s")
    assert cache.get("s") is None
    cache.delete("s")  # idempotent


def test_cache_write_is_atomic_and_leaves_no_temp_files(
    cache: VoiceCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = DesignedVoiceConfig(name="s", provider="google", description="warm")
    cache.put("s", _entry(cfg))
    before = cache.path.read_text()

    def boom(*_args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("voice_playground.voices.os.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        cache.put("t", _entry(cfg, remote_id="other"))
    assert cache.path.read_text() == before  # original untouched
    assert [p.name for p in cache.cache_dir.iterdir()] == ["voices.json"]


def test_cache_json_shape(cache: VoiceCache) -> None:
    cfg = DesignedVoiceConfig(name="s", provider="google", description="warm")
    cache.put("s", _entry(cfg, expires_at=None))
    data = json.loads(cache.path.read_text())
    assert set(data["s"]) == {"provider", "remote_id", "config_hash", "created_at", "expires_at"}
    assert data["s"]["expires_at"] is None


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"s": {"provider": "x"}}'])
def test_corrupt_cache_warns_and_is_empty(
    cache: VoiceCache, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    cache.cache_dir.mkdir(parents=True)
    cache.path.write_text(content)
    assert cache.get("s") is None
    assert "corrupt" in capsys.readouterr().err
    cfg = DesignedVoiceConfig(name="s", provider="google", description="warm")
    cache.put("s", _entry(cfg))  # recovers by overwriting
    assert cache.get("s") is not None


def test_cache_record_builds_entry(cache: VoiceCache) -> None:
    cfg = DesignedVoiceConfig(name="s", provider="google", description="warm")
    expires = NOW + timedelta(days=365)
    entry = cache.record(cfg, "google", CreatedVoice("voice_abc", expires), now=NOW)
    assert entry == CacheEntry(
        provider="google",
        remote_id="voice_abc",
        config_hash=config_hash(cfg),
        created_at=NOW,
        expires_at=expires,
    )
    assert cache.get("s") == entry
    assert cache.is_valid("s", cfg, NOW)


def test_is_valid_states(cache: VoiceCache, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = DesignedVoiceConfig(name="s", provider="google", description="warm")
    assert cache.status("s", cfg, NOW) == "missing"
    assert not cache.is_valid("s", cfg, NOW)

    cache.put("s", _entry(cfg))
    assert cache.is_valid("s", cfg, NOW)
    assert cache.is_valid("s", cfg, NOW.replace(tzinfo=None))  # naive = UTC
    assert capsys.readouterr().err == ""

    changed = DesignedVoiceConfig(name="s", provider="google", description="cold")
    assert cache.status("s", changed, NOW) == "stale"
    assert not cache.is_valid("s", changed, NOW)
    assert "changed" in capsys.readouterr().err

    assert cache.status("s", cfg, NOW + timedelta(days=6)) == "expired"  # expires_at == now
    assert not cache.is_valid("s", cfg, NOW + timedelta(days=7))
    assert "expired" in capsys.readouterr().err

    assert cache.status("s", cfg, NOW, provider="elevenlabs") == "provider_mismatch"

    cache.put("s", _entry(cfg, expires_at=None))
    assert cache.is_valid("s", cfg, NOW + timedelta(days=10_000))


# --- resolve_voice -----------------------------------------------------------------------


def test_resolve_raw_voice_fallback(voices_dir: Path, cache: VoiceCache) -> None:
    provider, voice, needs_create = resolve_voice(
        "Kore", "google", voices_dir=voices_dir, cache=cache
    )
    assert (provider, voice.provider_voice, voice.config_name, needs_create) == (
        "google",
        "Kore",
        None,
        False,
    )
    provider, voice, _ = resolve_voice("coral", None, voices_dir=voices_dir, cache=cache)
    assert provider is None
    assert voice.provider_voice == "coral"


@pytest.mark.parametrize("raw", ["../etc/passwd", "a/b", ".hidden"])
def test_resolve_path_like_names_are_raw(voices_dir: Path, cache: VoiceCache, raw: str) -> None:
    (voices_dir.parent / "etc").mkdir(exist_ok=True)
    _, voice, _ = resolve_voice(raw, "fake", voices_dir=voices_dir, cache=cache)
    assert voice.provider_voice == raw
    assert voice.config_name is None


def test_resolve_empty_voice(voices_dir: Path, cache: VoiceCache) -> None:
    with pytest.raises(ConfigError):
        resolve_voice("  ", None, voices_dir=voices_dir, cache=cache)


def test_resolve_prebuilt_uses_config(voices_dir: Path, cache: VoiceCache) -> None:
    write(
        voices_dir,
        "narrator",
        "name: narrator\nprovider: google\ntype: prebuilt\nvoice: Kore\nstyle: calm\n"
        "language: en-US\nsample_rate: 16000\nprovider_options: {a: 1}\n",
    )
    provider, voice, needs_create = resolve_voice(
        "narrator", None, voices_dir=voices_dir, cache=cache
    )
    assert provider == "google"
    assert not needs_create
    assert voice.provider_voice == "Kore"
    assert (voice.style, voice.language, voice.sample_rate) == ("calm", "en-US", 16000)
    assert voice.options == {"a": 1}
    assert voice.config_name == "narrator"
    # Matching flag is fine; config without provider takes the flag.
    assert resolve_voice("narrator", "google", voices_dir=voices_dir, cache=cache)[0] == "google"
    write(voices_dir, "any", "name: any\ntype: prebuilt\nvoice: v\n")
    assert resolve_voice("any", "openai", voices_dir=voices_dir, cache=cache)[0] == "openai"
    assert resolve_voice("any", None, voices_dir=voices_dir, cache=cache)[0] is None


def test_resolve_provider_conflict(voices_dir: Path, cache: VoiceCache) -> None:
    path = write(
        voices_dir, "narrator", "name: narrator\nprovider: google\ntype: prebuilt\nvoice: K\n"
    )
    with pytest.raises(ConfigError) as excinfo:
        resolve_voice("narrator", "openai", voices_dir=voices_dir, cache=cache)
    message = str(excinfo.value)
    assert "google" in message and "openai" in message and str(path) in message
    assert excinfo.value.exit_code == 2


def test_resolve_invalid_config_raises(voices_dir: Path, cache: VoiceCache) -> None:
    write(voices_dir, "narrator", "name: other\ntype: prebuilt\nvoice: K\n")
    with pytest.raises(ConfigError, match="must match"):
        resolve_voice("narrator", None, voices_dir=voices_dir, cache=cache)


def _designed(voices_dir: Path, description: str = "warm") -> DesignedVoiceConfig:
    write(
        voices_dir,
        "story",
        f"name: story\nprovider: fake\ntype: designed\ndescription: {description}\nstyle: soft\n",
    )
    cfg = load_voice("story", voices_dir)
    assert isinstance(cfg, DesignedVoiceConfig)
    return cfg


def test_resolve_designed_needs_create_then_uses_cache(voices_dir: Path, cache: VoiceCache) -> None:
    cfg = _designed(voices_dir)
    provider, voice, needs_create = resolve_voice("story", None, voices_dir=voices_dir, cache=cache)
    assert (provider, voice.provider_voice, needs_create) == ("fake", "", True)
    assert voice.style == "soft"

    cache.record(cfg, "fake", CreatedVoice("fake_voice_story"))
    final = dataclasses.replace(voice, provider_voice="fake_voice_story")
    provider, voice, needs_create = resolve_voice("story", None, voices_dir=voices_dir, cache=cache)
    assert (provider, needs_create) == ("fake", False)
    assert voice == final


def test_resolve_designed_stale_hash_warns_and_recreates(
    voices_dir: Path, cache: VoiceCache, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _designed(voices_dir)
    cache.record(cfg, "fake", CreatedVoice("fake_voice_story"))
    _designed(voices_dir, description="cold and distant")
    _, voice, needs_create = resolve_voice("story", None, voices_dir=voices_dir, cache=cache)
    assert needs_create
    assert voice.provider_voice == ""
    assert "story" in capsys.readouterr().err


def test_resolve_designed_expired_recreates(voices_dir: Path, cache: VoiceCache) -> None:
    cfg = _designed(voices_dir)
    cache.record(cfg, "fake", CreatedVoice("id", NOW + timedelta(days=7)), now=NOW)
    for days, expected in ((1, False), (8, True)):
        later = NOW + timedelta(days=days)
        result = resolve_voice("story", None, voices_dir=voices_dir, cache=cache, now=later)
        assert result[2] is expected


def test_resolve_designed_needs_provider(voices_dir: Path, cache: VoiceCache) -> None:
    path = write(voices_dir, "d", "name: d\ntype: designed\ndescription: warm\n")
    with pytest.raises(ConfigError, match="needs a provider") as excinfo:
        resolve_voice("d", None, voices_dir=voices_dir, cache=cache)
    assert str(path) in str(excinfo.value)
    assert resolve_voice("d", "fake", voices_dir=voices_dir, cache=cache)[2] is True


def test_resolve_cloned_checks_reference_audio(
    voices_dir: Path, cache: VoiceCache, tmp_path: Path
) -> None:
    path = write(
        voices_dir,
        "me",
        "name: me\nprovider: fake\ntype: cloned\nreference_audio: voices/audio/me.wav\n",
    )
    with pytest.raises(ConfigError, match="reference_audio not found") as excinfo:
        resolve_voice("me", None, voices_dir=voices_dir, cache=cache, root=tmp_path)
    assert str(path) in str(excinfo.value)

    audio = tmp_path / "voices" / "audio" / "me.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF....")
    _, _, needs_create = resolve_voice(
        "me", None, voices_dir=voices_dir, cache=cache, root=tmp_path
    )
    assert needs_create
    cfg = load_voice("me", voices_dir)
    cache.record(cfg, "fake", CreatedVoice("clone_id"), root=tmp_path)
    _, voice, needs_create = resolve_voice(
        "me", None, voices_dir=voices_dir, cache=cache, root=tmp_path
    )
    assert (voice.provider_voice, needs_create) == ("clone_id", False)
    audio.write_bytes(b"RIFF-new-recording")  # re-recorded -> stale
    assert resolve_voice("me", None, voices_dir=voices_dir, cache=cache, root=tmp_path)[2]


def test_resolve_cloned_default_root_is_cwd(
    voices_dir: Path, cache: VoiceCache, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(voices_dir, "me", "name: me\nprovider: fake\ntype: cloned\nreference_audio: me.wav\n")
    (tmp_path / "me.wav").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert resolve_voice("me", None, voices_dir=voices_dir, cache=cache)[2] is True


def test_resolve_cache_from_other_provider_recreates(voices_dir: Path, cache: VoiceCache) -> None:
    write(voices_dir, "d", "name: d\ntype: designed\ndescription: warm\n")
    cfg = load_voice("d", voices_dir)
    cache.record(cfg, "google", CreatedVoice("g_id"))
    assert resolve_voice("d", "google", voices_dir=voices_dir, cache=cache)[1].provider_voice == (
        "g_id"
    )
    assert resolve_voice("d", "fake", voices_dir=voices_dir, cache=cache)[2] is True
