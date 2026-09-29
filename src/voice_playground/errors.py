"""Error hierarchy. Each error carries the process exit code the CLI should use.

Exit codes (PLAN.md §2 rule 7): 0 ok, 1 provider/runtime error, 2 usage/config error.
Error messages are shown as one line on stderr and must never contain API keys.
"""

from __future__ import annotations


class VPError(Exception):
    """Base class for all expected voice_playground errors."""

    exit_code: int = 1


class ConfigError(VPError):
    """Usage or configuration problem (bad flags, missing env var, invalid voice config)."""

    exit_code = 2


class ProviderError(VPError):
    """A provider SDK/API call, or another runtime step (ffmpeg, playback, HA), failed."""

    exit_code = 1


class UnsupportedCapability(VPError):
    """The selected provider does not support the requested capability.

    This is treated as a usage error (exit code 2): the user asked for something the
    chosen provider cannot do (e.g. voice cloning on OpenAI).
    """

    exit_code = 2
