"""Run a skill's bundled Python scripts in a subprocess.

The safety-critical core, kept free of any Tool imports so it can be tested on
its own; ``tools/script_tools.py`` wraps it. The split mirrors ``workspace.py``
vs ``tools/fs_tools.py``.

What may run is deliberately narrow: a ``.py`` file under a loaded skill's
``scripts/`` directory, which is operator-curated and mounted read-only. Code
the model wrote is never executed — that is a different feature with a different
threat model. See docs/plan-skill-scripts.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agentharness.config import Config, as_bool
from agentharness.skills.model import ENV_NAME_RE

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_TIMEOUT = 120
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024

# Unknown keys are an error rather than ignored, as in mcp/config.py: there is
# no adapter downstream to absorb a typo, so it would be a silently missing
# setting — and here the missing setting could be a security control.
_POLICY_KEYS = frozenset(
    {"enabled", "default_timeout", "max_timeout", "max_output_bytes", "env_allowlist"}
)


class ScriptConfigError(ValueError):
    """A malformed ``skill_scripts`` block. Raised at startup, not at call time."""


@dataclass(frozen=True)
class ScriptPolicy:
    """What the operator permits. Disabled is the default in every direction."""

    enabled: bool = False
    default_timeout: int = DEFAULT_TIMEOUT
    max_timeout: int = DEFAULT_MAX_TIMEOUT
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    env_allowlist: frozenset[str] = frozenset()

    def clamp_timeout(self, requested: Any) -> int:
        """Seconds to allow, given what the model asked for (possibly nothing).

        An over-large request is clamped rather than refused: a model guessing
        600 wants a long run, not a failed call. A nonsensical one is refused,
        because silently substituting a default would hide the mistake.
        """
        if requested is None:
            return self.default_timeout
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise ValueError("'timeout' must be a positive whole number of seconds")
        return min(requested, self.max_timeout)


def _positive_int(block: dict[str, Any], key: str, default: int) -> int:
    value = block.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScriptConfigError(f"skill_scripts.{key} must be a positive integer")
    return value


def parse_policy(config: Config) -> ScriptPolicy:
    """Validate the ``skill_scripts`` block into a policy.

    Called once at startup so a typo is a clear error naming the key, rather
    than a surprise inside a tool call ten minutes later.
    """
    block = config.skill_scripts or {}
    if not isinstance(block, dict):
        raise ScriptConfigError("skill_scripts must be a mapping")
    unknown = sorted(set(block) - _POLICY_KEYS)
    if unknown:
        raise ScriptConfigError(
            f"unknown skill_scripts key(s): {', '.join(unknown)}; "
            f"expected any of {', '.join(sorted(_POLICY_KEYS))}"
        )

    enabled = bool(block.get("enabled", False))
    override = os.environ.get("AGENTHARNESS_SKILL_SCRIPTS_ENABLED")
    if override is not None:
        enabled = as_bool(override)

    default_timeout = _positive_int(block, "default_timeout", DEFAULT_TIMEOUT)
    max_timeout = _positive_int(block, "max_timeout", DEFAULT_MAX_TIMEOUT)
    if default_timeout > max_timeout:
        raise ScriptConfigError(
            f"skill_scripts.default_timeout ({default_timeout}) exceeds "
            f"max_timeout ({max_timeout})"
        )

    raw_allowlist = block.get("env_allowlist", [])
    if not isinstance(raw_allowlist, list) or not all(
        isinstance(v, str) for v in raw_allowlist
    ):
        raise ScriptConfigError("skill_scripts.env_allowlist must be a list of strings")
    for name in raw_allowlist:
        if not ENV_NAME_RE.match(name):
            raise ScriptConfigError(
                f"invalid skill_scripts.env_allowlist entry {name!r}: environment "
                "variable names must match [A-Z_][A-Z0-9_]*"
            )

    return ScriptPolicy(
        enabled=enabled,
        default_timeout=default_timeout,
        max_timeout=max_timeout,
        max_output_bytes=_positive_int(block, "max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES),
        env_allowlist=frozenset(raw_allowlist),
    )
