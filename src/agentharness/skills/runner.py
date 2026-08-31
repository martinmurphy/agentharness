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
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentharness.config import Config, as_bool
from agentharness.skills.model import ENV_NAME_RE, Skill
from agentharness.workspace import Workspace

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


def resolve_script(skill: Skill, rel_path: str) -> Path:
    """Resolve a runnable script inside a skill, refusing anything else.

    Resolving against ``scripts/`` rather than the skill root is strictly
    tighter than ``read_skill_file`` and subsumes its escape check: ``..``,
    absolute paths, and symlink escapes all fail the one ``is_relative_to``.
    The rule it leaves is a sentence a skill author can hold in their head —
    references/ is read, scripts/ is run — so a skill cannot be talked into
    executing its own documentation.
    """
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("'path' is required and must be a non-empty string")
    base = (skill.path / "scripts").resolve()
    target = (skill.path / rel_path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(
            f"path {rel_path!r} is not under the skill's scripts/ directory"
        )
    if target.suffix != ".py":
        raise ValueError(f"only .py files can be run: {rel_path}")
    if not target.is_file():
        raise ValueError(f"no such script: {rel_path}")
    return target


def child_env(skill: Skill, ws: Workspace, policy: ScriptPolicy) -> dict[str, str]:
    """The environment a script runs with: built from empty, never inherited.

    A script's stdout goes straight back into the model's context, so an
    inherited ANTHROPIC_API_KEY would be one ``print(os.environ)`` away from the
    transcript. What a skill declares is a *request*; what it gets is the
    intersection with the operator's allowlist, because skills are bind-mounted
    and may come from anywhere.

    PYTHONDONTWRITEBYTECODE earns its place: /skills is mounted read-only, and a
    __pycache__ write that silently fails on every run is noise nobody will chase.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(ws.root)),
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AGENTHARNESS_WORKSPACE_DIR": str(ws.root),
        "AGENTHARNESS_SKILL_DIR": str(skill.path),
    }
    for name in sorted(skill.script_env & policy.env_allowlist):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


@dataclass(frozen=True)
class ScriptResult:
    """One completed run. ``dropped`` counts bytes lost to the output cap."""

    exit_status: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


class ScriptTimeout(ValueError):
    """A script that outlived its timeout and was killed, with what it had said.

    The partial output travels with the error because that is usually where a
    hang explains itself.
    """

    def __init__(
        self,
        timeout: int,
        stdout: str,
        stderr: str,
        stdout_dropped: int,
        stderr_dropped: int,
    ) -> None:
        super().__init__(f"script exceeded its {timeout}s timeout and was killed")
        self.timeout = timeout
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_dropped = stdout_dropped
        self.stderr_dropped = stderr_dropped


def _truncate(text: str, limit: int) -> tuple[str, int]:
    """Cut ``text`` to ``limit`` bytes of UTF-8, on a character boundary."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, 0
    kept = raw[:limit].decode("utf-8", errors="ignore")
    return kept, len(raw) - len(kept.encode("utf-8"))


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def run_script(
    skill: Skill,
    rel_path: str,
    ws: Workspace,
    policy: ScriptPolicy,
    *,
    args: Sequence[str] = (),
    stdin: str | None = None,
    timeout: int | None = None,
) -> ScriptResult:
    """Run a script bundled with ``skill`` and return what it said.

    A non-zero exit is a *result*, not an exception: a script that exits 2 and
    explains itself on stderr has told the model something it can act on.
    Exceptions are for the harness failing to run the thing at all.

    Note the honest limit: a subprocess is not confined by
    ``workspace.resolve_in``. It can reach any path the container user can. The
    container is the confinement, which is why only vetted scripts may run.
    """
    if not policy.enabled:
        raise ValueError(
            "skill scripts are disabled; set skill_scripts.enabled: true in config "
            "(or AGENTHARNESS_SKILL_SCRIPTS_ENABLED=true) to allow them"
        )
    target = resolve_script(skill, rel_path)
    if not ws.root.is_dir():
        raise ValueError(f"workspace directory does not exist: {ws.root}")
    seconds = policy.clamp_timeout(timeout)

    # A list, never shell=True: there is no shell, so there is nothing to inject
    # into. sys.executable rather than a PATH lookup, so a `python` earlier on
    # PATH cannot substitute itself. start_new_session gives the child a process
    # group of its own — subprocess's timeout kills the direct child only, and a
    # script that forked would otherwise leave orphans holding the pipes open.
    proc = subprocess.Popen(
        [sys.executable, str(target), *args],
        cwd=str(ws.root),
        env=child_env(skill, ws, policy),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",  # a script writing non-UTF-8 has a bug; we don't raise
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(stdin, timeout=seconds)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired as escaped:
            # A grandchild escaped the group: the pipes never drained, so
            # communicate() never got to decode or close them itself. CPython
            # still hands back whatever it had already buffered (as raw
            # bytes, since decoding happens only on a clean return) — that is
            # worth keeping, since this is the case where a hang most needs
            # explaining. Close the pipes and reap what we can so a dangling
            # Popen doesn't warn on GC.
            out = (escaped.stdout or b"").decode("utf-8", errors="replace")
            err = (escaped.stderr or b"").decode("utf-8", errors="replace")
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
            proc.poll()
        kept_out, dropped_out = _truncate(out or "", policy.max_output_bytes)
        kept_err, dropped_err = _truncate(err or "", policy.max_output_bytes)
        raise ScriptTimeout(seconds, kept_out, kept_err, dropped_out, dropped_err) from None

    kept_out, dropped_out = _truncate(out, policy.max_output_bytes)
    kept_err, dropped_err = _truncate(err, policy.max_output_bytes)
    return ScriptResult(
        exit_status=proc.returncode,
        stdout=kept_out,
        stderr=kept_err,
        stdout_dropped=dropped_out,
        stderr_dropped=dropped_err,
    )
