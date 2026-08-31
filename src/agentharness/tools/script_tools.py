"""The ``run_skill_script`` tool: execution of a skill's bundled Python.

The thin half of the seam. Everything safety-critical — what may run, what
environment it gets, how it is killed — lives in ``skills/runner.py``; this
module owns the schema, the argument coercion, and the rendering the model
reads. A disabled policy returns no tools at all, so the model is never offered
a capability it cannot have, exactly as a read-only workspace withholds
``write_file``.
"""

from __future__ import annotations

from typing import Any

from agentharness.skills.loader import SkillSet
from agentharness.skills.runner import (
    ScriptPolicy,
    ScriptResult,
    ScriptTimeout,
    run_script,
)
from agentharness.tools.registry import Tool
from agentharness.workspace import Workspace


def _stream(text: str, dropped: int) -> str:
    body = text.rstrip("\n") or "(empty)"
    if dropped:
        body += f"\n... truncated: {dropped} more bytes"
    return body


def _render(result: ScriptResult) -> str:
    return "\n".join(
        [
            f"exit status: {result.exit_status}",
            "--- stdout ---",
            _stream(result.stdout, result.stdout_dropped),
            "--- stderr ---",
            _stream(result.stderr, result.stderr_dropped),
        ]
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(
            "'args' must be a list of strings — quote numbers, e.g. [\"3\"] not [3]"
        )
    return value


def make_script_tools(
    skillset: SkillSet, ws: Workspace, policy: ScriptPolicy
) -> list[Tool]:
    if not policy.enabled:
        return []

    def _run(args: dict[str, Any]) -> str:
        name = args.get("skill")
        path = args.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError("'skill' is required")
        if not isinstance(path, str) or not path:
            raise ValueError("'path' is required")
        skill = skillset.by_name(name)
        if skill is None:
            available = ", ".join(s.name for s in skillset.skills) or "(none)"
            raise ValueError(f"unknown skill {name!r}; available: {available}")
        stdin = args.get("stdin")
        if stdin is not None and not isinstance(stdin, str):
            raise ValueError("'stdin' must be a string")
        try:
            result = run_script(
                skill,
                path,
                ws,
                policy,
                args=_string_list(args.get("args")),
                stdin=stdin,
                timeout=args.get("timeout"),
            )
        except ScriptTimeout as exc:
            # An error, but the partial output goes with it: that is usually
            # where the hang explains itself.
            raise ValueError(
                f"{exc}\n--- stdout so far ---\n{_stream(exc.stdout, exc.stdout_dropped)}\n"
                f"--- stderr so far ---\n{_stream(exc.stderr, exc.stderr_dropped)}"
            ) from None
        return _render(result)

    return [
        Tool(
            name="run_skill_script",
            description=(
                "Run a Python script bundled with a skill, under its scripts/ "
                "directory, and get back its exit status, stdout, and stderr. Use "
                "this rather than reimplementing what a skill's script already "
                "does; the skill's instructions say which script to use and what "
                "arguments it takes. The script runs with the workspace as its "
                "working directory, so relative paths are workspace paths."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "The skill's name."},
                    "path": {
                        "type": "string",
                        "description": (
                            "The script, relative to the skill directory, e.g. "
                            "'scripts/analyse.py'."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command-line arguments, each a string.",
                        "default": [],
                    },
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Text to send to the script's standard input. Omit to "
                            "give it no input."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Seconds to allow (default {policy.default_timeout}, "
                            f"capped at {policy.max_timeout})."
                        ),
                    },
                },
                "required": ["skill", "path"],
                "additionalProperties": False,
            },
            handler=_run,
        )
    ]
