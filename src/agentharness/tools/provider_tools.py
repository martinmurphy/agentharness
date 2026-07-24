"""A tool that lists the model providers the harness supports."""

from __future__ import annotations

from typing import Any

from agentharness.providers.factory import provider_status
from agentharness.tools.registry import Tool


def _list_providers(_args: dict[str, Any]) -> str:
    lines = []
    for p in provider_status():
        state = "key set" if p.available else "no key"
        lines.append(f"{p.name} (key from {p.env_var}): {state}")
    return "\n".join(lines)


def list_providers_tool() -> Tool:
    return Tool(
        name="list_providers",
        description=(
            "List the model providers this harness supports and whether each one's "
            "API key is currently configured in the environment. 'no key' reports "
            "that the key env var is unset; it does not by itself mean the provider "
            "is unusable (a local OpenAI-compatible server may need no key)."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_list_providers,
    )
