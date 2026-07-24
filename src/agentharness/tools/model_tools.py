"""A tool that lists the models available on the active provider.

Unlike the other built-in tools, this one depends on live state (which provider
the current conversation uses) and makes a network call, so it is built by the
harness with a ``lister`` closure rather than by the state-free
``build_default_registry``.
"""

from __future__ import annotations

from collections.abc import Callable

from agentharness.tools.registry import Tool


def list_models_tool(lister: Callable[[], list[str]]) -> Tool:
    """Build a ``list_models`` tool. ``lister`` returns the active provider's
    model IDs (and may raise on an unsupported provider or a network/auth
    error — the registry surfaces that to the model as a tool error)."""

    def _handler(_args: dict) -> str:
        models = lister()
        if not models:
            return "No models were returned by the provider."
        return "\n".join(models)

    return Tool(
        name="list_models",
        description=(
            "List the model IDs available on the current provider (the one this "
            "conversation is using). Use it to discover valid model names, e.g. "
            "before suggesting the user switch models."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_handler,
    )
