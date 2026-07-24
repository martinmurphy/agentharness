"""A tool that lists models — on the active provider, a named one, or all.

Depends on live state (which provider the conversation uses) and makes network
calls, so the harness builds it with closures rather than the state-free
``build_default_registry``.

``provider`` argument:
- omitted        -> the active conversation's provider
- a provider name -> that provider (needs its API key set)
- ``"all"``      -> every provider whose key is set (unavailable ones are noted,
                    and a per-provider failure is reported without aborting)
"""

from __future__ import annotations

from collections.abc import Callable

from agentharness.providers.factory import provider_status
from agentharness.tools.registry import Tool


def _format_one(name: str, models: list[str]) -> str:
    if not models:
        return f"{name}: no models returned"
    return f"{name}:\n" + "\n".join(f"  {m}" for m in models)


def list_models_tool(
    active_provider_name: Callable[[], str],
    list_for: Callable[[str], list[str]],
) -> Tool:
    """Build the ``list_models`` tool.

    ``active_provider_name()`` gives the current conversation's provider;
    ``list_for(name)`` returns that provider's model IDs (and may raise on a
    missing key, unknown provider, or network/auth error).
    """
    known = [p.name for p in provider_status()]

    def _handler(args: dict) -> str:
        provider = args.get("provider")

        if provider is None:
            name = active_provider_name()
            return _format_one(name, list_for(name))

        if provider == "all":
            blocks: list[str] = []
            for p in provider_status():
                if not p.available:
                    blocks.append(f"{p.name}: (no key set — set {p.env_var} to list)")
                    continue
                try:
                    blocks.append(_format_one(p.name, list_for(p.name)))
                except Exception as exc:  # noqa: BLE001 - report per provider, don't abort "all"
                    blocks.append(f"{p.name}: error: {exc}")
            return "\n".join(blocks)

        if provider not in known:
            raise ValueError(
                f"unknown provider {provider!r}; choose one of "
                f"{', '.join(known)}, or 'all'"
            )
        return _format_one(provider, list_for(provider))

    return Tool(
        name="list_models",
        description=(
            "List available model IDs. Omit 'provider' for the current "
            "conversation's provider; pass a provider name for a specific one "
            "(its API key must be set); or pass 'all' to list every provider "
            "whose key is set. Use list_providers first to see which keys are set."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": [*known, "all"],
                    "description": "Provider to list, or 'all'. Omit for the active provider.",
                }
            },
            "additionalProperties": False,
        },
        handler=_handler,
    )
