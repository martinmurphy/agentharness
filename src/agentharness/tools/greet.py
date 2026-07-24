"""The example ``greet`` tool — trivial, exists to prove the loop end to end."""

from __future__ import annotations

from typing import Any

from agentharness.tools.registry import Tool

_TEMPLATES = {
    "formal": "Good day, {name}. It is a pleasure to make your acquaintance.",
    "casual": "Hey {name}! Nice to meet you.",
    "enthusiastic": "{name}!! So great to meet you!! 🎉",
}


def _greet(args: dict[str, Any]) -> str:
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' is required and must be a non-empty string")
    style = args.get("style", "casual")
    template = _TEMPLATES.get(style)
    if template is None:
        raise ValueError(f"unknown style {style!r}; choose one of {sorted(_TEMPLATES)}")
    return template.format(name=name.strip())


def greet_tool() -> Tool:
    return Tool(
        name="greet",
        description="Generate a greeting for a person in a chosen style.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The person's name."},
                "style": {
                    "type": "string",
                    "enum": ["formal", "casual", "enthusiastic"],
                    "description": "Tone of the greeting.",
                    "default": "casual",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=_greet,
    )
