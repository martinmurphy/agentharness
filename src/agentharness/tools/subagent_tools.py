"""A tool that lets the model delegate a task to a fresh subagent.

The subagent is a new conversation state with its own provider/model that runs
its own tool + skill loop until it produces an answer, which is returned to the
caller as the tool result. It is a stateful tool (it needs the harness to create
states and drive the loop), so the harness builds it with a ``runner`` closure —
see ``Harness._spawn_subagent`` in ``repl.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from agentharness.tools.registry import Tool

# runner(task, provider, model) -> the subagent's final answer
SubagentRunner = Callable[[str, "str | None", "str | None"], str]


def spawn_subagent_tool(runner: SubagentRunner, providers: list[str]) -> Tool:
    def _handler(args: dict) -> str:
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("'task' is required and must be a non-empty string")
        provider = args.get("provider")
        model = args.get("model")
        return runner(task.strip(), provider, model)

    return Tool(
        name="spawn_subagent",
        description=(
            "Delegate a task to a fresh subagent that runs its own tool and skill "
            "loop until it produces an answer, then returns that answer to you. "
            "By default the subagent uses the same provider and model as you; "
            "optionally set 'provider' and/or 'model' to run it on a different "
            "backend. Use this to isolate a subtask or get a second model's take."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What the subagent should work on until it has an answer.",
                },
                "provider": {
                    "type": "string",
                    "enum": providers,
                    "description": "Provider for the subagent (default: same as yours).",
                },
                "model": {
                    "type": "string",
                    "description": "Model ID for the subagent (default: same as yours).",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        handler=_handler,
    )
