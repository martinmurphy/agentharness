"""OpenAI-compatible adapter for the neutral message model.

Works against any server speaking the OpenAI Chat Completions dialect — OpenAI
itself, or a local Ollama / vLLM endpoint via ``base_url``.

Two mapping differences from Anthropic are load-bearing:

- the system prompt is a leading ``{"role": "system"}`` message, not a field;
- a neutral tool turn carrying N results fans out to N separate
  ``{"role": "tool"}`` messages (Anthropic packs them into one).
"""

from __future__ import annotations

import json
from typing import Any

from agentharness.providers.base import (
    Message,
    ProviderResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

_FINISH_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        **options: Any,
    ) -> None:
        self.model = model
        if client is None:
            import openai

            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            if "api_key" in options:
                kwargs["api_key"] = options["api_key"]
            client = openai.OpenAI(**kwargs)
        self._client = client

    # ---- request building ---------------------------------------------------

    @staticmethod
    def _tool_specs(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    @classmethod
    def _assistant_wire(cls, msg: Message) -> dict[str, Any]:
        if msg.provider_raw is not None:
            return msg.provider_raw
        text = msg.text()
        wire: dict[str, Any] = {"role": "assistant", "content": text or None}
        calls = [b for b in msg.blocks if isinstance(b, ToolCall)]
        if calls:
            wire["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in calls
            ]
        return wire

    @classmethod
    def _to_wire_messages(cls, system: str, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.role == "assistant":
                wire.append(cls._assistant_wire(msg))
            elif msg.role == "tool":
                # Fan out: one {"role": "tool"} message per result.
                for b in msg.blocks:
                    if isinstance(b, ToolResult):
                        content = b.content
                        if b.is_error and not content.lower().startswith("error"):
                            content = f"Error: {content}"
                        wire.append(
                            {"role": "tool", "tool_call_id": b.call_id, "content": content}
                        )
            else:  # user
                wire.append({"role": "user", "content": msg.text()})
        return wire

    # ---- response parsing ---------------------------------------------------

    @staticmethod
    def _parse_message(raw_message: Any) -> tuple[list, dict[str, Any]]:
        blocks: list = []
        content = getattr(raw_message, "content", None)
        if content:
            blocks.append(TextBlock(text=content))

        tool_calls = getattr(raw_message, "tool_calls", None) or []
        for tc in tool_calls:
            fn = tc.function
            try:
                args = json.loads(fn.arguments) if fn.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            blocks.append(ToolCall(id=tc.id, name=fn.name, arguments=args))

        # The assistant turn is replayed from this dict, so it has to keep
        # fields this adapter knows nothing about. Gemini 3.x behind an
        # OpenAI-compatible endpoint is the case in point: it returns a
        # thought_signature alongside each function call and rejects a replayed
        # call that has lost it ("Function call is missing a thought_signature
        # in functionCall parts"). The SDK's models allow extra fields, so
        # dumping one preserves whatever the server sent; naming the fields by
        # hand silently drops the rest, which is the bug this replaced.
        dump = getattr(raw_message, "model_dump", None)
        if dump is not None:
            raw_wire: dict[str, Any] = dump(exclude_none=True)
            # A tool-call-only turn has always gone out with an explicit null
            # rather than no content key; keep that wire shape.
            raw_wire.setdefault("content", None)
            return blocks, raw_wire

        # A client that is not an SDK model (a test double, an injected stub)
        # has no extras to preserve, so the fields we know are all there is.
        raw_wire = {"role": "assistant", "content": content}
        if tool_calls:
            raw_wire["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        return blocks, raw_wire

    # ---- the protocol method ------------------------------------------------

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._to_wire_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = self._tool_specs(tools)
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        blocks, raw_wire = self._parse_message(choice.message)
        stop_reason: StopReason = _FINISH_MAP.get(choice.finish_reason or "", "other")
        assistant = Message(role="assistant", blocks=blocks, provider_raw=raw_wire)

        usage_obj = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        )
        return ProviderResponse(message=assistant, stop_reason=stop_reason, usage=usage)

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list().data)
