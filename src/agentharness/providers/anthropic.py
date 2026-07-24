"""Anthropic adapter for the neutral message model.

Uses ``client.messages.create`` directly — deliberately not the SDK's tool
runner, which would own the agent loop we keep provider-neutral.

Assistant turns are replayed from ``provider_raw`` (the SDK's own content
blocks). That is what preserves adaptive-thinking blocks, which carry
signatures and must be sent back unmodified; they have no neutral form.
"""

from __future__ import annotations

from typing import Any

from agentharness.providers.base import (
    Message,
    ProviderResponse,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

_STOP_MAP: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        effort: str = "high",
        show_thinking: bool = False,
        client: Any = None,
        **options: Any,
    ) -> None:
        self.model = model
        self.effort = effort
        self.show_thinking = show_thinking
        if client is None:
            import anthropic

            # base_url is optional; api_key comes from ANTHROPIC_API_KEY.
            kwargs = {k: v for k, v in options.items() if k in ("base_url", "api_key")}
            client = anthropic.Anthropic(**kwargs)
        self._client = client

    # ---- request building ---------------------------------------------------

    def _thinking(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"type": "adaptive"}
        if self.show_thinking:
            cfg["display"] = "summarized"
        return cfg

    @staticmethod
    def _tool_specs(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

    @classmethod
    def _to_wire_messages(cls, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "assistant":
                # Replay our own content blocks verbatim (keeps signed thinking).
                if msg.provider_raw is not None:
                    wire.append({"role": "assistant", "content": msg.provider_raw})
                else:
                    wire.append(
                        {"role": "assistant", "content": cls._assistant_fallback(msg)}
                    )
            elif msg.role == "tool":
                # tool_result blocks must ride in a user message.
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": b.call_id,
                        "content": b.content,
                        "is_error": b.is_error,
                    }
                    for b in msg.blocks
                    if isinstance(b, ToolResult)
                ]
                wire.append({"role": "user", "content": content})
            else:  # user
                wire.append({"role": "user", "content": msg.text()})
        return wire

    @staticmethod
    def _assistant_fallback(msg: Message) -> list[dict[str, Any]]:
        """Reconstruct assistant content when provider_raw is absent.

        Thinking blocks are omitted here — they cannot be rebuilt without their
        signature. In normal operation provider_raw is always set, so this runs
        only for synthetic messages (e.g. tests).
        """
        content: list[dict[str, Any]] = []
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                content.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolCall):
                content.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments}
                )
        return content

    # ---- response parsing ---------------------------------------------------

    @staticmethod
    def _parse_content(raw_content: Any) -> list:
        blocks: list = []
        for block in raw_content:
            btype = getattr(block, "type", None)
            if btype == "text":
                blocks.append(TextBlock(text=block.text))
            elif btype == "thinking":
                # display=omitted yields empty text; keep the block for order.
                blocks.append(ThinkingBlock(text=getattr(block, "thinking", "") or ""))
            elif btype == "tool_use":
                blocks.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
            # redacted_thinking and any future block types are preserved only in
            # provider_raw, not surfaced as neutral blocks.
        return blocks

    # ---- the protocol method ------------------------------------------------

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ProviderResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=self._to_wire_messages(messages),
            tools=self._tool_specs(tools),
            thinking=self._thinking(),
            output_config={"effort": self.effort},
        )

        stop_reason: StopReason = _STOP_MAP.get(resp.stop_reason or "", "other")
        blocks = self._parse_content(resp.content)
        assistant = Message(role="assistant", blocks=blocks, provider_raw=resp.content)
        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )
        return ProviderResponse(message=assistant, stop_reason=stop_reason, usage=usage)
