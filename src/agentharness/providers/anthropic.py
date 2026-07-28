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


def _adaptive_unsupported(exc: Exception) -> bool:
    """True if a 400 says adaptive thinking / effort is unsupported by the model."""
    msg = str(getattr(exc, "message", None) or exc).lower()
    return "not supported" in msg and ("thinking" in msg or "effort" in msg)


def _build_client(vertex: bool, options: dict[str, Any]) -> Any:
    """Construct the SDK client for the first-party API, or for Vertex AI.

    Vertex is the same Messages API behind Google's endpoint — same request
    shape, same response shape, so the whole adapter below is shared. Only the
    client class and how it authenticates differ: Vertex uses Google
    Application Default Credentials, so there is no api_key to pass and no
    ANTHROPIC_API_KEY to read. ``project_id`` and ``region`` may be omitted,
    in which case the SDK reads ANTHROPIC_VERTEX_PROJECT_ID and CLOUD_ML_REGION.
    """
    import anthropic

    if vertex:
        allowed = ("project_id", "region", "base_url")
        return anthropic.AnthropicVertex(**{k: v for k, v in options.items() if k in allowed})
    # base_url is optional; api_key comes from ANTHROPIC_API_KEY.
    return anthropic.Anthropic(
        **{k: v for k, v in options.items() if k in ("base_url", "api_key")}
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        effort: str = "high",
        show_thinking: bool = False,
        thinking_budget: int | None = None,
        client: Any = None,
        vertex: bool = False,
        **options: Any,
    ) -> None:
        self.model = model
        self.effort = effort
        self.show_thinking = show_thinking
        # Fixed thinking budget for models that reject adaptive thinking; used
        # only on the non-adaptive fallback path. None = no thinking there.
        self.thinking_budget = thinking_budget
        # Adaptive thinking + effort are supported on current models (Opus 4.6+,
        # Sonnet 4.6+, …) but 400 on older ones (Haiku 4.5, Sonnet 4.5, …). We
        # try them, and on that specific 400 drop them and remember it for this
        # instance so later turns skip straight to the plain request.
        self._supports_adaptive = True
        self.vertex = vertex
        if client is None:
            client = _build_client(vertex, options)
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

    def _create(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
        adaptive: bool,
    ):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_wire_messages(messages),
            "tools": self._tool_specs(tools),
        }
        if adaptive:
            kwargs["thinking"] = self._thinking()
            kwargs["output_config"] = {"effort": self.effort}
        else:
            # Older models: optional fixed-budget extended thinking (no effort —
            # they reject output_config). Budget must satisfy 1024 <= N < max_tokens.
            budget = self._enabled_budget(max_tokens)
            if budget is not None:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return self._client.messages.create(**kwargs)

    def _enabled_budget(self, max_tokens: int) -> int | None:
        """Clamp the configured thinking budget to the API's bounds, or None if
        unset or if max_tokens leaves no room for the 1024-token minimum."""
        if not self.thinking_budget:
            return None
        budget = min(self.thinking_budget, max_tokens - 1)
        return budget if budget >= 1024 else None

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ProviderResponse:
        import anthropic

        call = {"system": system, "messages": messages, "tools": tools, "max_tokens": max_tokens}
        try:
            resp = self._create(**call, adaptive=self._supports_adaptive)
        except anthropic.BadRequestError as exc:
            if not (self._supports_adaptive and _adaptive_unsupported(exc)):
                raise
            # This model rejects adaptive thinking / effort; retry without them
            # and skip them from now on.
            self._supports_adaptive = False
            resp = self._create(**call, adaptive=False)

        stop_reason: StopReason = _STOP_MAP.get(resp.stop_reason or "", "other")
        blocks = self._parse_content(resp.content)
        assistant = Message(role="assistant", blocks=blocks, provider_raw=resp.content)
        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )
        return ProviderResponse(message=assistant, stop_reason=stop_reason, usage=usage)

    def list_models(self) -> list[str]:
        # Vertex serves the Messages API but not the Models API, so there is
        # nothing to enumerate there — the model has to be named in config.
        models = getattr(self._client, "models", None)
        if models is None:
            raise NotImplementedError(
                "Vertex AI does not serve the Models API; name the model in config "
                "(see https://cloud.google.com/vertex-ai/generative-ai/docs for what "
                "your project can reach)"
            )
        return sorted(m.id for m in models.list())
