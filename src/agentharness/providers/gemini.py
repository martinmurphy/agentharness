"""Google Gemini adapter (AI Studio key) for the neutral message model.

Uses the ``google-genai`` SDK against the Gemini Developer API — the API key
comes from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in the environment, the
same env-only pattern as the other adapters. This is the AI Studio path, not
Vertex AI.

Mapping notes specific to Gemini:

- roles are ``user`` / ``model`` (assistant -> ``model``); tool results ride in
  a ``user``-role content as ``function_response`` parts;
- a tool result references its call by function *name* (and id when present),
  not an opaque id, so we recover the name from the preceding model turn;
- Gemini reports ``finish_reason: STOP`` even when it emits function calls, so
  tool-use is detected by the presence of function-call parts, not the reason.
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

# finish_reason (as string) -> neutral stop reason. Anything safety-related is a
# refusal; tool-use is handled separately (see chat()).
_REFUSAL_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
}


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        show_thinking: bool = False,
        **options: Any,
    ) -> None:
        self.model = model
        self.show_thinking = show_thinking
        if client is None:
            from google import genai

            # api_key defaults to GEMINI_API_KEY / GOOGLE_API_KEY from the env.
            kwargs = {k: v for k, v in options.items() if k in ("api_key",)}
            client = genai.Client(**kwargs)
        self._client = client

    # ---- request building ---------------------------------------------------

    @staticmethod
    def _tools_config(tools: list[ToolSpec]):
        from google.genai import types

        if not tools:
            return None
        declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters_json_schema=t.input_schema,
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _id_to_name(messages: list[Message]) -> dict[str, str]:
        """Map tool_call id -> function name, from the model turns in history."""
        mapping: dict[str, str] = {}
        for msg in messages:
            if msg.role == "assistant":
                for b in msg.blocks:
                    if isinstance(b, ToolCall):
                        mapping[b.id] = b.name
        return mapping

    @classmethod
    def _to_contents(cls, messages: list[Message]) -> list:
        from google.genai import types

        id_to_name = cls._id_to_name(messages)
        contents: list = []
        for msg in messages:
            if msg.role == "assistant":
                if msg.provider_raw is not None:
                    contents.append(msg.provider_raw)  # verbatim (keeps thought sigs)
                else:
                    contents.append(
                        types.Content(role="model", parts=cls._assistant_parts(msg))
                    )
            elif msg.role == "tool":
                parts = []
                for b in msg.blocks:
                    if isinstance(b, ToolResult):
                        payload = {"error": b.content} if b.is_error else {"result": b.content}
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=b.call_id,
                                    name=id_to_name.get(b.call_id, b.call_id),
                                    response=payload,
                                )
                            )
                        )
                contents.append(types.Content(role="user", parts=parts))
            else:  # user
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=msg.text())])
                )
        return contents

    @staticmethod
    def _assistant_parts(msg: Message) -> list:
        """Reconstruct model parts when provider_raw is absent (e.g. tests)."""
        from google.genai import types

        parts: list = []
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                parts.append(types.Part(text=b.text))
            elif isinstance(b, ToolCall):
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=b.id, name=b.name, args=b.arguments
                        )
                    )
                )
        return parts

    # ---- response parsing ---------------------------------------------------

    @staticmethod
    def _parse_parts(parts: Any) -> list:
        blocks: list = []
        for part in parts or []:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = dict(getattr(fc, "args", None) or {})
                call_id = getattr(fc, "id", None) or fc.name
                blocks.append(ToolCall(id=call_id, name=fc.name, arguments=args))
                continue
            text = getattr(part, "text", None)
            if text:
                if getattr(part, "thought", False):
                    blocks.append(ThinkingBlock(text=text))
                else:
                    blocks.append(TextBlock(text=text))
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
        from google.genai import types

        # Thought parts are only returned when explicitly requested.
        thinking_config = (
            types.ThinkingConfig(include_thoughts=True) if self.show_thinking else None
        )
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            tools=self._tools_config(tools),
            thinking_config=thinking_config,
            # We drive the loop ourselves; never let the SDK auto-execute tools.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=self._to_contents(messages),
            config=config,
        )

        candidates = getattr(resp, "candidates", None) or []
        content = candidates[0].content if candidates else None
        raw_parts = getattr(content, "parts", None) if content is not None else None
        blocks = self._parse_parts(raw_parts)

        finish = candidates[0].finish_reason if candidates else None
        finish_str = getattr(finish, "value", finish) or ""
        assistant = Message(role="assistant", blocks=blocks, provider_raw=content)

        if any(isinstance(b, ToolCall) for b in blocks):
            stop_reason: StopReason = "tool_use"
        elif finish_str == "STOP":
            stop_reason = "end_turn"
        elif finish_str == "MAX_TOKENS":
            stop_reason = "max_tokens"
        elif finish_str in _REFUSAL_REASONS:
            stop_reason = "refusal"
        else:
            stop_reason = "other"

        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        )
        return ProviderResponse(message=assistant, stop_reason=stop_reason, usage=usage)

    def list_models(self) -> list[str]:
        # Keep only models that can generate content (excludes embeddings, etc.),
        # and strip the "models/" prefix so the ids are usable with /new --model.
        out: list[str] = []
        for m in self._client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions and m.name:
                out.append(m.name.split("/")[-1])
        return sorted(out)
