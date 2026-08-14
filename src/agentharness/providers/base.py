"""The provider-neutral message model and the Provider protocol.

Everything above the providers layer speaks in these types; each provider
adapter maps them to and from its own wire format. Keeping the agent loop in
terms of these neutral types is what lets a new backend be a new file rather
than a refactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# ---- content blocks ---------------------------------------------------------


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ThinkingBlock:
    """Model reasoning, when a provider surfaces it. Rendered but not required."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


Block = TextBlock | ThinkingBlock | ToolCall | ToolResult


# ---- messages ---------------------------------------------------------------


@dataclass
class Message:
    """One turn in a conversation.

    ``provider_raw`` holds the original wire-format content that produced an
    assistant message. On replay a provider re-emits its own ``provider_raw``
    verbatim and ignores anyone else's — this is how Anthropic's signed thinking
    blocks survive a round trip, since they have no neutral representation. A
    state is bound to one provider, so raw content is only ever replayed to its
    author.
    """

    role: Literal["user", "assistant", "tool"]
    blocks: list[Block]
    provider_raw: Any = None

    def text(self) -> str:
        """Concatenated text of all TextBlocks (ignores thinking and tools)."""
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.blocks if isinstance(b, ToolCall)]


# ---- tools ------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """A tool advertised to the model. ``input_schema`` is JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ---- responses --------------------------------------------------------------

StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ProviderResponse:
    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)


# ---- the protocol -----------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """A model backend. Implementations own the wire-format mapping only —
    never the agent loop."""

    name: str

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ProviderResponse: ...

    def list_models(self) -> list[str]:
        """Model IDs this provider's credentials can reach (text-capable).

        Optional: providers may raise ``NotImplementedError``. Callers should
        also tolerate its absence and any SDK/network error.
        """
        ...


# ---- error classification ----------------------------------------------------


def is_model_not_found(exc: BaseException) -> bool:
    """True if this failure means "that model ID is not one I serve".

    Worth telling apart from every other failure because it is the one with an
    obvious next move — look the names up — and the caller that hit it is
    usually a model, which will only make that move if something says so.

    Deliberately one duck-typed rule rather than a method on each adapter. The
    SDKs differ only in which attribute carries the HTTP status (the Anthropic
    and OpenAI clients use ``status_code``; ``google-genai`` uses ``code``), and
    a rule per adapter would be three copies of one rule — including for the
    next adapter, which will also be HTTP.

    The status alone is not enough: a wrong ``base_url`` 404s as readily as a
    wrong model, and answering that with "call list_models" sends the caller
    somewhere that will fail too. Requiring the word *model* in the message
    separates them — every SDK names the thing it could not find, whether or not
    it echoes the ID (Cerebras returns "Model does not exist" and no ID at all).

    Only the exception itself is examined, not its ``__cause__`` chain: an SDK
    that wrapped a 404 in something else would be missed, and the caller simply
    gets the unhinted message it got before.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if str(status) != "404":
        return False
    return "model" in str(exc).lower()
