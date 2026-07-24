"""Conversation states — multiple independent, in-memory sessions.

Each state has its own history, base system prompt, provider, and model. A
state is bound to one provider at creation because assistant turns are replayed
via ``Message.provider_raw``, which only its author can interpret.

State is in-memory only: everything is lost when the REPL process exits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentharness.providers.base import Message, Provider, Usage

# Builds a provider bound to (provider_name, model).
ProviderBuilder = Callable[[str, str], Provider]


@dataclass
class ConversationState:
    name: str
    provider_name: str
    model: str
    system: str  # base system prompt; the skills catalog is appended at call time
    # Zero-arg builder; the Provider (and its SDK client) is created lazily on
    # first use, so the REPL can start and run non-model commands with no API key.
    provider_builder: Callable[[], Provider]
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    _provider: Provider | None = field(default=None, repr=False)

    @property
    def provider(self) -> Provider:
        if self._provider is None:
            self._provider = self.provider_builder()
        return self._provider

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def reset(self) -> None:
        self.messages.clear()
        self.usage = Usage()


class StateManager:
    def __init__(
        self,
        provider_builder: ProviderBuilder,
        *,
        default_provider: str,
        default_model: str,
        default_system: str,
    ) -> None:
        self._build = provider_builder
        self._default_provider = default_provider
        self._default_model = default_model
        self._default_system = default_system
        self._states: dict[str, ConversationState] = {}
        self._active: str | None = None
        self.new("default")

    # ---- lifecycle ----------------------------------------------------------

    def new(
        self,
        name: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        system: str | None = None,
    ) -> ConversationState:
        if name in self._states:
            raise ValueError(f"state {name!r} already exists")
        provider_name = provider or self._default_provider
        model_name = model or self._default_model
        state = ConversationState(
            name=name,
            provider_name=provider_name,
            model=model_name,
            system=system if system is not None else self._default_system,
            provider_builder=lambda: self._build(provider_name, model_name),
        )
        self._states[name] = state
        self._active = name
        return state

    def switch(self, name: str) -> ConversationState:
        if name not in self._states:
            raise ValueError(f"no such state: {name}")
        self._active = name
        return self._states[name]

    def delete(self, name: str) -> None:
        if name not in self._states:
            raise ValueError(f"no such state: {name}")
        if len(self._states) == 1:
            raise ValueError("cannot delete the last remaining state")
        del self._states[name]
        if self._active == name:
            self._active = next(iter(self._states))

    # ---- access -------------------------------------------------------------

    @property
    def active(self) -> ConversationState:
        assert self._active is not None
        return self._states[self._active]

    def names(self) -> list[str]:
        return list(self._states)

    def __iter__(self):
        return iter(self._states.values())
