"""Conversation states — multiple independent, in-memory sessions.

Each state has its own history, base system prompt, provider, and model. A
state is bound to one provider at creation because assistant turns are replayed
via ``Message.provider_raw``, which only its author can interpret.

State is in-memory only: everything is lost when the REPL process exits.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentharness.providers.base import Message, Provider, Usage

# Builds a provider bound to (provider_name, model).
ProviderBuilder = Callable[[str, str], Provider]

# The model a state on a given provider should bind to when none was asked for.
# Injected rather than imported: a provider may declare its own default model,
# which is a config concern, and this module deliberately knows only the neutral
# provider protocol.
ModelResolver = Callable[[str], str]


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
    _provider_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def provider(self) -> Provider:
        """The state's Provider, built on first use.

        Locked because parallel tool calls can reach a state's first use at the
        same moment (four subagents on one provider, say), and an unguarded
        check-then-build would construct — and leak — a second SDK client.
        """
        if self._provider is None:
            with self._provider_lock:
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
        default_model_for: ModelResolver,
        default_system: str,
    ) -> None:
        self._build = provider_builder
        self._default_provider = default_provider
        self._default_model_for = default_model_for
        self._default_system = default_system
        self._states: dict[str, ConversationState] = {}
        self._active: str | None = None
        # Guards the table and the active pointer. Background jobs and parallel
        # subagent spawns all create states, so name allocation has to be
        # atomic: without this, two spawns can both pass the duplicate check
        # and one silently replaces the other.
        self._lock = threading.Lock()
        self.new("default")

    # ---- lifecycle ----------------------------------------------------------

    def new(
        self,
        name: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        system: str | None = None,
        activate: bool = True,
    ) -> ConversationState:
        """Create a state, by default switching to it.

        ``activate=False`` is for states created *by* a turn rather than by the
        user — a subagent's. A background job's subagent must not flip the
        state the REPL prompt is pointing at while someone is typing into it.
        """
        provider_name = provider or self._default_provider
        # Resolved against the chosen provider, not globally: switching provider
        # without naming a model must not carry the old provider's model over.
        model_name = model or self._default_model_for(provider_name)
        with self._lock:
            if name in self._states:
                raise ValueError(f"state {name!r} already exists")
            state = ConversationState(
                name=name,
                provider_name=provider_name,
                model=model_name,
                system=system if system is not None else self._default_system,
                provider_builder=lambda: self._build(provider_name, model_name),
            )
            self._states[name] = state
            if activate:
                self._active = name
        return state

    def switch(self, name: str) -> ConversationState:
        with self._lock:
            if name not in self._states:
                raise ValueError(f"no such state: {name}")
            self._active = name
            return self._states[name]

    def delete(self, name: str) -> None:
        with self._lock:
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
        with self._lock:
            assert self._active is not None
            return self._states[self._active]

    def names(self) -> list[str]:
        with self._lock:
            return list(self._states)

    def get(self, name: str) -> ConversationState | None:
        with self._lock:
            return self._states.get(name)

    def __iter__(self):
        # A snapshot, not a live view: a subagent spawned mid-iteration would
        # otherwise turn `/states` or `/usage --all` into a RuntimeError.
        with self._lock:
            return iter(list(self._states.values()))
