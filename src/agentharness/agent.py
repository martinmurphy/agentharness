"""The provider-neutral agent loop.

``run_turn`` drives one user turn to completion, yielding events as it goes so
that rendering stays out of the loop (and the loop stays testable against a
fake provider). The caller appends the user Message before calling; the loop
appends the assistant and tool messages it produces.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from agentharness.providers.base import (
    Message,
    Provider,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from agentharness.state import ConversationState
from agentharness.tools.registry import ToolRegistry


class MaxIterationsExceeded(RuntimeError):
    def __init__(self, limit: int) -> None:
        super().__init__(f"tool loop exceeded {limit} iterations")
        self.limit = limit


# ---- events -----------------------------------------------------------------


@dataclass(frozen=True)
class ThinkingEvent:
    text: str


@dataclass(frozen=True)
class TextEvent:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    call: ToolCall


@dataclass(frozen=True)
class ToolResultEvent:
    result: ToolResult


@dataclass(frozen=True)
class UsageEvent:
    usage: Usage


@dataclass(frozen=True)
class DoneEvent:
    stop_reason: StopReason


Event = ThinkingEvent | TextEvent | ToolCallEvent | ToolResultEvent | UsageEvent | DoneEvent


def _events_for_message(message: Message) -> Iterator[Event]:
    for block in message.blocks:
        if isinstance(block, ThinkingBlock):
            if block.text:
                yield ThinkingEvent(block.text)
        elif isinstance(block, TextBlock):
            if block.text:
                yield TextEvent(block.text)
        elif isinstance(block, ToolCall):
            yield ToolCallEvent(block)


def run_turn(
    *,
    provider: Provider,
    state: ConversationState,
    registry: ToolRegistry,
    system: str,
    max_tokens: int,
    max_iterations: int,
) -> Iterator[Event]:
    """Run one turn: call the model, execute any tools, repeat until it stops.

    Assumes the triggering user Message is already appended to ``state``.
    """
    for _ in range(max_iterations):
        resp = provider.chat(
            system=system,
            messages=state.messages,
            tools=registry.specs(),
            max_tokens=max_tokens,
        )
        state.add(resp.message)
        state.usage = state.usage + resp.usage

        yield from _events_for_message(resp.message)
        yield UsageEvent(resp.usage)

        calls = resp.message.tool_calls()
        if not calls:
            yield DoneEvent(resp.stop_reason)
            return

        results: list[ToolResult] = []
        for call in calls:
            result = registry.dispatch(call)
            results.append(result)
            yield ToolResultEvent(result)
        state.add(Message(role="tool", blocks=list(results)))

    raise MaxIterationsExceeded(max_iterations)
