"""The provider-neutral agent loop.

``run_turn`` drives one user turn to completion, yielding events as it goes so
that rendering stays out of the loop (and the loop stays testable against a
fake provider). The caller appends the user Message before calling; the loop
appends the assistant and tool messages it produces.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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


def _dispatch_all(
    registry: ToolRegistry, calls: list[ToolCall], max_concurrency: int
) -> Iterator[tuple[int, ToolResult]]:
    """Run one message's tool calls, yielding ``(index, result)`` as each lands.

    One call runs inline: the common case must not pay for a pool. Several run
    concurrently — a model that asks four questions at once should wait for the
    slowest, not the sum.

    The index is what keeps the two orderings apart. Results are yielded in
    *completion* order so the terminal shows liveness, but the caller files them
    back into *call* order before they reach the history: a transcript whose
    tool results are ordered by whichever server answered first would replay
    differently every run.

    The pool is per-turn on purpose. A ``spawn_subagent`` handler blocks its
    worker for the whole of the subagent's turn, which needs workers of its own;
    one process-wide pool would deadlock as soon as outer calls filled it.
    Nesting is bounded at two levels — subagents get a registry with no
    ``spawn_subagent`` — so a turn's ceiling is ``n + n²`` mostly-blocked
    threads.
    """
    if len(calls) == 1 or max_concurrency <= 1:
        for index, call in enumerate(calls):
            yield index, registry.dispatch(call)
        return

    # A pool worker starts with an *empty* context, so the run context (which
    # state is calling, where its output goes) has to be carried in explicitly
    # or every handler that reads it would see nothing. One snapshot *per call*:
    # a Context cannot be entered twice, so a shared one would fail the moment
    # two workers ran at once.
    with ThreadPoolExecutor(
        max_workers=min(len(calls), max_concurrency),
        thread_name_prefix="tool",
    ) as pool:
        futures: dict[Future[ToolResult], int] = {
            pool.submit(contextvars.copy_context().run, registry.dispatch, call): index
            for index, call in enumerate(calls)
        }
        for future in as_completed(futures):
            yield futures[future], future.result()


def run_turn(
    *,
    provider: Provider,
    state: ConversationState,
    registry: ToolRegistry,
    system: str,
    max_tokens: int,
    max_iterations: int,
    max_concurrency: int = 1,
) -> Iterator[Event]:
    """Run one turn: call the model, execute any tools, repeat until it stops.

    Assumes the triggering user Message is already appended to ``state``.
    ``max_concurrency`` caps how many of one message's tool calls run at once.
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

        results: list[ToolResult | None] = [None] * len(calls)
        for index, result in _dispatch_all(registry, calls, max_concurrency):
            results[index] = result
            yield ToolResultEvent(result)
        state.add(Message(role="tool", blocks=[r for r in results if r is not None]))

    raise MaxIterationsExceeded(max_iterations)
