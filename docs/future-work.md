# Future work

Deferred improvements, kept here so they aren't lost. Each entry notes the
problem, why it's deferred, and a sketch of the fix.

## Surface the adaptive-thinking fallback (currently silent)

**Problem.** When an Anthropic model rejects `thinking={"type":"adaptive"}` +
`output_config.effort` (older models — Haiku 4.5, Sonnet 4.5), the adapter
catches the 400, drops those parameters, and retries — either with no thinking
or, if `thinking_budget` is set, with fixed-budget extended thinking. This all
happens **silently**: no log line, no on-screen notice. The user gets no signal
that the model doesn't support adaptive thinking or that a different thinking
mode was substituted; the only observable effect is a slightly slower first turn
(one extra round trip) on such a model.

See `AnthropicProvider.chat` / `_create` in `src/agentharness/providers/anthropic.py`.

**Why deferred.** The provider layer is intentionally decoupled from rendering
(it returns a `ProviderResponse`; the REPL owns all terminal output via the
event stream), so surfacing this cleanly is a small design choice rather than a
one-liner, and it isn't required for correctness.

**Sketch of the fix.** Either (or both):
- A `logging.info` in the adapter on fallback (e.g. "model X rejected adaptive
  thinking; retrying with budget_tokens=N" / "…with no thinking"). No coupling;
  off by default in the REPL, visible via log level / `ANTHROPIC_LOG`.
- A one-time dim REPL notice the first time a state falls back — thread the fact
  from the provider to the REPL (a small "last fallback" flag the REPL checks
  after a turn, or a dedicated event the loop renders).
