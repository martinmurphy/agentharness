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

## SSRF guardrails for the web_fetch tool

**Problem.** The `web_fetch` tool (`src/agentharness/tools/web_tools.py`) lets
the model make arbitrary outbound HTTP(S) requests. It restricts the scheme to
http/https, caps the response body, and sets a timeout, but it does **not**
block requests to private/loopback addresses or cloud metadata endpoints
(169.254.169.254, `localhost`, RFC 1918 ranges, `.internal`, …). A model that is
prompt-injected by fetched content could be steered into probing internal
services or exfiltrating instance credentials.

**Why deferred.** Blocking loopback/private ranges outright would break a
legitimate dev-harness use — hitting a local API or `localhost` test server. The
right design is a *configurable* policy (default deny internal, opt-in
allowlist), which is more than a one-liner and wasn't in the tool's initial ask.

**Sketch of the fix.** A config option, e.g. `web_fetch: {allow_private: false,
allowed_hosts: [...]}`. Resolve the URL's host to its IPs before the request and
reject private/loopback/link-local/ULA ranges unless the host is on the
allowlist (resolve-then-check, and guard against DNS-rebinding by pinning the
resolved IP for the actual connection). Surface rejections as a normal tool
error. Keep the default safe (deny internal) so the capability is opt-in for
internal targets.
