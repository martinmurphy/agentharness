"""Wire-format mapping tests for both providers. No network.

Each provider is given a fake client that records the request it was handed and
returns a canned SDK-shaped response, so we test the mapping in both directions
without a live model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentharness.providers.anthropic import AnthropicProvider
from agentharness.providers.base import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from agentharness.providers.openai_compatible import OpenAICompatibleProvider

TOOLS = [
    ToolSpec(
        name="greet",
        description="Greet someone.",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
    )
]


# ---- Anthropic --------------------------------------------------------------


class FakeAnthropicClient:
    def __init__(self, response, models=None):
        self._response = response
        self.captured = None

        self.messages = SimpleNamespace(create=self._create)
        # models.list() returns an iterable of objects with .id
        self.models = SimpleNamespace(
            list=lambda: [SimpleNamespace(id=m) for m in (models or [])]
        )

    def _create(self, **kwargs):
        self.captured = kwargs
        return self._response


def _anthropic_response(content, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def test_anthropic_request_shape():
    text = SimpleNamespace(type="text", text="hi")
    client = FakeAnthropicClient(_anthropic_response([text]))
    provider = AnthropicProvider("claude-opus-4-8", client=client, effort="high")

    resp = provider.chat(
        system="SYS",
        messages=[Message(role="user", blocks=[TextBlock("hello")])],
        tools=TOOLS,
        max_tokens=100,
    )

    req = client.captured
    assert req["model"] == "claude-opus-4-8"
    assert req["system"] == "SYS"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "high"}
    assert "temperature" not in req and "top_p" not in req  # rejected on this model
    assert req["messages"] == [{"role": "user", "content": "hello"}]
    assert req["tools"][0]["name"] == "greet"
    assert "input_schema" in req["tools"][0]

    assert resp.stop_reason == "end_turn"
    assert resp.message.text() == "hi"
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 7


def test_anthropic_show_thinking_flag():
    client = FakeAnthropicClient(_anthropic_response([]))
    provider = AnthropicProvider("m", client=client, show_thinking=True)
    provider.chat(system="", messages=[], tools=[], max_tokens=10)
    assert client.captured["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_anthropic_parses_tool_use():
    tu = SimpleNamespace(type="tool_use", id="toolu_1", name="greet", input={"name": "Ada"})
    client = FakeAnthropicClient(_anthropic_response([tu], stop_reason="tool_use"))
    provider = AnthropicProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)
    assert resp.stop_reason == "tool_use"
    calls = resp.message.tool_calls()
    assert len(calls) == 1
    assert calls[0].id == "toolu_1"
    assert calls[0].arguments == {"name": "Ada"}


def test_anthropic_refusal_stop_reason():
    client = FakeAnthropicClient(_anthropic_response([], stop_reason="refusal"))
    provider = AnthropicProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=[], max_tokens=10)
    assert resp.stop_reason == "refusal"


class _AdaptiveRejectingClient:
    """Raises BadRequestError whenever a request includes adaptive thinking,
    succeeds otherwise. Models the older-model (e.g. Haiku 4.5) behaviour."""

    def __init__(self, response):
        import httpx

        self._response = response
        self._httpx = httpx
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        import anthropic

        self.calls.append(kwargs)
        # Older models reject *adaptive* thinking but accept enabled/budget_tokens.
        if kwargs.get("thinking", {}).get("type") == "adaptive":
            resp = self._httpx.Response(400, request=self._httpx.Request("POST", "http://x"))
            raise anthropic.BadRequestError(
                "adaptive thinking is not supported on this model",
                response=resp,
                body=None,
            )
        return self._response


def test_anthropic_falls_back_when_adaptive_unsupported():
    client = _AdaptiveRejectingClient(_anthropic_response([SimpleNamespace(type="text", text="hi")]))
    provider = AnthropicProvider("claude-haiku-4-5", client=client)

    resp = provider.chat(system="S", messages=[], tools=[], max_tokens=10)
    assert resp.message.text() == "hi"
    # first attempt sent adaptive thinking; retry dropped thinking + output_config
    assert len(client.calls) == 2
    assert "thinking" in client.calls[0]
    assert "thinking" not in client.calls[1]
    assert "output_config" not in client.calls[1]
    assert provider._supports_adaptive is False

    # subsequent turn goes straight to the plain request — no wasted 400
    client.calls.clear()
    provider.chat(system="S", messages=[], tools=[], max_tokens=10)
    assert len(client.calls) == 1
    assert "thinking" not in client.calls[0]


def test_anthropic_fallback_uses_thinking_budget_when_set():
    client = _AdaptiveRejectingClient(_anthropic_response([SimpleNamespace(type="text", text="hi")]))
    provider = AnthropicProvider("claude-haiku-4-5", client=client, thinking_budget=4000)

    provider.chat(system="S", messages=[], tools=[], max_tokens=16000)
    retry = client.calls[1]  # the non-adaptive retry
    assert retry["thinking"] == {"type": "enabled", "budget_tokens": 4000}
    assert "output_config" not in retry  # older models reject effort


def test_anthropic_fallback_budget_clamped_below_max_tokens():
    client = _AdaptiveRejectingClient(_anthropic_response([SimpleNamespace(type="text", text="hi")]))
    provider = AnthropicProvider("claude-haiku-4-5", client=client, thinking_budget=100000)

    provider.chat(system="S", messages=[], tools=[], max_tokens=8000)
    assert client.calls[1]["thinking"] == {"type": "enabled", "budget_tokens": 7999}


def test_anthropic_fallback_budget_skipped_when_no_room_for_minimum():
    client = _AdaptiveRejectingClient(_anthropic_response([SimpleNamespace(type="text", text="hi")]))
    provider = AnthropicProvider("claude-haiku-4-5", client=client, thinking_budget=4000)

    provider.chat(system="S", messages=[], tools=[], max_tokens=1000)  # < 1024 min
    assert "thinking" not in client.calls[1]


def test_anthropic_current_model_ignores_thinking_budget():
    # A current model accepts adaptive, so the fallback (and its budget) never runs.
    client = FakeAnthropicClient(_anthropic_response([SimpleNamespace(type="text", text="hi")]))
    provider = AnthropicProvider("claude-opus-4-8", client=client, thinking_budget=4000)
    provider.chat(system="S", messages=[], tools=[], max_tokens=16000)
    req = client.captured
    assert req["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in req.get("thinking", {})


def test_anthropic_other_bad_request_not_swallowed():
    import anthropic
    import httpx

    class _AlwaysBad:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
            raise anthropic.BadRequestError("model: invalid model name", response=resp, body=None)

    provider = AnthropicProvider("bogus", client=_AlwaysBad())
    with pytest.raises(anthropic.BadRequestError):
        provider.chat(system="", messages=[], tools=[], max_tokens=10)


def test_anthropic_provider_raw_replayed_verbatim():
    # A thinking block has no neutral form; it must survive via provider_raw.
    thinking = SimpleNamespace(type="thinking", thinking="secret reasoning", signature="sig")
    tu = SimpleNamespace(type="tool_use", id="toolu_1", name="greet", input={"name": "Ada"})
    client = FakeAnthropicClient(_anthropic_response([thinking, tu], stop_reason="tool_use"))
    provider = AnthropicProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)

    # provider_raw holds the original SDK blocks, including the signed thinking.
    assert resp.message.provider_raw == [thinking, tu]

    # Replaying that assistant message sends the raw blocks back unchanged.
    follow = provider.chat(
        system="",
        messages=[
            Message(role="user", blocks=[TextBlock("go")]),
            resp.message,
            Message(role="tool", blocks=[ToolResult(call_id="toolu_1", content="ok")]),
        ],
        tools=TOOLS,
        max_tokens=10,
    )
    sent = client.captured["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] is resp.message.provider_raw  # verbatim, incl. thinking
    # tool result packed into ONE user message
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok", "is_error": False}
        ],
    }
    assert follow.message.text() == ""


def test_anthropic_multiple_tool_results_packed_into_one_message():
    client = FakeAnthropicClient(_anthropic_response([]))
    provider = AnthropicProvider("m", client=client)
    provider.chat(
        system="",
        messages=[
            Message(
                role="tool",
                blocks=[
                    ToolResult(call_id="a", content="1"),
                    ToolResult(call_id="b", content="2", is_error=True),
                ],
            )
        ],
        tools=[],
        max_tokens=10,
    )
    sent = client.captured["messages"]
    assert len(sent) == 1
    assert sent[0]["role"] == "user"
    assert len(sent[0]["content"]) == 2  # both results in one message


# ---- OpenAI-compatible ------------------------------------------------------


class FakeOpenAIClient:
    def __init__(self, response, models=None):
        self._response = response
        self.captured = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        # models.list() returns an object with a .data list of objects with .id
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in (models or [])])
        )

    def _create(self, **kwargs):
        self.captured = kwargs
        return self._response


def _openai_response(*, content=None, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
    )


def _oa_tool_call(id_, name, args_json):
    return SimpleNamespace(
        id=id_, function=SimpleNamespace(name=name, arguments=args_json)
    )


def test_openai_request_shape_and_system_message():
    client = FakeOpenAIClient(_openai_response(content="hi"))
    provider = OpenAICompatibleProvider("gpt-4o", client=client)
    resp = provider.chat(
        system="SYS",
        messages=[Message(role="user", blocks=[TextBlock("hello")])],
        tools=TOOLS,
        max_tokens=100,
    )
    req = client.captured
    assert req["messages"][0] == {"role": "system", "content": "SYS"}
    assert req["messages"][1] == {"role": "user", "content": "hello"}
    assert req["tools"][0]["type"] == "function"
    assert req["tools"][0]["function"]["parameters"] == TOOLS[0].input_schema
    assert req["tool_choice"] == "auto"
    assert resp.message.text() == "hi"
    assert resp.usage.input_tokens == 5


def test_openai_parses_tool_call_json_arguments():
    tc = _oa_tool_call("call_1", "greet", json.dumps({"name": "Ada"}))
    client = FakeOpenAIClient(_openai_response(tool_calls=[tc], finish_reason="tool_calls"))
    provider = OpenAICompatibleProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)
    assert resp.stop_reason == "tool_use"
    calls = resp.message.tool_calls()
    assert calls[0].arguments == {"name": "Ada"}  # parsed from JSON string


def test_openai_bad_arguments_json_becomes_empty_dict():
    tc = _oa_tool_call("call_1", "greet", "{not json")
    client = FakeOpenAIClient(_openai_response(tool_calls=[tc], finish_reason="tool_calls"))
    provider = OpenAICompatibleProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)
    assert resp.message.tool_calls()[0].arguments == {}


def test_openai_tool_results_fan_out_to_separate_messages():
    client = FakeOpenAIClient(_openai_response(content="done"))
    provider = OpenAICompatibleProvider("m", client=client)
    provider.chat(
        system="S",
        messages=[
            Message(
                role="tool",
                blocks=[
                    ToolResult(call_id="a", content="1"),
                    ToolResult(call_id="b", content="2"),
                ],
            )
        ],
        tools=[],
        max_tokens=10,
    )
    sent = client.captured["messages"]
    # system + two separate tool messages
    tool_msgs = [m for m in sent if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "a"
    assert tool_msgs[1]["tool_call_id"] == "b"


def test_openai_assistant_replay_uses_provider_raw():
    tc = _oa_tool_call("call_1", "greet", json.dumps({"name": "Ada"}))
    client = FakeOpenAIClient(_openai_response(tool_calls=[tc], finish_reason="tool_calls"))
    provider = OpenAICompatibleProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)

    provider.chat(
        system="S",
        messages=[Message(role="user", blocks=[TextBlock("hi")]), resp.message],
        tools=TOOLS,
        max_tokens=10,
    )
    sent = client.captured["messages"]
    assistant = sent[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "greet"


def test_anthropic_assistant_fallback_without_provider_raw():
    # A synthetic assistant message (no provider_raw) reconstructs from blocks.
    client = FakeAnthropicClient(_anthropic_response([]))
    provider = AnthropicProvider("m", client=client)
    synthetic = Message(
        role="assistant",
        blocks=[TextBlock("thinking done"), ThinkingBlock("hidden"),
                ToolCall(id="t1", name="greet", arguments={"name": "X"})],
    )
    provider.chat(system="", messages=[synthetic], tools=TOOLS, max_tokens=10)
    content = client.captured["messages"][0]["content"]
    types = [b["type"] for b in content]
    assert "text" in types and "tool_use" in types
    assert "thinking" not in types  # cannot reconstruct signed thinking


# ---- Gemini -----------------------------------------------------------------


class FakeGeminiClient:
    def __init__(self, response, models=None):
        self._response = response
        self.captured = None
        # models is a list of (name, supported_actions) tuples
        self._models = models or []
        self.models = SimpleNamespace(
            generate_content=self._generate, list=self._list
        )

    def _generate(self, *, model, contents, config):
        self.captured = {"model": model, "contents": contents, "config": config}
        return self._response

    def _list(self):
        return [
            SimpleNamespace(name=name, supported_actions=actions)
            for name, actions in self._models
        ]


def _gemini_response(parts, finish_reason="STOP"):
    content = SimpleNamespace(role="model", parts=parts)
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=4),
    )


def _gemini_text_part(text, *, thought=False):
    return SimpleNamespace(text=text, thought=thought, function_call=None)


def _gemini_fc_part(name, args, id_=None):
    fc = SimpleNamespace(name=name, args=args, id=id_)
    return SimpleNamespace(text=None, thought=False, function_call=fc)


def test_gemini_request_shape():
    from agentharness.providers.gemini import GeminiProvider

    client = FakeGeminiClient(_gemini_response([_gemini_text_part("hi there")]))
    provider = GeminiProvider("gemini-2.5-flash", client=client)
    resp = provider.chat(
        system="SYS",
        messages=[Message(role="user", blocks=[TextBlock("hello")])],
        tools=TOOLS,
        max_tokens=100,
    )
    cap = client.captured
    assert cap["model"] == "gemini-2.5-flash"
    cfg = cap["config"]
    assert cfg.system_instruction == "SYS"
    assert cfg.max_output_tokens == 100
    assert cfg.automatic_function_calling.disable is True  # loop stays ours
    # one Tool with our function declaration carrying the JSON schema verbatim
    decl = cfg.tools[0].function_declarations[0]
    assert decl.name == "greet"
    assert decl.parameters_json_schema == TOOLS[0].input_schema
    # contents: a single user Content with a text part
    assert cap["contents"][0].role == "user"
    assert cap["contents"][0].parts[0].text == "hello"

    assert resp.stop_reason == "end_turn"
    assert resp.message.text() == "hi there"
    assert resp.usage.input_tokens == 9
    assert resp.usage.output_tokens == 4


def test_gemini_parses_function_call_as_tool_use():
    from agentharness.providers.gemini import GeminiProvider

    part = _gemini_fc_part("greet", {"name": "Ada"}, id_="fc_1")
    client = FakeGeminiClient(_gemini_response([part]))  # finish STOP even with a call
    provider = GeminiProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)
    assert resp.stop_reason == "tool_use"  # detected by the call, not finish_reason
    call = resp.message.tool_calls()[0]
    assert call.id == "fc_1"
    assert call.name == "greet"
    assert call.arguments == {"name": "Ada"}


def test_gemini_function_call_without_id_uses_name():
    from agentharness.providers.gemini import GeminiProvider

    part = _gemini_fc_part("greet", {"name": "Ada"}, id_=None)
    client = FakeGeminiClient(_gemini_response([part]))
    provider = GeminiProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)
    assert resp.message.tool_calls()[0].id == "greet"


def test_gemini_thought_part_becomes_thinking_block():
    from agentharness.providers.gemini import GeminiProvider

    parts = [_gemini_text_part("reasoning", thought=True), _gemini_text_part("answer")]
    client = FakeGeminiClient(_gemini_response(parts))
    provider = GeminiProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=[], max_tokens=10)
    kinds = [type(b).__name__ for b in resp.message.blocks]
    assert kinds == ["ThinkingBlock", "TextBlock"]
    assert resp.message.text() == "answer"  # thought excluded from text()


def test_gemini_refusal_stop_reason():
    from agentharness.providers.gemini import GeminiProvider

    client = FakeGeminiClient(_gemini_response([], finish_reason="SAFETY"))
    provider = GeminiProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=[], max_tokens=10)
    assert resp.stop_reason == "refusal"


def test_gemini_tool_result_roundtrip_and_provider_raw():
    from agentharness.providers.gemini import GeminiProvider

    part = _gemini_fc_part("greet", {"name": "Ada"}, id_="fc_1")
    client = FakeGeminiClient(_gemini_response([part]))
    provider = GeminiProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)

    # provider_raw is the original model Content, replayed verbatim.
    assert resp.message.provider_raw is not None

    provider.chat(
        system="",
        messages=[
            Message(role="user", blocks=[TextBlock("go")]),
            resp.message,
            Message(role="tool", blocks=[ToolResult(call_id="fc_1", content="Hi Ada")]),
        ],
        tools=TOOLS,
        max_tokens=10,
    )
    contents = client.captured["contents"]
    # assistant model turn replayed as the raw Content object
    assert contents[1] is resp.message.provider_raw
    # tool result -> user Content with a function_response naming the function
    fr_content = contents[2]
    assert fr_content.role == "user"
    fr = fr_content.parts[0].function_response
    assert fr.name == "greet"        # recovered from the model turn's call
    assert fr.id == "fc_1"
    assert fr.response == {"result": "Hi Ada"}


def test_gemini_error_tool_result_wrapped_as_error():
    from agentharness.providers.gemini import GeminiProvider

    part = _gemini_fc_part("greet", {}, id_="fc_1")
    client = FakeGeminiClient(_gemini_response([part]))
    provider = GeminiProvider("m", client=client)
    resp = provider.chat(system="", messages=[], tools=TOOLS, max_tokens=10)
    provider.chat(
        system="",
        messages=[
            resp.message,
            Message(
                role="tool",
                blocks=[ToolResult(call_id="fc_1", content="boom", is_error=True)],
            ),
        ],
        tools=TOOLS,
        max_tokens=10,
    )
    fr = client.captured["contents"][1].parts[0].function_response
    assert fr.response == {"error": "boom"}


# ---- list_models ------------------------------------------------------------


def test_anthropic_list_models():
    client = FakeAnthropicClient(_anthropic_response([]), models=["claude-opus-4-8", "claude-haiku-4-5"])
    provider = AnthropicProvider("m", client=client)
    assert provider.list_models() == ["claude-haiku-4-5", "claude-opus-4-8"]  # sorted


def test_openai_list_models():
    client = FakeOpenAIClient(_openai_response(content="x"), models=["gpt-4o", "gpt-4o-mini"])
    provider = OpenAICompatibleProvider("m", client=client)
    assert provider.list_models() == ["gpt-4o", "gpt-4o-mini"]


def test_gemini_list_models_filters_to_generate_content():
    from agentharness.providers.gemini import GeminiProvider

    client = FakeGeminiClient(
        _gemini_response([]),
        models=[
            ("models/gemini-3.5-flash", ["generateContent", "countTokens"]),
            ("models/text-embedding-004", ["embedContent"]),  # excluded
            ("models/gemini-3.5-pro", ["generateContent"]),
        ],
    )
    provider = GeminiProvider("m", client=client)
    # sorted, prefix stripped, embeddings excluded
    assert provider.list_models() == ["gemini-3.5-flash", "gemini-3.5-pro"]


# ---- provider_status / _KNOWN -----------------------------------------------


def test_known_derived_from_env_vars():
    from agentharness.providers.factory import _KNOWN

    assert _KNOWN == ("anthropic", "openai", "gemini")


def test_provider_status_reflects_env(monkeypatch):
    from agentharness.providers.factory import provider_status

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "y")

    status = {p.name: p for p in provider_status()}
    assert status["anthropic"].available is True
    assert status["anthropic"].env_var == "ANTHROPIC_API_KEY"
    assert status["openai"].available is False
    assert status["gemini"].available is True


# ---- provider aliases --------------------------------------------------------


def _alias_config(**blocks):
    from agentharness.config import Config

    return Config(providers=blocks)


def _fake_openai(monkeypatch):
    """Capture the kwargs the OpenAI SDK client would be constructed with."""
    import openai

    captured = {}

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", _Client)
    return captured


def test_alias_builds_its_declared_adapter(monkeypatch):
    from agentharness.providers.factory import build_provider
    from agentharness.providers.openai_compatible import OpenAICompatibleProvider

    captured = _fake_openai(monkeypatch)
    cfg = _alias_config(
        lmstudio={"type": "openai", "base_url": "http://localhost:1234/v1"}
    )
    provider = build_provider("lmstudio", "some-model", cfg)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "some-model"
    assert captured["base_url"] == "http://localhost:1234/v1"
    # 'type' is a harness key and must not reach the SDK
    assert "type" not in captured


def test_two_aliases_keep_separate_base_urls(monkeypatch):
    from agentharness.providers.factory import build_provider

    cfg = _alias_config(
        one={"type": "openai", "base_url": "http://a:1/v1"},
        two={"type": "openai", "base_url": "http://b:2/v1"},
    )
    captured = _fake_openai(monkeypatch)
    build_provider("one", "m", cfg)
    assert captured["base_url"] == "http://a:1/v1"
    captured = _fake_openai(monkeypatch)
    build_provider("two", "m", cfg)
    assert captured["base_url"] == "http://b:2/v1"


def test_alias_api_key_env_is_read_from_environment(monkeypatch):
    from agentharness.providers.factory import build_provider

    monkeypatch.setenv("TOGETHER_API_KEY", "secret-value")
    captured = _fake_openai(monkeypatch)
    cfg = _alias_config(
        together={
            "type": "openai",
            "base_url": "https://api.together.xyz/v1",
            "api_key_env": "TOGETHER_API_KEY",
        }
    )
    build_provider("together", "m", cfg)

    assert captured["api_key"] == "secret-value"
    # the env var *name* is a harness key, not an SDK one
    assert "api_key_env" not in captured


def test_alias_missing_api_key_env_errors_clearly(monkeypatch):
    from agentharness.providers.factory import build_provider

    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    cfg = _alias_config(
        together={"type": "openai", "api_key_env": "TOGETHER_API_KEY"}
    )
    with pytest.raises(ValueError, match="TOGETHER_API_KEY, which is not set"):
        build_provider("together", "m", cfg)


def test_keyless_local_endpoint_gets_placeholder(monkeypatch):
    """A base_url with no resolvable key: the OpenAI SDK still needs something."""
    from agentharness.providers.factory import _PLACEHOLDER_KEY, build_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = _fake_openai(monkeypatch)
    cfg = _alias_config(lmstudio={"type": "openai", "base_url": "http://localhost:1234/v1"})
    build_provider("lmstudio", "m", cfg)
    assert captured["api_key"] == _PLACEHOLDER_KEY


def test_builtin_openai_with_base_url_also_gets_placeholder(monkeypatch):
    from agentharness.providers.factory import _PLACEHOLDER_KEY, build_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = _fake_openai(monkeypatch)
    cfg = _alias_config(openai={"base_url": "http://localhost:1234/v1"})
    build_provider("openai", "m", cfg)
    assert captured["api_key"] == _PLACEHOLDER_KEY


def test_builtin_key_in_env_is_not_overridden_by_placeholder(monkeypatch):
    """A proxy base_url must not clobber the key the SDK would find itself."""
    from agentharness.providers.factory import build_provider

    monkeypatch.setenv("OPENAI_API_KEY", "real-key")
    captured = _fake_openai(monkeypatch)
    cfg = _alias_config(openai={"base_url": "https://proxy.example/v1"})
    build_provider("openai", "m", cfg)
    assert "api_key" not in captured  # left to the SDK


def test_literal_api_key_still_wins(monkeypatch):
    from agentharness.providers.factory import build_provider

    monkeypatch.setenv("SOME_KEY", "from-env")
    captured = _fake_openai(monkeypatch)
    cfg = _alias_config(
        local={
            "type": "openai",
            "base_url": "http://localhost:1/v1",
            "api_key": "literal",
            "api_key_env": "SOME_KEY",
        }
    )
    build_provider("local", "m", cfg)
    assert captured["api_key"] == "literal"


def test_unknown_provider_names_aliases_in_its_error():
    from agentharness.providers.factory import build_provider

    cfg = _alias_config(lmstudio={"type": "openai", "base_url": "http://x/v1"})
    with pytest.raises(ValueError, match="unknown provider 'bogus'") as exc:
        build_provider("bogus", "m", cfg)
    assert "lmstudio" in str(exc.value)  # the alias is offered as a choice


def test_alias_without_type_is_rejected():
    from agentharness.providers.factory import build_provider

    cfg = _alias_config(mystery={"base_url": "http://x/v1"})
    with pytest.raises(ValueError, match=r"providers\.mystery\.type"):
        build_provider("mystery", "m", cfg)


def test_alias_with_unknown_type_is_rejected():
    from agentharness.providers.factory import build_provider

    cfg = _alias_config(weird={"type": "llamafile", "base_url": "http://x/v1"})
    with pytest.raises(ValueError, match="unknown type 'llamafile'"):
        build_provider("weird", "m", cfg)


def test_provider_status_includes_aliases(monkeypatch):
    from agentharness.providers.factory import known_providers, provider_status

    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    cfg = _alias_config(
        lmstudio={"type": "openai", "base_url": "http://x/v1"},
        together={"type": "openai", "api_key_env": "TOGETHER_API_KEY"},
    )
    status = {p.name: p for p in provider_status(cfg)}

    assert status["lmstudio"].keyless is True
    assert status["lmstudio"].available is True  # needs no key, so it is ready
    assert status["together"].keyless is False
    assert status["together"].available is False  # env var unset
    monkeypatch.setenv("TOGETHER_API_KEY", "x")
    assert {p.name: p for p in provider_status(cfg)}["together"].available is True

    assert known_providers(cfg) == ["anthropic", "openai", "gemini", "lmstudio", "together"]


def test_provider_status_without_config_is_builtins_only():
    from agentharness.providers.factory import provider_status

    assert [p.name for p in provider_status()] == ["anthropic", "openai", "gemini"]
