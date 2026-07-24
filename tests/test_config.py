"""Tests for config resolution and env overrides."""

from __future__ import annotations

from agentharness.config import load_config


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no ./config.yaml here
    monkeypatch.delenv("AGENTHARNESS_CONFIG", raising=False)
    cfg = load_config()
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-opus-4-8"
    assert cfg.max_tokens == 16000


def test_yaml_file(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("provider: openai\nmodel: gpt-4o\nmax_tokens: 2048\n", encoding="utf-8")
    monkeypatch.setenv("AGENTHARNESS_CONFIG", str(path))
    cfg = load_config()
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"
    assert cfg.max_tokens == 2048


def test_env_override_beats_file(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model: gpt-4o\n", encoding="utf-8")
    monkeypatch.setenv("AGENTHARNESS_CONFIG", str(path))
    monkeypatch.setenv("AGENTHARNESS_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("AGENTHARNESS_SHOW_THINKING", "true")
    cfg = load_config()
    assert cfg.model == "claude-opus-4-8"
    assert cfg.show_thinking is True


def test_provider_options(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "providers:\n  openai:\n    base_url: http://localhost:11434/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTHARNESS_CONFIG", str(path))
    cfg = load_config()
    assert cfg.provider_options("openai") == {"base_url": "http://localhost:11434/v1"}
    assert cfg.provider_options("missing") == {}
