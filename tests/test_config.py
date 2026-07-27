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


def test_thinking_budget_default_and_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTHARNESS_CONFIG", raising=False)
    assert load_config().thinking_budget is None  # off by default

    path = tmp_path / "c.yaml"
    path.write_text("thinking_budget: 4000\n", encoding="utf-8")
    monkeypatch.setenv("AGENTHARNESS_CONFIG", str(path))
    assert load_config().thinking_budget == 4000

    monkeypatch.setenv("AGENTHARNESS_THINKING_BUDGET", "2048")
    assert load_config().thinking_budget == 2048  # env overrides file


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


def test_workspace_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTHARNESS_CONFIG", raising=False)
    monkeypatch.delenv("AGENTHARNESS_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("AGENTHARNESS_WORKSPACE_WRITABLE", raising=False)
    cfg = load_config()
    assert cfg.workspace_dir == "./workspace"
    assert cfg.workspace_writable is True  # writes are on by default


def test_workspace_from_yaml_and_env(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("workspace_dir: /data\nworkspace_writable: false\n", encoding="utf-8")
    monkeypatch.setenv("AGENTHARNESS_CONFIG", str(path))
    monkeypatch.delenv("AGENTHARNESS_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("AGENTHARNESS_WORKSPACE_WRITABLE", raising=False)
    cfg = load_config()
    assert cfg.workspace_dir == "/data"
    assert cfg.workspace_writable is False

    monkeypatch.setenv("AGENTHARNESS_WORKSPACE_DIR", "/elsewhere")
    monkeypatch.setenv("AGENTHARNESS_WORKSPACE_WRITABLE", "yes")
    cfg = load_config()
    assert cfg.workspace_dir == "/elsewhere"
    assert cfg.workspace_writable is True  # env overrides the file, both ways
