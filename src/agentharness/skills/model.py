"""The Skill dataclass — one loaded, validated Agent Skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    """A single Agent Skill loaded from a directory containing SKILL.md.

    ``body`` is the Markdown content after the frontmatter. Bundled files under
    ``scripts/``, ``references/``, ``assets/`` are read on demand via the loader,
    not eagerly held here.
    """

    name: str
    description: str
    body: str
    path: Path
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def catalog_line(self) -> str:
        """One-line entry for the discovery catalog in the system prompt."""
        return f"- {self.name}: {self.description}"
