"""The Skill dataclass — one loaded, validated Agent Skill."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# What a valid environment variable name looks like. Shared by the loader (a
# skill's metadata.env request) and the runner (the operator's allowlist), so
# the two halves of the intersection cannot disagree about what a name is.
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


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
    # Environment variables the skill asks be forwarded to its scripts, parsed
    # from `metadata.env`. A request, not a grant: what actually reaches a
    # child is the intersection with the operator's allowlist. See
    # skills/runner.py.
    script_env: frozenset[str] = frozenset()

    def catalog_line(self) -> str:
        """One-line entry for the discovery catalog in the system prompt."""
        return f"- {self.name}: {self.description}"
