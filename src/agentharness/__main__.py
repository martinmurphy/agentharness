"""CLI entrypoint.

    agentharness                 start the interactive REPL
    agentharness --list-skills   print the skill catalog and exit (no API key needed)
    agentharness --version       print the version and exit
"""

from __future__ import annotations

import argparse
import sys

from agentharness import __version__
from agentharness.config import load_config
from agentharness.skills.loader import load_skills


def _list_skills() -> int:
    config = load_config()
    skillset = load_skills(config.skills_dir)
    print(f"skills directory: {skillset.root}")
    if skillset.skills:
        print(f"\n{len(skillset.skills)} skill(s) loaded:\n")
        for skill in sorted(skillset.skills, key=lambda s: s.name):
            print(f"  {skill.name}")
            print(f"    {skill.description}")
    else:
        print("\nno skills loaded")
    if skillset.errors:
        print(f"\n{len(skillset.errors)} skill(s) failed to load:\n")
        for err in skillset.errors:
            print(f"  {err.path.name}: {err.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentharness", description=__doc__)
    parser.add_argument("--version", action="version", version=f"agentharness {__version__}")
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="list discovered skills and exit (no model call, no API key required)",
    )
    args = parser.parse_args(argv)

    if args.list_skills:
        return _list_skills()

    # Interactive REPL — imported lazily so the flags above work without the
    # provider SDKs or an API key being available.
    from agentharness.repl import run_repl

    return run_repl()


if __name__ == "__main__":
    sys.exit(main())
