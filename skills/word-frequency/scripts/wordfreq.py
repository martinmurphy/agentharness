"""Count word frequencies in a workspace text file.

Run via the harness's run_skill_script tool. The working directory is the
workspace root, so PATH arguments are workspace-relative.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="text file to read, relative to the workspace")
    parser.add_argument("-n", "--top", type=int, default=20, help="how many to show")
    parser.add_argument("--min-length", type=int, default=1, help="ignore shorter words")
    args = parser.parse_args()

    try:
        text = Path(args.path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {args.path}: {exc}", file=sys.stderr)
        return 1

    words = [w for w in re.findall(r"[\w'-]+", text.lower()) if len(w) >= args.min_length]
    if not words:
        print("no words found", file=sys.stderr)
        return 1

    counts = Counter(words)
    print(f"{len(words)} words, {len(counts)} distinct")
    for word, n in counts.most_common(args.top):
        print(f"{n:>7}  {word}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
