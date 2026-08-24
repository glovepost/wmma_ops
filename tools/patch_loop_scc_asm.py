#!/usr/bin/env python3
"""Use the K-loop decrement's SCC for its back-edge branch."""

from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT OUTPUT")
    text = Path(sys.argv[1]).read_text()
    needle = "\n\ts_cmp_eq_u32 s6, 0\n"
    if text.count(needle) != 1:
        raise SystemExit("expected one K-loop equality compare")
    Path(sys.argv[2]).write_text(text.replace(needle, "\n", 1))


if __name__ == "__main__":
    main()
