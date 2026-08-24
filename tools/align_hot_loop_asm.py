#!/usr/bin/env python3
"""Align the retained block-prepacked K loop without changing its body."""

from pathlib import Path
import sys


HOT_LOOP = ".LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s ALIGNMENT_BYTES",
            file=sys.stderr,
        )
        return 2

    alignment = int(sys.argv[3])
    if alignment not in (64, 128, 256, 512):
        raise ValueError("ALIGNMENT_BYTES must be 64, 128, 256, or 512")

    source = Path(sys.argv[1]).read_text()
    if source.count(HOT_LOOP) != 1:
        raise ValueError("expected exactly one retained K-loop header")

    exponent = alignment.bit_length() - 1
    aligned = source.replace(HOT_LOOP, f"\t.p2align\t{exponent}\n{HOT_LOOP}", 1)
    Path(sys.argv[2]).write_text(aligned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
