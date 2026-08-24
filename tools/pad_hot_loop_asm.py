#!/usr/bin/env python3
"""Shift the retained K loop by one-time SALU no-ops."""

from pathlib import Path
import sys


HOT_LOOP = ".LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"
ALLOWED_PADDING = (8, 24, 40, 56)


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s PADDING_BYTES",
            file=sys.stderr,
        )
        return 2

    padding = int(sys.argv[3])
    if padding not in ALLOWED_PADDING:
        raise ValueError(f"PADDING_BYTES must be one of {ALLOWED_PADDING}")

    source = Path(sys.argv[1]).read_text()
    if source.count(HOT_LOOP) != 1:
        raise ValueError("expected exactly one retained K-loop header")

    no_ops = "\ts_nop\t0\n" * (padding // 4)
    padded = source.replace(HOT_LOOP, no_ops + HOT_LOOP, 1)
    Path(sys.argv[2]).write_text(padded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
