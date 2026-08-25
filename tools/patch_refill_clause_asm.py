#!/usr/bin/env python3
"""Patch only the fixed K16 hot-loop global-refill S_CLAUSE."""

import argparse
from pathlib import Path


LOOP = ".LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"
BACK_EDGE = "\ts_cbranch_scc0 .LBB0_13\n"
CLAUSE = "\ts_clause 0x1\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("all-three", "nop", "drop"))
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(LOOP) != 1 or source.count(BACK_EDGE) != 1:
        raise ValueError("expected one fixed K16 hot loop")
    start = source.index(LOOP)
    end = source.index(BACK_EDGE, start) + len(BACK_EDGE)
    loop = source[start:end]
    if loop.count(CLAUSE) != 1:
        raise ValueError("expected one two-load refill clause in the hot loop")

    replacement = {
        "all-three": "\ts_clause 0x2\n",
        "nop": "\ts_nop 0\n",
        "drop": "",
    }[args.mode]
    loop = loop.replace(CLAUSE, replacement, 1)
    args.output.write_text(source[:start] + loop + source[end:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
