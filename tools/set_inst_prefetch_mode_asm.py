#!/usr/bin/env python3
"""Set the RDNA 3.5 wave instruction-prefetch mode before the K16 loop."""

import argparse
from pathlib import Path


LOOP = ".LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("1", "2", "3", "nop"))
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(LOOP) != 1:
        raise ValueError("expected one fixed K16 loop label")
    instruction = (
        "\ts_nop 0\n"
        if args.mode == "nop"
        else f"\ts_set_inst_prefetch_distance {args.mode}\n"
    )
    # The back edge targets LOOP, so this state change executes only on the
    # initial fall-through and remains active for all 255 repeated bodies.
    source = source.replace(LOOP, instruction + LOOP, 1)
    args.output.write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
