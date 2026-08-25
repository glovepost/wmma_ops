#!/usr/bin/env python3
"""Set gfx1151 MUBUF cache-policy bits on the selected hot-loop refills."""

import argparse
from pathlib import Path


LOADS = (
    "\tbuffer_load_b128 v[76:79], v71, s[0:3], s7 offen\n"
    "\tbuffer_load_b128 v[80:83], v71, s[0:3], s7 offen offset:16\n"
    "\tbuffer_load_b128 v[116:119], v73, s[8:11], s18 offen\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "policy",
        choices=("none", "glc", "slc", "dlc", "glc-slc", "glc-dlc",
                 "slc-dlc", "glc-slc-dlc"),
    )
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(LOADS) != 1:
        raise ValueError("expected exactly one selected hot-loop refill trio")

    modifiers = "" if args.policy == "none" else " ".join(
        args.policy.split("-")
    )
    if modifiers:
        replacement = "".join(
            line + " " + modifiers + "\n"
            for line in LOADS.rstrip("\n").split("\n")
        )
    else:
        replacement = LOADS

    args.output.write_text(source.replace(LOADS, replacement, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
