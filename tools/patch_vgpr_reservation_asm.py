#!/usr/bin/env python3
"""Safely over-reserve VGPRs in a fixed gfx1151 assembly image."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("count", type=int, choices=(121, 124, 128))
    args = parser.parse_args()

    source = args.input.read_text()
    fields = (
        ("\t\t.amdhsa_next_free_vgpr 120\n",
         f"\t\t.amdhsa_next_free_vgpr {args.count}\n"),
        ("    .vgpr_count:     120\n",
         f"    .vgpr_count:     {args.count}\n"),
    )
    for old, new in fields:
        if source.count(old) != 1:
            raise ValueError(f"expected one source field: {old.strip()}")
        source = source.replace(old, new, 1)

    args.output.write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
