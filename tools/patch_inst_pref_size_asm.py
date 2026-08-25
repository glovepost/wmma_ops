#!/usr/bin/env python3
"""Patch only the gfx11 code-object initial instruction-prefetch size."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("size", type=int, choices=(0, 8, 12, 16, 32))
    args = parser.parse_args()

    source = args.input.read_text()
    old = "\t\t.amdhsa_inst_pref_size 63\n"
    new = f"\t\t.amdhsa_inst_pref_size {args.size}\n"
    if source.count(old) != 1:
        raise ValueError("expected one inst_pref_size=63 descriptor field")
    args.output.write_text(source.replace(old, new, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
