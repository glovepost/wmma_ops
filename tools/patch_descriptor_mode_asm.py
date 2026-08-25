#!/usr/bin/env python3
"""Patch gfx11 kernel-descriptor scheduling fields without changing ISA."""

import argparse
from pathlib import Path


def replace_field(source: str, field: str, value: int) -> str:
    old = f"\t\t.{field} 1\n"
    new = f"\t\t.{field} {value}\n"
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one {field}=1 descriptor field")
    return source.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--forward-progress", type=int, choices=(0, 1), default=1)
    parser.add_argument("--memory-ordered", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    source = args.input.read_text()
    if args.forward_progress == 0:
        source = replace_field(source, "amdhsa_forward_progress", 0)
    if args.memory_ordered == 0:
        source = replace_field(source, "amdhsa_memory_ordered", 0)
    if args.forward_progress == 1 and args.memory_ordered == 1:
        raise ValueError("requested descriptor is identical to the input")

    args.output.write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
