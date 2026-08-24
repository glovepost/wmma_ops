#!/usr/bin/env python3
"""Permute the ten independent initial LDS reads in the p8 hot loop.

The transformation preserves every destination, address and wait threshold. It
only changes issue order, allowing a targeted test of LDS-bank/queue phase
without changing the WMMA operand order or numerical contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SCHEDULES = {
    "bfirst": (4, 5, 0, 1, 2, 3, 6, 7, 8, 9),
    "bmiddle": (0, 1, 4, 5, 2, 3, 6, 7, 8, 9),
    "astripe": (0, 2, 4, 6, 8, 1, 3, 5, 7, 9),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("schedule", choices=sorted(SCHEDULES))
    args = parser.parse_args()

    text = args.source.read_text()
    lines = text.splitlines(keepends=True)
    marker = ".LBB0_13:"
    start = next(i for i, line in enumerate(lines) if line.startswith(marker)) + 1
    load_indices = []
    for i in range(start, min(start + 20, len(lines))):
        if "\tds_load_b128 " not in lines[i]:
            break
        load_indices.append(i)
    if len(load_indices) != 10:
        raise SystemExit("expected ten contiguous initial LDS reads")
    loads = [lines[i] for i in load_indices]
    ordered = [loads[i] for i in SCHEDULES[args.schedule]]
    for index, line in zip(load_indices, ordered):
        lines[index] = line
    args.output.write_text("".join(lines))


if __name__ == "__main__":
    main()
