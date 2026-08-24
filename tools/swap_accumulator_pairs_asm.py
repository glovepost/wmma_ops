#!/usr/bin/env python3
"""Swap two 16-VGPR accumulator banks from initialization through epilogue."""

from pathlib import Path
import re
import sys


PAIR_WIDTH = 16
VALID_BASES = (17, 33, 49)
INIT_ANCHOR = "\tv_mov_b32_e32 v57, 0\n"


def main() -> int:
    if len(sys.argv) != 5:
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s FIRST_BASE SECOND_BASE",
            file=sys.stderr,
        )
        return 2
    first_base = int(sys.argv[3])
    second_base = int(sys.argv[4])
    if first_base not in VALID_BASES or second_base not in VALID_BASES:
        raise ValueError(
            f"bases must be selected from {VALID_BASES}; v1 is tied to v0 "
            "by a contiguous epilogue range"
        )
    if first_base >= second_base:
        raise ValueError("FIRST_BASE must be less than SECOND_BASE")

    source = Path(sys.argv[1]).read_text()
    if source.count(INIT_ANCHOR) != 1:
        raise ValueError("expected one accumulator initialization anchor")
    start = source.index(INIT_ANCHOR)
    suffix = source[start:]

    def swap(register: int) -> int:
        if first_base <= register < first_base + PAIR_WIDTH:
            return register + second_base - first_base
        if second_base <= register < second_base + PAIR_WIDTH:
            return register + first_base - second_base
        return register

    def swap_range(match: re.Match[str]) -> str:
        first = int(match.group(1))
        last = int(match.group(2))
        mapped_first = swap(first)
        mapped_last = swap(last)
        if mapped_last - mapped_first != last - first:
            raise ValueError(f"range crosses an accumulator pair: {match.group(0)}")
        return f"v[{mapped_first}:{mapped_last}]"

    suffix = re.sub(r"\bv\[(\d+):(\d+)\]", swap_range, suffix)
    suffix = re.sub(
        r"\bv(\d+)\b",
        lambda match: f"v{swap(int(match.group(1)))}",
        suffix,
    )

    Path(sys.argv[2]).write_text(source[:start] + suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
