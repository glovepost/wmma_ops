#!/usr/bin/env python3
"""Shift non-accumulator VGPRs to sweep WMMA register-bank phase."""

from pathlib import Path
import re
import sys


DEFAULT_BOUNDARY = 65
ORIGINAL_NEXT_FREE = 118


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s DELTA [BOUNDARY]",
            file=sys.stderr,
        )
        return 2
    delta = int(sys.argv[3])
    if delta not in (2, 4, 6):
        raise ValueError("DELTA must be one of 2, 4, or 6")
    boundary = int(sys.argv[4]) if len(sys.argv) == 5 else DEFAULT_BOUNDARY
    if boundary not in range(1, ORIGINAL_NEXT_FREE + 1):
        raise ValueError("BOUNDARY is outside the allocated VGPR range")

    source = Path(sys.argv[1]).read_text()

    def shift_range(match: re.Match[str]) -> str:
        first = int(match.group(1))
        last = int(match.group(2))
        if first < boundary <= last:
            raise ValueError(f"VGPR range crosses shift boundary: {match.group(0)}")
        if first >= boundary:
            first += delta
            last += delta
        return f"v[{first}:{last}]"

    source = re.sub(r"\bv\[(\d+):(\d+)\]", shift_range, source)

    def shift_scalar(match: re.Match[str]) -> str:
        register = int(match.group(1))
        if register >= boundary:
            register += delta
        return f"v{register}"

    source = re.sub(r"\bv(\d+)\b", shift_scalar, source)

    old_next_free = f"\t\t.amdhsa_next_free_vgpr {ORIGINAL_NEXT_FREE}\n"
    new_next_free = (
        f"\t\t.amdhsa_next_free_vgpr {ORIGINAL_NEXT_FREE + delta}\n"
    )
    if source.count(old_next_free) != 1:
        raise ValueError("expected one HSA next-free VGPR declaration")
    source = source.replace(old_next_free, new_next_free, 1)

    old_count = f"    .vgpr_count:     {ORIGINAL_NEXT_FREE}\n"
    new_count = f"    .vgpr_count:     {ORIGINAL_NEXT_FREE + delta}\n"
    if source.count(old_count) != 1:
        raise ValueError("expected one metadata VGPR count")
    source = source.replace(old_count, new_count, 1)

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
