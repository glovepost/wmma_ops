#!/usr/bin/env python3
"""Swap two eight-VGPR banks only within the steady WMMA loop."""

from pathlib import Path
import re
import sys


B_BASE = 92
GROUP_WIDTH = 8
VALID_OTHER_BASES = (76, 84, 100, 108)


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s OTHER_BASE", file=sys.stderr)
        return 2
    other_base = int(sys.argv[3])
    if other_base not in VALID_OTHER_BASES:
        raise ValueError(f"OTHER_BASE must be one of {VALID_OTHER_BASES}")

    source = Path(sys.argv[1]).read_text()
    start = source.index(".LBB0_13:")
    end = source.index("; %bb.14:", start)
    hot = source[start:end]

    def swap(register: int) -> int:
        if B_BASE <= register < B_BASE + GROUP_WIDTH:
            return register + other_base - B_BASE
        if other_base <= register < other_base + GROUP_WIDTH:
            return register + B_BASE - other_base
        return register

    def swap_range(match: re.Match[str]) -> str:
        first = int(match.group(1))
        last = int(match.group(2))
        mapped_first = swap(first)
        mapped_last = swap(last)
        if mapped_last - mapped_first != last - first:
            raise ValueError(f"range crosses a swap boundary: {match.group(0)}")
        return f"v[{mapped_first}:{mapped_last}]"

    hot = re.sub(r"\bv\[(\d+):(\d+)\]", swap_range, hot)
    hot = re.sub(r"\bv(\d+)\b", lambda match: f"v{swap(int(match.group(1)))}", hot)

    source = source[:start] + hot + source[end:]
    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
