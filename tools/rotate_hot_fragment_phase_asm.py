#!/usr/bin/env python3
"""Rotate the selected hot-loop fragment banks without raising VGPR count.

The qualified delta-2 image places all five eight-VGPR WMMA input banks at
bases congruent to four modulo eight.  Earlier wider boundary shifts changed
both this phase and the kernel's reported residency.  This transform isolates
every remaining modulo-eight phase by moving the five banks one to four
registers in either direction while rotating the four-register B refill into
the vacated slot.  Overlapping loop-invariant vector addresses are shadowed in
registers that are dead throughout the hot loop.  The tail remains
byte-for-byte unchanged.
"""

from pathlib import Path
import re
import sys


LOOP_LABEL = ".LBB0_13:"
LOOP_EXIT = "; %bb.14:"


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s PHASE_DELTA",
            file=sys.stderr,
        )
        return 2

    delta = int(sys.argv[3])
    if delta not in (-4, -3, -2, -1, 1, 2, 3, 4):
        raise ValueError("PHASE_DELTA must be a nonzero value from -4 to 4")

    source = Path(sys.argv[1]).read_text()
    start = source.index(LOOP_LABEL)
    end = source.index(LOOP_EXIT, start)
    prefix = source[:start]
    hot = source[start:end]

    required = (
        "\tds_load_b128 v[76:79], v68\n",
        "\tds_load_b128 v[92:95], v67 offset:12288\n",
        "\tbuffer_load_b128 v[116:119], v73, s[8:11], s18 offen\n",
        "\tds_store_b128 v75, v[116:119]\n",
    )
    for anchor in required:
        if hot.count(anchor) != 1:
            raise ValueError(f"selected-loop anchor {anchor.strip()!r} changed")

    if delta == 4:
        shadow_map = {71: 65, 73: 66, 75: 72}
        refill_base = 76
    elif delta == 3:
        shadow_map = {75: 65}
        refill_base = 75
    elif delta == 2:
        shadow_map = {75: 72}
        refill_base = 74
    elif delta == 1:
        shadow_map = {73: 65, 75: 66}
        refill_base = 73
    elif delta == -1:
        shadow_map = {75: 65}
        refill_base = 115
    elif delta == -2:
        shadow_map = {75: 65}
        refill_base = 114
    elif delta == -3:
        shadow_map = {73: 65, 75: 66}
        refill_base = 113
    else:
        shadow_map = {71: 65, 73: 66, 75: 116}
        refill_base = 112

    def map_register(register: int) -> int:
        if 76 <= register <= 115:
            return register + delta
        if 116 <= register <= 119:
            return refill_base + register - 116
        return shadow_map.get(register, register)

    def map_range(match: re.Match[str]) -> str:
        first = int(match.group(1))
        last = int(match.group(2))
        mapped = [map_register(register)
                  for register in range(first, last + 1)]
        if mapped != list(range(mapped[0], mapped[-1] + 1)):
            raise ValueError(f"range crosses rotation boundary: {match.group(0)}")
        return f"v[{mapped[0]}:{mapped[-1]}]"

    hot = re.sub(r"\bv\[(\d+):(\d+)\]", map_range, hot)
    hot = re.sub(
        r"\bv(\d+)\b",
        lambda match: f"v{map_register(int(match.group(1)))}",
        hot,
    )

    shadows = "".join(
        f"\tv_mov_b32_e32 v{destination}, v{register}\n"
        for register, destination in shadow_map.items()
    )
    source = prefix + shadows + hot + source[end:]

    registers = [int(value) for value in re.findall(r"\bv(\d+)\b", source)]
    registers.extend(
        int(value)
        for pair in re.findall(r"v\[(\d+):(\d+)\]", source)
        for value in pair
    )
    if max(registers) > 119:
        raise ValueError(f"rotated image addresses v{max(registers)}")
    if hot.count("\tv_wmma_f16_16x16x16_f16") != 16:
        raise ValueError("hot loop must retain 16 WMMA instructions")
    if hot.count("\tds_load_b128") != 16:
        raise ValueError("hot loop must retain 16 b128 LDS reads")
    if hot.count("\tbuffer_load_b128") != 3:
        raise ValueError("hot loop must retain three MUBUF refills")
    if hot.count("\tds_store_b128") != 3:
        raise ValueError("hot loop must retain three b128 LDS stores")
    if hot.count("\ts_barrier") != 2:
        raise ValueError("hot loop must retain two workgroup barriers")

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
