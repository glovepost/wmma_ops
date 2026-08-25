#!/usr/bin/env python3
"""Dephase alternate 40-CU scheduling rounds before the kernel prologue.

The record launch contains 512 one-dimensional workgroups on a 40-CU gfx1151
device.  Assuming the ordinary sequential dispatch order, workgroups 0..39
fill the first resident slot and 40..79 fill the second.  Alternating
``(workgroup_id / 40) & 1`` therefore gives the two initially co-resident
workgroups different startup phases without changing the repeated K16 loop.
"""

import argparse
from pathlib import Path


ENTRY = "; %bb.0:\n\ts_load_b128 s[16:19], s[0:1], 0x18\n"
MAGIC_DIVIDE_40 = 0xCCCCCCCD


def scheduling_round(workgroup_id: int) -> int:
    """Mirror the unsigned scalar magic-number division emitted in assembly."""

    high = (workgroup_id * MAGIC_DIVIDE_40) >> 32
    return high >> 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "delay",
        help="S_SLEEP immediate, or 'nop' for a same-size classifier control",
    )
    args = parser.parse_args()

    if args.delay == "nop":
        delay_instruction = "\ts_nop 0\n"
    else:
        delay = int(args.delay, 0)
        if not 1 <= delay <= 127:
            raise ValueError("S_SLEEP immediate must be in [1, 127]")
        delay_instruction = f"\ts_sleep {delay}\n"

    # The record grid is small, but verify the exact magic division over a much
    # wider range so a future shape extension cannot silently change parity.
    for workgroup_id in range(1 << 20):
        if scheduling_round(workgroup_id) != workgroup_id // 40:
            raise AssertionError(f"bad divide-by-40 result for {workgroup_id}")

    source = args.input.read_text()
    if source.count(ENTRY) != 1:
        raise ValueError("expected one fixed kernel entry sequence")
    if ".Lphase_skew_done:" in source:
        raise ValueError("input already contains a phase-skew transform")

    classifier = (
        "; One-time 40-CU residency-round phase skew. s19 is dead until the\n"
        "; K-loop trip-count setup, where the original image overwrites it.\n"
        f"\ts_mul_hi_u32 s19, s2, 0x{MAGIC_DIVIDE_40:08x}\n"
        "\ts_lshr_b32 s19, s19, 5\n"
        "\ts_and_b32 s19, s19, 1\n"
        "\ts_cmp_eq_u32 s19, 0\n"
        "\ts_cbranch_scc1 .Lphase_skew_done\n"
        + delay_instruction
        + ".Lphase_skew_done:\n"
    )
    args.output.write_text(source.replace(ENTRY, "; %bb.0:\n" + classifier + "\ts_load_b128 s[16:19], s[0:1], 0x18\n", 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
