#!/usr/bin/env python3
"""Give alternating 40-CU scheduling rounds different wave priorities."""

import argparse
from pathlib import Path


ENTRY = "; %bb.0:\n\ts_load_b128 s[16:19], s[0:1], 0x18\n"
MAGIC_DIVIDE_40 = 0xCCCCCCCD


def scheduling_round(workgroup_id: int) -> int:
    high = (workgroup_id * MAGIC_DIVIDE_40) >> 32
    return high >> 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("policy", choices=("even-high", "odd-high", "nop"))
    args = parser.parse_args()

    for workgroup_id in range(1 << 20):
        if scheduling_round(workgroup_id) != workgroup_id // 40:
            raise AssertionError(f"bad divide-by-40 result for {workgroup_id}")

    source = args.input.read_text()
    if source.count(ENTRY) != 1:
        raise ValueError("expected one fixed kernel entry sequence")
    if ".Lpriority_partition_done:" in source:
        raise ValueError("input already contains a priority transform")

    branch = (
        "\ts_cbranch_scc0 .Lpriority_partition_done\n"
        if args.policy == "even-high"
        else "\ts_cbranch_scc1 .Lpriority_partition_done\n"
    )
    priority_instruction = (
        "\ts_nop 0\n" if args.policy == "nop" else "\ts_setprio 1\n"
    )
    classifier = (
        "; Assign persistent priority by 40-CU scheduling-round parity. s19\n"
        "; is dead here and overwritten by the original K-loop setup.\n"
        f"\ts_mul_hi_u32 s19, s2, 0x{MAGIC_DIVIDE_40:08x}\n"
        "\ts_lshr_b32 s19, s19, 5\n"
        "\ts_and_b32 s19, s19, 1\n"
        "\ts_cmp_eq_u32 s19, 0\n"
        + branch
        + priority_instruction
        + ".Lpriority_partition_done:\n"
    )
    replacement = "; %bb.0:\n" + classifier + ENTRY.split("\n", 1)[1]
    args.output.write_text(source.replace(ENTRY, replacement, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
