#!/usr/bin/env python3
"""Narrow the hot-loop publication wait after VMEM is already retired."""

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s LGKMCNT",
            file=sys.stderr,
        )
        return 2

    threshold = int(sys.argv[3])
    if threshold not in (0, 1, 2):
        raise ValueError("LGKMCNT must be 0, 1, or 2")

    source = Path(sys.argv[1]).read_text()
    old = (
        "\tds_store_b128 v75, v[116:119]\n"
        "\ts_waitcnt vmcnt(0) lgkmcnt(0)\n"
        "\ts_barrier\n"
        "\ts_cbranch_scc0 .LBB0_13\n"
    )
    new = (
        "\tds_store_b128 v75, v[116:119]\n"
        f"\ts_waitcnt lgkmcnt({threshold})\n"
        "\ts_barrier\n"
        "\ts_cbranch_scc0 .LBB0_13\n"
    )
    if source.count(old) != 1:
        raise ValueError("expected one hot-loop publication wait")
    Path(sys.argv[2]).write_text(source.replace(old, new, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
