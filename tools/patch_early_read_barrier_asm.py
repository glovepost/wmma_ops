#!/usr/bin/env python3
"""Overlap final register-only WMMAs with the single-buffer refill commit."""

from pathlib import Path
import sys


WMMAS = (
    "\tv_wmma_f16_16x16x16_f16 v[33:40], v[84:91], v[92:99], "
    "v[33:40] op_sel:[0,0,1]\n"
    "\tv_wmma_f16_16x16x16_f16 v[17:24], v[100:107], v[92:99], "
    "v[17:24] op_sel:[0,0,1]\n"
    "\tv_wmma_f16_16x16x16_f16 v[1:8], v[108:115], v[92:99], "
    "v[1:8] op_sel:[0,0,1]\n"
)
COMMITS = (
    "\ts_waitcnt vmcnt(2)\n"
    "\tds_store_b128 v70, v[76:79]\n"
    "\ts_waitcnt vmcnt(1)\n"
    "\tds_store_b128 v70, v[80:83] offset:16\n"
    "\ts_waitcnt vmcnt(0)\n"
    "\tds_store_b128 v75, v[116:119]\n"
)
OLD = "\ts_cmp_eq_u32 s6, 0\n" + WMMAS + "\ts_barrier\n" + COMMITS


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s "
            "{wmma-interleave|store-interleave|stores-first|"
            "stores-first-lgkm|stores-first-no-wait}",
            file=sys.stderr,
        )
        return 2

    mode = sys.argv[3]
    wmma = WMMAS.splitlines(keepends=True)
    commit = COMMITS.splitlines(keepends=True)
    if mode == "wmma-interleave":
        schedule = (
            wmma[0:1] + commit[0:2]
            + wmma[1:2] + commit[2:4]
            + wmma[2:3] + commit[4:6]
        )
    elif mode == "store-interleave":
        schedule = (
            commit[0:2] + wmma[0:1]
            + commit[2:4] + wmma[1:2]
            + commit[4:6] + wmma[2:3]
        )
    elif mode in ("stores-first", "stores-first-lgkm", "stores-first-no-wait"):
        schedule = commit + wmma
    else:
        raise ValueError("unknown early-barrier schedule")

    source = Path(sys.argv[1]).read_text()
    if source.count(OLD) != 1:
        raise ValueError("expected one final-WMMA/refill handoff")
    replacement = "\ts_cmp_eq_u32 s6, 0\n\ts_barrier\n" + "".join(schedule)
    output = source.replace(OLD, replacement, 1)
    if mode in ("stores-first-lgkm", "stores-first-no-wait"):
        old_tail = (
            "\tv_wmma_f16_16x16x16_f16 v[1:8], v[108:115], v[92:99], "
            "v[1:8] op_sel:[0,0,1]\n"
            "\ts_waitcnt vmcnt(0) lgkmcnt(0)\n"
            "\ts_barrier\n"
        )
        wait = "\ts_waitcnt lgkmcnt(0)\n" if mode == "stores-first-lgkm" else ""
        new_tail = (
            "\tv_wmma_f16_16x16x16_f16 v[1:8], v[108:115], v[92:99], "
            "v[1:8] op_sel:[0,0,1]\n"
            + wait
            + "\ts_barrier\n"
        )
        if output.count(old_tail) != 1:
            raise ValueError("expected one stores-first publication tail")
        output = output.replace(old_tail, new_tail, 1)
    Path(sys.argv[2]).write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
