#!/usr/bin/env python3
"""Pair-unroll the fixed-K=4096 single-buffer K16 hot loop.

The original loop executes 255 identical hot bodies after the initial K16
tile.  K=4096 leaves an even 254 bodies after peeling one body, so the
remaining bodies can run in pairs with one compare/branch per pair.  This
transform deliberately preserves every load, wait, WMMA, store, and barrier;
it is a record-shape specialization, not a general-K kernel transform.
"""

from pathlib import Path
import sys


LOOP_LABEL = ".LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"
COMPARE = "\ts_cmp_eq_u32 s6, 0\n"
BRANCH = "\ts_cbranch_scc0 .LBB0_13\n"
PAIR_LABEL = ".LWMMA_KPAIR:\n"
PAIR_BRANCH = "\ts_cbranch_scc0 .LWMMA_KPAIR\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s", file=sys.stderr)
        return 2

    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    source = source_path.read_text()

    if source.count(LOOP_LABEL) != 1:
        raise ValueError("expected exactly one K16 hot-loop label")
    if source.count(BRANCH) != 1:
        raise ValueError("expected exactly one K16 hot-loop back edge")
    if PAIR_LABEL in source or PAIR_BRANCH in source:
        raise ValueError("input already contains the pair-unrolled loop")

    start = source.index(LOOP_LABEL)
    branch_start = source.index(BRANCH, start)
    end = branch_start + len(BRANCH)
    loop_block = source[start:end]
    body = loop_block[len(LOOP_LABEL) : -len(BRANCH)]

    if body.count(COMPARE) != 1:
        raise ValueError("expected exactly one loop-counter compare in hot body")
    if body.count("\ts_add_i32 s6, s6, -1\n") != 1:
        raise ValueError("expected exactly one loop-counter decrement in hot body")
    if body.count("\ts_barrier\n") != 2:
        raise ValueError("expected both correctness barriers in hot body")

    body_without_compare = body.replace(COMPARE, "", 1)
    replacement = (
        LOOP_LABEL
        + body_without_compare  # peeled body: 255 -> 254
        + PAIR_LABEL
        + body_without_compare  # first body in pair: no control test
        + body                  # second body in pair: test for zero
        + PAIR_BRANCH
    )

    if replacement.count("\ts_add_i32 s6, s6, -1\n") != 3:
        raise AssertionError("pair-unrolled replacement must contain three bodies")
    if replacement.count(COMPARE) != 1:
        raise AssertionError("pair-unrolled replacement must contain one compare")
    if replacement.count("\ts_barrier\n") != 6:
        raise AssertionError("pair-unrolled replacement lost a correctness barrier")

    output_path.write_text(source[:start] + replacement + source[end:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
