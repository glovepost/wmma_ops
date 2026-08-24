#!/usr/bin/env python3
"""Reorder independent WMMA rows after the progressive first fragment."""

from pathlib import Path
import sys


HOT_LOOP = ".LBB0_13:"
HOT_LOOP_END = "; %bb.14:"


def main() -> int:
    if len(sys.argv) not in (4, 6):
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s ORDER [ORDER2 ORDER3]",
            file=sys.stderr,
        )
        return 2

    order_texts = [sys.argv[3]] if len(sys.argv) == 4 else sys.argv[3:]
    orders = []
    for order_text in order_texts:
        if len(order_text) != 4 or set(order_text) != set("0123"):
            raise ValueError("each ORDER must be a permutation of 0123")
        if order_text[0] != "0":
            raise ValueError(
                "each ORDER must start with row 0 because the final fragment "
                "refills row 0 immediately after its WMMA"
            )
        orders.append([int(character) for character in order_text])
    if len(orders) == 1:
        orders *= 3

    source = Path(sys.argv[1]).read_text()
    start = source.index(HOT_LOOP)
    end = source.index(HOT_LOOP_END, start)
    hot = source[start:end]
    lines = hot.splitlines(keepends=True)
    wmma_indices = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("v_wmma_f16_16x16x16_f16 ")
    ]
    if len(wmma_indices) != 16:
        raise ValueError(f"expected 16 hot-loop WMMAs, found {len(wmma_indices)}")

    # Fragment 0 is progressively released by lgkmcnt(4/2/0), so its row
    # order is a real load dependency and stays untouched. Fragments 1--3
    # have all operands ready before their first WMMA. Fragment 3 must still
    # consume row 0 first because the following global refill overwrites A0.
    for group in range(1, 4):
        indices = wmma_indices[group * 4 : group * 4 + 4]
        original = [lines[index] for index in indices]
        for destination, row in zip(indices, orders[group - 1]):
            lines[destination] = original[row]

    transformed_hot = "".join(lines)
    transformed = source[:start] + transformed_hot + source[end:]
    Path(sys.argv[2]).write_text(transformed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
