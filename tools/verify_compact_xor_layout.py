#!/usr/bin/env python3
"""Exhaustively prove the padding-free half-row XOR LDS layout."""

from collections import Counter


def slot(row: int, half: int) -> int:
    return 2 * row + (half ^ ((row >> 2) & 1))


def verify(rows: int) -> None:
    slots = [slot(row, half) for row in range(rows) for half in range(2)]
    assert sorted(slots) == list(range(2 * rows))
    for fragment_base in range(0, rows, 16):
        for half in range(2):
            phases = Counter(
                slot(fragment_base + lane, half) % 8
                for lane in range(16)
            )
            assert phases == Counter({phase: 2 for phase in range(8)})
        for lane in range(16):
            row = fragment_base + lane
            assert {slot(row, 0), slot(row, 1)} == {2 * row, 2 * row + 1}


def main() -> int:
    verify(256)
    verify(128)
    print("compact XOR layout: bijective; every K16 half spans 8 LDS phases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
