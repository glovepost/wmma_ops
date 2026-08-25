#!/usr/bin/env python3
"""Add VALU/LDS clauses to natural same-type runs in the selected K loop.

RDNA 3.5 S_CLAUSE encodes ``instruction_count - 1``. A clause may contain
only one instruction type, so this transform never moves instructions and
only prefixes already-contiguous WMMA or DS_LOAD_B128 runs.
"""

import argparse
from pathlib import Path


LOOP_START = ".LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"
LOOP_END = "; %bb.14:\n"
EXPECTED_RUNS = {
    "wmma4": [("v_wmma_f16_16x16x16_f16", 4)] * 2,
    "wmma-all": [
        ("v_wmma_f16_16x16x16_f16", 2),
        ("v_wmma_f16_16x16x16_f16", 4),
        ("v_wmma_f16_16x16x16_f16", 4),
        ("v_wmma_f16_16x16x16_f16", 3),
    ],
    "lds10": [("ds_load_b128", 10)],
    "lds-all": [
        ("ds_load_b128", 10),
        ("ds_load_b128", 2),
        ("ds_load_b128", 2),
        ("ds_load_b128", 2),
    ],
}


def opcode(line: str) -> str:
    stripped = line.strip()
    return stripped.split(maxsplit=1)[0] if stripped else ""


def selected_runs(lines: list[str], policy: str) -> list[tuple[int, int, str]]:
    want_wmma = policy in {
        "wmma4", "wmma-all", "wmma4-lds10", "both-all"
    }
    want_lds = policy in {
        "lds10", "lds-all", "wmma4-lds10", "both-all"
    }
    runs: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        op = opcode(lines[index])
        if op not in {"v_wmma_f16_16x16x16_f16", "ds_load_b128"}:
            index += 1
            continue
        end = index + 1
        while end < len(lines) and opcode(lines[end]) == op:
            end += 1
        count = end - index
        select = (
            (op == "v_wmma_f16_16x16x16_f16" and want_wmma
             and (policy not in {"wmma4", "wmma4-lds10"} or count >= 4))
            or (op == "ds_load_b128" and want_lds
                and (policy not in {"lds10", "wmma4-lds10"} or count >= 10))
        )
        if select and count >= 2:
            runs.append((index, count, op))
        index = end
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "policy",
        choices=[
            "wmma4", "wmma-all", "lds10", "lds-all", "wmma4-lds10",
            "both-all",
        ],
    )
    parser.add_argument("instruction", choices=["clause", "nop"])
    args = parser.parse_args()

    source = args.input.read_text()
    if source.count(LOOP_START) != 1 or source.count(LOOP_END) < 1:
        raise ValueError("expected the selected kernel's fixed hot-loop boundaries")
    prefix, remainder = source.split(LOOP_START, 1)
    loop, suffix = remainder.split(LOOP_END, 1)
    lines = loop.splitlines(keepends=True)
    runs = selected_runs(lines, args.policy)

    if args.policy == "both-all":
        expected = EXPECTED_RUNS["wmma-all"] + EXPECTED_RUNS["lds-all"]
    elif args.policy == "wmma4-lds10":
        expected = EXPECTED_RUNS["wmma4"] + EXPECTED_RUNS["lds10"]
    else:
        expected = EXPECTED_RUNS[args.policy]
    actual = [(op, count) for _, count, op in runs]
    # both-all is discovered in program order rather than type-grouped order.
    if args.policy in {"both-all", "wmma4-lds10"}:
        if sorted(actual) != sorted(expected):
            raise ValueError(f"unexpected natural runs: {actual}")
    elif actual != expected:
        raise ValueError(f"unexpected natural runs: {actual}")

    starts = {start: count for start, count, _ in runs}
    transformed: list[str] = []
    for index, line in enumerate(lines):
        if index in starts:
            count = starts[index]
            inserted = f"\ts_clause 0x{count - 1:x}\n"
            if args.instruction == "nop":
                inserted = "\ts_nop 0\n"
            transformed.append(inserted)
        transformed.append(line)

    args.output.write_text(
        prefix + LOOP_START + "".join(transformed) + LOOP_END + suffix
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
