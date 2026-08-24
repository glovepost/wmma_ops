#!/usr/bin/env python3
"""Retarget the validated p4 two-address LDS schedule to p8.

The ds_load_2addr_b64 offsets are eight-byte units and only eight bits wide.
A p8 row is six units, so the fourth 16-row fragment begins at unit 288 and
does not fit directly.  Two loop-invariant bases shifted by 64 units keep that
fragment at offsets 224--227 without adding hot-loop address arithmetic.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def restore_b128_stores(source: str) -> str:
    pattern = re.compile(
        r"^(\s*)ds_store_2addr_b64 (v\d+), "
        r"v\[(\d+):(\d+)\], v\[(\d+):(\d+)\](.*)$"
    )
    output: list[str] = []
    converted = 0
    for raw_line in source.splitlines(keepends=True):
        newline = "\n" if raw_line.endswith("\n") else ""
        line = raw_line[:-1] if newline else raw_line
        match = pattern.match(line)
        if not match:
            output.append(raw_line)
            continue
        indent, base, first_lo, first_hi, second_lo, second_hi, suffix = match.groups()
        registers = tuple(map(int, (first_lo, first_hi, second_lo, second_hi)))
        if registers[1] != registers[0] + 1 \
                or registers[2] != registers[1] + 1 \
                or registers[3] != registers[2] + 1:
            raise ValueError(f"non-contiguous paired store: {line}")
        offset0_match = re.search(r"offset0:(\d+)", suffix)
        offset1_match = re.search(r"offset1:(\d+)", suffix)
        offset0 = int(offset0_match.group(1)) if offset0_match else 0
        offset1 = int(offset1_match.group(1)) if offset1_match else 1
        if offset1 != offset0 + 1:
            raise ValueError(f"non-adjacent paired store: {line}")
        offset = f" offset:{8 * offset0}" if offset0 else ""
        output.append(
            f"{indent}ds_store_b128 {base}, "
            f"v[{registers[0]}:{registers[3]}]{offset}{newline}"
        )
        converted += 1
    if converted != 6:
        raise ValueError(f"expected six paired stores, found {converted}")
    return "".join(output)


def transform(source: str, *, b128_stores: bool = False) -> str:
    if source.count(".group_segment_fixed_size: 15360") != 1:
        raise ValueError("expected one p4 metadata LDS size")
    if source.count(".amdhsa_group_segment_fixed_size 15360") != 1:
        raise ValueError("expected one p4 descriptor LDS size")
    if source.count(".LBB0_13:") != 1:
        raise ValueError("expected one hot-loop label")

    text = source.replace(", 40,", ", 48,")
    text = text.replace("0x2800", "0x3000")
    text = text.replace(
        ".amdhsa_group_segment_fixed_size 15360",
        ".amdhsa_group_segment_fixed_size 18432",
    )
    text = text.replace(
        ".group_segment_fixed_size: 15360",
        ".group_segment_fixed_size: 18432",
    )

    offsets = {80: 96, 81: 97, 82: 98, 83: 99,
               160: 192, 161: 193, 162: 194, 163: 195,
               240: 224, 241: 225, 242: 226, 243: 227}
    high_lines = 0
    transformed: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("ds_load_2addr_b64"):
            original = line
            high = any(re.search(rf"offset[01]:{value}(?:\D|$)", original)
                       for value in (240, 241, 242, 243))
            for old, new in offsets.items():
                line = re.sub(rf"(offset[01]:){old}(?=\D|$)", rf"\g<1>{new}", line)
            if high:
                if ", v66 " in line:
                    line = line.replace(", v66 ", ", v119 ", 1)
                elif ", v65 " in line:
                    line = line.replace(", v65 ", ", v120 ", 1)
                else:
                    raise ValueError(f"unexpected high-offset base: {original.rstrip()}")
                high_lines += 1
        transformed.append(line)
    if high_lines != 8:
        raise ValueError(f"expected eight high-offset loads, found {high_lines}")
    text = "".join(transformed)

    setup = ("\tv_add_nc_u32_e32 v119, 512, v66\n"
             "\tv_add_nc_u32_e32 v120, 512, v65\n")
    text = text.replace(".LBB0_13:", setup + ".LBB0_13:", 1)

    text = text.replace(".amdhsa_next_free_vgpr 119",
                        ".amdhsa_next_free_vgpr 121")
    text = text.replace(".num_vgpr, 119", ".num_vgpr, 121")
    text = text.replace(".vgpr_count:     119", ".vgpr_count:     121")

    if "offset0:240" in text or "offset0:242" in text:
        raise ValueError("unconverted high offset remains")
    if text.count("v119") < 3 or text.count("v120") < 3:
        raise ValueError("shifted LDS bases were not used")
    if b128_stores:
        text = restore_b128_stores(text)
    return text


def main() -> int:
    args = sys.argv[1:]
    b128_stores = False
    if args and args[0] == "--b128-stores":
        b128_stores = True
        args.pop(0)
    if len(args) != 2:
        print(
            f"usage: {sys.argv[0]} [--b128-stores] INPUT.s OUTPUT.s",
            file=sys.stderr,
        )
        return 2
    source_path = Path(args[0])
    output_path = Path(args[1])
    output_path.write_text(
        transform(source_path.read_text(), b128_stores=b128_stores)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
