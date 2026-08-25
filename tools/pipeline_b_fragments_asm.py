#!/usr/bin/env python3
"""Double-buffer the four B fragments in the fixed gfx1151 K16 hot loop."""

from pathlib import Path
import sys


START = ".LBB0_13:                               ; =>This Inner Loop Header: Depth=1\n"
END = "\ts_clause 0x1\n"


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s [GROUP_SEGMENT_BYTES]",
            file=sys.stderr,
        )
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    group_segment_bytes = int(sys.argv[3]) if len(sys.argv) == 4 else 18432
    if group_segment_bytes < 18432:
        raise ValueError("group reservation cannot be smaller than the used LDS")
    source = input_path.read_text()
    if source.count(START) != 1:
        raise ValueError("expected one fixed K16 loop")

    start = source.index(START)
    end = source.index(END, start)
    old = source[start:end]
    lines = old.splitlines(keepends=True)
    if len(lines) != 37:
        raise ValueError(f"unexpected compute prefix shape: {len(lines)} lines")
    if sum("ds_load_b128" in line for line in lines) != 16:
        raise ValueError("expected ten A/B0 loads and three two-load B fragments")
    if sum("v_wmma_f16_16x16x16_f16" in line for line in lines) != 13:
        raise ValueError("expected 13 WMMAs before the refill clause")

    # Preserve the first ten LDS loads and add B1 into the otherwise unused
    # v120:v127 allocation tail.  The final three B groups are then alternated
    # between v92:v99 and v120:v127.  lgkmcnt(2) retires the older pair while
    # leaving the newly issued pair in flight during four independent WMMAs.
    prefix = lines[:11]
    replacement = prefix + [
        "\tds_load_b128 v[124:127], v67 offset:13072\n",
        "\tds_load_b128 v[120:123], v67 offset:13056\n",
        "\ts_add_i32 s6, s6, -1\n",
        "\ts_waitcnt lgkmcnt(6)\n",
        "\tv_wmma_f16_16x16x16_f16 v[57:64], v[76:83], v[92:99], v[57:64]\n",
        "\tv_wmma_f16_16x16x16_f16 v[41:48], v[84:91], v[92:99], v[41:48]\n",
        "\ts_waitcnt lgkmcnt(4)\n",
        "\tv_wmma_f16_16x16x16_f16 v[25:32], v[100:107], v[92:99], v[25:32]\n",
        "\ts_waitcnt lgkmcnt(2)\n",
        "\tv_wmma_f16_16x16x16_f16 v[9:16], v[108:115], v[92:99], v[9:16]\n",
        "\tds_load_b128 v[96:99], v67 offset:13840\n",
        "\tds_load_b128 v[92:95], v67 offset:13824\n",
        "\ts_waitcnt lgkmcnt(2)\n",
        "\tv_wmma_f16_16x16x16_f16 v[49:56], v[76:83], v[120:127], v[49:56]\n",
        "\tv_wmma_f16_16x16x16_f16 v[33:40], v[84:91], v[120:127], v[33:40]\n",
        "\tv_wmma_f16_16x16x16_f16 v[17:24], v[100:107], v[120:127], v[17:24]\n",
        "\tv_wmma_f16_16x16x16_f16 v[1:8], v[108:115], v[120:127], v[1:8]\n",
        "\tds_load_b128 v[124:127], v67 offset:14608\n",
        "\tds_load_b128 v[120:123], v67 offset:14592\n",
        "\ts_waitcnt lgkmcnt(2)\n",
        "\tv_wmma_f16_16x16x16_f16 v[57:64], v[76:83], v[92:99], v[57:64] op_sel:[0,0,1]\n",
        "\tv_wmma_f16_16x16x16_f16 v[41:48], v[84:91], v[92:99], v[41:48] op_sel:[0,0,1]\n",
        "\tv_wmma_f16_16x16x16_f16 v[25:32], v[100:107], v[92:99], v[25:32] op_sel:[0,0,1]\n",
        "\tv_wmma_f16_16x16x16_f16 v[9:16], v[108:115], v[92:99], v[9:16] op_sel:[0,0,1]\n",
        "\ts_waitcnt lgkmcnt(0)\n",
        "\tv_wmma_f16_16x16x16_f16 v[49:56], v[76:83], v[120:127], v[49:56] op_sel:[0,0,1]\n",
    ]

    new = "".join(replacement)
    if new.count("\tv_wmma_f16_16x16x16_f16") != 13:
        raise AssertionError("pipeline replacement lost a WMMA")
    if new.count("\tds_load_b128") != 16:
        raise AssertionError("pipeline replacement changed the LDS load count")

    source = source[:start] + new + source[end:]

    # LLVM placed the refill clause between the first and final three B3
    # WMMAs.  Retarget those remaining consumers to the alternate fragment as
    # well; the global refill destinations are intentionally unchanged.
    post_start = source.index(END, start) + len(END)
    post_end = source.index("\ts_barrier\n", post_start)
    post_clause = source[post_start:post_end]
    for accumulator, a_fragment in (
        ("v[33:40]", "v[84:91]"),
        ("v[17:24]", "v[100:107]"),
        ("v[1:8]", "v[108:115]"),
    ):
        old_wmma = (
            f"\tv_wmma_f16_16x16x16_f16 {accumulator}, {a_fragment}, "
            f"v[92:99], {accumulator} op_sel:[0,0,1]\n"
        )
        new_wmma = old_wmma.replace("v[92:99]", "v[120:127]", 1)
        if post_clause.count(old_wmma) != 1:
            raise ValueError(f"expected one post-clause {accumulator} WMMA")
        post_clause = post_clause.replace(old_wmma, new_wmma, 1)
    source = source[:post_start] + post_clause + source[post_end:]
    metadata = (
        ("\t\t.amdhsa_next_free_vgpr 120\n", "\t\t.amdhsa_next_free_vgpr 128\n"),
        ("    .vgpr_count:     120\n", "    .vgpr_count:     128\n"),
    )
    for old_field, new_field in metadata:
        if source.count(old_field) != 1:
            raise ValueError(f"expected one metadata field: {old_field.strip()}")
        source = source.replace(old_field, new_field, 1)
    if group_segment_bytes != 18432:
        group_metadata = (
            (
                "\t\t.amdhsa_group_segment_fixed_size 18432\n",
                f"\t\t.amdhsa_group_segment_fixed_size {group_segment_bytes}\n",
            ),
            (
                "    .group_segment_fixed_size: 18432\n",
                f"    .group_segment_fixed_size: {group_segment_bytes}\n",
            ),
        )
        for old_field, new_field in group_metadata:
            if source.count(old_field) != 1:
                raise ValueError(f"expected one metadata field: {old_field.strip()}")
            source = source.replace(old_field, new_field, 1)

    output_path.write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
