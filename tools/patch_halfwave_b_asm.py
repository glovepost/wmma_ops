#!/usr/bin/env python3
"""Halve hot-loop B LDS bytes by rebuilding replicated halves in-wave."""

from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def rebuild_fragment(low: int, high: int) -> str:
    lines = [
        f"\tv_permlanex16_b32 v{high + i}, v{low + i}, s20, s21"
        for i in range(4)
    ]
    lines.append("\ts_mov_b32 exec_lo, 0xffff0000")
    lines.extend(
        f"\tv_swap_b32 v{low + i}, v{high + i}" for i in range(4)
    )
    lines.append("\ts_mov_b32 exec_lo, -1")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).read_text()

    # The leader uses s0:s19 explicitly.  The two new selectors raise the
    # ordinary allocation by one four-SGPR granule; this remains far below the
    # occupancy limit imposed by 120 VGPR and 18 KiB LDS.
    source = replace_once(
        source,
        "\t.amdhsa_next_free_sgpr 20\n",
        "\t.amdhsa_next_free_sgpr 22\n",
        "kernel SGPR declaration",
    )
    source = replace_once(
        source,
        "    .sgpr_count:     22\n",
        "    .sgpr_count:     24\n",
        "metadata SGPR count",
    )

    # v67 addresses the same B row in both wave halves.  Add 16 bytes for
    # lanes 16--31 so one b128 load fetches the low half in row 0 and the high
    # half in row 1.  Identity selectors gather the corresponding lane from
    # the opposite row.
    source = replace_once(
        source,
        "\tv_mul_u32_u24_e32 v67, 48, v69\n"
        "\tv_mov_b32_e32 v59, v57\n",
        "\tv_mul_u32_u24_e32 v67, 48, v69\n"
        "\tv_and_b32_e32 v65, 16, v0\n"
        "\tv_add_nc_u32_e32 v65, v67, v65\n"
        "\ts_mov_b32 s20, 0x76543210\n"
        "\ts_mov_b32 s21, 0xfedcba98\n"
        "\tv_mov_b32_e32 v59, v57\n",
        "half-wave B address and selectors",
    )

    # B0 is the fifth of nine LDS operations after the transform.  lgkmcnt(4)
    # therefore preserves the leader's exact readiness point.
    source = replace_once(
        source,
        "\tds_load_b128 v[92:95], v67 offset:12288\n"
        "\tds_load_b128 v[96:99], v67 offset:12304\n"
        "\tds_load_b128 v[104:107], v68 offset:1552\n",
        "\tds_load_b128 v[92:95], v65 offset:12288\n"
        "\tds_load_b128 v[104:107], v68 offset:1552\n",
        "hot-loop B0 load",
    )
    source = replace_once(
        source,
        "\ts_waitcnt lgkmcnt(4)\n"
        "\tv_wmma_f16_16x16x16_f16 v[57:64], v[76:83], v[92:99], v[57:64]\n",
        "\ts_waitcnt lgkmcnt(4)\n"
        + rebuild_fragment(92, 96)
        + "\tv_wmma_f16_16x16x16_f16 v[57:64], v[76:83], v[92:99], v[57:64]\n",
        "hot-loop B0 reconstruction",
    )

    for name, offset in (("B1", 13056), ("B2", 13824), ("B3", 14592)):
        source = replace_once(
            source,
            f"\tds_load_b128 v[96:99], v67 offset:{offset + 16}\n"
            f"\tds_load_b128 v[92:95], v67 offset:{offset}\n"
            "\ts_waitcnt lgkmcnt(0)\n",
            f"\tds_load_b128 v[92:95], v65 offset:{offset}\n"
            "\ts_waitcnt lgkmcnt(0)\n"
            + rebuild_fragment(92, 96),
            f"hot-loop {name} load/reconstruction",
        )

    # The final, non-loop K slice is deliberately untouched.  These exact
    # counts prove only the 255 steady-state B pairs were replaced.
    if source.count("\tv_permlanex16_b32 ") != 16:
        raise SystemExit("expected four dword gathers for each hot-loop B fragment")
    if source.count("\tv_swap_b32 ") != 16:
        raise SystemExit("expected four upper-row swaps for each hot-loop B fragment")

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
