#!/usr/bin/env python3
"""Pipeline one future p8 B fragment across the current WMMA group."""

from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(
            f"usage: {sys.argv[0]} INPUT.s OUTPUT.s [--base=VGPR]",
            file=sys.stderr,
        )
        return 2
    base = 118
    if len(sys.argv) == 4:
        prefix = "--base="
        if not sys.argv[3].startswith(prefix):
            raise ValueError("expected --base=VGPR")
        base = int(sys.argv[3][len(prefix):])
        if base not in (118, 120, 122, 124):
            raise ValueError("base must be one of 118, 120, 122, or 124")

    source = Path(sys.argv[1]).read_text()
    old = """\
\tds_load_b128 v[74:77], v66
\tds_load_b128 v[78:81], v66 offset:16
\tds_load_b128 v[86:89], v66 offset:784
\tds_load_b128 v[82:85], v66 offset:768
\tds_load_b128 v[90:93], v65 offset:12288
\tds_load_b128 v[94:97], v65 offset:12304
\tds_load_b128 v[102:105], v66 offset:1552
\tds_load_b128 v[98:101], v66 offset:1536
\tds_load_b128 v[110:113], v66 offset:2320
\tds_load_b128 v[106:109], v66 offset:2304
\ts_add_i32 s6, s6, -1
\ts_waitcnt lgkmcnt(4)
\tv_wmma_f16_16x16x16_f16 v[57:64], v[74:81], v[90:97], v[57:64]
\tv_wmma_f16_16x16x16_f16 v[41:48], v[82:89], v[90:97], v[41:48]
\ts_waitcnt lgkmcnt(2)
\tv_wmma_f16_16x16x16_f16 v[25:32], v[98:105], v[90:97], v[25:32]
\ts_waitcnt lgkmcnt(0)
\tv_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], v[9:16]
\tds_load_b128 v[94:97], v65 offset:13072
\tds_load_b128 v[90:93], v65 offset:13056
\ts_waitcnt lgkmcnt(0)
\tv_wmma_f16_16x16x16_f16 v[49:56], v[74:81], v[90:97], v[49:56]
\tv_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[90:97], v[33:40]
\tv_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[90:97], v[17:24]
\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], v[1:8]
\tds_load_b128 v[94:97], v65 offset:13840
\tds_load_b128 v[90:93], v65 offset:13824
\ts_waitcnt lgkmcnt(0)
\tv_wmma_f16_16x16x16_f16 v[57:64], v[74:81], v[90:97], v[57:64] op_sel:[0,0,1]
\tv_wmma_f16_16x16x16_f16 v[41:48], v[82:89], v[90:97], v[41:48] op_sel:[0,0,1]
\tv_wmma_f16_16x16x16_f16 v[25:32], v[98:105], v[90:97], v[25:32] op_sel:[0,0,1]
\tv_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], v[9:16] op_sel:[0,0,1]
\tds_load_b128 v[94:97], v65 offset:14608
\tds_load_b128 v[90:93], v65 offset:14592
\ts_waitcnt lgkmcnt(0)
\tv_wmma_f16_16x16x16_f16 v[49:56], v[74:81], v[90:97], v[49:56] op_sel:[0,0,1]
"""
    new = """\
\tds_load_b128 v[74:77], v66
\tds_load_b128 v[78:81], v66 offset:16
\tds_load_b128 v[86:89], v66 offset:784
\tds_load_b128 v[82:85], v66 offset:768
\tds_load_b128 v[90:93], v65 offset:12288
\tds_load_b128 v[94:97], v65 offset:12304
\tds_load_b128 v[102:105], v66 offset:1552
\tds_load_b128 v[98:101], v66 offset:1536
\tds_load_b128 v[110:113], v66 offset:2320
\tds_load_b128 v[106:109], v66 offset:2304
\tds_load_b128 v[122:125], v65 offset:13072
\tds_load_b128 v[118:121], v65 offset:13056
\ts_add_i32 s6, s6, -1
\ts_waitcnt lgkmcnt(6)
\tv_wmma_f16_16x16x16_f16 v[57:64], v[74:81], v[90:97], v[57:64]
\tv_wmma_f16_16x16x16_f16 v[41:48], v[82:89], v[90:97], v[41:48]
\ts_waitcnt lgkmcnt(4)
\tv_wmma_f16_16x16x16_f16 v[25:32], v[98:105], v[90:97], v[25:32]
\ts_waitcnt lgkmcnt(2)
\tv_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], v[9:16]
\ts_waitcnt lgkmcnt(0)
\tds_load_b128 v[94:97], v65 offset:13840
\tds_load_b128 v[90:93], v65 offset:13824
\tv_wmma_f16_16x16x16_f16 v[49:56], v[74:81], v[118:125], v[49:56]
\tv_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[118:125], v[33:40]
\tv_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[118:125], v[17:24]
\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[118:125], v[1:8]
\ts_waitcnt lgkmcnt(0)
\tds_load_b128 v[122:125], v65 offset:14608
\tds_load_b128 v[118:121], v65 offset:14592
\tv_wmma_f16_16x16x16_f16 v[57:64], v[74:81], v[90:97], v[57:64] op_sel:[0,0,1]
\tv_wmma_f16_16x16x16_f16 v[41:48], v[82:89], v[90:97], v[41:48] op_sel:[0,0,1]
\tv_wmma_f16_16x16x16_f16 v[25:32], v[98:105], v[90:97], v[25:32] op_sel:[0,0,1]
\tv_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], v[9:16] op_sel:[0,0,1]
\ts_waitcnt lgkmcnt(0)
\tv_wmma_f16_16x16x16_f16 v[49:56], v[74:81], v[118:125], v[49:56] op_sel:[0,0,1]
"""
    source = replace_once(source, old, new, "hot-loop B lookahead")
    source = replace_once(
        source,
        "\tv_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[90:97], "
        "v[33:40] op_sel:[0,0,1]\n"
        "\tv_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[90:97], "
        "v[17:24] op_sel:[0,0,1]\n"
        "\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], "
        "v[1:8] op_sel:[0,0,1]\n",
        "\tv_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[118:125], "
        "v[33:40] op_sel:[0,0,1]\n"
        "\tv_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[118:125], "
        "v[17:24] op_sel:[0,0,1]\n"
        "\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[118:125], "
        "v[1:8] op_sel:[0,0,1]\n",
        "final B lookahead consumers",
    )
    if base != 118:
        replacements = (
            ("v[118:125]", f"v[{base}:{base + 7}]", 8),
            ("v[122:125]", f"v[{base + 4}:{base + 7}]", 2),
            ("v[118:121]", f"v[{base}:{base + 3}]", 2),
        )
        for old_registers, new_registers, expected in replacements:
            count = source.count(old_registers)
            if count != expected:
                raise ValueError(
                    f"{old_registers}: expected {expected} uses, found {count}"
                )
            source = source.replace(old_registers, new_registers)
    source = replace_once(
        source,
        "\t\t.amdhsa_next_free_vgpr 118\n",
        f"\t\t.amdhsa_next_free_vgpr {base + 8}\n",
        "HSA next-free VGPR",
    )
    source = replace_once(
        source,
        "    .vgpr_count:     118\n",
        f"    .vgpr_count:     {base + 8}\n",
        "metadata VGPR count",
    )
    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
