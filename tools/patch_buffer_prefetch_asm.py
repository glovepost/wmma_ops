#!/usr/bin/env python3
"""Patch the pinned gfx1151 p8 assembly to use scalar-base buffer loads."""

from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    modes = set(sys.argv[3:])
    valid_modes = {
        "--scalar-offset",
        "--no-clause",
        "--late-recurrence",
        "--early-b-4",
        "--early-b-8",
    }
    if len(sys.argv) < 3 or len(modes) != len(sys.argv[3:]) or not modes <= valid_modes:
        raise SystemExit(
            "usage: patch_buffer_prefetch_asm.py INPUT.s OUTPUT.s "
            "[--scalar-offset] [--no-clause] [--late-recurrence] "
            "[--early-b-4 | --early-b-8]"
        )
    if {"--early-b-4", "--early-b-8"} <= modes:
        raise SystemExit("choose only one early-B placement")

    source = Path(sys.argv[1]).read_text()

    source = replace_once(
        source,
        "\tv_mov_b32_e32 v57, 0\n"
        "\tv_add_co_u32 v69, s0, s0, v70\n",
        "\tv_mov_b32_e32 v57, 0\n"
        "\ts_mov_b64 s[8:9], s[0:1]\n"
        "\ts_mov_b64 s[10:11], s[2:3]\n"
        "\tv_add_co_u32 v69, s0, s0, v70\n",
        "descriptor initialization",
    )

    source = replace_once(
        source,
        "\tv_add_co_ci_u32_e64 v72, null, s3, 0, s0\n"
        "\tv_dual_mov_b32 v58, v57 :: v_dual_add_nc_u32 v73, 0x3000, v73\n",
        "\tv_add_co_ci_u32_e64 v72, null, s3, 0, s0\n"
        "\ts_mov_b64 s[0:1], s[8:9]\n"
        "\ts_mov_b32 s2, -1\n"
        "\ts_mov_b32 s3, 0x31004000\n"
        "\ts_mov_b64 s[8:9], s[10:11]\n"
        "\ts_mov_b32 s10, -1\n"
        "\ts_mov_b32 s11, 0x31004000\n"
        "\tv_lshlrev_b32_e32 v69, 5, v0\n"
        "\tv_add_nc_u32_e32 v69, 0x2000, v69\n"
        "\tv_lshlrev_b32_e32 v71, 4, v0\n"
        "\tv_add_nc_u32_e32 v71, 0x1000, v71\n"
        "\tv_dual_mov_b32 v58, v57 :: v_dual_add_nc_u32 v73, 0x3000, v73\n",
        "descriptor finalization",
    )

    source = replace_once(
        source,
        "\ts_max_i32 s0, s6, 2\n"
        "\ts_movk_i32 s2, 0x1000\n"
        "\ts_add_i32 s6, s0, -1\n"
        "\ts_movk_i32 s0, 0x800\n"
        "\ts_mov_b32 s3, 0\n",
        "\ts_max_i32 s19, s6, 2\n"
        "\ts_add_i32 s6, s19, -1\n",
        "flat-offset initialization",
    )

    source = replace_once(
        source,
        "\ts_lshl_b64 s[8:9], s[2:3], 1\n"
        "\ts_mov_b32 s1, s3\n"
        "\tv_add_co_u32 v114, vcc_lo, v69, s8\n"
        "\ts_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)\n"
        "\tv_add_co_ci_u32_e64 v115, null, s9, v70, vcc_lo\n"
        "\ts_lshl_b64 s[8:9], s[0:1], 1\n"
        "\ts_add_i32 s6, s6, -1\n"
        "\tv_add_co_u32 v116, vcc_lo, v71, s8\n"
        "\tv_add_co_ci_u32_e64 v117, null, s9, v72, vcc_lo\n"
        "\ts_addk_i32 s2, 0x1000\n"
        "\ts_addk_i32 s0, 0x800\n"
        "\ts_cmp_eq_u32 s6, 0\n",
        "\ts_add_i32 s6, s6, -1\n"
        "\ts_cmp_eq_u32 s6, 0\n",
        "flat-address recurrence",
    )

    source = replace_once(
        source,
        "\ts_clause 0x1\n"
        "\tglobal_load_b128 v[74:77], v[114:115], off\n"
        "\tglobal_load_b128 v[78:81], v[114:115], off offset:16\n"
        "\tglobal_load_b128 v[114:117], v[116:117], off\n",
        "\ts_clause 0x1\n"
        "\tbuffer_load_b128 v[74:77], v69, s[0:3], 0 offen\n"
        "\tbuffer_load_b128 v[78:81], v69, s[0:3], 0 offen offset:16\n"
        "\tbuffer_load_b128 v[114:117], v71, s[8:11], 0 offen\n"
        "\tv_add_nc_u32_e32 v69, 0x2000, v69\n"
        "\tv_add_nc_u32_e32 v71, 0x1000, v71\n",
        "hot-loop buffer loads",
    )

    if "--scalar-offset" in modes:
        source = replace_once(
            source,
            "\ts_add_i32 s6, s6, -1\n"
            "\ts_cmp_eq_u32 s6, 0\n",
            "\ts_add_i32 s6, s6, -1\n",
            "scalar-offset loop compare",
        )
        source = replace_once(
            source,
            "\ts_mov_b32 s11, 0x31004000\n"
            "\tv_lshlrev_b32_e32 v69, 5, v0\n"
            "\tv_add_nc_u32_e32 v69, 0x2000, v69\n"
            "\tv_lshlrev_b32_e32 v71, 4, v0\n"
            "\tv_add_nc_u32_e32 v71, 0x1000, v71\n",
            "\ts_mov_b32 s11, 0x31004000\n"
            "\ts_movk_i32 s7, 0x2000\n"
            "\ts_movk_i32 s18, 0x1000\n"
            "\tv_lshlrev_b32_e32 v69, 5, v0\n"
            "\tv_lshlrev_b32_e32 v71, 4, v0\n",
            "scalar-offset initialization",
        )
        source = replace_once(
            source,
            "\tbuffer_load_b128 v[74:77], v69, s[0:3], 0 offen\n"
            "\tbuffer_load_b128 v[78:81], v69, s[0:3], 0 offen offset:16\n"
            "\tbuffer_load_b128 v[114:117], v71, s[8:11], 0 offen\n"
            "\tv_add_nc_u32_e32 v69, 0x2000, v69\n"
            "\tv_add_nc_u32_e32 v71, 0x1000, v71\n",
            "\tbuffer_load_b128 v[74:77], v69, s[0:3], s7 offen\n"
            "\tbuffer_load_b128 v[78:81], v69, s[0:3], s7 offen offset:16\n"
            "\tbuffer_load_b128 v[114:117], v71, s[8:11], s18 offen\n"
            "\ts_addk_i32 s7, 0x2000\n"
            "\ts_addk_i32 s18, 0x1000\n"
            "\ts_cmp_eq_u32 s6, 0\n",
            "scalar-offset hot loop",
        )

    if "--no-clause" in modes:
        source = replace_once(
            source,
            "\ts_clause 0x1\n"
            "\tbuffer_load_b128 v[74:77], v69, s[0:3],",
            "\tbuffer_load_b128 v[74:77], v69, s[0:3],",
            "hot-loop clause",
        )

    if "--late-recurrence" in modes:
        if "--scalar-offset" not in modes:
            raise SystemExit("--late-recurrence requires --scalar-offset")
        source = replace_once(
            source,
            "\ts_addk_i32 s7, 0x2000\n"
            "\ts_addk_i32 s18, 0x1000\n"
            "\ts_cmp_eq_u32 s6, 0\n"
            "\tv_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[90:97], "
            "v[33:40] op_sel:[0,0,1]\n"
            "\tv_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[90:97], "
            "v[17:24] op_sel:[0,0,1]\n"
            "\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], "
            "v[1:8] op_sel:[0,0,1]\n",
            "\tv_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[90:97], "
            "v[33:40] op_sel:[0,0,1]\n"
            "\tv_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[90:97], "
            "v[17:24] op_sel:[0,0,1]\n"
            "\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], "
            "v[1:8] op_sel:[0,0,1]\n"
            "\ts_addk_i32 s7, 0x2000\n"
            "\ts_addk_i32 s18, 0x1000\n"
            "\ts_cmp_eq_u32 s6, 0\n",
            "scalar recurrence placement",
        )

    early_b = modes & {"--early-b-4", "--early-b-8"}
    if early_b:
        if "--scalar-offset" not in modes:
            raise SystemExit("early-B placement requires --scalar-offset")
        source = replace_once(
            source,
            "\tbuffer_load_b128 v[114:117], v71, s[8:11], s18 offen\n",
            "",
            "remove original B refill",
        )
        if "--early-b-4" in modes:
            anchor = (
                "\tv_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], "
                "v[9:16] op_sel:[0,0,1]\n"
                "\tds_load_b128 v[94:97], v65 offset:14608\n"
            )
            replacement = (
                "\tv_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], "
                "v[9:16] op_sel:[0,0,1]\n"
                "\tbuffer_load_b128 v[114:117], v71, s[8:11], s18 offen\n"
                "\tds_load_b128 v[94:97], v65 offset:14608\n"
            )
        else:
            anchor = (
                "\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], "
                "v[1:8]\n"
                "\tds_load_b128 v[94:97], v65 offset:13840\n"
            )
            replacement = (
                "\tv_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], "
                "v[1:8]\n"
                "\tbuffer_load_b128 v[114:117], v71, s[8:11], s18 offen\n"
                "\tds_load_b128 v[94:97], v65 offset:13840\n"
            )
        source = replace_once(source, anchor, replacement, "early B refill")

    Path(sys.argv[2]).write_text(source)


if __name__ == "__main__":
    main()
