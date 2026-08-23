#!/usr/bin/env python3
"""Move patched MUBUF prefetches before the full WMMA cluster."""

from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: patch_early_buffer_prefetch_asm.py INPUT.s OUTPUT.s"
        )
    source = Path(sys.argv[1]).read_text()
    scalar_offset = "s18 offen" in source

    if scalar_offset:
        early = (
            "\tbuffer_load_b128 v[118:121], v69, s[0:3], s7 offen\n"
            "\tbuffer_load_b128 v[122:125], v69, s[0:3], s7 offen offset:16\n"
            "\tbuffer_load_b128 v[126:129], v71, s[8:11], s18 offen\n"
            "\ts_addk_i32 s7, 0x2000\n"
            "\ts_addk_i32 s18, 0x1000\n"
            "\ts_cmp_eq_u32 s6, 0\n"
        )
        late = (
            "\ts_clause 0x1\n"
            "\tbuffer_load_b128 v[74:77], v69, s[0:3], s7 offen\n"
            "\tbuffer_load_b128 v[78:81], v69, s[0:3], s7 offen offset:16\n"
            "\tbuffer_load_b128 v[114:117], v71, s[8:11], s18 offen\n"
            "\ts_addk_i32 s7, 0x2000\n"
            "\ts_addk_i32 s18, 0x1000\n"
            "\ts_cmp_eq_u32 s6, 0\n"
        )
        source = replace_once(
            source,
            "\ts_add_i32 s6, s6, -1\n\ts_waitcnt lgkmcnt(4)\n",
            "\ts_add_i32 s6, s6, -1\n" + early
            + "\ts_waitcnt lgkmcnt(4)\n",
            "scalar-offset early insertion",
        )
    else:
        early = (
            "\tbuffer_load_b128 v[118:121], v69, s[0:3], 0 offen\n"
            "\tbuffer_load_b128 v[122:125], v69, s[0:3], 0 offen offset:16\n"
            "\tbuffer_load_b128 v[126:129], v71, s[8:11], 0 offen\n"
            "\tv_add_nc_u32_e32 v69, 0x2000, v69\n"
            "\tv_add_nc_u32_e32 v71, 0x1000, v71\n"
        )
        late = (
            "\ts_clause 0x1\n"
            "\tbuffer_load_b128 v[74:77], v69, s[0:3], 0 offen\n"
            "\tbuffer_load_b128 v[78:81], v69, s[0:3], 0 offen offset:16\n"
            "\tbuffer_load_b128 v[114:117], v71, s[8:11], 0 offen\n"
            "\tv_add_nc_u32_e32 v69, 0x2000, v69\n"
            "\tv_add_nc_u32_e32 v71, 0x1000, v71\n"
        )
        source = replace_once(
            source,
            "\ts_cmp_eq_u32 s6, 0\n\ts_waitcnt lgkmcnt(4)\n",
            "\ts_cmp_eq_u32 s6, 0\n" + early
            + "\ts_waitcnt lgkmcnt(4)\n",
            "vector-offset early insertion",
        )

    source = replace_once(source, late, "", "late buffer prefetch removal")
    source = replace_once(
        source,
        "\tds_store_b128 v68, v[74:77]\n"
        "\tds_store_b128 v68, v[78:81] offset:16\n"
        "\tds_store_b128 v73, v[114:117]\n",
        "\tds_store_b128 v68, v[118:121]\n"
        "\tds_store_b128 v68, v[122:125] offset:16\n"
        "\tds_store_b128 v73, v[126:129]\n",
        "prefetch commit registers",
    )
    for old, new, label in (
        ("\t\t.amdhsa_next_free_vgpr 118\n",
         "\t\t.amdhsa_next_free_vgpr 130\n", "HSA next-free VGPR"),
        ("; NumVgprs: 118\n", "; NumVgprs: 130\n", "resource comment"),
        ("    .vgpr_count:     118\n",
         "    .vgpr_count:     130\n", "metadata VGPR count"),
    ):
        source = replace_once(source, old, new, label)
    Path(sys.argv[2]).write_text(source)


if __name__ == "__main__":
    main()
