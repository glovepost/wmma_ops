#!/usr/bin/env python3
"""Move p8 next-tile VMEM loads ahead of the full WMMA cluster.

The ROCm 7.14 allocator aliases the three b128 prefetch destinations with dead
fragment/address registers.  That is register-efficient, but it leaves only
three WMMAs between the loads and vmcnt(0).  This patch dedicates v118:v129 to
the prefetched A0/A1/B vectors and launches them as soon as their addresses are
ready, before the 16-WMMA cluster.  The resulting 130-VGPR kernel remains in
the retained geometry's two-block LDS occupancy class.
"""

from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_early_prefetch_asm.py INPUT.s OUTPUT.s")

    source = Path(sys.argv[1]).read_text()
    source = replace_once(
        source,
        "\ts_cmp_eq_u32 s6, 0\n"
        "\ts_waitcnt lgkmcnt(4)\n",
        "\ts_cmp_eq_u32 s6, 0\n"
        "\tglobal_load_b128 v[118:121], v[114:115], off\n"
        "\tglobal_load_b128 v[122:125], v[114:115], off offset:16\n"
        "\tglobal_load_b128 v[126:129], v[116:117], off\n"
        "\ts_waitcnt lgkmcnt(4)\n",
        "early prefetch insertion",
    )
    source = replace_once(
        source,
        "\ts_clause 0x1\n"
        "\tglobal_load_b128 v[74:77], v[114:115], off\n"
        "\tglobal_load_b128 v[78:81], v[114:115], off offset:16\n"
        "\tglobal_load_b128 v[114:117], v[116:117], off\n",
        "",
        "late prefetch removal",
    )
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
    source = replace_once(
        source,
        "\t\t.amdhsa_next_free_vgpr 118\n",
        "\t\t.amdhsa_next_free_vgpr 130\n",
        "HSA next-free VGPR",
    )
    source = replace_once(
        source,
        "; NumVgprs: 118\n",
        "; NumVgprs: 130\n",
        "resource comment",
    )
    source = replace_once(
        source,
        "    .vgpr_count:     118\n",
        "    .vgpr_count:     130\n",
        "metadata VGPR count",
    )
    Path(sys.argv[2]).write_text(source)


if __name__ == "__main__":
    main()
