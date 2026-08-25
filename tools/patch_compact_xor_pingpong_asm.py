#!/usr/bin/env python3
"""Compress the fixed gfx1151 compact-XOR ping-pong hot loop.

Four B fragments remain independent: attempts to reload a B bank and consume
it again in the same phase were not deterministic.  Once a fragment has made
its final WMMA contribution, its dead VGPRs instead stage the next global A/B
vectors.  That preserves exactness while eliminating dedicated refill banks.
"""

from pathlib import Path
import re
import sys


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def with_offset(register: int, offset: int) -> str:
    return f"v{register}" + (f" offset:{offset}" if offset else "")


def load_pair(destination: int, first_base: int, second_base: int,
              offset: int) -> list[str]:
    return [
        f"\tds_load_b128 v[{destination}:{destination + 3}], "
        f"{with_offset(first_base, offset)}",
        f"\tds_load_b128 v[{destination + 4}:{destination + 7}], "
        f"{with_offset(second_base, offset)}",
    ]


def wmma_group(b_source: int, outputs: tuple[int, ...],
               op_sel: bool = False) -> list[str]:
    a_sources = (73, 81, 89, 97)
    suffix = " op_sel:[0,0,1]" if op_sel else ""
    return [
        f"\tv_wmma_f16_16x16x16_f16 v[{output}:{output + 7}], "
        f"v[{a_source}:{a_source + 7}], "
        f"v[{b_source}:{b_source + 7}], "
        f"v[{output}:{output + 7}]{suffix}"
        for a_source, output in zip(a_sources, outputs)
    ]


def phase_loads(a_first: int, a_second: int,
                b_first: int, b_second: int) -> list[str]:
    lines: list[str] = []
    lines += load_pair(73, a_first, a_second, 0)
    lines += load_pair(81, a_first, a_second, 512)
    # Match the selected kernel's readiness order: B0 follows A0/A1 so the
    # first two WMMAs can issue with four LDS operations still outstanding.
    lines += load_pair(105, b_first, b_second, 0)
    lines += load_pair(89, a_first, a_second, 1024)
    lines += load_pair(97, a_first, a_second, 1536)
    return lines


def phase_compute(b_first: int, b_second: int,
                  a_soffset: int, b_soffset: int) -> list[str]:
    first_group = wmma_group(105, (57, 41, 25, 9), False)
    lines: list[str] = ["\ts_waitcnt lgkmcnt(4)"]
    lines += first_group[:2]
    lines.append("\ts_waitcnt lgkmcnt(2)")
    lines.append(first_group[2])
    lines.append("\ts_waitcnt lgkmcnt(0)")
    lines.append(first_group[3])
    # B0 is dead after this group.  Stage both next-tile A halves in its eight
    # VGPRs while the remaining twelve WMMAs execute.
    lines += [
        "\ts_clause 0x1",
        f"\tbuffer_load_b128 v[105:108], v66, s[0:3], s{a_soffset} offen",
        f"\tbuffer_load_b128 v[109:112], v70, s[0:3], s{a_soffset} offen",
    ]
    lines += load_pair(113, b_first, b_second, 512)
    lines.append("\ts_waitcnt lgkmcnt(0)")
    lines += wmma_group(113, (49, 33, 17, 1), False)
    # B1 is likewise dead; its low half is enough for the cooperative B
    # refill.  No subsequent WMMA reads either recycled bank.
    lines += [
        f"\tbuffer_load_b128 v[113:116], v67, s[4:7], s{b_soffset} offen",
    ]
    lines += load_pair(121, b_first, b_second, 1024)
    lines.append("\ts_waitcnt lgkmcnt(0)")
    lines += wmma_group(121, (57, 41, 25, 9), True)
    lines += load_pair(129, b_first, b_second, 1536)
    lines.append("\ts_waitcnt lgkmcnt(0)")
    lines += wmma_group(129, (49, 33, 17, 1), True)
    return lines


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).read_text()
    loop_start = source.index(".LBB0_13:")
    loop_end = source.index(".LBB0_15:", loop_start)
    prefix = source[:loop_start]
    body = source[loop_start:loop_end]
    suffix = source[loop_end:]

    # These anchors deliberately describe the compiler's valid raw-buffer
    # resource layout.  In particular, the descriptors in s[0:3]/s[4:7] come
    # from __builtin_amdgcn_make_buffer_rsrc; reconstructing them from flat
    # pointers loses the AMDGPU aperture bits and faults on gfx1151.
    for anchor in (
        "\tbuffer_load_b128 v[168:171], v66, s[0:3], s11 offen\n",
        "\tbuffer_load_b128 v[176:179], v67, s[4:7], s14 offen\n",
        "\tds_load_b128 v[104:107], v72\n",
        "\tds_load_b128 v[104:107], v88\n",
        "\ts_cbranch_scc0 .LBB0_13\n",
    ):
        if body.count(anchor) != 1:
            raise SystemExit(f"source loop anchor {anchor.strip()!r} changed")

    control0 = (
        ".LBB0_13:\n"
        "\ts_cmp_lt_i32 s15, s10\n"
        "\ts_cselect_b32 s18, s15, 0\n"
        "\ts_lshl_b32 s19, s18, 13\n"
        "\ts_lshl_b32 s18, s18, 12\n"
    )
    publish0 = (
        "\ts_addk_i32 s14, 0x2000\n"
        "\ts_addk_i32 s11, 0x4000\n"
        "\ts_waitcnt vmcnt(2)\n"
        "\tds_store_b128 v68, v[105:108] offset:8192\n"
        "\ts_waitcnt vmcnt(1)\n"
        "\tds_store_b128 v69, v[109:112] offset:8192\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\tds_store_b128 v71, v[113:116] offset:4096\n"
        "\ts_waitcnt lgkmcnt(0)\n"
        "\ts_barrier\n"
    )
    publish1 = (
        "\ts_add_i32 s18, s15, 2\n"
        "\ts_cmp_ge_i32 s15, s10\n"
        "\ts_mov_b32 s15, s18\n"
        "\ts_waitcnt vmcnt(2)\n"
        "\tds_store_b128 v68, v[105:108]\n"
        "\ts_waitcnt vmcnt(1)\n"
        "\tds_store_b128 v69, v[109:112]\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\tds_store_b128 v71, v[113:116]\n"
        "\ts_waitcnt lgkmcnt(0)\n"
        "\ts_barrier\n"
    )
    loop_footer = (
        "\ts_cbranch_scc0 .LBB0_13\n"
        "; %bb.14:\n"
        "\tv_mov_b32_e32 v70, v65\n"
    )

    new_body = "\n".join(
        control0.rstrip("\n").splitlines()
        + phase_loads(137, 138, 139, 140)
        + phase_compute(139, 140, 11, 14)
        + publish0.rstrip("\n").splitlines()
        + phase_loads(141, 142, 143, 144)
        + phase_compute(143, 144, 19, 18)
        + publish1.rstrip("\n").splitlines()
        + loop_footer.rstrip("\n").splitlines()
    ) + "\n"

    prefix = replace_once(
        prefix,
        "\ts_movk_i32 s11, 0x2000\n"
        "\ts_mov_b32 s15, 2\n"
        "\ts_movk_i32 s14, 0x1000\n"
        "\ts_mov_b32 s7, s3\n",
        "\ts_movk_i32 s11, 0x2000\n"
        "\ts_mov_b32 s15, 2\n"
        "\ts_movk_i32 s14, 0x1000\n"
        "\ts_mov_b32 s7, s3\n"
        "\tv_mov_b32_e32 v137, v72\n"
        "\tv_mov_b32_e32 v138, v73\n"
        "\tv_mov_b32_e32 v139, v80\n"
        "\tv_mov_b32_e32 v140, v81\n"
        "\tv_mov_b32_e32 v141, v88\n"
        "\tv_mov_b32_e32 v142, v89\n"
        "\tv_mov_b32_e32 v143, v96\n"
        "\tv_mov_b32_e32 v144, v97\n",
        "canonical XOR bases",
    )

    source = prefix + new_body + suffix
    source = replace_once(
        source,
        "\t\t.amdhsa_next_free_vgpr 180\n",
        "\t\t.amdhsa_next_free_vgpr 145\n",
        "VGPR declaration",
    )
    source = replace_once(
        source,
        "    .vgpr_count:     180\n",
        "    .vgpr_count:     145\n",
        "VGPR metadata",
    )

    registers = [int(value) for value in re.findall(r"\bv([0-9]+)\b", source)]
    registers.extend(
        int(value)
        for pair in re.findall(r"v\[([0-9]+):([0-9]+)\]", source)
        for value in pair
    )
    if max(registers) > 144:
        raise SystemExit(f"compressed image still addresses v{max(registers)}")
    if new_body.count("\tds_load_b128") != 32:
        raise SystemExit("hot pair must contain 32 b128 LDS reads")
    if new_body.count("\tv_wmma_f16_16x16x16_f16") != 32:
        raise SystemExit("hot pair must contain 32 WMMAs")
    if new_body.count("\ts_barrier") != 2:
        raise SystemExit("hot pair must contain one barrier per K16")
    if new_body.count("\tbuffer_load_b128") != 6:
        raise SystemExit("hot pair must contain three refills per K16")
    if new_body.count("\tds_store_b128") != 6:
        raise SystemExit("hot pair must publish three vectors per K16")

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
