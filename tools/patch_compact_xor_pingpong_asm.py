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


def remap_control(text: str) -> str:
    register_map = {
        **{105 + i: 145 + i for i in range(4)},
        **{181 + i: 149 + i for i in range(4)},
    }

    def remap_range(match: re.Match[str]) -> str:
        low = int(match.group(1))
        high = int(match.group(2))
        mapped = [register_map.get(value, value)
                  for value in range(low, high + 1)]
        if mapped != list(range(mapped[0], mapped[-1] + 1)):
            raise SystemExit(f"non-contiguous control range v[{low}:{high}]")
        return f"v[{mapped[0]}:{mapped[-1]}]"

    text = re.sub(r"v\[([0-9]+):([0-9]+)\]", remap_range, text)
    return re.sub(
        r"\bv([0-9]+)\b",
        lambda match: f"v{register_map.get(int(match.group(1)), int(match.group(1)))}",
        text,
    )


def phase_loads(a_first: int, a_second: int,
                b_first: int, b_second: int) -> list[str]:
    lines: list[str] = []
    lines += load_pair(73, a_first, a_second, 0)
    lines += load_pair(81, a_first, a_second, 512)
    lines += load_pair(89, a_first, a_second, 1024)
    lines += load_pair(97, a_first, a_second, 1536)
    lines += load_pair(105, b_first, b_second, 0)
    lines += load_pair(113, b_first, b_second, 512)
    lines += load_pair(121, b_first, b_second, 1024)
    lines += load_pair(129, b_first, b_second, 1536)
    return lines


def phase_compute(a_address: int, b_address: int) -> list[str]:
    lines: list[str] = ["\ts_waitcnt lgkmcnt(0)"]
    lines += wmma_group(105, (57, 41, 25, 9), False)
    # B0 is dead after this group.  Stage both next-tile A halves in its eight
    # VGPRs while the remaining twelve WMMAs execute.
    lines += [
        "\ts_clause 0x1",
        f"\tglobal_load_b128 v[105:108], v[{a_address}:{a_address + 1}], off",
        f"\tglobal_load_b128 v[109:112], v[{a_address}:{a_address + 1}], off offset:16",
    ]
    lines += wmma_group(113, (49, 33, 17, 1), False)
    # B1 is likewise dead; its low half is enough for the cooperative B
    # refill.  No subsequent WMMA reads either recycled bank.
    lines += [
        f"\tglobal_load_b128 v[113:116], v[{b_address}:{b_address + 1}], off",
    ]
    lines += wmma_group(121, (57, 41, 25, 9), True)
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

    first_load = body.index("\tds_load_b128 v[105:108], v73\n")
    first_commit = body.index(
        "\t;;#ASMSTART\n\ts_waitcnt vmcnt(0)\n", first_load)
    following_load_end = body.index(
        "\tglobal_load_b128 v[177:180], v[183:184], off\n",
        first_commit,
    ) + len("\tglobal_load_b128 v[177:180], v[183:184], off\n")
    second_load = body.index("\tds_load_b128 v[105:108], v89\n")
    second_commit = body.index(
        "\t;;#ASMSTART\n\ts_waitcnt vmcnt(0)\n", second_load)
    second_compute = body.index(
        "\tv_wmma_f16_16x16x16_f16", second_commit)
    footer = body.index(
        "\t;;#ASMSTART\n\ts_waitcnt lgkmcnt(0)\n", second_compute)

    control0 = remap_control(body[:first_load])
    publish0 = remap_control(body[first_commit:following_load_end])
    publish1 = remap_control(body[second_commit:second_compute])
    initial_prefetch = (
        "\ts_clause 0x1\n"
        "\tglobal_load_b128 v[169:172], v[105:106], off\n"
        "\tglobal_load_b128 v[173:176], v[105:106], off offset:16\n"
        "\tglobal_load_b128 v[177:180], v[107:108], off\n"
    )
    following_prefetch = (
        "\ts_clause 0x1\n"
        "\tglobal_load_b128 v[169:172], v[181:182], off\n"
        "\tglobal_load_b128 v[173:176], v[181:182], off offset:16\n"
        "\tglobal_load_b128 v[177:180], v[183:184], off\n"
    )
    control0 = replace_once(
        control0, remap_control(initial_prefetch), "", "initial prefetch")
    publish0 = replace_once(
        publish0, remap_control(following_prefetch), "", "following prefetch")
    for old, new in (
        ("v[169:172]", "v[105:108]"),
        ("v[173:176]", "v[109:112]"),
        ("v[177:180]", "v[113:116]"),
    ):
        publish0 = publish0.replace(old, new)
        publish1 = publish1.replace(old, new)
    publish1 = replace_once(
        publish1,
        "\ts_waitcnt lgkmcnt(11)\n",
        "",
        "compiler phase-1 readiness wait",
    )
    loop_footer = body[footer:]

    new_body = "\n".join(
        control0.rstrip("\n").splitlines()
        + phase_loads(137, 138, 139, 140)
        + phase_compute(145, 147)
        + publish0.rstrip("\n").splitlines()
        + phase_loads(141, 142, 143, 144)
        + phase_compute(149, 151)
        + publish1.rstrip("\n").splitlines()
        + loop_footer.rstrip("\n").splitlines()
    ) + "\n"

    prefix = replace_once(
        prefix,
        "\ts_mov_b32 s3, 0\n",
        "\ts_mov_b32 s3, 0\n"
        "\tv_mov_b32_e32 v137, v73\n"
        "\tv_mov_b32_e32 v138, v74\n"
        "\tv_mov_b32_e32 v139, v81\n"
        "\tv_mov_b32_e32 v140, v82\n"
        "\tv_mov_b32_e32 v141, v89\n"
        "\tv_mov_b32_e32 v142, v90\n"
        "\tv_mov_b32_e32 v143, v97\n"
        "\tv_mov_b32_e32 v144, v98\n",
        "canonical XOR bases",
    )

    source = prefix + new_body + suffix
    source = replace_once(
        source,
        "\t\t.amdhsa_next_free_vgpr 185\n",
        "\t\t.amdhsa_next_free_vgpr 153\n",
        "VGPR declaration",
    )
    source = replace_once(
        source,
        "    .vgpr_count:     185\n",
        "    .vgpr_count:     153\n",
        "VGPR metadata",
    )

    registers = [int(value) for value in re.findall(r"\bv([0-9]+)\b", source)]
    registers.extend(
        int(value)
        for pair in re.findall(r"v\[([0-9]+):([0-9]+)\]", source)
        for value in pair
    )
    if max(registers) > 152:
        raise SystemExit(f"compressed image still addresses v{max(registers)}")
    if new_body.count("\tds_load_b128") != 32:
        raise SystemExit("hot pair must contain 32 b128 LDS reads")
    if new_body.count("\tv_wmma_f16_16x16x16_f16") != 32:
        raise SystemExit("hot pair must contain 32 WMMAs")
    if new_body.count("\ts_barrier") != 2:
        raise SystemExit("hot pair must contain one barrier per K16")
    if new_body.count("\tglobal_load_b128") != 6:
        raise SystemExit("hot pair must contain three refills per K16")
    if new_body.count("\tds_store_b128") != 6:
        raise SystemExit("hot pair must publish three vectors per K16")

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
