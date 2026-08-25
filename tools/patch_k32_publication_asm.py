#!/usr/bin/env python3
"""Compress the fixed gfx1151 K32 publication image to the 120-VGPR class."""

from pathlib import Path
import re
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def with_offset(base: int, offset: int) -> str:
    return f"v{base}" + (f" offset:{offset}" if offset else "")


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.s OUTPUT.s", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).read_text()

    # Initial K32 publication: retain one A-row and one B-row base.
    for line in (
        "\tv_mad_u32_u24 v70, 0x50, v0, 16\n",
        "\tv_mad_u32_u24 v71, 0x50, v0, 32\n",
        "\tv_mad_u32_u24 v72, 0x50, v0, 48\n",
        "\tv_add_nc_u32_e32 v74, 0x5010, v28\n",
    ):
        source = replace_once(source, line, "", "initial LDS address removal")
    source = replace_once(
        source,
        "\tv_add_nc_u32_e32 v73, 0x5000, v28\n",
        "\tv_add_nc_u32_e32 v70, 0x5000, v28\n",
        "initial B LDS base",
    )
    initial_stores = {
        "\tds_store_b128 v70, v[8:11]\n":
            "\tds_store_b128 v69, v[8:11] offset:16\n",
        "\tds_store_b128 v71, v[12:15]\n":
            "\tds_store_b128 v69, v[12:15] offset:32\n",
        "\tds_store_b128 v72, v[16:19]\n":
            "\tds_store_b128 v69, v[16:19] offset:48\n",
        "\tds_store_b128 v73, v[20:23]\n":
            "\tds_store_b128 v70, v[20:23]\n",
        "\tds_store_b128 v74, v[24:27]\n":
            "\tds_store_b128 v70, v[24:27] offset:16\n",
    }
    for old, new in initial_stores.items():
        source = replace_once(source, old, new, "initial LDS store base")

    # Split the one steady-state body from the source-specialized final tile.
    loop_start = source.index(".LBB0_13:")
    loop_end = source.index(".LBB0_15:", loop_start)
    prefix = source[:loop_start]
    body = source[loop_start:loop_end]
    suffix = source[loop_end:]

    # All fragment rows are multiples of the 80-byte physical pitch.  v65 is
    # the A row base and v66 becomes B's row base.  Immediate offsets cover
    # both K16 slices and all four M/N fragments.
    address_map = {
        65: (65, 0), 79: (65, 16), 80: (65, 1280), 81: (65, 1296),
        85: (65, 2560), 86: (65, 2576), 82: (65, 3840), 87: (65, 3856),
        94: (65, 32), 95: (65, 48), 96: (65, 1312), 97: (65, 1328),
        100: (65, 2592), 101: (65, 2608), 102: (65, 3872), 103: (65, 3888),
        83: (66, 0), 84: (66, 16), 88: (66, 1280), 89: (66, 1296),
        90: (66, 2560), 91: (66, 2576), 92: (66, 3840), 93: (66, 3856),
        98: (66, 32), 99: (66, 48), 104: (66, 1312), 105: (66, 1328),
        106: (66, 2592), 107: (66, 2608), 108: (66, 3872), 109: (66, 3888),
    }

    load_pattern = re.compile(
        r"(?m)^(\tds_load_b128 v\[[0-9]+:[0-9]+\]), v([0-9]+)$"
    )
    seen_addresses: list[int] = []

    def rewrite_load(match: re.Match[str]) -> str:
        address = int(match.group(2))
        if address not in address_map:
            return match.group(0)
        seen_addresses.append(address)
        base, offset = address_map[address]
        return f"{match.group(1)}, {with_offset(base, offset)}"

    body = load_pattern.sub(rewrite_load, body)
    if len(seen_addresses) != 32 or set(seen_addresses) != set(address_map):
        raise SystemExit(
            f"expected 32 K32 fragment loads and every address base, got "
            f"{len(seen_addresses)} loads/{len(set(seen_addresses))} bases"
        )

    hot_stores = {
        "\tds_store_b128 v70, v[114:117]\n":
            "\tds_store_b128 v69, v[114:117] offset:16\n",
        "\tds_store_b128 v71, v[118:121]\n":
            "\tds_store_b128 v69, v[118:121] offset:32\n",
        "\tds_store_b128 v72, v[122:125]\n":
            "\tds_store_b128 v69, v[122:125] offset:48\n",
        "\tds_store_b128 v73, v[134:137]\n":
            "\tds_store_b128 v70, v[134:137]\n",
        "\tds_store_b128 v74, v[138:141]\n":
            "\tds_store_b128 v70, v[138:141] offset:16\n",
    }
    for old, new in hot_stores.items():
        body = replace_once(body, old, new, "steady-state LDS store base")

    # Compress the five live fragment banks and the global-address pair.  The
    # next-tile loads already reuse A0/A1/A2 after their final WMMAs.
    register_map = {}
    for old_start, new_start in (
        (110, 80), (118, 88), (126, 96), (134, 104), (142, 112)
    ):
        register_map.update(
            {old_start + i: new_start + i for i in range(8)}
        )
    register_map.update({150: 71, 151: 72})

    def remap_range(match: re.Match[str]) -> str:
        low = register_map.get(int(match.group(1)), int(match.group(1)))
        high = register_map.get(int(match.group(2)), int(match.group(2)))
        if high - low != int(match.group(2)) - int(match.group(1)):
            raise SystemExit("VGPR range remap is not contiguous")
        return f"v[{low}:{high}]"

    body = re.sub(r"v\[([0-9]+):([0-9]+)\]", remap_range, body)
    body = re.sub(
        r"v([0-9]+)",
        lambda m: f"v{register_map.get(int(m.group(1)), int(m.group(1)))}",
        body,
    )

    # The source allocator temporarily used the second B-stage destination as
    # its own flat-address pair.  After compression, reuse v71:v72 for both
    # A and B address calculations; the A requests have already issued before
    # this sequence and the pair is otherwise dead.
    body = replace_once(
        body,
        "\tv_add_co_u32 v108, vcc_lo, v77, s8\n"
        "\tv_add_co_ci_u32_e64 v109, null, s9, v78, vcc_lo\n"
        "\ts_clause 0x1\n"
        "\tglobal_load_b128 v[104:107], v[108:109], off\n"
        "\tglobal_load_b128 v[108:111], v[108:109], off offset:16\n",
        "\tv_add_co_u32 v71, vcc_lo, v77, s8\n"
        "\tv_add_co_ci_u32_e64 v72, null, s9, v78, vcc_lo\n"
        "\ts_clause 0x1\n"
        "\tglobal_load_b128 v[104:107], v[71:72], off\n"
        "\tglobal_load_b128 v[108:111], v[71:72], off offset:16\n",
        "compressed B global address",
    )

    # Inline LDS assembly makes the data registers explicit but leaves LLVM
    # unable to place asynchronous-memory waits.  Restore the selected p8
    # readiness schedule in each slice: B0 is ready at lgkmcnt(4), A2 at 2,
    # A3 at 0, and each later two-load B fragment retires at 0 before use.
    lines = body.splitlines()
    remove: set[int] = set()
    insert_before: dict[int, list[str]] = {}
    wait4_indices = [
        i for i, line in enumerate(lines) if line == "\ts_waitcnt lgkmcnt(4)"
    ]
    if len(wait4_indices) != 2:
        raise SystemExit("expected one initial progressive wait per K16 slice")
    for wait4 in wait4_indices:
        wmmas = [
            i for i in range(wait4 + 1, len(lines))
            if lines[i].startswith("\tv_wmma_f16_16x16x16_f16")
        ][:4]
        if len(wmmas) != 4:
            raise SystemExit("initial slice WMMA group is incomplete")
        insert_before.setdefault(wmmas[2], []).append("\ts_waitcnt lgkmcnt(2)")
        insert_before.setdefault(wmmas[3], []).append("\ts_waitcnt lgkmcnt(0)")
        cursor = wmmas[3] + 1
        removed_waits: list[int] = []
        while cursor < len(lines) and not lines[cursor].startswith("\tds_load_b128"):
            if lines[cursor] in {
                "\ts_waitcnt lgkmcnt(2)",
                "\ts_waitcnt lgkmcnt(0)",
            }:
                removed_waits.append(cursor)
            cursor += 1
        if [lines[i] for i in removed_waits] != [
            "\ts_waitcnt lgkmcnt(2)",
            "\ts_waitcnt lgkmcnt(0)",
        ]:
            raise SystemExit("failed to find compiler-displaced progressive waits")
        remove.update(removed_waits)
    lines = [
        inserted
        for i, line in enumerate(lines)
        for inserted in (insert_before.get(i, []) + ([] if i in remove else [line]))
    ]

    scheduled: list[str] = []
    pending_b = 0
    inserted_b_waits = 0
    for line in lines:
        if line.startswith("\tds_load_b128") and ", v66" in line:
            pending_b += 1
        elif line.startswith("\ts_waitcnt lgkmcnt"):
            pending_b = 0
        elif line.startswith("\tv_wmma_f16_16x16x16_f16") and pending_b:
            if pending_b != 2:
                raise SystemExit(
                    f"later B fragment has {pending_b} pending LDS operations"
                )
            scheduled.append("\ts_waitcnt lgkmcnt(0)")
            inserted_b_waits += 1
            pending_b = 0
        scheduled.append(line)
    if inserted_b_waits != 6:
        raise SystemExit(f"expected six just-in-time B waits, got {inserted_b_waits}")
    body = "\n".join(scheduled) + "\n"

    # The address builder is now dead.  Preserve only the accumulator-zeroing
    # half of dual instructions, and materialize one B LDS base before v1 is
    # repurposed as an accumulator.
    address_destinations = set(address_map) - {65}
    prefix_lines = prefix.splitlines(keepends=True)
    rewritten: list[str] = []
    b_base_written = False
    for line in prefix_lines:
        dual = re.fullmatch(
            r"\tv_dual_mov_b32 v([0-9]+), ([^ ]+) :: "
            r"v_dual_(?:add_nc_u32|lshlrev_b32) v([0-9]+),.*\n",
            line,
        )
        if dual and int(dual.group(3)) in address_destinations:
            rewritten.append(
                f"\tv_mov_b32_e32 v{dual.group(1)}, {dual.group(2)}\n"
            )
            if int(dual.group(3)) == 83:
                rewritten.append("\tv_add_nc_u32_e32 v66, 0x5000, v1\n")
                b_base_written = True
            continue
        add = re.fullmatch(
            r"\tv_add_nc_u32_e32 v([0-9]+),.*\n", line
        )
        if add and int(add.group(1)) in address_destinations:
            continue
        if line == "\tv_mul_u32_u24_e32 v66, 40, v79\n":
            continue
        rewritten.append(line)
    if not b_base_written:
        raise SystemExit("failed to materialize compact B LDS base")
    prefix = "".join(rewritten)

    # The source-specialized final tile shared two row-address registers with
    # values repurposed by the compressed hot loop.  Give the tail immutable A
    # and B bases in its otherwise-unused v111/v112, then express every load
    # with the same canonical offsets as the steady-state body.  This also
    # makes the no-hot-loop K=32 path independent of the prefetch setup.
    suffix = replace_once(
        suffix,
        ".LBB0_15:\n",
        ".LBB0_15:\n"
        "\tv_mov_b32_e32 v111, v65\n"
        "\tv_lshlrev_b32_e32 v112, 1, v67\n"
        "\tv_add_nc_u32_e32 v112, 0x5000, v112\n",
        "final-tile compact LDS bases",
    )
    tail_offsets = [
        (111, 0), (111, 16), (111, 1280), (111, 1296),
        (112, 0), (112, 16), (111, 2560), (111, 2576),
        (111, 3840), (111, 3856), (112, 1280), (112, 1296),
        (112, 2560), (112, 2576), (112, 3840), (112, 3856),
        (111, 32), (111, 48), (111, 1312), (111, 1328),
        (112, 32), (112, 48), (111, 2592), (111, 2608),
        (111, 3872), (111, 3888), (112, 1312), (112, 1328),
        (112, 2592), (112, 2608), (112, 3872), (112, 3888),
    ]
    tail_load_index = 0

    def rewrite_tail_load(match: re.Match[str]) -> str:
        nonlocal tail_load_index
        if tail_load_index >= len(tail_offsets):
            raise SystemExit("final tile has more than 32 fragment loads")
        base, offset = tail_offsets[tail_load_index]
        tail_load_index += 1
        return f"{match.group(1)}, {with_offset(base, offset)}"

    suffix = load_pattern.sub(rewrite_tail_load, suffix)
    if tail_load_index != 32:
        raise SystemExit(
            f"expected 32 final-tile fragment loads, got {tail_load_index}"
        )

    # Inline DS assembly again hides async readiness from LLVM.  A conservative
    # wait before every tail WMMA keeps this rare path unambiguous while the
    # hot loop retains the progressive schedule above.
    suffix_lines = suffix.splitlines()
    scheduled_suffix: list[str] = []
    tail_waits = 0
    for line in suffix_lines:
        if line.startswith("\tv_wmma_f16_16x16x16_f16"):
            scheduled_suffix.append("\ts_waitcnt lgkmcnt(0)")
            tail_waits += 1
        scheduled_suffix.append(line)
    if tail_waits != 32:
        raise SystemExit(f"expected 32 final-tile WMMAs, got {tail_waits}")
    suffix = "\n".join(scheduled_suffix) + "\n"

    source = prefix + body + suffix
    source = replace_once(
        source,
        "\t.amdhsa_next_free_vgpr 169\n",
        "\t.amdhsa_next_free_vgpr 120\n",
        "VGPR declaration",
    )
    source = replace_once(
        source,
        "    .vgpr_count:     152\n",
        "    .vgpr_count:     120\n",
        "VGPR metadata",
    )

    registers = [int(value) for value in re.findall(r"\bv([0-9]+)\b", source)]
    registers.extend(
        int(value)
        for pair in re.findall(r"v\[([0-9]+):([0-9]+)\]", source)
        for value in pair
    )
    if max(registers) > 119:
        raise SystemExit(f"compressed image still addresses v{max(registers)}")
    if body.count("\tds_load_b128") != 32:
        raise SystemExit("steady-state body must retain 32 K32 fragment loads")
    if body.count("\tv_wmma_f16_16x16x16_f16") != 32:
        raise SystemExit("steady-state body must retain 32 WMMAs")
    if body.count("\ts_barrier") != 2:
        raise SystemExit("steady-state body must contain exactly two barriers")

    Path(sys.argv[2]).write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
