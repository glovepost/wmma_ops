#!/usr/bin/env python3
"""Remap selected-kernel workgroup IDs to fixed 5x8 CU-sized supertiles."""

import argparse
from pathlib import Path


ENTRY = (
    "; %bb.0:\n"
    "\ts_load_b128 s[16:19], s[0:1], 0x18\n"
    "\ts_abs_i32 s6, s2\n"
)


def selected_mapping(tile_id: int) -> tuple[int, int]:
    """Period-16 XOR-snake mapping specialized to the 16x32 grid."""
    source_m = tile_id & 15
    source_n = (tile_id >> 4) & 15
    outer_n = tile_id >> 8
    row = 15 - source_n if outer_n else source_n
    col = (source_m ^ source_n) + 16 * outer_n
    return row, col


def target_mapping(tile_id: int, skew: bool) -> tuple[int, int]:
    """Model cu_5x8_record_mapping's exact-shape branch."""
    if tile_id >= 480:
        return 15, 511 - tile_id
    group, local = divmod(tile_id, 40)
    outer_m, logical_outer_n = divmod(group, 4)
    outer_n = 3 - logical_outer_n if outer_m & 1 else logical_outer_n
    local_n, local_m = divmod(local, 5)
    if skew:
        local_m = (local_m + local_n) % 5
    return outer_m * 5 + local_m, outer_n * 8 + local_n


def inverse_selected(row: int, col: int) -> int:
    outer_n = col >> 4
    source_n = 15 - row if outer_n else row
    source_m = (col & 15) ^ source_n
    return source_m | (source_n << 4) | (outer_n << 8)


def verify_mapping(skew: bool) -> None:
    transformed = []
    for tile_id in range(512):
        target = target_mapping(tile_id, skew)
        remapped_id = inverse_selected(*target)
        if selected_mapping(remapped_id) != target:
            raise AssertionError(f"inverse mismatch at workgroup {tile_id}")
        transformed.append(remapped_id)
    if len(set(transformed)) != 512:
        raise AssertionError("workgroup remap is not bijective")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skew", action="store_true")
    args = parser.parse_args()

    verify_mapping(args.skew)

    source = args.input.read_text()
    if source.count(ENTRY) != 1:
        raise ValueError("expected one selected-kernel entry")
    if ".Lbp_map_done:" in source:
        raise ValueError("input is already remapped")

    skew = ""
    if args.skew:
        skew = (
            "\ts_add_u32 s27, s27, s26\n"
            "\ts_cmp_ge_u32 s27, 5\n"
            "\ts_add_u32 s21, s27, -5\n"
            "\ts_cselect_b32 s27, s21, s27\n"
            "\ts_cmp_ge_u32 s27, 5\n"
            "\ts_add_u32 s21, s27, -5\n"
            "\ts_cselect_b32 s27, s21, s27\n"
        )

    # For 0 <= x < 512, floor(x / 40) == (x * 205) >> 13;
    # for 0 <= x < 40, floor(x / 5) == (x * 205) >> 10.
    # The final block computes f^-1(row, col), where f is the selected
    # 16-wide XOR-snake mapper. Feeding f^-1(g(id)) to the unmodified kernel
    # produces the desired 5x8 mapping without changing its SGPR contract.
    remap = (
        "; Fixed 16x32-grid workgroup remap: selected XOR-snake inverse of\n"
        "; the 5x8 (40-workgroup) target traversal. Hot-loop ISA is untouched.\n"
        "\ts_cmp_ge_u32 s2, 480\n"
        "\ts_cbranch_scc1 .Lbp_map_tail\n"
        "\ts_mul_i32 s21, s2, 0xcd\n"
        "\ts_lshr_b32 s21, s21, 13\n"       # group = id / 40
        "\ts_mul_i32 s22, s21, 40\n"
        "\ts_sub_u32 s22, s2, s22\n"       # local = id % 40
        "\ts_lshr_b32 s23, s21, 2\n"       # outer_m
        "\ts_and_b32 s24, s21, 3\n"        # logical outer_n
        "\ts_sub_u32 s25, 3, s24\n"
        "\ts_and_b32 s20, s23, 1\n"
        "\ts_cmp_eq_u32 s20, 0\n"
        "\ts_cselect_b32 s25, s24, s25\n"  # snake outer_n
        "\ts_mul_i32 s26, s22, 0xcd\n"
        "\ts_lshr_b32 s26, s26, 10\n"      # local_n = local / 5
        "\ts_mul_i32 s27, s26, 5\n"
        "\ts_sub_u32 s27, s22, s27\n"      # local_m = local % 5
        + skew
        + "\ts_mul_i32 s28, s23, 5\n"
        "\ts_add_u32 s28, s28, s27\n"      # row
        "\ts_lshl_b32 s29, s25, 3\n"
        "\ts_add_u32 s29, s29, s26\n"      # col
        "\ts_branch .Lbp_map_inverse\n"
        ".Lbp_map_tail:\n"
        "\ts_movk_i32 s28, 15\n"
        "\ts_sub_u32 s29, 511, s2\n"
        ".Lbp_map_inverse:\n"
        "\ts_lshr_b32 s30, s29, 4\n"       # outer = col / 16
        "\ts_sub_u32 s31, 15, s28\n"
        "\ts_cmp_eq_u32 s30, 0\n"
        "\ts_cselect_b32 s31, s28, s31\n"  # n_inner
        "\ts_and_b32 s22, s29, 15\n"
        "\ts_xor_b32 s22, s22, s31\n"      # idx_m
        "\ts_lshl_b32 s21, s31, 4\n"
        "\ts_or_b32 s22, s22, s21\n"
        "\ts_lshl_b32 s30, s30, 8\n"
        "\ts_or_b32 s2, s22, s30\n"
        ".Lbp_map_done:\n"
    )

    replacement = ENTRY.replace(
        "\ts_abs_i32 s6, s2\n", remap + "\ts_abs_i32 s6, s2\n"
    )
    source = source.replace(ENTRY, replacement, 1)

    replacements = (
        ("\t\t.amdhsa_next_free_sgpr 20\n",
         "\t\t.amdhsa_next_free_sgpr 32\n"),
        (".numbered_sgpr, 20\n", ".numbered_sgpr, 32\n"),
        ("; TotalNumSgprs: 22\n", "; TotalNumSgprs: 34\n"),
        ("; NumSGPRsForWavesPerEU: 22\n", "; NumSGPRsForWavesPerEU: 34\n"),
        ("    .sgpr_count:     22\n", "    .sgpr_count:     34\n"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"expected one resource marker: {old.strip()}")
        source = source.replace(old, new, 1)

    args.output.write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
