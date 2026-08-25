#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Ivendor-waveprivate/include -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=128 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DWMMA_BP_WARP_TILE_M=2 -DWMMA_BP_WARP_TILE2_REPAIR=1
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=2 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

build_candidate() {
    local candidate=$1
    local prefetch_a=$2
    local prefetch_b=$3
    local flags=("${common[@]}"
                 -DWMMA_BP_PREFETCH_A0_WN="${prefetch_a}"
                 -DWMMA_BP_PREFETCH_B_WN="${prefetch_b}")

    hipcc "${flags[@]}" tools/rocwmma_record.hip -lrocblas -o "${candidate}"
    hipcc "${flags[@]}" --cuda-device-only -S tools/rocwmma_record.hip \
        -o "traces/${candidate}.s"
    echo "${candidate}:"
    grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):' \
        "traces/${candidate}.s"
}

# a0b1 is the first fully repaired mapping.  The remaining candidates move
# one or both 16-byte global refills across the four two-WMMA N steps.
build_candidate bp-warp-tile2-a0b0 0 0
build_candidate bp-warp-tile2-a0b1 0 1
build_candidate bp-warp-tile2-a0b2 0 2
build_candidate bp-warp-tile2-a0b3 0 3
build_candidate bp-warp-tile2-a1b0 1 0
