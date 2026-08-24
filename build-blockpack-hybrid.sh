#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=256 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

build_candidate() {
    local candidate=$1
    shift
    local flags=("${common[@]}" "$@")

    hipcc "${flags[@]}" rocwmma_half_record.hip -lrocblas -o "${candidate}"
    hipcc "${flags[@]}" --cuda-device-only -S rocwmma_half_record.hip \
        -o "traces/${candidate}.s"
    echo "${candidate}:"
    grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):' \
        "traces/${candidate}.s"
}

build_candidate bp-hybrid-a-split -DWMMA_BP_HYBRID_A_PINGPONG=1
build_candidate bp-hybrid-a-grouped \
    -DWMMA_BP_HYBRID_A_PINGPONG=1 \
    -DWMMA_BP_HYBRID_A_GROUP_LOADS=1
build_candidate bp-hybrid-a-late \
    -DWMMA_BP_HYBRID_A_PINGPONG=1 \
    -DWMMA_BP_HYBRID_A_GROUP_LOADS=1 \
    -DWMMA_BP_HYBRID_A_LATE_COMMIT=1
build_candidate bp-hybrid-b -DWMMA_BP_HYBRID_B_PINGPONG=1
