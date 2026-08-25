#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Ivendor-waveprivate/include -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=256 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

build_candidate() {
    local name=$1
    local skew_a=$2
    local skew_b=$3
    local defines=(-DWMMA_BP_FRAGMENT_SKEW_A="${skew_a}"
                   -DWMMA_BP_FRAGMENT_SKEW_B="${skew_b}")
    hipcc "${common[@]}" "${defines[@]}" --cuda-device-only -S \
        tools/rocwmma_record.hip -o "traces/${name}.s"
    hipcc "${common[@]}" "${defines[@]}" \
        tools/rocwmma_record.hip -lrocblas -o "${name}"
    echo "${name}:"
    grep -E '^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
        "traces/${name}.s"
}

build_candidate bp-fragment-skew-control 0 0
build_candidate bp-fragment-skew-a8 8 0
build_candidate bp-fragment-skew-a16 16 0
build_candidate bp-fragment-skew-b8 0 8
build_candidate bp-fragment-skew-b16 0 16
build_candidate bp-fragment-skew-ab8 8 8
build_candidate bp-fragment-skew-ab16 16 16
