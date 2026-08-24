#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=128 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_STREAM_B_BARRIER=1
        -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

build_candidate() {
    local candidate=$1
    local padding_a=$2
    local padding_b=$3
    shift 3
    local flags=("${common[@]}"
                 -DWMMA_BP_PADDING_A="${padding_a}"
                 -DWMMA_BP_PADDING_B="${padding_b}"
                 "$@")

    hipcc "${flags[@]}" rocwmma_half_record.hip -lrocblas -o "${candidate}"
    hipcc "${flags[@]}" --cuda-device-only -S rocwmma_half_record.hip \
        -o "traces/${candidate}.s"
    echo "${candidate}:"
    grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):' \
        "traces/${candidate}.s"
}

build_candidate bp-4wave-stream-p8p2 8 2
build_candidate bp-4wave-stream-p4p4 4 4
build_candidate bp-4wave-stream-p2p0 2 0
build_candidate bp-4wave-stream-p0p0 0 0

# Cross both allocation thresholds at once.  The standalone experiments showed
# that 12 KiB LDS caps the 120-VGPR refill kernel at five blocks, while the
# 144-register allocation class caps the smaller-LDS streamed kernels at five.
combined=(-DWMMA_BP_LATE_B_REFILL=1 -DWMMA_BP_BUFFER_A_PREFETCH=1)
build_candidate bp-4wave-combined-p8p2 8 2 "${combined[@]}"
build_candidate bp-4wave-combined-p4p4 4 4 "${combined[@]}"
build_candidate bp-4wave-combined-p2p0 2 0 "${combined[@]}"
build_candidate bp-4wave-combined-p0p0 0 0 "${combined[@]}"
