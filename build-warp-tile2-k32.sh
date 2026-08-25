#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Ivendor-waveprivate/include -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=128 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DWMMA_BP_WARP_TILE_M=2 -DWMMA_BP_WARP_TILE2_REPAIR=1
        -DWMMA_BP_K_SLICES=2
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=2 -DRECORD_WARP_TILE_N=4
        -DRECORD_K_SLICES=2 -DRECORD_SINGLE_BUFFER=1)

candidate=bp-warp-tile2-k32
hipcc "${common[@]}" tools/rocwmma_record.hip -lrocblas -o "${candidate}"
hipcc "${common[@]}" --cuda-device-only -S tools/rocwmma_record.hip \
    -o "traces/${candidate}.s"

grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):' \
    "traces/${candidate}.s"
