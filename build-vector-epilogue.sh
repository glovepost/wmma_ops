#!/usr/bin/env bash
set -euo pipefail

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1 \
    -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8 \
    -DWMMA_BP_VECTOR_EPILOGUE=1 \
    -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4 \
    -DRECORD_SINGLE_BUFFER=1 \
    rocwmma_half_record.hip -lrocblas -o bp-vector-epilogue
