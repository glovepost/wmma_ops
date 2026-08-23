#!/usr/bin/env bash
set -euo pipefail

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DWMMA_PACK_C=1 -DWMMA_WAVES_MAX=16 \
    -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=8 -DRECORD_WARP_TILE_N=4 \
    -DRECORD_SINGLE_BUFFER=1 rocwmma_half_record.hip -lrocblas \
    -o half-single-packc-4wave-8x4

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DWMMA_PACK_C=1 -DWMMA_WAVES_MAX=16 \
    -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=1 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=8 \
    -DRECORD_SINGLE_BUFFER=1 rocwmma_half_record.hip -lrocblas \
    -o half-single-packc-4wave-4x8
