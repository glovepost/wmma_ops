#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_SINGLE_BUFFER=1)

for chains in 5 6 7; do
    hipcc "${common[@]}" \
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
        -DRECORD_WARP_TILE_M=2 -DRECORD_WARP_TILE_N="${chains}" \
        rocwmma_half_record.hip -lrocblas \
        -o "staircase-2x${chains}-4x2"

    hipcc "${common[@]}" \
        -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=4 \
        -DRECORD_WARP_TILE_M="${chains}" -DRECORD_WARP_TILE_N=2 \
        rocwmma_half_record.hip -lrocblas \
        -o "staircase-${chains}x2-2x4"
done
