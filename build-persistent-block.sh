#!/usr/bin/env bash
set -euo pipefail

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_PERSISTENT_BLOCK=1 \
    -DRECORD_GRID_BLOCKS=80 \
    -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4 \
    rocwmma_half_record.hip -lrocblas -o bp-persistent-block
