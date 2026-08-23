#!/usr/bin/env bash
set -euo pipefail

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_PAIRED_A_HANDOFF=1 \
    -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4 \
    -DRECORD_SINGLE_BUFFER=1 \
    rocwmma_half_record.hip -lrocblas -o bp-paired-a-handoff

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_PAIRED_A_HANDOFF=1 \
    -DWMMA_PAH_CYCLIC_FINAL_REFILL=1 \
    -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4 \
    -DRECORD_SINGLE_BUFFER=1 \
    rocwmma_half_record.hip -lrocblas -o bp-paired-a-handoff-cyclic

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_PAIRED_A_HANDOFF=1 \
    -DWMMA_PAH_CYCLIC_PREFETCH_ONLY=1 \
    -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4 \
    -DRECORD_SINGLE_BUFFER=1 \
    rocwmma_half_record.hip -lrocblas -o bp-paired-a-handoff-cyclic-prefetch
