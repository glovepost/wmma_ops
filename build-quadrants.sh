#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2)

for tile in 2x4 4x2; do
    case "${tile}" in
        2x4) tile_m=2; tile_n=4 ;;
        4x2) tile_m=4; tile_n=2 ;;
    esac
    for single in 0 1; do
        hipcc "${common[@]}" \
            -DRECORD_WARP_TILE_M="${tile_m}" \
            -DRECORD_WARP_TILE_N="${tile_n}" \
            -DRECORD_SINGLE_BUFFER="${single}" \
            rocwmma_half_record.hip -lrocblas \
            -o "quadrant-${tile}-s${single}"
    done
done
