#!/usr/bin/env bash
set -euo pipefail

for tile in 2x2 2x4 4x2; do
    case "${tile}" in
        2x2) tm=2; tn=2 ;;
        2x4) tm=2; tn=4 ;;
        4x2) tm=4; tn=2 ;;
    esac
    echo "building tile=${tile}"
    hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
        -Iinclude-waveprivate -DRECORD_WARP_TILE_M="${tm}" \
        -DRECORD_WARP_TILE_N="${tn}" -DRECORD_SWIZZLE=16 \
        rocwmma_half_record.hip -lrocblas -o "half-waveprivate-t${tile}"
done
