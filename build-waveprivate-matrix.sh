#!/usr/bin/env bash
set -euo pipefail

for s in 1 2 4 8 16 32; do
    echo "building swizzle=${s}"
    hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
        -Iinclude-waveprivate -DRECORD_SWIZZLE="${s}" \
        rocwmma_half_record.hip -lrocblas -o "half-waveprivate-2x2-s${s}"
done

for geom in 4x1 1x4; do
    case "${geom}" in
        4x1) wm=4; wn=1 ;;
        1x4) wm=1; wn=4 ;;
    esac
    echo "building geometry=${geom}"
    hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
        -Iinclude-waveprivate -DRECORD_WARPS_M="${wm}" -DRECORD_WARPS_N="${wn}" \
        -DRECORD_SWIZZLE=16 rocwmma_half_record.hip -lrocblas \
        -o "half-waveprivate-${geom}-s16"
done
