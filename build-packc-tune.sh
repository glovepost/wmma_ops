#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20 -Iinclude-waveprivate
        -DWMMA_PACK_C=1 -DWMMA_WAVES_MAX=16 -DRECORD_SINGLE_BUFFER=1)

for bits in 32 64 256; do
    echo "building bits=${bits}"
    hipcc "${common[@]}" -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
        -DRECORD_BITS="${bits}" rocwmma_half_record.hip -lrocblas \
        -o "half-single-packc-b${bits}"
done

for swizzle in 1 4 8 32; do
    echo "building swizzle=${swizzle}"
    hipcc "${common[@]}" -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 \
        -DRECORD_SWIZZLE="${swizzle}" rocwmma_half_record.hip -lrocblas \
        -o "half-single-packc-s${swizzle}"
done

for geom in 2x4 4x4; do
    case "${geom}" in
        2x4) wm=2; wn=4 ;;
        4x4) wm=4; wn=4 ;;
    esac
    echo "building geometry=${geom}"
    hipcc "${common[@]}" -DRECORD_WARPS_M="${wm}" -DRECORD_WARPS_N="${wn}" \
        rocwmma_half_record.hip -lrocblas -o "half-single-packc-g${geom}"
done
