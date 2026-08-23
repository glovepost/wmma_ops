#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DWMMA_PACK_N=1 -DWMMA_FORCE_N_MAJOR=1
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_SINGLE_BUFFER=1)

hipcc "${common[@]}" rocwmma_half_record.hip -lrocblas -o packn-base

for padding in 2 4 6 10 12 14; do
    hipcc "${common[@]}" -DWMMA_PADDING_A="${padding}" \
        rocwmma_half_record.hip -lrocblas -o "packn-pa${padding}"
    hipcc "${common[@]}" -DWMMA_PADDING_B="${padding}" \
        rocwmma_half_record.hip -lrocblas -o "packn-pb${padding}"
done

for bits in 64 256; do
    hipcc "${common[@]}" -DRECORD_BITS="${bits}" \
        rocwmma_half_record.hip -lrocblas -o "packn-bits${bits}"
done

for swizzle in 8 32; do
    hipcc "${common[@]}" -DRECORD_SWIZZLE="${swizzle}" \
        rocwmma_half_record.hip -lrocblas -o "packn-swizzle${swizzle}"
done

hipcc "${common[@]}" -DWMMA_WAVE_EAGER=1 \
    rocwmma_half_record.hip -lrocblas -o packn-eager
hipcc "${common[@]}" -DRECORD_K_SLICES=2 \
    rocwmma_half_record.hip -lrocblas -o packn-k2

hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DWMMA_PACK_N=1 -DWMMA_FORCE_N_MAJOR=1 \
    -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=4 -DRECORD_SINGLE_BUFFER=1 \
    rocwmma_half_record.hip -lrocblas -o packn-warps2x4
