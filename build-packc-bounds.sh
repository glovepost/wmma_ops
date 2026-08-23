#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20 -Iinclude-waveprivate
        -DWMMA_PACK_C=1 -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_SINGLE_BUFFER=1)

for waves_min in 1 4 8; do
    hipcc "${common[@]}" -DWMMA_WAVES_MIN="${waves_min}" -DWMMA_WAVES_MAX=16 \
        rocwmma_half_record.hip -lrocblas -o "half-single-packc-min${waves_min}"
done

hipcc "${common[@]}" -DWMMA_NO_WAVE_BOUNDS=1 rocwmma_half_record.hip -lrocblas \
    -o half-single-packc-nobounds
