#!/usr/bin/env bash
set -euo pipefail

for waves in 10 12 16; do
    echo "building waves-max=${waves}"
    hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
        -Iinclude-waveprivate -DWMMA_PACK_C=1 -DWMMA_WAVES_MAX="${waves}" \
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2 -DRECORD_SINGLE_BUFFER=1 \
        rocwmma_half_record.hip -lrocblas -o "half-single-packc-w${waves}"
done
