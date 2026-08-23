#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

for padding in 0 2 4 6 8 10 12 14; do
    hipcc "${common[@]}" \
        -DWMMA_BP_PADDING_A="${padding}" \
        -DWMMA_BP_PADDING_B="${padding}" \
        rocwmma_half_record.hip -lrocblas -o "blockpack-p${padding}"
done

for padding in 0 2 4 6 8; do
    hipcc "${common[@]}" -DWMMA_BP_DOUBLE_BUFFER=1 \
        -DWMMA_BP_PADDING_A="${padding}" \
        -DWMMA_BP_PADDING_B="${padding}" \
        rocwmma_half_record.hip -lrocblas -o "blockpack-double-p${padding}"
done

for padding in 0 2 4 6 8 10 12 14; do
    hipcc "${common[@]}" -DWMMA_BP_PACK_N=1 \
        -DWMMA_BP_PADDING_A="${padding}" \
        -DWMMA_BP_PADDING_B="${padding}" \
        rocwmma_half_record.hip -lrocblas -o "blockpack-n-p${padding}"
done
