#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

hipcc "${common[@]}" rocwmma_half_record.hip -lrocblas -o bp-tune-base

for pair in 0:8 8:0 8:16 16:8 16:16; do
    pa=${pair%%:*}
    pb=${pair##*:}
    hipcc "${common[@]}" -UWMMA_BP_PADDING_A -UWMMA_BP_PADDING_B \
        -DWMMA_BP_PADDING_A="${pa}" -DWMMA_BP_PADDING_B="${pb}" \
        rocwmma_half_record.hip -lrocblas -o "bp-pa${pa}-pb${pb}"
done

for mode in 1 2 3 4 5 6; do
    hipcc "${common[@]}" -DWMMA_MAPPING_MODE="${mode}" \
        rocwmma_half_record.hip -lrocblas -o "bp-map${mode}"
done

for swizzle in 2 4 8 32; do
    hipcc "${common[@]}" -DWMMA_BP_SWIZZLE="${swizzle}" \
        rocwmma_half_record.hip -lrocblas -o "bp-swizzle${swizzle}"
done

hipcc "${common[@]}" -DWMMA_BP_SPLIT_BARRIER=1 \
    rocwmma_half_record.hip -lrocblas -o bp-split-barrier
hipcc "${common[@]}" -DWMMA_BP_WAIT_AFTER_BARRIER=1 \
    rocwmma_half_record.hip -lrocblas -o bp-wait-after-barrier
hipcc "${common[@]}" -DWMMA_BP_NO_EXPLICIT_VMWAIT=1 \
    rocwmma_half_record.hip -lrocblas -o bp-no-explicit-vmwait
hipcc "${common[@]}" -DWMMA_BP_SET_PRIO=1 \
    rocwmma_half_record.hip -lrocblas -o bp-setprio

for bounds in 1:8 4:8 2:4; do
    minimum=${bounds%%:*}
    maximum=${bounds##*:}
    hipcc "${common[@]}" \
        -DWMMA_BP_WAVES_MIN="${minimum}" -DWMMA_BP_WAVES_MAX="${maximum}" \
        rocwmma_half_record.hip -lrocblas -o "bp-waves${minimum}-${maximum}"
done
