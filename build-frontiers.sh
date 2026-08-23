#!/usr/bin/env bash
set -euo pipefail

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DWMMA_PACK_C=1
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_SINGLE_BUFFER=1)

hipcc "${common[@]}" rocwmma_half_record.hip -lrocblas \
    -o frontier-baseline
hipcc "${common[@]}" -DRECORD_PREPACKED=1 rocwmma_half_record.hip \
    -lrocblas -o frontier-prepacked
for padding in 2 6 10 14; do
    hipcc "${common[@]}" -DRECORD_PREPACKED=1 \
        -DWMMA_PADDING_A="${padding}" -DWMMA_PADDING_B="${padding}" \
        rocwmma_half_record.hip -lrocblas \
        -o "frontier-prepacked-p${padding}"
done
for operand in a b; do
    case "${operand}" in
        a) prepack=(-DRECORD_PREPACK_A=1); padding_name=WMMA_PADDING_A ;;
        b) prepack=(-DRECORD_PREPACK_B=1); padding_name=WMMA_PADDING_B ;;
    esac
    for padding in 2 6 10 14; do
        hipcc "${common[@]}" "${prepack[@]}" \
            -D"${padding_name}"="${padding}" \
            rocwmma_half_record.hip -lrocblas \
            -o "frontier-prepacked-${operand}-p${padding}"
    done
done
hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_DIRECT_PREPACKED=1 -DRECORD_PREPACKED=1 \
    -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4 \
    rocwmma_half_record.hip -lrocblas -o frontier-prepacked-direct
hipcc "${common[@]}" -DWMMA_VECTOR_TRANSPOSE=1 rocwmma_half_record.hip \
    -lrocblas -o frontier-transpose
hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_PRODUCER_CONSUMER=1 \
    -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4 \
    rocwmma_half_record.hip -lrocblas -o frontier-producer
hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_PRODUCER_CONSUMER=1 \
    -DRECORD_PRODUCER_WAVES=9 -DWMMA_PC_WIDE=1 \
    -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=4 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=2 \
    rocwmma_half_record.hip -lrocblas -o frontier-producer-wide
hipcc -O3 --offload-arch=gfx1151 -mcumode -std=c++20 \
    -Iinclude-waveprivate -DRECORD_PRODUCER_CONSUMER=1 \
    -DRECORD_PRODUCER_WAVES=5 -DWMMA_PC_NARROW=1 \
    -DRECORD_WARPS_M=2 -DRECORD_WARPS_N=2 \
    -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=2 \
    rocwmma_half_record.hip -lrocblas -o frontier-producer-narrow
