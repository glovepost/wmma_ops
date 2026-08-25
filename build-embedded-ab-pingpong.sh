#!/usr/bin/env bash
set -euo pipefail

cd /work

flags=(--offload-arch=gfx1151 -mcumode -O3 -ffast-math
       -mllvm -amdgpu-unroll-threshold-local=700 -std=c++20
       -Ivendor-waveprivate/include
       -DRECORD_EMBEDDED_AB_PINGPONG=1
       -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
       -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
       -DRECORD_SINGLE_BUFFER=1)

mkdir -p traces
hipcc "${flags[@]}" --cuda-device-only -S tools/rocwmma_record.hip \
    -o traces/bp-embedded-ab-pingpong.s
hipcc "${flags[@]}" tools/rocwmma_record.hip -lrocblas \
    -o bp-embedded-ab-pingpong
grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-embedded-ab-pingpong.s
