#!/usr/bin/env bash
set -euo pipefail

cd /work/asm-k32-ring
llvm=/opt/rocm/llvm/bin

"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c bp-k32-ring-hand.s -o bp-k32-ring-hand.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o bp-k32-ring-hand.out bp-k32-ring-hand.o
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input=bp-k32-ring-hand.out -output=bp-k32-ring-hand.hipfb

common=(--offload-host-only --offload-arch=gfx1151 -O3 -std=c++20
        -I../include-waveprivate -DRECORD_BLOCK_K2_RING=1
        -DRECORD_BLOCK_SLICE_MAJOR=1 -DRECORD_K_SLICES=2
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4)
hipcc "${common[@]}" \
    -Xclang -fcuda-include-gpubinary -Xclang bp-k32-ring-hand.hipfb \
    -c ../rocwmma_half_record.hip -o bp-k32-ring-hand-host.o
hipcc bp-k32-ring-hand-host.o -lrocblas -o ../bp-k32-ring-hand

"${llvm}/llvm-readelf" -n bp-k32-ring-hand.out \
    | grep -E 'group_segment_fixed_size|vgpr_count|sgpr_count|spill_count'
