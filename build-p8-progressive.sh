#!/usr/bin/env bash
set -euo pipefail

cd /work/asm-p8-progressive
llvm=/opt/rocm/llvm/bin

"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c bp-p8-progressive.s -o bp-p8-progressive.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o bp-p8-progressive.out bp-p8-progressive.o
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input=bp-p8-progressive.out -output=bp-p8-progressive.hipfb

common=(--offload-host-only --offload-arch=gfx1151 -O3 -std=c++20
        -I../include-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)
hipcc "${common[@]}" \
    -Xclang -fcuda-include-gpubinary -Xclang bp-p8-progressive.hipfb \
    -c ../rocwmma_half_record.hip -o bp-p8-progressive-host.o
hipcc bp-p8-progressive-host.o -lrocblas -o ../bp-p8-progressive

"${llvm}/llvm-readelf" -n bp-p8-progressive.out \
    | grep -E 'group_segment_fixed_size|vgpr_count|sgpr_count'
