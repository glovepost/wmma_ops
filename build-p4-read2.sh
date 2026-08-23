#!/usr/bin/env bash
set -euo pipefail

cd /work/asm-p4-read2
llvm=/opt/rocm/llvm/bin

"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c bp-p4-read2.s -o bp-p4-read2.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o bp-p4-read2.out bp-p4-read2.o
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input=bp-p4-read2.out \
    -output=bp-p4-read2.hipfb

common=(--offload-host-only --offload-arch=gfx1151 -O3 -std=c++20
        -I../include-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=4 -DWMMA_BP_PADDING_B=4
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)
hipcc "${common[@]}" \
    -Xclang -fcuda-include-gpubinary -Xclang bp-p4-read2.hipfb \
    -c ../rocwmma_half_record.hip -o bp-p4-read2-host.o
hipcc bp-p4-read2-host.o -lrocblas -o ../bp-p4-read2

"${llvm}/llvm-readelf" -n bp-p4-read2.out \
    | grep -E 'group_segment_fixed_size|vgpr_count|sgpr_count'
