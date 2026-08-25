#!/usr/bin/env bash
set -euo pipefail

cd /work

flags=(--offload-arch=gfx1151 -mcumode -O3 -ffast-math
       -mllvm -amdgpu-unroll-threshold-local=700 -std=c++20
       -Ivendor-waveprivate/include
       -DRECORD_BLOCK_K32_PUBLICATION=1 -DRECORD_K_SLICES=2
       -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
       -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
       -DRECORD_SINGLE_BUFFER=1)

mkdir -p traces
hipcc "${flags[@]}" --cuda-device-only -S tools/rocwmma_record.hip \
    -o traces/bp-block-k32-publication-source.s
python3 tools/patch_k32_publication_asm.py \
    traces/bp-block-k32-publication-source.s \
    traces/bp-block-k32-publication.s

llvm=/opt/rocm/llvm/bin
"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c traces/bp-block-k32-publication.s \
    -o traces/bp-block-k32-publication.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o traces/bp-block-k32-publication.out \
    traces/bp-block-k32-publication.o
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input=traces/bp-block-k32-publication.out \
    -output=traces/bp-block-k32-publication.hipfb
hipcc --offload-host-only "${flags[@]}" \
    -Xclang -fcuda-include-gpubinary \
    -Xclang traces/bp-block-k32-publication.hipfb \
    -c tools/rocwmma_record.hip \
    -o traces/bp-block-k32-publication-host.o
hipcc traces/bp-block-k32-publication-host.o -lrocblas \
    -o bp-block-k32-publication
grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-block-k32-publication.s
