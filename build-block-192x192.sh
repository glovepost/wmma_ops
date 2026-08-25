#!/usr/bin/env bash
set -euo pipefail

cd /work

# Twelve waves compute 48x64 each. The packed M/N dimensions are rounded to
# 4224 outside timing; edge stores retain logical 4096x4096 bounds.
flags=(--offload-arch=gfx1151 -mcumode -O3 -ffast-math
       -mllvm -amdgpu-unroll-threshold-local=700 -std=c++20
       -Ivendor-waveprivate/include
       -DRECORD_BLOCK_PREPACKED=1 -DRECORD_PADDED_BLOCK_CONTRACT=1
       -DWMMA_BP_PADDED_EDGES=1
       -DWMMA_BP_BLOCK_M=192 -DWMMA_BP_BLOCK_N=192
       -DWMMA_BP_WARP_TILE_M=3
       -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
       -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=3
       -DRECORD_WARP_TILE_M=3 -DRECORD_WARP_TILE_N=4
       -DRECORD_SINGLE_BUFFER=1)

mkdir -p traces
hipcc "${flags[@]}" --cuda-device-only -S tools/rocwmma_record.hip \
    -o traces/bp-block-192x192.s
hipcc "${flags[@]}" tools/rocwmma_record.hip -lrocblas -o bp-block-192x192
grep -E '^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-block-192x192.s

# The 12-wave block leaves four CU-mode wave slots empty. Test the identical
# instruction image in WGP placement, where a second block may fit.
llvm=/opt/rocm/llvm/bin
python3 tools/set_workgroup_mode_asm.py \
    traces/bp-block-192x192.s traces/bp-block-192x192-wgp.s
"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c traces/bp-block-192x192-wgp.s \
    -o traces/bp-block-192x192-wgp.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o traces/bp-block-192x192-wgp.out traces/bp-block-192x192-wgp.o
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input=traces/bp-block-192x192-wgp.out \
    -output=traces/bp-block-192x192-wgp.hipfb
hipcc --offload-host-only "${flags[@]}" \
    -Xclang -fcuda-include-gpubinary \
    -Xclang traces/bp-block-192x192-wgp.hipfb \
    -c tools/rocwmma_record.hip -o traces/bp-block-192x192-wgp-host.o
hipcc traces/bp-block-192x192-wgp-host.o -lrocblas \
    -o bp-block-192x192-wgp
grep -E 'workgroup_processor_mode|^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-block-192x192-wgp.s
