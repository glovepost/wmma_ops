#!/usr/bin/env bash
set -euo pipefail

cd /work

# Eight waves compute 48x96 each. Every thread transfers exactly three b128
# vectors per K16 refill; late scheduling lets those stages reuse A-fragment
# registers after their final WMMAs.
flags=(--offload-arch=gfx1151 -mcumode -O3 -ffast-math
       -mllvm -amdgpu-unroll-threshold-local=700 -std=c++20
       -Ivendor-waveprivate/include
       -DRECORD_BLOCK_PREPACKED=1 -DRECORD_PADDED_BLOCK_CONTRACT=1
       -DWMMA_BP_PADDED_EDGES=1 -DWMMA_BP_WIDE_N_192=1
       -DWMMA_BP_BLOCK_M=192 -DWMMA_BP_BLOCK_N=192
       -DWMMA_BP_WARP_TILE_M=3 -DWMMA_BP_WARP_TILE_N=6
       -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
       -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
       -DRECORD_WARP_TILE_M=3 -DRECORD_WARP_TILE_N=6
       -DRECORD_SINGLE_BUFFER=1)

mkdir -p traces
hipcc "${flags[@]}" --cuda-device-only -S tools/rocwmma_record.hip \
    -o traces/bp-block-192x192-wide.s
hipcc "${flags[@]}" tools/rocwmma_record.hip -lrocblas \
    -o bp-block-192x192-wide
grep -E '^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-block-192x192-wide.s

llvm=/opt/rocm/llvm/bin
python3 tools/set_workgroup_mode_asm.py \
    traces/bp-block-192x192-wide.s traces/bp-block-192x192-wide-wgp.s
"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c traces/bp-block-192x192-wide-wgp.s \
    -o traces/bp-block-192x192-wide-wgp.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o traces/bp-block-192x192-wide-wgp.out \
    traces/bp-block-192x192-wide-wgp.o
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input=traces/bp-block-192x192-wide-wgp.out \
    -output=traces/bp-block-192x192-wide-wgp.hipfb
hipcc --offload-host-only "${flags[@]}" \
    -Xclang -fcuda-include-gpubinary \
    -Xclang traces/bp-block-192x192-wide-wgp.hipfb \
    -c tools/rocwmma_record.hip \
    -o traces/bp-block-192x192-wide-wgp-host.o
hipcc traces/bp-block-192x192-wide-wgp-host.o -lrocblas \
    -o bp-block-192x192-wide-wgp
grep -E 'workgroup_processor_mode|^    \.(group_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-block-192x192-wide-wgp.s
