#!/usr/bin/env bash
set -euo pipefail

cd /work
llvm=/opt/rocm/llvm/bin
mkdir -p traces
assembly=bp-interleaved-pingpong-hand.s
object=traces/bp-interleaved-pingpong-hand.o
code_object=traces/bp-interleaved-pingpong-hand.out
fat_binary=traces/bp-interleaved-pingpong-hand.hipfb

"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c "${assembly}" -o "${object}"
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o "${code_object}" "${object}"
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input="${code_object}" -output="${fat_binary}"

common=(--offload-host-only --offload-arch=gfx1151 -O3 -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)
hipcc "${common[@]}" \
    -Xclang -fcuda-include-gpubinary -Xclang "${fat_binary}" \
    -c rocwmma_half_record.hip -o traces/bp-interleaved-pingpong-hand-host.o
hipcc traces/bp-interleaved-pingpong-hand-host.o -lrocblas \
    -o bp-interleaved-pingpong-hand

"${llvm}/llvm-readelf" -n "${code_object}" \
    | grep -E 'group_segment_fixed_size|private_segment_fixed_size|vgpr_count|sgpr_count|spill_count'
