#!/usr/bin/env bash
set -euo pipefail

cd /work

flags=(--offload-arch=gfx1151 -mcumode -O3 -ffast-math -std=c++20
       -Ivendor-waveprivate/include
       -DRECORD_COMPACT_XOR_PINGPONG=1
       -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
       -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
       -DRECORD_SINGLE_BUFFER=1)

mkdir -p traces
python3 tools/verify_compact_xor_layout.py
hipcc "${flags[@]}" --cuda-device-only -S tools/rocwmma_record.hip \
    -o traces/bp-compact-xor-pingpong-source.s
hipcc "${flags[@]}" tools/rocwmma_record.hip -lrocblas \
    -o bp-compact-xor-pingpong-source
grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-compact-xor-pingpong-source.s

python3 tools/patch_compact_xor_pingpong_asm.py \
    traces/bp-compact-xor-pingpong-source.s \
    traces/bp-compact-xor-pingpong-hand.s
llvm=/opt/rocm/llvm/bin
"${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
    -mcode-object-version=6 -c traces/bp-compact-xor-pingpong-hand.s \
    -o traces/bp-compact-xor-pingpong-hand.o
"${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -o traces/bp-compact-xor-pingpong-hand.out \
    traces/bp-compact-xor-pingpong-hand.o
"${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
    -input=/dev/null -input=traces/bp-compact-xor-pingpong-hand.out \
    -output=traces/bp-compact-xor-pingpong-hand.hipfb
hipcc --offload-host-only "${flags[@]}" \
    -Xclang -fcuda-include-gpubinary \
    -Xclang traces/bp-compact-xor-pingpong-hand.hipfb \
    -c tools/rocwmma_record.hip \
    -o traces/bp-compact-xor-pingpong-hand-host.o
hipcc traces/bp-compact-xor-pingpong-hand-host.o -lrocblas \
    -o bp-compact-xor-pingpong-hand
grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|vgpr_count):' \
    traces/bp-compact-xor-pingpong-hand.s
