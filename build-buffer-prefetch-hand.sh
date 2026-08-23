#!/usr/bin/env bash
set -euo pipefail

cd /work
llvm=/opt/rocm/llvm/bin
mkdir -p traces

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Iinclude-waveprivate -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)

hipcc "${common[@]}" --cuda-device-only -S rocwmma_half_record.hip \
    -o traces/bp-buffer-prefetch-base.s
python3 tools/patch_buffer_prefetch_asm.py \
    traces/bp-buffer-prefetch-base.s traces/bp-buffer-prefetch-hand.s
python3 tools/patch_buffer_prefetch_asm.py \
    traces/bp-buffer-prefetch-base.s traces/bp-buffer-prefetch-soffset-hand.s \
    --scalar-offset
python3 tools/patch_buffer_prefetch_asm.py \
    traces/bp-buffer-prefetch-base.s \
    traces/bp-buffer-prefetch-soffset-no-clause-hand.s \
    --scalar-offset --no-clause
python3 tools/patch_buffer_prefetch_asm.py \
    traces/bp-buffer-prefetch-base.s \
    traces/bp-buffer-prefetch-soffset-late-hand.s \
    --scalar-offset --late-recurrence
python3 tools/patch_buffer_prefetch_asm.py \
    traces/bp-buffer-prefetch-base.s \
    traces/bp-buffer-prefetch-soffset-early-b4-hand.s \
    --scalar-offset --early-b-4
python3 tools/patch_buffer_prefetch_asm.py \
    traces/bp-buffer-prefetch-base.s \
    traces/bp-buffer-prefetch-soffset-early-b8-hand.s \
    --scalar-offset --early-b-8
for candidate in \
    bp-buffer-prefetch-hand \
    bp-buffer-prefetch-soffset-hand \
    bp-buffer-prefetch-soffset-no-clause-hand \
    bp-buffer-prefetch-soffset-late-hand \
    bp-buffer-prefetch-soffset-early-b4-hand \
    bp-buffer-prefetch-soffset-early-b8-hand; do
    "${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
        -mcode-object-version=6 -c "traces/${candidate}.s" \
        -o "traces/${candidate}.o"
    "${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
        -o "traces/${candidate}.out" "traces/${candidate}.o"
    "${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
        -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
        -input=/dev/null -input="traces/${candidate}.out" \
        -output="traces/${candidate}.hipfb"

    hipcc --offload-host-only "${common[@]}" \
        -Xclang -fcuda-include-gpubinary \
        -Xclang "traces/${candidate}.hipfb" \
        -c rocwmma_half_record.hip -o "traces/${candidate}-host.o"
    hipcc "traces/${candidate}-host.o" -lrocblas -o "${candidate}"

    echo "${candidate}:"
    "${llvm}/llvm-readelf" -n "traces/${candidate}.out" \
        | grep -E 'group_segment_fixed_size|private_segment_fixed_size|vgpr_count|sgpr_count|spill_count'
done
