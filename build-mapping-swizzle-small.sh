#!/usr/bin/env bash
set -euo pipefail

cd /work

common=(-O3 --offload-arch=gfx1151 -mcumode -std=c++20
        -Ivendor-waveprivate/include -DRECORD_BLOCK_PREPACKED=1
        -DWMMA_BP_BLOCK_M=256 -DWMMA_BP_BLOCK_N=128
        -DWMMA_BP_PACK_N=1 -DWMMA_BP_PADDING_A=8 -DWMMA_BP_PADDING_B=8
        -DRECORD_WARPS_M=4 -DRECORD_WARPS_N=2
        -DRECORD_WARP_TILE_M=4 -DRECORD_WARP_TILE_N=4
        -DRECORD_SINGLE_BUFFER=1)
llvm=/opt/rocm/llvm/bin

build_candidate() {
    local swizzle=$1
    local stem="bp-mapping-swizzle-${swizzle}"
    hipcc "${common[@]}" -DWMMA_BP_SWIZZLE="${swizzle}" \
        --cuda-device-only -S tools/rocwmma_record.hip \
        -o "traces/${stem}-source.s"
    python3 tools/patch_buffer_prefetch_asm.py \
        "traces/${stem}-source.s" "traces/${stem}-soffset.s" --scalar-offset
    python3 tools/patch_progressive_commit_asm.py \
        "traces/${stem}-soffset.s" "traces/${stem}-base.s"
    python3 tools/shift_vgpr_boundary_asm.py \
        "traces/${stem}-base.s" "traces/${stem}-d2.s" 2
    python3 tools/patch_publish_wait_asm.py \
        "traces/${stem}-d2.s" "traces/${stem}.s" 0

    "${llvm}/clang" -target amdgcn-amd-amdhsa -mcpu=gfx1151 \
        -mcode-object-version=6 -c "traces/${stem}.s" \
        -o "traces/${stem}.o"
    "${llvm}/lld" -flavor gnu -m elf64_amdgpu --no-undefined -shared \
        -o "traces/${stem}.out" "traces/${stem}.o"
    "${llvm}/clang-offload-bundler" -type=o -bundle-align=4096 \
        -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1151 \
        -input=/dev/null -input="traces/${stem}.out" \
        -output="traces/${stem}.hipfb"
    hipcc --offload-host-only "${common[@]}" \
        -DWMMA_BP_SWIZZLE="${swizzle}" \
        -Xclang -fcuda-include-gpubinary -Xclang "traces/${stem}.hipfb" \
        -c tools/rocwmma_record.hip -o "traces/${stem}-host.o"
    hipcc "traces/${stem}-host.o" -lrocblas -o "${stem}"

    echo "${stem}:"
    grep -E '^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|sgpr_spill_count|vgpr_count|vgpr_spill_count):' \
        "traces/${stem}.s"
}

build_candidate 2
build_candidate 4
